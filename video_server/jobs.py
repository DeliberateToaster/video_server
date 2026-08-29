"""Jono, job-tila ja worker-loop.

Rajoite, joka on hyvä tietää laajennettaessa: job-hakemisto on prosessin
muistissa ja worker-taskeja on täsmälleen yksi. Tämä riittää yhdelle GPU:lle ja
yhdelle ajolle kerrallaan, mikä on koko palvelimen lähtökohta. Useampi GPU
tarkoittaisi useampaa workeria, mikä puolestaan tarkoittaisi että jaettu tila ei
voi enää olla pelkkä dict - siinä vaiheessa tähän tulee ulkoinen jono (Redis tai
vastaava). Sitä ennen se olisi turhaa monimutkaisuutta.

Erillistä lukkoa ei tarvita: yksi kuluttaja on poissulkeminen.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from video_server.backends.base import (
    GenerationCancelled,
    GenerationParams,
    Progress,
    VideoBackend,
)
from video_server.settings import Settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED})


class QueueFullError(RuntimeError):
    """Jono on täynnä -> 429."""


class JobConflictError(RuntimeError):
    """Työtä ei voi enää peruuttaa -> 409."""


@dataclass
class Job:
    id: str
    params: GenerationParams
    created_at: datetime
    status: JobStatus = JobStatus.QUEUED
    step: int = 0
    total_steps: int = 0
    eta_seconds: float | None = None
    phase: str = "denoising"
    output_path: Path | None = None
    preview_path: Path | None = None
    error: str | None = None
    # Luetaan executor-säikeestä, kirjoitetaan event loopista. Yksi tavallinen
    # bool riittää: luku ja kirjoitus ovat atomisia, eikä väliaikainen viive
    # peruutuksen havaitsemisessa haittaa - se havaitaan viimeistään seuraavassa
    # askeleessa.
    cancel_requested: bool = field(default=False, repr=False)

    @property
    def video_filename(self) -> str:
        return f"{self.id}.mp4"


class _EtaTracker:
    """Askelkestojen liukuva keskiarvo.

    Ensimmäistä väliä ei mitata tarkoituksella: ensimmäinen askel sisältää
    lämmittelyn (kernelien kääntö, muistin varaus) ja vääristäisi arvion
    pahasti. Naiivi kulunut/askel antaisi ensimmäisten askelten jälkeen
    moninkertaisen ETA:n.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._last: float | None = None
        self._average: float | None = None

    def observe(self) -> float | None:
        now = time.perf_counter()
        if self._last is None:
            self._last = now
            return None
        delta = now - self._last
        self._last = now
        if self._average is None:
            self._average = delta
        else:
            self._average = self._average * (1 - self._alpha) + delta * self._alpha
        return self._average


class JobStore:
    def __init__(self, settings: Settings, backend: VideoBackend) -> None:
        self._settings = settings
        self._backend = backend
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self.outputs_dir = settings.outputs_dir

    # --- Elinkaari --------------------------------------------------------

    async def start(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._load_sidecars()
        self.sweep_retention()
        self._worker = asyncio.create_task(self._worker_loop(), name="wanflash-worker")

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None

    # --- Julkinen rajapinta ----------------------------------------------

    def submit(self, params: GenerationParams) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            params=params,
            created_at=datetime.now(UTC),
            total_steps=params.num_inference_steps,
        )
        try:
            self._queue.put_nowait(job.id)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"queue is full ({self._settings.max_queue_size} jobs); try again later"
            ) from None
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self._jobs[job_id]
        if job.status in TERMINAL_STATUSES:
            raise JobConflictError(f"job is already in state {job.status.value}")
        if job.status is JobStatus.QUEUED:
            # Jonossa olevaa ei tarvitse poistaa jonosta: worker ohittaa sen.
            job.status = JobStatus.CANCELLED
            self._write_sidecar(job)
        else:
            job.cancel_requested = True
        return job

    # --- Worker -----------------------------------------------------------

    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if job is None or job.status is not JobStatus.QUEUED:
                    continue  # peruttu jonossa ollessaan
                job.status = JobStatus.RUNNING
                try:
                    output = await loop.run_in_executor(None, self._run, job, loop)
                except GenerationCancelled:
                    job.status = JobStatus.CANCELLED
                    logger.info("job %s cancelled", job.id)
                except Exception as exc:  # virhe kuuluu jobiin, ei kaada workeria
                    job.status = JobStatus.FAILED
                    job.error = f"{exc.__class__.__name__}: {exc}"
                    logger.exception("job %s failed", job.id)
                else:
                    job.output_path = output
                    job.status = JobStatus.DONE
                    job.eta_seconds = 0.0
                    logger.info("job %s done: %s", job.id, output.name)
                self._write_sidecar(job)
                self.sweep_retention()
            finally:
                self._queue.task_done()

    def _run(self, job: Job, loop: asyncio.AbstractEventLoop) -> Path:
        """Suoritetaan executor-säikeessä; ei saa koskea job-tilaan suoraan."""
        eta = _EtaTracker()

        def on_progress(progress: Progress) -> None:
            if job.cancel_requested:
                raise GenerationCancelled(job.id)
            if progress.phase == "denoising":
                average = eta.observe()
                remaining = max(0, progress.total_steps - progress.step)
                eta_seconds = None if average is None else average * remaining
            else:
                # Dekoodauksesta ei saa askelittaista tietoa, joten kestoa ei
                # voi arvioida. None on rehellisempi kuin 0.
                eta_seconds = None
            # Job-tila elää event loopissa; päivitys viedään sinne sen sijaan
            # että kirjoitettaisiin suoraan toisesta säikeestä.
            loop.call_soon_threadsafe(self._apply_progress, job, progress, eta_seconds)

        output_path = self.outputs_dir / job.video_filename
        return self._backend.generate(job.params, on_progress, output_path)

    @staticmethod
    def _apply_progress(job: Job, progress: Progress, eta_seconds: float | None) -> None:
        job.step = progress.step
        job.total_steps = progress.total_steps
        job.eta_seconds = eta_seconds
        job.phase = progress.phase
        if progress.preview_path is not None:
            job.preview_path = progress.preview_path

    # --- Pysyvyys ---------------------------------------------------------

    def _sidecar_path(self, job_id: str) -> Path:
        return self.outputs_dir / f"{job_id}.json"

    def _write_sidecar(self, job: Job) -> None:
        """Job-tila on muistissa, mp4:t levyllä. Ilman tätä uudelleenkäynnistys
        jättäisi valmiit videot orvoiksi: tiedosto on olemassa mutta
        GET /jobs/{id} vastaisi 404."""
        payload = {
            "job_id": job.id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "params": job.params.model_dump(mode="json"),
            "video": job.video_filename if job.output_path else None,
            "error": job.error,
        }
        try:
            self._sidecar_path(job.id).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("sidecar write failed for job %s", job.id, exc_info=True)

    def _load_sidecars(self) -> None:
        for path in sorted(self.outputs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job = Job(
                    id=payload["job_id"],
                    params=GenerationParams.model_validate(payload["params"]),
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    status=JobStatus(payload["status"]),
                    error=payload.get("error"),
                )
            except (OSError, KeyError, ValueError) as exc:
                logger.warning("skipping malformed sidecar %s: %s", path.name, exc)
                continue

            video = self.outputs_dir / payload["video"] if payload.get("video") else None
            if video is not None and video.exists():
                job.output_path = video
                job.step = job.total_steps = job.params.num_inference_steps
            elif job.status is JobStatus.DONE:
                # Video on siivottu retentionin toimesta; jobia ei ole enää
                # mielekästä tarjota valmiina.
                continue
            self._jobs[job.id] = job

        if self._jobs:
            logger.info("loaded %d previous jobs from disk", len(self._jobs))

    # --- Retention --------------------------------------------------------

    def sweep_retention(self) -> int:
        """Poistaa vanhat ulostulot. Palauttaa poistettujen videoiden määrän.

        Kaksi sääntöä, kumpikin valinnainen: ikä ja yhteiskoko. Pelkkä ikäraja
        päästää läpi levyn täyttymisen ruuhkapäivänä, pelkkä kokoraja säilyttää
        ikuisesti jos käyttö on vähäistä.
        """
        max_age = self._settings.retention_max_age_days
        max_total = self._settings.retention_max_total_gb
        if max_age is None and max_total is None:
            return 0

        videos = sorted(
            (p for p in self.outputs_dir.glob("*.mp4") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        removed: list[Path] = []

        if max_age is not None:
            cutoff = time.time() - max_age * 86400
            for path in list(videos):
                if path.stat().st_mtime < cutoff:
                    removed.append(path)
                    videos.remove(path)

        if max_total is not None:
            limit = max_total * 1024**3
            total = sum(p.stat().st_size for p in videos)
            for path in list(videos):  # vanhin ensin
                if total <= limit:
                    break
                total -= path.stat().st_size
                removed.append(path)
                videos.remove(path)

        for path in removed:
            job_id = path.stem
            with contextlib.suppress(OSError):
                path.unlink()
            with contextlib.suppress(OSError):
                self._sidecar_path(job_id).unlink(missing_ok=True)
            self._jobs.pop(job_id, None)

        if removed:
            logger.info("retention removed %d videos", len(removed))
        return len(removed)
