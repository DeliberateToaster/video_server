"""FastAPI-app: reitit ja elinkaari.

Reittikerros on ohut tarkoituksella. Se tekee kolme asiaa: muuntaa pyynnön
parametreiksi, antaa työn jonolle ja muuntaa jobin vastaukseksi. Kaikki
mallikohtainen on backendin ja profiilin takana, kaikki ajastus jonon.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.staticfiles import StaticFiles

from video_server.backends import registry
from video_server.jobs import Job, JobConflictError, JobStatus, JobStore, QueueFullError
from video_server.schemas import (
    AcceptedResponse,
    ConstraintsResponse,
    GenerationRequest,
    HealthResponse,
    Img2VidRequest,
    JobResponse,
    ModelsResponse,
    ParamError,
    ProgressResponse,
    Txt2VidRequest,
    resolve_params,
)
from video_server.settings import Settings, get_settings

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"

router = APIRouter(prefix=API_PREFIX)


@dataclass
class BackendState:
    """Mallin lataus kestää minuutteja, joten 'palvelin vastaa' ja 'palvelin voi
    ottaa työtä vastaan' ovat eri asioita."""

    status: str = "loading"
    detail: str | None = None


# --- Apurit ---------------------------------------------------------------


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _require_ready(request: Request) -> None:
    state: BackendState = request.app.state.backend_state
    if state.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.detail or f"backend ei ole valmis (tila: {state.status})",
        )


def _to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=ProgressResponse(
            step=job.step,
            total_steps=job.total_steps,
            eta_seconds=job.eta_seconds,
            phase=job.phase,
        ),
        params=job.params.model_dump(mode="json"),
        created_at=job.created_at.isoformat(),
        video_url=f"/outputs/{job.video_filename}" if job.status is JobStatus.DONE else None,
        preview_url=f"/outputs/{job.preview_path.name}" if job.preview_path else None,
        error=job.error,
    )


def _require_mode(request: Request, mode: str) -> None:
    """Kaikki mallit eivät tue kumpaakin suuntaa: A14B on erikseen T2V- ja
    I2V-checkpointeina. Vastaus on 501 eikä 400, koska pyyntö on sinänsä
    kelvollinen - palvelin vain ei tarjoa tätä toimintoa tällä mallilla."""
    profile = request.app.state.backend.profile
    if not profile.supports_mode(mode):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"aktiivinen malli {profile.name} ei tue {mode}-generointia",
        )


def _submit(request: Request, payload: GenerationRequest) -> AcceptedResponse:
    _require_ready(request)
    backend = request.app.state.backend
    try:
        params = resolve_params(payload, backend.profile)
    except ParamError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None
    try:
        job = _store(request).submit(params)
    except QueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        ) from None
    return AcceptedResponse(job_id=job.id)


# --- Reitit ---------------------------------------------------------------


@router.post("/txt2vid", status_code=status.HTTP_202_ACCEPTED)
def txt2vid(request: Request, payload: Txt2VidRequest) -> AcceptedResponse:
    _require_mode(request, "t2v")
    return _submit(request, payload)


@router.post("/img2vid", status_code=status.HTTP_202_ACCEPTED)
def img2vid(request: Request, payload: Img2VidRequest) -> AcceptedResponse:
    _require_mode(request, "i2v")
    return _submit(request, payload)


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> JobResponse:
    try:
        return _to_response(_store(request).get(job_id))
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"tuntematon job {job_id}"
        ) from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> JobResponse:
    try:
        return _to_response(_store(request).cancel(job_id))
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"tuntematon job {job_id}"
        ) from None
    except JobConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None


@router.get("/models")
def list_models(request: Request) -> ModelsResponse:
    profile = request.app.state.backend.profile
    return ModelsResponse(
        active=profile.name,
        available=registry.available(),
        tier=request.app.state.tier,
        constraints=ConstraintsResponse(
            resolutions=list(profile.resolutions),
            frame_multiple=profile.frame_multiple,
            native_fps=profile.native_fps,
            supports_i2v=profile.supports_i2v,
        ),
    )


@router.get("/health")
def health(request: Request) -> HealthResponse:
    state: BackendState = request.app.state.backend_state
    return HealthResponse(
        status=state.status,
        backend=request.app.state.backend.profile.name,
        detail=state.detail,
    )


# --- Elinkaari ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    tier, reason = settings.resolve_tier()
    logger.info("tier=%s (%s)", tier, reason)

    backend = registry.create(settings)
    store = JobStore(settings, backend)
    state = BackendState()

    app.state.backend = backend
    app.state.store = store
    app.state.backend_state = state
    app.state.tier = tier

    await store.start()

    async def load_backend() -> None:
        try:
            await asyncio.to_thread(backend.load)
        except Exception as exc:  # lataus epäonnistuu -> /health kertoo miksi
            state.status = "error"
            state.detail = f"{exc.__class__.__name__}: {exc}"
            logger.exception("backendin lataus epäonnistui")
        else:
            state.status = "ready"
            logger.info("backend %s ladattu", backend.profile.name)

    load_task = asyncio.create_task(load_backend(), name="backend-load")

    try:
        yield
    finally:
        load_task.cancel()
        await store.stop()


async def require_api_key(request: Request, call_next):
    """Valinnainen API-avain.

    Oletuksena pois päältä: speksin v1-rajaus on "ei autentikointia, lisätään
    erikseen jos palvelin altistetaan verkkoon". Tämä on se erikseen-lisäys, ja
    se on kytkin eikä pakko - ilman `api_key`-asetusta käyttäytyminen ei muutu.

    `/api/v1/health` jätetään auki, jotta monitorointi ja käynnistyksen
    seuranta toimivat ilman avainta; se ei paljasta muuta kuin latauksen tilan.
    """
    key: str | None = request.app.state.settings.api_key
    if key:
        path = request.url.path
        guarded = path.startswith(("/api/v1", "/outputs")) and not path.endswith("/health")
        if guarded:
            provided = request.headers.get("x-api-key") or ""
            # compare_digest: vakioaikainen vertailu, ei vuoda avainta ajastuksella.
            if not secrets.compare_digest(provided, key):
                return Response(
                    content="puuttuva tai virheellinen X-API-Key",
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    media_type="text/plain; charset=utf-8",
                )
    return await call_next(request)


async def limit_body_size(request: Request, call_next):
    """img2vid ottaa kuvan base64:nä rungossa, joten runko on rajattava.
    Content-Length riittää tähän: se on ainoa mitä voi tarkistaa ennen kuin
    runko on luettu muistiin."""
    max_bytes: int = request.app.state.settings.max_request_bytes
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > max_bytes:
        return Response(
            content=f"pyyntörunko on liian suuri (max {max_bytes} tavua)",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            media_type="text/plain; charset=utf-8",
        )
    return await call_next(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Tehdasfunktio, jotta testit voivat luoda sovelluksen omalla
    konfiguraatiollaan. Staattinen /outputs-mount tarvitsee hakemiston jo
    luontihetkellä, joten sitä ei voi kiinnittää moduulitasolla."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    app = FastAPI(title="WanFlash", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(limit_body_size)
    app.middleware("http")(require_api_key)
    app.include_router(router)

    # Staattinen tarjoilu valmiille videoille. Ei autentikointia (ks. spec,
    # rajaukset v1) - tiedostonimi on job-UUID, mikä estää nimien arvaamisen
    # mutta ei ole pääsynhallintaa.
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=settings.outputs_dir), name="outputs")
    return app


app = create_app()
