"""GPU:ton testibackend.

Simuloi oikean backendin käyttäytymisen - askeleet, etenemiskutsut, vaiheet,
esikatselukuvat, peruutuksen ja mp4-ulostulon - ilman mallia ja ilman torchia.
Tämän ansiosta koko rajapinta, jono, peruutus ja retention ovat kehitettävissä
ja testattavissa sekunneissa. CI ajaa tätä vasten.

Ulostulo on oikea, toistettava mp4: liikkuva palkki värjätyllä taustalla, jonka
sävy määräytyy siemenestä. Näin video_url osoittaa tiedostoon, jonka voi
oikeasti avata, eikä testien tarvitse teeskennellä.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from video_server.backends.base import GenerationParams, Progress, VideoBackend
from video_server.config import PROFILES, ModelProfile
from video_server.video import preview_path_for, write_video


class MockBackend(VideoBackend):
    def __init__(
        self,
        profile: ModelProfile | None = None,
        step_seconds: float = 0.15,
        preview_every_n_steps: int = 0,
    ) -> None:
        self._profile = profile or PROFILES["mock"]
        self._step_seconds = step_seconds
        self._preview_every = preview_every_n_steps
        self._loaded = False

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def load(self) -> None:
        # Oikea backend lataa tässä kymmeniä gigatavuja; mock vain merkitsee
        # itsensä valmiiksi, jotta käynnistyspolku on sama molemmilla.
        self._loaded = True

    def generate(
        self,
        params: GenerationParams,
        on_progress: Callable[[Progress], None],
        output_path: Path,
    ) -> Path:
        if not self._loaded:
            raise RuntimeError("MockBackend.load() was not called")

        width, height = params.size
        hue = int(np.random.default_rng(params.seed).integers(0, 3))
        steps = params.num_inference_steps

        for step in range(1, steps + 1):
            if self._step_seconds:
                time.sleep(self._step_seconds)
            preview = None
            if self._preview_every and step % self._preview_every == 0:
                preview = self._write_preview(params, width, height, hue, step, steps, output_path)
            # Peruutus kulkee tämän kutsun kautta: jos jono on pyytänyt
            # peruutusta, callback nostaa GenerationCancelled emmekä nappaa sitä.
            on_progress(Progress(step=step, total_steps=steps, preview_path=preview))

        # Oikealla mallilla enkoodausta edeltää VAE-dekoodaus, joka vie
        # minuutteja. Mock raportoi saman vaiheen, jotta rajapinnan
        # käyttäytyminen on sama molemmilla backendeillä.
        on_progress(Progress(step=steps, total_steps=steps, phase="decoding"))

        frames = (
            self._frame(index, params.num_frames, width, height, hue)
            for index in range(params.num_frames)
        )
        return write_video(frames, output_path, params.fps)

    def _write_preview(
        self,
        params: GenerationParams,
        width: int,
        height: int,
        hue: int,
        step: int,
        steps: int,
        output_path: Path,
    ) -> Path:
        # Frame etenee askelten mukana, jotta peräkkäiset esikatselut eroavat
        # toisistaan - muuten testi ei erottaisi tuoretta kuvaa vanhasta.
        index = int((params.num_frames - 1) * step / max(1, steps))
        path = preview_path_for(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._frame(index, params.num_frames, width, height, hue)).save(path)
        return path

    @staticmethod
    def _frame(index: int, total: int, width: int, height: int, hue: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        gradient = np.linspace(30, 200, width, dtype=np.uint8)
        frame[:, :, hue] = gradient[None, :]

        # Liikkuva valkoinen palkki tekee liikkeen näkyväksi, jolloin
        # silmämääräinen tarkistus kertoo heti onko fps ja frame-määrä järkevä.
        bar_width = max(2, width // 20)
        position = int((width - bar_width) * index / max(1, total - 1))
        frame[:, position : position + bar_width, :] = 255
        return frame
