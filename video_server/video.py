"""Yhteinen mp4-kirjoitus.

Sekä mock- että Wan-backend kirjoittavat ulostulonsa tämän kautta, jotta
tiedostomuoto, koodekki ja laatuasetukset ovat samat riippumatta siitä mikä
backend ajoi. Rajapinnan lupaus on mp4; sen tarkka muoto kuuluu yhteen paikkaan.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _as_uint8(frame: np.ndarray) -> np.ndarray:
    """Diffusers palauttaa float32-framet välillä 0..1, mock uint8-framet."""
    array = np.asarray(frame)
    if array.dtype == np.uint8:
        return array
    return (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)


def write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1: älä pakota resoluutiota 16:n monikertaan. Ulostulon
    # koon ei saa hiljaa muuttua siitä mitä pyynnössä pyydettiin.
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(_as_uint8(frame))
    finally:
        writer.close()
    return path


def preview_path_for(output_path: Path) -> Path:
    """Esikatselukuva valmiin videon vierelle samaan hakemistoon.

    Samassa hakemistossa siksi, että se tarjoillaan samaa staattista reittiä
    pitkin eikä vaadi omaa mounttia.
    """
    return output_path.with_name(f"{output_path.stem}-preview.png")
