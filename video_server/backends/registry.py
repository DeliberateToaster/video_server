"""Backend-rekisteri: nimi -> tehdasfunktio.

Rekisteröity backend on sellainen, joka on oikeasti ajettavissa.

Wan-backendit rekisteröidään laiskasti: tehdasfunktio tuo `wan22`-moduulin
sisään vasta kutsuttaessa. Näin perusasennus (uv sync ilman --extra gpu) ei
kaadu importtiin, ja mock-backendillä ajettava testisetti pysyy täysin
GPU-riippumattomana.
"""

from __future__ import annotations

from collections.abc import Callable

from video_server.backends.base import VideoBackend
from video_server.backends.mock import MockBackend
from video_server.config import PROFILES
from video_server.settings import Settings

BackendFactory = Callable[[Settings], VideoBackend]

# Profiilit, jotka Wan-backend osaa ajaa.
WAN_PROFILES = ("wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.2-i2v-a14b")


def _wan_backend(settings: Settings) -> VideoBackend:
    from video_server.backends.wan22 import Wan22Backend

    return Wan22Backend(profile=settings.profile(), settings=settings)


_REGISTRY: dict[str, BackendFactory] = {
    "mock": lambda s: MockBackend(
        profile=s.profile(),
        step_seconds=s.mock_step_seconds,
        preview_every_n_steps=s.preview_every_n_steps,
    ),
}
for _name in WAN_PROFILES:
    assert _name in PROFILES, _name
    _REGISTRY[_name] = _wan_backend


def register(name: str, factory: BackendFactory) -> None:
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def create(settings: Settings) -> VideoBackend:
    # resolve_backend hoitaa auto-valinnan tierin kautta.
    name = settings.resolve_backend()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(available())
        raise ValueError(f"tuntematon backend {name!r}; rekisteröidyt: {known}") from None
    return factory(settings)
