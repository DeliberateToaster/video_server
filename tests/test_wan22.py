"""Wan-backendin logiikkatestit ilman malleja ja ilman GPU:ta.

Painojen lataus ja oikea inferenssi eivät ole testattavissa CI:ssä, mutta se
osa backendistä joka päättää MITÄ pipelinelle välitetään on - ja juuri siellä
virheet ovat kalliita: väärä kutsuparametri huomataan muuten vasta kymmenen
minuutin ajon jälkeen tai, pahempaa, hiljaisena laatuvirheenä.

Pipeline korvataan tuplalla, joka tallentaa saamansa kutsuparametrit.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from video_server.backends.base import GenerationCancelled, Progress
from video_server.config import PROFILES
from video_server.schemas import Img2VidRequest, Txt2VidRequest, resolve_params
from video_server.settings import Settings

pytest.importorskip("torch", reason="Wan-backend vaatii GPU-riippuvuudet")

from video_server.backends.wan22 import Wan22Backend

TI2V = PROFILES["wan2.2-ti2v-5b"]
A14B = PROFILES["wan2.2-t2v-a14b"]


class FakeScheduler:
    def __init__(self, config: dict) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> FakeScheduler:
        return cls({**config, **kwargs})


class FakePipe:
    """Tallentaa kutsuparametrit ja ajaa callbackin joka askeleelta."""

    def __init__(self, flow_shift: float = 5.0, temporal: int = 4) -> None:
        self.scheduler = FakeScheduler({"flow_shift": flow_shift})
        self.vae_scale_factor_temporal = temporal
        self.config = {"boundary_ratio": None, "expand_timesteps": True}
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs["callback_on_step_end"]
        for index in range(kwargs["num_inference_steps"]):
            result = callback(self, index, 0, {})
            assert isinstance(result, dict), "pipeline tekee paluuarvolle .pop()"
        frames = np.zeros(
            (kwargs["num_frames"], kwargs["height"], kwargs["width"], 3), dtype=np.uint8
        )
        return SimpleNamespace(frames=[frames])


def _backend(profile=TI2V, **setting_overrides) -> Wan22Backend:
    options = {"backend": profile.name, "tier": "test"}
    options.update(setting_overrides)
    settings = Settings(**options)
    # Rakennetaan kuten registry: profiili tulee asetusten läpi, jotta
    # konfiguraation ohitukset ovat mukana samalla tavalla kuin ajossa.
    backend = Wan22Backend(profile=settings.profile(), settings=settings)
    backend._pipe = FakePipe()
    backend._scheduler_config = dict(backend._pipe.scheduler.config)
    return backend


def _png_base64(size: tuple[int, int] = (64, 32)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# --- Kutsuparametrit ------------------------------------------------------


def test_single_expert_never_receives_guidance_scale_2(tmp_path) -> None:
    """TI2V-5B:n boundary_ratio on None, jolloin diffusers NOSTAA ValueErrorin
    jos guidance_scale_2 annetaan. Sen ei siis saa päätyä kutsuun edes silloin
    kun asiakas lähetti sen."""
    backend = _backend(TI2V)
    params = resolve_params(
        Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=2, guidance_scale_2=7.0),
        TI2V,
    )
    backend.generate(params, lambda p: None, tmp_path / "out.mp4")

    assert "guidance_scale_2" not in backend._pipe.calls[0]


def test_moe_model_passes_both_guidance_values(tmp_path) -> None:
    backend = _backend(A14B)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=2), A14B)
    backend.generate(params, lambda p: None, tmp_path / "out.mp4")

    call = backend._pipe.calls[0]
    assert call["guidance_scale"] == A14B.default_guidance
    assert call["guidance_scale_2"] == A14B.default_guidance_2


def test_resolution_is_split_into_width_and_height(tmp_path) -> None:
    backend = _backend(TI2V)
    params = resolve_params(
        Txt2VidRequest(prompt="x", resolution="704x1280", num_frames=5, num_inference_steps=2),
        TI2V,
    )
    backend.generate(params, lambda p: None, tmp_path / "out.mp4")

    call = backend._pipe.calls[0]
    assert (call["width"], call["height"]) == (704, 1280)


def test_empty_negative_prompt_is_passed_as_none(tmp_path) -> None:
    backend = _backend(TI2V)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=2), TI2V)
    backend.generate(params, lambda p: None, tmp_path / "out.mp4")

    assert backend._pipe.calls[0]["negative_prompt"] is None


# --- Shift ja scheduler ---------------------------------------------------


def test_shift_rebuilds_scheduler_without_compounding(tmp_path) -> None:
    """Scheduler rakennetaan aina alkuperäisestä configista. Jos lähtökohtana
    olisi edellinen scheduler, arvo kumuloituisi pyyntöjen yli."""
    backend = _backend(TI2V)
    for shift in (3.0, 7.0):
        params = resolve_params(
            Txt2VidRequest(prompt="x", shift=shift, num_frames=5, num_inference_steps=2),
            TI2V,
        )
        backend.generate(params, lambda p: None, tmp_path / "out.mp4")
        assert backend._pipe.scheduler.config["flow_shift"] == shift

    assert backend._scheduler_config["flow_shift"] == 5.0  # alkuperäinen ennallaan


# --- Etenemä ja peruutus --------------------------------------------------


def test_progress_is_reported_one_based_for_every_step(tmp_path) -> None:
    backend = _backend(TI2V)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=4), TI2V)
    seen: list[Progress] = []
    backend.generate(params, seen.append, tmp_path / "out.mp4")

    # diffusers indeksoi nollasta, rajapinta raportoi ykkösestä.
    denoising = [p for p in seen if p.phase == "denoising"]
    assert [p.step for p in denoising] == [1, 2, 3, 4]
    assert {p.total_steps for p in seen} == {4}


def test_decoding_phase_is_reported_after_last_step(tmp_path) -> None:
    """VAE-dekoodaus vie minuutteja eikä tuota callbackeja. Ilman erillistä
    vaihetta job näyttäisi tilaa n/n kesken ajon."""
    backend = _backend(TI2V)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=3), TI2V)
    seen: list[Progress] = []
    backend.generate(params, seen.append, tmp_path / "out.mp4")

    assert [p.phase for p in seen] == ["denoising"] * 3 + ["decoding"]


def test_cancellation_propagates_out_of_the_pipeline(tmp_path) -> None:
    """Peruutus toteutetaan nostamalla callbackissa. Backend ei saa napata
    poikkeusta, muuten peruutus jäisi huomaamatta."""
    backend = _backend(TI2V)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=10), TI2V)

    def cancel_at_third(progress: Progress) -> None:
        if progress.step == 3:
            raise GenerationCancelled("test")

    with pytest.raises(GenerationCancelled):
        backend.generate(params, cancel_at_third, tmp_path / "out.mp4")


def test_video_is_written_on_success(tmp_path) -> None:
    backend = _backend(TI2V)
    params = resolve_params(Txt2VidRequest(prompt="x", num_frames=5, num_inference_steps=2), TI2V)
    output = tmp_path / "out.mp4"
    assert backend.generate(params, lambda p: None, output) == output
    assert output.stat().st_size > 0


# --- Aloituskuvan dekoodaus -----------------------------------------------


def test_init_image_is_decoded_and_resized() -> None:
    image = Wan22Backend._decode_image(_png_base64((64, 32)), 128, 64)
    assert image.size == (128, 64)
    assert image.mode == "RGB"


def test_init_image_accepts_data_uri_prefix() -> None:
    """Speksi sanoo ilman prefiksiä, mutta sen kanssa lähettäminen on yleinen
    asiakasvirhe eikä hylkäys hyödyttäisi ketään."""
    payload = "data:image/png;base64," + _png_base64()
    assert Wan22Backend._decode_image(payload, 32, 32).size == (32, 32)


def test_invalid_base64_gives_clear_error() -> None:
    with pytest.raises(ValueError, match="base64"):
        Wan22Backend._decode_image("ei tämä ole base64!!", 32, 32)


def test_valid_base64_that_is_not_an_image_gives_clear_error() -> None:
    payload = base64.b64encode(b"tama ei ole kuva").decode()
    with pytest.raises(ValueError, match="image"):
        Wan22Backend._decode_image(payload, 32, 32)


def test_i2v_request_passes_image_to_pipeline(tmp_path) -> None:
    backend = _backend(TI2V)
    params = resolve_params(
        Img2VidRequest(prompt="x", init_image=_png_base64(), num_frames=5, num_inference_steps=2),
        TI2V,
    )
    # TI2V-5B tukee molempia suuntia, joten I2V-pipeline rakennettaisiin
    # komponenteista. Testissä primääri tuplaa riittää.
    backend._i2v_pipe = backend._pipe
    backend.generate(params, lambda p: None, tmp_path / "out.mp4")

    assert backend._pipe.calls[0]["image"].size == params.size


# --- Profiilin synkronointi checkpointista --------------------------------


def test_checkpoint_overrides_profile_defaults() -> None:
    backend = _backend(TI2V)
    backend._pipe = FakePipe(flow_shift=3.0, temporal=8)
    backend._sync_profile_from_checkpoint(backend._pipe)

    assert backend.profile.frame_multiple == 8
    assert backend.profile.default_shift == 3.0


def test_explicit_shift_setting_survives_checkpoint_sync() -> None:
    """Käyttäjän eksplisiittistä asetusta ei saa ylikirjoittaa checkpointin
    oletuksella."""
    backend = _backend(TI2V, shift=9.0)
    backend._pipe = FakePipe(flow_shift=3.0)
    backend._sync_profile_from_checkpoint(backend._pipe)

    assert backend.profile.default_shift == 9.0


# --- Offload-päätös -------------------------------------------------------


def test_explicit_offload_setting_wins_over_tier() -> None:
    assert _backend(TI2V, cpu_offload=True)._resolve_load_mode()[0] == "offload"
    assert _backend(A14B, cpu_offload=False)._resolve_load_mode()[0] == "bf16"


def test_insufficient_vram_falls_back_to_offload(monkeypatch) -> None:
    """Turvaverkko: jos kortti ei riitä profiilille, siirrytään offloadiin sen
    sijaan että ajo kaatuisi OOM:iin vasta minuuttien päästä."""
    backend = _backend(A14B)  # vaatii 30 GB
    monkeypatch.setattr(backend, "_vram_gib", lambda: 24.0)

    mode, reason = backend._resolve_load_mode()
    assert mode == "offload"
    assert "is below" in reason


def test_sufficient_vram_keeps_bf16(monkeypatch) -> None:
    backend = _backend(A14B)
    monkeypatch.setattr(backend, "_vram_gib", lambda: 48.0)
    assert backend._resolve_load_mode()[0] == "bf16"
