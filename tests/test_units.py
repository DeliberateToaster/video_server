"""Yksikkötestit: parametrien ratkaisu, tier-päättely, ETA ja retention."""

from __future__ import annotations

import os
import time

import pytest

from video_server.backends.mock import MockBackend
from video_server.config import PROFILES, TIERS, suggest_tier
from video_server.jobs import JobStore, _EtaTracker
from video_server.schemas import (
    GenerationRequest,
    Img2VidRequest,
    ParamError,
    Txt2VidRequest,
    resolve_params,
)
from video_server.settings import Settings

A14B = PROFILES["wan2.2-t2v-a14b"]
TI2V = PROFILES["wan2.2-ti2v-5b"]


# --- Parametrien ratkaisu -------------------------------------------------


def test_defaults_are_taken_from_profile() -> None:
    params = resolve_params(Txt2VidRequest(prompt="x"), TI2V)
    assert params.fps == TI2V.native_fps
    assert params.resolution == TI2V.resolutions[0]
    assert params.shift == TI2V.default_shift
    assert params.num_inference_steps == TI2V.default_steps


def test_explicit_values_win_over_defaults() -> None:
    params = resolve_params(
        Txt2VidRequest(prompt="x", fps=12, shift=3.5, num_inference_steps=7), TI2V
    )
    assert (params.fps, params.shift, params.num_inference_steps) == (12, 3.5, 7)


def test_seed_minus_one_is_resolved_to_concrete_value() -> None:
    params = resolve_params(Txt2VidRequest(prompt="x", seed=-1), TI2V)
    assert params.seed >= 0


def test_explicit_seed_is_preserved() -> None:
    assert resolve_params(Txt2VidRequest(prompt="x", seed=42), TI2V).seed == 42


@pytest.mark.parametrize("num_frames", [81, 5, 1, 121])
def test_valid_frame_counts_accepted(num_frames: int) -> None:
    assert resolve_params(Txt2VidRequest(prompt="x", num_frames=num_frames), TI2V)


@pytest.mark.parametrize("num_frames", [80, 50, 2])
def test_invalid_frame_counts_rejected(num_frames: int) -> None:
    with pytest.raises(ParamError, match="num_frames"):
        resolve_params(Txt2VidRequest(prompt="x", num_frames=num_frames), TI2V)


def test_frame_error_names_nearest_valid_values() -> None:
    with pytest.raises(ParamError) as excinfo:
        resolve_params(Txt2VidRequest(prompt="x", num_frames=50), TI2V)
    assert "49" in str(excinfo.value) and "53" in str(excinfo.value)


def test_unsupported_resolution_rejected_with_allowed_list() -> None:
    with pytest.raises(ParamError) as excinfo:
        resolve_params(Txt2VidRequest(prompt="x", resolution="1920x1080"), TI2V)
    assert TI2V.resolutions[0] in str(excinfo.value)


def test_guidance_2_ignored_on_single_expert_model() -> None:
    """TI2V-5B:llä on yksi asiantuntija: kenttä ei ole virhe, sillä ei vain ole
    vaikutusta."""
    params = resolve_params(Txt2VidRequest(prompt="x", guidance_scale_2=9.0), TI2V)
    assert params.guidance_scale_2 is None


def test_guidance_2_defaults_on_moe_model() -> None:
    params = resolve_params(Txt2VidRequest(prompt="x"), A14B)
    assert params.guidance_scale == A14B.default_guidance
    assert params.guidance_scale_2 == A14B.default_guidance_2


def test_guidance_2_override_on_moe_model() -> None:
    params = resolve_params(Txt2VidRequest(prompt="x", guidance_scale_2=6.5), A14B)
    assert params.guidance_scale_2 == 6.5


def test_init_image_rejected_on_t2v_only_model() -> None:
    request = Img2VidRequest(prompt="x", init_image="AAAA", resolution=A14B.resolutions[0])
    with pytest.raises(ParamError, match="i2v"):
        resolve_params(request, A14B)


def test_init_image_sets_mode() -> None:
    params = resolve_params(Img2VidRequest(prompt="x", init_image="AAAA"), TI2V)
    assert params.mode == "i2v"
    assert resolve_params(Txt2VidRequest(prompt="x"), TI2V).mode == "t2v"


def test_init_image_excluded_from_dump() -> None:
    params = resolve_params(Img2VidRequest(prompt="x", init_image="AAAA"), TI2V)
    assert "init_image" not in params.model_dump()


def test_size_parsed_from_resolution() -> None:
    params = resolve_params(GenerationRequest(resolution="832x480"), A14B)
    assert params.size == (832, 480)


# --- Tier-päättely ---------------------------------------------------------


def test_tier_high_for_large_card() -> None:
    tier, _ = suggest_tier(48.0, 128.0)
    assert tier == "high"


def test_tier_mid_for_3090_and_notes_offload_is_impossible() -> None:
    """24 GB VRAM + 32 GB RAM on kehityskoneen kokoonpano: A14B-offload ei
    mahdu, ja perustelun on sanottava se ääneen."""
    tier, reason = suggest_tier(24.0, 32.0)
    assert tier == "mid"
    assert "offload will not fit" in reason


def test_tier_mid_allows_offload_with_enough_ram() -> None:
    _, reason = suggest_tier(24.0, 64.0)
    assert "with offload" in reason


def test_tier_low_for_small_card() -> None:
    assert suggest_tier(12.0, 32.0)[0] == "low"


def test_tier_unsupported_for_tiny_card() -> None:
    assert suggest_tier(8.0, 32.0)[0] == "unsupported"


# --- ETA -------------------------------------------------------------------


def test_eta_tracker_skips_first_interval() -> None:
    """Ensimmäinen askel sisältää lämmittelyn eikä sitä saa laskea mukaan."""
    tracker = _EtaTracker()
    assert tracker.observe() is None  # ei vielä mitattavaa väliä
    time.sleep(0.02)
    first = tracker.observe()
    assert first is not None and first > 0


def test_eta_tracker_smooths_outliers() -> None:
    tracker = _EtaTracker(alpha=0.3)
    tracker.observe()
    for _ in range(3):
        time.sleep(0.01)
        tracker.observe()
    steady = tracker.observe()
    assert steady is not None and steady < 0.05


# --- Retention -------------------------------------------------------------


def _store(tmp_path, **overrides) -> JobStore:
    options = {"backend": "mock", "outputs_dir": tmp_path, "tier": "test"}
    options.update(overrides)
    settings = Settings(**options)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    return JobStore(settings, MockBackend())


def _fake_output(tmp_path, name: str, size: int = 1024, age_days: float = 0.0) -> None:
    path = tmp_path / f"{name}.mp4"
    path.write_bytes(b"\0" * size)
    (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))


def test_retention_removes_old_files(tmp_path) -> None:
    _fake_output(tmp_path, "vanha", age_days=30)
    _fake_output(tmp_path, "uusi", age_days=0)
    store = _store(tmp_path, retention_max_age_days=14, retention_max_total_gb=None)

    assert store.sweep_retention() == 1
    assert not (tmp_path / "vanha.mp4").exists()
    assert not (tmp_path / "vanha.json").exists()  # sidecar siivotaan mukana
    assert (tmp_path / "uusi.mp4").exists()


def test_retention_enforces_total_size_oldest_first(tmp_path) -> None:
    _fake_output(tmp_path, "a", size=800, age_days=3)
    _fake_output(tmp_path, "b", size=800, age_days=2)
    _fake_output(tmp_path, "c", size=800, age_days=1)
    # Raja 2 kt: kolmesta 800 tavun tiedostosta vanhimman on väistyttävä.
    store = _store(tmp_path, retention_max_age_days=None, retention_max_total_gb=2048 / 1024**3)

    assert store.sweep_retention() == 1
    assert not (tmp_path / "a.mp4").exists()
    assert (tmp_path / "b.mp4").exists() and (tmp_path / "c.mp4").exists()


def test_retention_disabled_by_default_config(tmp_path) -> None:
    _fake_output(tmp_path, "vanha", age_days=999)
    store = _store(tmp_path, retention_max_age_days=None, retention_max_total_gb=None)
    assert store.sweep_retention() == 0
    assert (tmp_path / "vanha.mp4").exists()


def test_t2v_rejected_on_i2v_only_model() -> None:
    """I2V-A14B on erillinen checkpoint joka vaatii aloituskuvan; pelkkä teksti
    ei kelpaa sille."""
    i2v_only = PROFILES["wan2.2-i2v-a14b"]
    assert i2v_only.supports_mode("i2v") and not i2v_only.supports_mode("t2v")
    with pytest.raises(ParamError, match="t2v"):
        resolve_params(Txt2VidRequest(prompt="x"), i2v_only)


# --- Tier ohjaa mallia ja lataustapaa -------------------------------------


def test_explicit_backend_is_not_overridden_by_tier() -> None:
    settings = Settings(backend="mock", tier="high")
    assert settings.resolve_backend() == "mock"


def test_auto_backend_follows_tier(monkeypatch) -> None:
    """Tier ei saa jäädä pelkäksi logiriviksi: sen on valittava malli."""
    for tier, expected in (
        ("mid", "wan2.2-ti2v-5b"),
        ("high", "wan2.2-t2v-a14b"),
        ("a14b-offload", "wan2.2-t2v-a14b"),
    ):
        monkeypatch.setattr(Settings, "resolve_tier", lambda self, t=tier: (t, "testi"))
        assert Settings(backend="auto").resolve_backend() == expected


def test_auto_backend_refuses_to_guess_without_gpu(monkeypatch) -> None:
    """Hiljainen putoaminen mockiin olisi pahin oletus: palvelin näyttäisi
    toimivan mutta tuottaisi väärennettyä videota."""
    monkeypatch.setattr(Settings, "resolve_tier", lambda self: ("cpu", "ei CUDAa"))
    with pytest.raises(ValueError, match="VIDEO_SERVER_BACKEND"):
        Settings(backend="auto").resolve_backend()


def test_tier_selects_load_mode(monkeypatch) -> None:
    for tier, expected in (("low", "quantized"), ("mid", "bf16"), ("a14b-offload", "offload")):
        monkeypatch.setattr(Settings, "resolve_tier", lambda self, t=tier: (t, "testi"))
        mode, reason = Settings(backend="auto").load_mode()
        assert mode == expected
        assert tier in reason


def test_explicit_cpu_offload_overrides_tier(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "resolve_tier", lambda self: ("low", "testi"))
    assert Settings(backend="mock", cpu_offload=True).load_mode()[0] == "offload"
    assert Settings(backend="mock", cpu_offload=False).load_mode()[0] == "bf16"


def test_every_tier_policy_points_at_a_real_profile() -> None:
    for policy in TIERS.values():
        assert policy.profile in PROFILES
        assert policy.load_mode in {"bf16", "offload", "quantized"}
