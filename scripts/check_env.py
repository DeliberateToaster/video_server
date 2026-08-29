"""Ympäristön tarkistus (Vaihe 0).

Ajo:  uv run python scripts/check_env.py

Kertoo onko kone valmis oikeaan inferenssiin ja minkä VRAM-tierin se saisi.
Tier-päättely tulee video_server.config-moduulista: sama logiikka jota palvelin
käyttää käynnistyksessä, ei kopio. Skripti toimii myös ilman GPU-riippuvuuksia:
silloin se raportoi mitä puuttuu sen sijaan että kaatuisi.

Tuloste on englanniksi, koska se on käyttäjälle näkyvää; kommentit ovat
suomeksi kuten muuallakin koodissa.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from video_server.config import GIB, suggest_tier

# Windows-konsolin oletuskoodisivu ei ole UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _line(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    _line("Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else "  <-- need 3.11 or 3.12"))
    return ok


def check_ffmpeg() -> bool:
    """imageio-ffmpeg riittää videon kirjoittamiseen; järjestelmän ffmpeg on
    kehityksen mukavuus (ffprobe ulostulojen tarkistukseen)."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        try:
            first = subprocess.run(
                [system_ffmpeg, "-version"], capture_output=True, text=True, timeout=15, check=False
            ).stdout.splitlines()[0]
            _line("ffmpeg (system)", first)
        except (subprocess.SubprocessError, IndexError):
            _line("ffmpeg (system)", f"found: {system_ffmpeg} (version unreadable)")
    else:
        _line("ffmpeg (system)", "not on PATH (optional)")

    try:
        import imageio_ffmpeg

        _line("imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_version())
        return True
    except Exception as exc:  # noqa: BLE001 - halutaan raportti, ei traceback
        _line("imageio-ffmpeg", f"MISSING ({exc.__class__.__name__}) - run: uv sync")
        return False


def system_ram_gib() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / GIB
    except Exception:  # noqa: BLE001
        return None


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        _line("torch", "MISSING - run: uv sync --extra gpu")
        return False

    _line("torch", torch.__version__)
    _line("CUDA (torch build)", torch.version.cuda or "not a CUDA build")

    if not torch.cuda.is_available():
        _line("CUDA available", "NO - check the driver and that torch is a CUDA build")
        return False

    props = torch.cuda.get_device_properties(0)
    ram = system_ram_gib()
    _line("GPU", props.name)
    _line("VRAM", f"{props.total_memory / GIB:.1f} GB")
    _line("Compute capability", f"sm_{props.major}{props.minor}")
    _line("System memory", "unknown" if ram is None else f"{ram:.1f} GB")

    # FP8-tensoriytimet vaativat sm_89 (Ada) tai uudemman. Amperella (sm_86)
    # FP8 on vain tallennusmuoto - VRAM-säästö on todellinen, nopeutus ei.
    fp8 = (props.major, props.minor) >= (8, 9)
    _line("FP8 compute", "yes" if fp8 else "no (Ampere or older): FP8 is storage only")

    tier, reason = suggest_tier(props.total_memory / GIB, ram)
    print()
    _line("Suggested tier", tier)
    _line("Reason", reason)

    # Pieni todellinen laskutoimitus: varmistaa että kernelit ajavat, ei pelkkä
    # is_available()-lippu.
    try:
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        torch.matmul(x, x)
        torch.cuda.synchronize()
        _line("bf16 matmul on GPU", "OK")
    except Exception as exc:  # noqa: BLE001
        _line("bf16 matmul on GPU", f"FAILED: {exc}")
        return False

    return tier != "unsupported"


def main() -> int:
    print("\nWanFlash: environment check\n" + "=" * 42)
    print("\nBase environment")
    ok_py = check_python()
    ok_ff = check_ffmpeg()
    print("\nGPU")
    ok_gpu = check_torch()

    print("\n" + "=" * 42)
    if ok_py and ok_ff and ok_gpu:
        print("Ready for real inference.\n")
        return 0
    if ok_py and ok_ff:
        print("Ready for the mock backend and API development.")
        print("Real inference needs the items flagged above.\n")
        return 1
    print("Base environment is incomplete; fix the items flagged above.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
