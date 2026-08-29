"""Ympäristön tarkistus (Vaihe 0).

Ajo:  uv run python scripts/check_env.py

Kertoo onko kone valmis oikeaan inferenssiin ja minkä VRAM-tierin se saisi.
Tier-päättely tulee video_server.config-moduulista: sama logiikka jota
palvelin käyttää käynnistyksessä, ei kopio. Skripti toimii myös ilman GPU-riippuvuuksia:
silloin se raportoi mitä puuttuu sen sijaan että kaatuisi.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from video_server.config import GIB, suggest_tier

# Windows-konsolin oletuskoodisivu ei ole UTF-8, ja raportissa on ääkkösiä.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _line(label: str, value: str) -> None:
    print(f"  {label:<22} {value}")


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 11) <= (v.major, v.minor) < (3, 13)
    _line(
        "Python", f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else "  <-- vaaditaan 3.11 tai 3.12")
    )
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
            _line("ffmpeg (järjestelmä)", first)
        except (subprocess.SubprocessError, IndexError):
            _line("ffmpeg (järjestelmä)", f"löytyi: {system_ffmpeg} (versio ei luettavissa)")
    else:
        _line("ffmpeg (järjestelmä)", "ei löydy PATH:sta (ei pakollinen)")

    try:
        import imageio_ffmpeg

        _line("imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_version())
        return True
    except Exception as exc:  # noqa: BLE001 - halutaan raportti, ei traceback
        _line("imageio-ffmpeg", f"PUUTTUU ({exc.__class__.__name__}) - aja: uv sync")
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
        _line("torch", "PUUTTUU - aja: uv sync --extra gpu")
        return False

    _line("torch", torch.__version__)
    _line("CUDA (torch build)", torch.version.cuda or "ei CUDA-buildi")

    if not torch.cuda.is_available():
        _line("CUDA saatavilla", "EI - tarkista ajuri ja että torch on CUDA-buildi")
        return False

    props = torch.cuda.get_device_properties(0)
    ram = system_ram_gib()
    _line("GPU", props.name)
    _line("VRAM", f"{props.total_memory / GIB:.1f} GB")
    _line("Compute capability", f"sm_{props.major}{props.minor}")
    _line("Järjestelmämuisti", "tuntematon" if ram is None else f"{ram:.1f} GB")

    # FP8-tensoriytimet vaativat sm_89 (Ada) tai uudemman. Amperella (sm_86)
    # FP8 on vain tallennusmuoto - VRAM-säästö on todellinen, nopeutus ei.
    fp8 = (props.major, props.minor) >= (8, 9)
    _line("FP8-laskenta", "kyllä" if fp8 else "ei (Ampere tai vanhempi): FP8 vain tallennusmuotona")

    tier, reason = suggest_tier(props.total_memory / GIB, ram)
    print()
    _line("Ehdotettu tier", tier)
    _line("Perustelu", reason)

    # Pieni todellinen laskutoimitus: varmistaa että kernelit ajavat, ei pelkkä
    # is_available()-lippu.
    try:
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        torch.matmul(x, x)
        torch.cuda.synchronize()
        _line("bf16-matmul GPU:lla", "OK")
    except Exception as exc:  # noqa: BLE001
        _line("bf16-matmul GPU:lla", f"EPÄONNISTUI: {exc}")
        return False

    return tier != "unsupported"


def main() -> int:
    print("\nWanFlash: ympäristön tarkistus\n" + "=" * 42)
    print("\nPerusympäristö")
    ok_py = check_python()
    ok_ff = check_ffmpeg()
    print("\nGPU")
    ok_gpu = check_torch()

    print("\n" + "=" * 42)
    if ok_py and ok_ff and ok_gpu:
        print("Valmis oikeaan inferenssiin (Vaihe 2).\n")
        return 0
    if ok_py and ok_ff:
        print("Valmis mock-backendiin ja rajapintakehitykseen (Vaihe 1).")
        print("Oikea inferenssi vaatii yllä merkityt korjaukset.\n")
        return 1
    print("Perusympäristö on puutteellinen; korjaa yllä merkityt kohdat.\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
