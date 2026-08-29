"""Download the selected profile weights from HuggingFace into the local cache.

Usage:
    uv run python scripts/download_model.py                    # default profile
    uv run python scripts/download_model.py wan2.2-t2v-a14b
    uv run python scripts/download_model.py --check            # already on disk?

Lataus on tarkoituksella eksplisiittinen eikä tapahdu palvelimen
käynnistyksessä: kymmenien gigatavujen hiljainen lataus ensimmäisen API-kutsun
yhteydessä on huono oletus. Palvelin tarkistaa painojen olemassaolon ja kertoo
tämän komennon jos ne puuttuvat.
"""

from __future__ import annotations

import argparse
import sys
import time

from video_server.config import DEFAULT_PROFILE, PROFILES

DOWNLOADABLE = sorted(name for name, profile in PROFILES.items() if profile.repo_id)


def download_with_retries(repo_id: str, attempts: int = 6) -> str:
    """Lataa repon ja yrittää uudelleen jos siirto katkeaa.

    Kymmenien gigatavujen lataus kestää minuutteja tai tunteja, ja HuggingFacen
    Xet-siirto katkeaa satunnaiseen verkkovirheeseen (CAS Client Error). Ilman
    uudelleenyritystä yksi katkos hukkaa koko latauksen käyttäjän kannalta.

    snapshot_download jatkaa keskeneräisistä tiedostoista, joten uudelleenyritys
    ei aloita alusta vaan jatkaa siitä mihin edellinen jäi.
    """
    from huggingface_hub import snapshot_download

    delay = 5
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(repo_id)
        except (RuntimeError, OSError) as exc:
            if attempt == attempts:
                raise
            print(f"\nDownload interrupted: {exc.__class__.__name__}: {exc}")
            print(f"Attempt {attempt}/{attempts}. Resuming in {delay} s from where")
            print("the download stopped (completed files are kept).\n")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "profile",
        nargs="?",
        default=DEFAULT_PROFILE,
        choices=DOWNLOADABLE,
        help=f"model profile to download (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only check whether the model is already cached, do not download",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is missing. Install GPU dependencies: uv sync --extra gpu",
            file=sys.stderr,
        )
        return 2

    if args.check:
        try:
            path = snapshot_download(profile.repo_id, local_files_only=True)
        except OSError as exc:  # LocalEntryNotFoundError periytyy OSErrorista
            print(f"{profile.name}: NOT in the local cache ({exc.__class__.__name__})")
            return 1
        print(f"{profile.name}: found in {path}")
        return 0

    print(f"Downloading {profile.name} ({profile.repo_id})")
    print("This is a multi-gigabyte download and may take a long time.")
    path = download_with_retries(profile.repo_id)
    print(f"\nDone: {path}")
    print(f"Start the server with VIDEO_SERVER_BACKEND={profile.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
