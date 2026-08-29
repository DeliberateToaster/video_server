"""Ajonaikaiset asetukset: .env tai config.yaml -> Settings.

Työnjako `config.py`:n kanssa: siellä on skeema ja fallback-oletukset, täällä
lopulliset arvot. Tavoite (ks. spec, "Generalisointi ja konfiguroitavuus") on
että toinen kehittäjä saa projektin käyntiin omalla laitteistollaan
muokkaamatta Python-koodia.

Prioriteetti korkeimmasta alimpaan:
  1. suoraan konstruktorille annetut arvot (testit)
  2. ympäristömuuttujat (VIDEO_SERVER_*)
  3. .env
  4. config.yaml
  5. tässä määritellyt oletukset
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from video_server.config import PROFILES, TIERS, ModelProfile, suggest_tier

_PACKAGE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDEO_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config.yaml",
        extra="ignore",
    )

    # --- Malli ja laitteisto ---------------------------------------------
    # "auto" = valitse malli tunnistetun tierin mukaan. Tämä on speksin
    # yleistystavoite: sama konfiguraatio toimii eri laitteistoilla ilman
    # koodimuutosta. Kehitys ja CI ajavat mockilla: VIDEO_SERVER_BACKEND=mock.
    backend: str = "auto"
    tier: str = "auto"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    # None = päättele laitteistosta (VRAM alle profiilin vaatimuksen -> offload).
    cpu_offload: bool | None = None
    # Painojen automaattilataus. Oletus false: kymmenien gigatavujen lataus ei
    # saa tapahtua vahingossa ensimmäisen API-kutsun yhteydessä.
    allow_download: bool = False
    # Tiilitetty VAE-dekoodaus. Videon dekoodaus on ajon suurin yksittäinen
    # muistipiikki (koko latenttitensori kerralla), ja se on se kohta joka
    # kaataa 24 GB:n kortin 720p-videolla. Tiilitys pilkkoo dekoodauksen
    # limittäisiin paloihin. Oletuksena päällä: ilman sitä ajo on riippuvainen
    # siitä ettei kortilla ole mitään muuta.
    vae_tiling: bool = True

    # --- Vaihe 4: valinnaiset ---------------------------------------------
    # Esikatselukuva N askeleen välein. 0 = pois (speksin oletus: "ei
    # oletuksena"). Maksaa yhden ylimääräisen VAE-dekoodauksen per kuva.
    preview_every_n_steps: int = Field(default=0, ge=0)
    # API-avain. None = ei autentikointia, kuten v1-rajaus edellyttää. Jos
    # asetettu, /api/v1- ja /outputs-polut vaativat X-API-Key-otsakkeen.
    api_key: str | None = None

    # --- Ohitukset profiilin oletuksille ---------------------------------
    # None = käytä aktiivisen profiilin arvoa.
    resolutions: list[str] | None = None
    num_inference_steps: int | None = None
    shift: float | None = None
    guidance_scale: float | None = None

    # --- Tiedostot ja retention ------------------------------------------
    outputs_dir: Path = _PACKAGE_DIR / "outputs"
    retention_max_age_days: float | None = 14.0
    retention_max_total_gb: float | None = 20.0

    # --- Jono ja pyynnöt --------------------------------------------------
    max_queue_size: int = Field(default=8, ge=1)
    max_request_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)

    # --- Mock-backendin ajoitus (testit asettavat tämän nollaan) ----------
    mock_step_seconds: float = Field(default=0.15, ge=0.0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # --- Johdetut arvot ---------------------------------------------------

    def resolve_backend(self) -> str:
        """Konkreettinen backendin nimi, myös kun asetuksena on "auto".

        Auto-valinta kulkee tierin kautta: laitteisto -> tier -> profiili.
        Jos tierille ei ole profiilia (ei CUDAa, tai liian pieni kortti),
        valintaa ei arvata vaan kaadutaan ohjeen kanssa - hiljainen putoaminen
        mock-backendiin olisi pahin mahdollinen oletus, koska palvelin
        näyttäisi toimivan mutta tuottaisi väärennettyä videota.
        """
        if self.backend != "auto":
            return self.backend

        tier, reason = self.resolve_tier()
        policy = TIERS.get(tier)
        if policy is None:
            raise ValueError(
                f"backend=auto cannot pick a model for tier {tier!r} ({reason}). "
                f"Set VIDEO_SERVER_BACKEND explicitly; for development use "
                f"VIDEO_SERVER_BACKEND=mock."
            )
        return policy.profile

    def load_mode(self) -> tuple[str, str]:
        """Palauttaa (lataustapa, perustelu).

        Prioriteetti: eksplisiittinen cpu_offload > tierin politiikka > bf16.
        Perustelu logitetaan, jottei valinta ole hiljainen.
        """
        if self.cpu_offload is True:
            return "offload", "set explicitly (cpu_offload=true)"
        if self.cpu_offload is False:
            return "bf16", "set explicitly (cpu_offload=false)"

        tier, reason = self.resolve_tier()
        policy = TIERS.get(tier)
        if policy is None:
            return "bf16", f"no policy defined for tier {tier!r} ({reason})"
        return policy.load_mode, f"tier {tier}: {policy.note}"

    def profile(self) -> ModelProfile:
        """Aktiivinen malliprofiili, konfiguraation ohitukset sovellettuina.

        Nostaa KeyError jos backend-nimeä ei tunneta - se on käynnistysvirhe,
        ja parempi kaatua heti kuin ajautua väärään malliin.
        """
        base = PROFILES[self.resolve_backend()]
        overrides: dict[str, object] = {}
        if self.resolutions:
            overrides["resolutions"] = tuple(self.resolutions)
        if self.num_inference_steps is not None:
            overrides["default_steps"] = self.num_inference_steps
        if self.shift is not None:
            overrides["default_shift"] = self.shift
        if self.guidance_scale is not None:
            overrides["default_guidance"] = self.guidance_scale
        if not overrides:
            return base
        return ModelProfile(**{**base.__dict__, **overrides})

    def resolve_tier(self) -> tuple[str, str]:
        """Palauttaa (tier, perustelu).

        `tier="auto"` tunnistaa laitteiston; mikä tahansa muu arvo on
        käyttäjän eksplisiittinen ohitus, jota kunnioitetaan sellaisenaan.
        Torch tuodaan sisään vasta täällä, jotta koko moduuli toimii ilman
        GPU-riippuvuuksia.
        """
        if self.tier != "auto":
            return self.tier, "set explicitly in configuration"

        try:
            import torch
        except ImportError:
            return "cpu", "torch missing - GPU dependencies are not installed"

        if not torch.cuda.is_available():
            return "cpu", "CUDA is not available"

        index = int(self.device.split(":")[1]) if ":" in self.device else 0
        props = torch.cuda.get_device_properties(index)
        vram_gib = props.total_memory / (1024**3)

        ram_gib: float | None
        try:
            import psutil

            ram_gib = psutil.virtual_memory().total / (1024**3)
        except Exception:  # noqa: BLE001 - RAM on lisätieto, ei pakollinen
            ram_gib = None

        tier, reason = suggest_tier(vram_gib, ram_gib)
        return tier, f"{props.name}: {reason}"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
