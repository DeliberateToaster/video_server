"""Abstrakti backend-rajapinta.

Tämän takana voi olla mikä tahansa moottori (Wan 2.2, myöhemmin esim. LTX)
ilman että `main.py` tai `jobs.py` muuttuu. Ks. spec, "Backend-abstraktio".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from video_server.config import ModelProfile, parse_resolution


class GenerationCancelled(Exception):
    """Nostetaan on_progress-callbackista kun peruutus on pyydetty.

    Kulkee ulos diffusersin pipelinesta normaalina poikkeuksena, mikä katkaisee
    ajon kesken kuluvan askeleen. Tämä on syy siihen, ettei backendillä ole
    erillistä cancel()-metodia: peruutukselle on yksi reitti, ei kahta.
    """


@dataclass
class Progress:
    """Etenemistieto yhden askeleen jälkeen.

    Olio eikä kaksi int-argumenttia siksi, että esikatselukuvan lisääminen
    myöhemmin (preview_path) on additiivinen muutos eikä riko backendejä.
    """

    step: int
    total_steps: int
    # Denoising-askeleet ovat vain osa ajosta: VAE-dekoodaus vie 720p-videolla
    # minuutteja eikä siitä ole askelittaista tietoa. Ilman tätä kenttää job
    # näyttäisi tilaa n/n ja ETA 0 vaikka ajoa on reilusti jäljellä.
    phase: str = "denoising"
    preview_path: Path | None = None


class GenerationParams(BaseModel):
    """Backendille menevät parametrit, kaikki oletukset jo ratkaistuina.

    Erotus pyyntöskeemasta (`schemas.py`) on tarkoituksellinen: pyynnössä kentät
    ovat valinnaisia ja tarkoittavat "käytä profiilin oletusta", täällä ne ovat
    konkreettisia arvoja. Backendin ei kuulu tietää mikä oli oletus ja mikä
    käyttäjän valinta.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int
    fps: int
    resolution: str
    seed: int = Field(ge=0)
    shift: float
    guidance_scale: float
    guidance_scale_2: float | None = None
    num_inference_steps: int

    # Base64-koodattu aloituskuva. exclude=True: tätä ei kirjoiteta job-vastaukseen
    # eikä sidecar-metadataan - megatavun base64-merkkijono ei kuulu kumpaankaan.
    init_image: str | None = Field(default=None, exclude=True, repr=False)

    @property
    def mode(self) -> str:
        return "i2v" if self.init_image else "t2v"

    @property
    def size(self) -> tuple[int, int]:
        return parse_resolution(self.resolution)


class VideoBackend(ABC):
    """Yksi malli, yksi ajo kerrallaan. Toteutus saa olettaa, ettei generate()
    ole rinnakkain kutsuttavissa - jono huolehtii siitä."""

    @property
    @abstractmethod
    def profile(self) -> ModelProfile:
        """Aktiivisen mallin rajoitteet ja oletukset.

        Luettavissa jo ennen load()-kutsua, jotta pyyntöjen validointi ja
        GET /api/v1/models toimivat mallin latautuessa.
        """

    @abstractmethod
    def load(self) -> None:
        """Lataa mallin muistiin. Kutsutaan kerran käynnistyksessä.

        Hidas ja blokkaava; kutsutaan taustasäikeessä.
        """

    @abstractmethod
    def generate(
        self,
        params: GenerationParams,
        on_progress: Callable[[Progress], None],
        output_path: Path,
    ) -> Path:
        """Ajaa generoinnin ja palauttaa polun valmiiseen mp4-tiedostoon.

        Kutsuu `on_progress` jokaisen askeleen jälkeen. Jos callback nostaa
        GenerationCancelled, sitä ei saa napata - sen on annettava edetä ulos.
        """
