"""Pyyntö- ja vastausmallit sekä pyyntöjen ratkaisu backend-parametreiksi.

Suunnitteluperiaate: pyynnössä generointikentät ovat valinnaisia ja `None`
tarkoittaa "käytä aktiivisen profiilin oletusta". Ratkaisu konkreettisiksi
arvoiksi tapahtuu kerran, `resolve_params()`:ssa, ja tulos tallennetaan jobiin.
Näin job-vastaus voi kertoa mitä oikeasti ajettiin - erityisesti ratkaistun
siemenen - jolloin onnistunut ajo on toistettavissa.
"""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field

from video_server.backends.base import GenerationParams
from video_server.config import DEFAULT_NUM_FRAMES, ModelProfile, parse_resolution

MAX_SEED = 2**32 - 1


class ParamError(ValueError):
    """Validointivirhe, joka riippuu ajonaikaisesta profiilista.

    Ei Pydantic-validaattori, koska sallitut arvot riippuvat siitä mikä malli on
    ladattu - staattinen Literal-tyyppi ei tähän taivu. Reittikerros muuntaa
    tämän 400-vastaukseksi.
    """


# --- Pyynnöt --------------------------------------------------------------


class GenerationRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int | None = Field(default=None, ge=1)
    fps: int | None = Field(default=None, ge=1, le=120)
    resolution: str | None = None
    # -1 = satunnainen, kuten Forgessa.
    seed: int = Field(default=-1, ge=-1, le=MAX_SEED)
    shift: float | None = Field(default=None, gt=0)
    guidance_scale: float | None = Field(default=None, ge=0)
    guidance_scale_2: float | None = Field(default=None, ge=0)
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)


class Txt2VidRequest(GenerationRequest):
    prompt: str = Field(min_length=1)


class Img2VidRequest(GenerationRequest):
    # Base64-koodattu PNG tai JPEG ilman data-URI-prefiksiä. Päätös base64:n
    # puolesta multipartin sijaan: ks. spec, "Ratkaistut avoimet kysymykset".
    init_image: str = Field(min_length=1)


# --- Vastaukset -----------------------------------------------------------


class AcceptedResponse(BaseModel):
    job_id: str


class ProgressResponse(BaseModel):
    step: int
    total_steps: int
    eta_seconds: float | None = None
    # Denoising ei ole koko ajo: dekoodausvaihe näkyy tässä erikseen, jottei
    # job vaikuta valmiilta kesken VAE-dekoodauksen.
    phase: Literal["denoising", "decoding"] = "denoising"


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: ProgressResponse
    params: dict
    created_at: str
    video_url: str | None = None
    # Vain jos esikatselu on kytketty päälle konfiguraatiosta.
    preview_url: str | None = None
    error: str | None = None


class ConstraintsResponse(BaseModel):
    resolutions: list[str]
    frame_multiple: int
    native_fps: int
    supports_i2v: bool


class ModelsResponse(BaseModel):
    active: str
    available: list[str]
    tier: str
    constraints: ConstraintsResponse


class HealthResponse(BaseModel):
    status: Literal["loading", "ready", "error"]
    backend: str
    detail: str | None = None


# --- Ratkaisu -------------------------------------------------------------


def resolve_params(request: GenerationRequest, profile: ModelProfile) -> GenerationParams:
    """Täydentää puuttuvat kentät profiilin oletuksilla ja validoi rajoitteet.

    Nostaa ParamError jos pyyntö on kelvoton. Virheviestissä kerrotaan sekä
    sääntö että kelvolliset vaihtoehdot: 400 joka vain toteaa "invalid" pakottaa
    asiakkaan arvaamaan.
    """
    resolution = request.resolution or profile.resolutions[0]
    if resolution not in profile.resolutions:
        allowed = ", ".join(profile.resolutions)
        raise ParamError(
            f"resolution {resolution!r} is not supported by {profile.name}; allowed: {allowed}"
        )
    try:
        parse_resolution(resolution)
    except ValueError:
        raise ParamError(f"resolution {resolution!r} is not of the form WIDTHxHEIGHT") from None

    num_frames = request.num_frames if request.num_frames is not None else DEFAULT_NUM_FRAMES
    if not profile.is_valid_frame_count(num_frames):
        lower, upper = profile.nearest_frame_counts(num_frames)
        raise ParamError(
            f"num_frames {num_frames} is invalid for {profile.name}: requires "
            f"n * {profile.frame_multiple} + 1. Nearest valid values: {lower} and {upper}"
        )

    init_image = getattr(request, "init_image", None)
    mode = "i2v" if init_image else "t2v"
    if not profile.supports_mode(mode):
        raise ParamError(f"model {profile.name} does not support {mode} generation")

    seed = request.seed if request.seed >= 0 else random.randrange(0, MAX_SEED + 1)

    # guidance_scale_2 on merkityksellinen vain MoE-malleilla. Yhden
    # asiantuntijan mallilla se on pakko jättää None:ksi: diffusers nostaa
    # ValueErrorin jos se annetaan pipelinelle jonka boundary_ratio on None.
    guidance_2 = None
    if profile.has_second_expert:
        guidance_2 = (
            request.guidance_scale_2
            if request.guidance_scale_2 is not None
            else profile.default_guidance_2
        )

    return GenerationParams(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_frames=num_frames,
        fps=request.fps if request.fps is not None else profile.native_fps,
        resolution=resolution,
        seed=seed,
        shift=request.shift if request.shift is not None else profile.default_shift,
        guidance_scale=(
            request.guidance_scale
            if request.guidance_scale is not None
            else profile.default_guidance
        ),
        guidance_scale_2=guidance_2,
        num_inference_steps=(
            request.num_inference_steps
            if request.num_inference_steps is not None
            else profile.default_steps
        ),
        init_image=init_image,
    )
