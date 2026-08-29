"""Wan 2.2 -backend diffusersin päällä.

Kaikki torch- ja diffusers-tuonnit ovat funktioiden sisällä. Näin tämä moduuli
voidaan tuoda sisään ilman GPU-riippuvuuksia, ja perusasennus (uv sync ilman
--extra gpu) pysyy toimivana - vain tämän backendin käyttö vaatii ne.

Toteutuksen kannalta olennaiset yksityiskohdat, jotka on tarkistettu
diffusersin lähdekoodista eikä oletettu:

- `shift` EI ole pipelinen kutsuparametri vaan schedulerin `flow_shift`.
  Scheduler rakennetaan uudelleen per pyyntö alkuperäisestä configista.
- `guidance_scale_2` on sallittu vain kun pipelinen `boundary_ratio` ei ole
  None. Yhden asiantuntijan mallilla sen välittäminen nostaa ValueErrorin,
  joten profiilin `has_second_expert` ratkaisee - ei arvaus.
- `callback_on_step_end(pipe, i, t, kwargs)` kutsutaan askeleen lopussa ja sen
  paluuarvon on oltava dict; pipeline tekee sille .pop()-kutsun.
- diffusers PYÖRISTÄÄ kelvottoman num_frames-arvon hiljaisesti varoituksen
  kanssa. Speksi kieltää hiljaisen pyöristyksen, joten arvo on validoitu jo
  rajapintakerroksessa eikä sen pitäisi koskaan päätyä tänne virheellisenä.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from video_server.backends.base import GenerationParams, Progress, VideoBackend
from video_server.config import GIB, ModelProfile
from video_server.settings import Settings
from video_server.video import preview_path_for, write_video

logger = logging.getLogger(__name__)


class WeightsMissingError(RuntimeError):
    """Painot puuttuvat levyltä eikä automaattilataus ole päällä.

    Kymmenien gigatavujen hiljainen lataus ensimmäisen API-kutsun yhteydessä on
    huono oletus, joten tämä on eksplisiittinen virhe, joka kertoo komennon.
    """


class Wan22Backend(VideoBackend):
    def __init__(self, profile: ModelProfile, settings: Settings) -> None:
        self._profile = profile
        self._settings = settings
        self._pipe: Any = None
        self._i2v_pipe: Any = None
        self._scheduler_config: dict[str, Any] | None = None
        self._offloaded = False

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    # --- Lataus -----------------------------------------------------------

    def load(self) -> None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        repo = self._profile.repo_id
        local_only = not self._settings.allow_download
        # I2V-only-profiilin checkpoint on tallennettu WanImageToVideoPipelinena;
        # muut ladataan WanPipelinena ja I2V johdetaan komponenteista.
        pipeline_cls = WanPipeline if self._profile.supports_t2v else WanImageToVideoPipeline

        logger.info(
            "ladataan %s (%s, dtype=%s, local_files_only=%s)",
            self._profile.name,
            repo,
            self._settings.dtype,
            local_only,
        )

        mode, reason = self._resolve_load_mode()
        logger.info("lataustapa=%s (%s)", mode, reason)

        try:
            # VAE erikseen float32:na: Wanin VAE on numeerisesti herkkä, ja
            # virallinen esimerkki lataa sen nimenomaan täydellä tarkkuudella
            # vaikka transformer ajetaan bf16:na.
            vae = AutoencoderKLWan.from_pretrained(
                repo,
                subfolder="vae",
                torch_dtype=torch.float32,
                local_files_only=local_only,
            )
            extra: dict[str, Any] = {}
            if mode == "quantized":
                extra["transformer"] = self._load_quantized_transformer(repo, local_only)
            pipe = pipeline_cls.from_pretrained(
                repo,
                vae=vae,
                torch_dtype=self._torch_dtype(),
                local_files_only=local_only,
                **extra,
            )
        except OSError as exc:
            if local_only:
                raise WeightsMissingError(
                    f"mallin {repo} painoja ei löydy paikallisesta cachesta. "
                    f"Lataa ne komennolla: uv run python scripts/download_model.py "
                    f"{self._profile.name}  (tai salli automaattilataus asetuksella "
                    f"VIDEO_SERVER_ALLOW_DOWNLOAD=true)"
                ) from exc
            raise

        if self._settings.vae_tiling:
            # Dekoodaus tehdään koko videotensorille kerralla, ja se on ajon
            # suurin muistipiikki - juuri siinä 24 GB:n kortti kaatuu 720p:llä
            # vaikka denoising olisi mennyt läpi. Tiilitys pilkkoo sen
            # limittäisiin paloihin.
            pipe.vae.enable_tiling()

        if mode in ("offload", "quantized"):
            # offload: painot pysyvät RAM:issa ja kerrokset siirretään GPU:lle
            # tarpeen mukaan - mahtuu pienempään VRAM:iin mutta on hitaampi.
            # quantized: 4-bit-moduulia ei voi siirtää .to()-kutsulla, joten
            # sijoittelu jätetään offload-hookeille.
            pipe.enable_model_cpu_offload(device=self._settings.device)
            self._offloaded = True
        else:
            pipe.to(self._settings.device)

        self._pipe = pipe
        self._scheduler_config = dict(pipe.scheduler.config)
        self._sync_profile_from_checkpoint(pipe)
        logger.info("malli %s ladattu (lataustapa=%s)", self._profile.name, mode)

    def _torch_dtype(self) -> Any:
        import torch

        dtype = getattr(torch, self._settings.dtype, None)
        if dtype is None:
            raise ValueError(f"tuntematon dtype {self._settings.dtype!r}")
        return dtype

    def _vram_gib(self) -> float | None:
        import torch

        if not torch.cuda.is_available():
            return None
        index = int(self._settings.device.split(":")[1]) if ":" in self._settings.device else 0
        return torch.cuda.get_device_properties(index).total_memory / GIB

    def _resolve_load_mode(self) -> tuple[str, str]:
        """Lataustapa ja sen perustelu.

        Asetukset päättävät ensisijaisesti (eksplisiittinen cpu_offload, sitten
        tierin politiikka). Tämän päällä on turvaverkko: jos bf16 valikoitui
        mutta kortti ei riitä profiilille, siirrytään offloadiin sen sijaan että
        ajo kaatuisi OOM:iin vasta minuuttien päästä.
        """
        mode, reason = self._settings.load_mode()
        if mode != "bf16" or self._settings.cpu_offload is not None:
            # Eksplisiittistä asetusta ei ylikirjoiteta. Jos käyttäjä on
            # kieltänyt offloadin, se on hänen päätöksensä - sama periaate kuin
            # shiftin kanssa checkpoint-synkronoinnissa.
            return mode, reason

        vram = self._vram_gib()
        if vram is not None and vram < self._profile.min_vram_gib:
            return "offload", (
                f"VRAM {vram:.1f} GB alittaa profiilin {self._profile.name} vaatimuksen "
                f"{self._profile.min_vram_gib:.0f} GB (ohittaa: {reason})"
            )
        return mode, reason

    def _load_quantized_transformer(self, repo: str, local_only: bool) -> Any:
        """4-bit-kvantisoitu transformer pienemmille korteille.

        Huom: kvantisoidaan vain `transformer`. MoE-malleilla (A14B) on lisäksi
        `transformer_2`, joka jäisi kvantisoimatta - siksi quantized-tier
        osoittaa yhden asiantuntijan malliin. Tätä polkua ei ole voitu ajaa
        tällä laitteistolla (24 GB riittää bf16:een), joten se on toteutettu
        mutta testaamaton.
        """
        from diffusers import BitsAndBytesConfig, WanTransformer3DModel

        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "quantized-lataustapa vaatii bitsandbytes-paketin: "
                "uv sync --extra gpu --extra quantized. Vaihtoehtoisesti pakota "
                "toinen lataustapa asetuksella VIDEO_SERVER_CPU_OFFLOAD=true."
            ) from None

        logger.info("ladataan transformer 4-bit-kvantisoituna (nf4)")
        return WanTransformer3DModel.from_pretrained(
            repo,
            subfolder="transformer",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._torch_dtype(),
            ),
            torch_dtype=self._torch_dtype(),
            local_files_only=local_only,
        )

    def _sync_profile_from_checkpoint(self, pipe: Any) -> None:
        """Checkpoint on totuuden lähde niille arvoille jotka se kantaa.

        frame_multiple tulee VAE:n scale_factor_temporal-arvosta ja shift
        schedulerin flow_shiftistä. Profiilin arvot ovat vain fallback, jota
        tarvitaan validointiin ennen kuin malli on ladattu. Käyttäjän
        eksplisiittistä shift-asetusta ei ylikirjoiteta.
        """
        frame_multiple = int(
            getattr(pipe, "vae_scale_factor_temporal", self._profile.frame_multiple)
        )
        updates: dict[str, Any] = {}

        if frame_multiple != self._profile.frame_multiple:
            logger.info(
                "frame_multiple %s -> %s (VAE scale_factor_temporal)",
                self._profile.frame_multiple,
                frame_multiple,
            )
            updates["frame_multiple"] = frame_multiple

        if self._settings.shift is None:
            shift = float(pipe.scheduler.config.get("flow_shift", self._profile.default_shift))
            if shift != self._profile.default_shift:
                logger.info(
                    "default_shift %s -> %s (schedulerin flow_shift)",
                    self._profile.default_shift,
                    shift,
                )
                updates["default_shift"] = shift

        if updates:
            self._profile = replace(self._profile, **updates)

    # --- Generointi -------------------------------------------------------

    def generate(
        self,
        params: GenerationParams,
        on_progress: Callable[[Progress], None],
        output_path: Path,
    ) -> Path:
        import torch

        if self._pipe is None:
            raise RuntimeError("Wan22Backend.load() kutsumatta")

        pipe = self._pipeline_for(params.mode)
        self._apply_shift(pipe, params.shift)

        width, height = params.size
        # CPU-generaattori: sama siemen tuottaa saman tuloksen riippumatta
        # siitä ajetaanko offload-tilassa vai ei.
        generator = torch.Generator(device="cpu").manual_seed(params.seed)

        total = params.num_inference_steps
        preview_every = self._settings.preview_every_n_steps
        preview_target = preview_path_for(output_path)

        def callback(pipeline: Any, step_index: int, timestep: Any, kwargs: dict) -> dict:
            # Kutsutaan askeleen lopussa, indeksi alkaa nollasta. Jos jono on
            # pyytänyt peruutusta, on_progress nostaa GenerationCancelled-
            # poikkeuksen, jonka annetaan edetä ulos pipelinesta.
            step = step_index + 1
            preview = None
            if preview_every and step % preview_every == 0:
                preview = self._write_preview(pipeline, kwargs.get("latents"), preview_target)
            on_progress(Progress(step=step, total_steps=total, preview_path=preview))

            if step == total:
                # Denoising on ohi ja pipeline siirtyy VAE-dekoodaukseen, joka
                # vie 720p-videolla minuutteja eikä tuota yhtään callbackia.
                # Ilman tätä job näyttäisi valmiilta kesken ajon.
                on_progress(Progress(step=total, total_steps=total, phase="decoding"))
            return {}

        call_kwargs: dict[str, Any] = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt or None,
            "height": height,
            "width": width,
            "num_frames": params.num_frames,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "generator": generator,
            "output_type": "np",
            "callback_on_step_end": callback,
        }
        if self._profile.has_second_expert and params.guidance_scale_2 is not None:
            call_kwargs["guidance_scale_2"] = params.guidance_scale_2
        if params.mode == "i2v":
            call_kwargs["image"] = self._decode_image(params.init_image, width, height)

        try:
            result = pipe(**call_kwargs)
        finally:
            # Peruutus ja virhe jättävät välitensorit VRAM:iin; seuraava ajo
            # alkaa muuten pienemmällä vapaalla muistilla.
            torch.cuda.empty_cache()

        return write_video(result.frames[0], output_path, params.fps)

    def _write_preview(self, pipe: Any, latents: Any, path: Path) -> Path | None:
        """Dekoodaa yhden latenttiframen esikatselukuvaksi.

        Normalisointi on sama kuin pipelinen omassa lopputuloksen
        dekoodauksessa (latents_mean / latents_std); ilman sitä kuva olisi
        roskaa.

        Epäonnistuminen ei saa kaataa ajoa. Esikatselu on mukavuus, ja
        ylimääräinen VAE-kutsu voi kaatua muistiin juuri silloin kun kortti on
        täynnä - silloin logitetaan ja jatketaan generointia.
        """
        if latents is None:
            return None

        import torch

        try:
            with torch.no_grad():
                single = latents[:, :, :1].to(pipe.vae.dtype)
                mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(single.device, single.dtype)
                )
                inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                    1, pipe.vae.config.z_dim, 1, 1, 1
                ).to(single.device, single.dtype)
                decoded = pipe.vae.decode(single / inv_std + mean, return_dict=False)[0]
                frames = pipe.video_processor.postprocess_video(decoded, output_type="pil")

            path.parent.mkdir(parents=True, exist_ok=True)
            frames[0][0].save(path)
            return path
        except (RuntimeError, ValueError, OSError, IndexError, AttributeError) as exc:
            logger.warning("esikatselukuvan dekoodaus epäonnistui, jatketaan: %s", exc)
            return None

    def _pipeline_for(self, mode: str) -> Any:
        if mode == "i2v" and self._profile.supports_t2v:
            # Primääri pipeline on T2V. I2V rakennetaan samoista komponenteista:
            # ei uutta latausta eikä lisää VRAM-kulutusta.
            if self._i2v_pipe is None:
                self._i2v_pipe = self._build_i2v_pipeline()
            return self._i2v_pipe
        return self._pipe

    def _build_i2v_pipeline(self) -> Any:
        from diffusers import WanImageToVideoPipeline

        config = self._pipe.config
        pipe = WanImageToVideoPipeline(
            **self._pipe.components,
            boundary_ratio=config.get("boundary_ratio"),
            # TI2V-5B asettaa tämän lipun model_indexissään. Ilman sitä I2V-polku
            # käsittelee timestepit väärin.
            expand_timesteps=config.get("expand_timesteps", False),
        )
        if self._offloaded:
            pipe.enable_model_cpu_offload(device=self._settings.device)
        return pipe

    def _apply_shift(self, pipe: Any, shift: float) -> None:
        """Rakentaa schedulerin uudelleen pyynnön shift-arvolla.

        Lähtökohtana on aina latauksessa talteen otettu alkuperäinen config,
        ei edellisen pyynnön scheduler - muuten arvo kumuloituisi pyyntöjen yli.
        """
        scheduler_cls = type(pipe.scheduler)
        pipe.scheduler = scheduler_cls.from_config(self._scheduler_config, flow_shift=shift)

    @staticmethod
    def _decode_image(data: str | None, width: int, height: int) -> Any:
        from PIL import Image

        if not data:
            raise ValueError("i2v-generointi vaatii init_image-kentän")

        # Speksi sanoo ilman data-URI-prefiksiä, mutta sen mukaan lähettäminen on
        # yleinen asiakasvirhe eikä sen hylkääminen hyödytä ketään.
        if data.startswith("data:"):
            _, _, data = data.partition(",")

        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"init_image ei ole kelvollista base64-dataa: {exc}") from None

        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except OSError as exc:
            raise ValueError(f"init_image ei ole tunnistettava kuva: {exc}") from None

        return image.convert("RGB").resize((width, height), Image.LANCZOS)
