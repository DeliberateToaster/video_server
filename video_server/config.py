"""Skeema, fallback-oletukset ja malliprofiilit.

Tämä moduuli EI lue ympäristöä eikä tuo torchia sisään. Se määrittelee mitä
arvoja on olemassa ja mitkä ovat niiden oletukset; lopulliset ajonaikaiset arvot
tulevat `settings.py`:stä. Ks. spec, "Konfiguraatio".

Torch-riippumattomuus on tarkoituksellista: `scripts/check_env.py` ja koko
mock-backendillä ajettava testisetti tuovat tämän moduulin sisään ilman että
GPU-riippuvuuksia tarvitsee olla asennettuna.

Profiilien arvot on tarkistettu mallien omista konfiguraatioista
(model_index.json, vae/config.json, scheduler/scheduler_config.json) ja
diffusersin lähdekoodista, ei muistinvaraisesti. Osa arvoista synkronoidaan
vielä latausvaiheessa checkpointista - ks. `backends/wan22.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3

# Speksin oletusarvo. Kelvollinen kaikilla profiileilla (81 = 4*20+1).
DEFAULT_NUM_FRAMES = 81


@dataclass(frozen=True)
class ModelProfile:
    """Mallikohtaiset rajoitteet ja oletukset.

    Nämä olivat alkuperäisessä speksissä globaaleja vakioita, mutta ne ovat
    mallikohtaisia: A14B ja TI2V-5B eroavat sekä fps:n, resoluutioiden,
    shiftin että guidance-rakenteen osalta. Pyynnöt validoidaan aktiivisen
    profiilin kenttiä vasten, ei koodiin kirjoitettuja vakioita vasten.
    """

    name: str
    repo_id: str
    # num_frames-sääntö: kelvollinen arvo on n * frame_multiple + 1.
    # Tulee VAE:n scale_factor_temporal-arvosta; wan22-backend synkronoi tämän
    # ladatusta VAE:sta, koska variantit voivat käyttää eri VAE:ta.
    frame_multiple: int
    native_fps: int
    resolutions: tuple[str, ...]
    default_steps: int
    # Schedulerin flow_shift. Jokainen checkpoint kantaa oman oletuksensa
    # (scheduler_config.json), joten tämä on vain fallback ennen latausta.
    default_shift: float
    default_guidance: float
    # MoE-malleilla on kaksi asiantuntijaa (high-noise / low-noise) ja kummallakin
    # oma guidance-arvo. Yhden asiantuntijan mallilla guidance_scale_2:n
    # välittäminen pipelinelle NOSTAA ValueErrorin, joten tätä lippua on
    # noudatettava, ei arvattava.
    has_second_expert: bool
    default_guidance_2: float | None
    supports_t2v: bool
    supports_i2v: bool
    # Karkea VRAM-vaatimus bf16-tarkkuudella, tierin päättelyä varten.
    min_vram_gib: float
    # CPU-offload pitää painot järjestelmämuistissa. None = ei offload-tarvetta.
    offload_ram_gib: float | None = None

    def is_valid_frame_count(self, num_frames: int) -> bool:
        return num_frames >= 1 and (num_frames - 1) % self.frame_multiple == 0

    def nearest_frame_counts(self, num_frames: int) -> tuple[int, int]:
        """Lähimmät kelvolliset frame-määrät molemmin puolin, virheviestiä varten."""
        k = self.frame_multiple
        n = max(0, (num_frames - 1) // k)
        lower = n * k + 1
        upper = (n + 1) * k + 1
        return lower, upper

    def supports_mode(self, mode: str) -> bool:
        return self.supports_i2v if mode == "i2v" else self.supports_t2v


# Tarkistetut lähteet per profiili on merkitty kommentteihin. Kaikki kolme
# Wan-repoa on olemassa ja niiden model_index.json on luettu.
PROFILES: dict[str, ModelProfile] = {
    "mock": ModelProfile(
        name="mock",
        repo_id="",
        frame_multiple=4,
        native_fps=24,
        # Pienet resoluutiot ovat mukana testejä varten: 1280x704 x 81 framea
        # enkoodattuna jokaisessa testissä olisi tarpeetonta odottelua.
        resolutions=("256x144", "144x256", "832x480", "480x832", "1280x704", "704x1280"),
        default_steps=8,
        default_shift=5.0,
        default_guidance=5.0,
        has_second_expert=False,
        default_guidance_2=None,
        supports_t2v=True,
        supports_i2v=True,
        min_vram_gib=0.0,
    ),
    # model_index: WanPipeline, boundary_ratio=null, transformer_2=[null,null]
    # vae/config: scale_factor_temporal=4, scale_factor_spatial=16 (704 = 16*44)
    # scheduler_config: flow_shift=5.0 | mallikortti: 1280x704, fps 24, guidance 5.0
    "wan2.2-ti2v-5b": ModelProfile(
        name="wan2.2-ti2v-5b",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        frame_multiple=4,
        native_fps=24,
        resolutions=("1280x704", "704x1280"),
        default_steps=50,
        default_shift=5.0,
        default_guidance=5.0,
        # Yksi asiantuntija: guidance_scale_2 EI saa mennä pipelinelle läpi.
        has_second_expert=False,
        default_guidance_2=None,
        supports_t2v=True,
        # image_encoder ja image_processor ovat diffusersissa valinnaisia
        # komponentteja, ja Wan 2.2 I2V ei käytä niitä lainkaan. TI2V-5B:n
        # komponentit riittävät siis WanImageToVideoPipelinelle.
        supports_i2v=True,
        # Mitattu: ~23 GB varattuna latauksen jälkeen vapaalla 24 GB:n kortilla
        # (bf16-transformer + UMT5-XXL-tekstikooderi + fp32-VAE, mukana torchin
        # allokaattorin varaukset). Arvio 15 GB oli liian optimistinen: 16 GB:n
        # kortilla se olisi luvannut mahtumisen ja kaatunut ajossa.
        min_vram_gib=20.0,
    ),
    # model_index: WanPipeline, boundary_ratio=0.875, transformer_2 mukana
    # scheduler_config: flow_shift=3.0 | mallikortti: guidance 4.0 / 3.0, fps 16
    "wan2.2-t2v-a14b": ModelProfile(
        name="wan2.2-t2v-a14b",
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        frame_multiple=4,
        native_fps=16,
        resolutions=("832x480", "480x832", "1280x720", "720x1280"),
        default_steps=40,
        default_shift=3.0,
        default_guidance=4.0,
        has_second_expert=True,
        default_guidance_2=3.0,
        supports_t2v=True,
        supports_i2v=False,
        min_vram_gib=30.0,
        offload_ram_gib=60.0,
    ),
    # model_index: WanImageToVideoPipeline, boundary_ratio=0.9,
    # image_encoder=[null,null] -> Wan 2.2 I2V ei käytä CLIP-kuvakooderia.
    "wan2.2-i2v-a14b": ModelProfile(
        name="wan2.2-i2v-a14b",
        repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        frame_multiple=4,
        native_fps=16,
        resolutions=("832x480", "480x832", "1280x720", "720x1280"),
        default_steps=40,
        default_shift=3.0,
        default_guidance=4.0,
        has_second_expert=True,
        default_guidance_2=3.0,
        # I2V-variantti vaatii aloituskuvan; pelkkä teksti ei kelpaa.
        supports_t2v=False,
        supports_i2v=True,
        min_vram_gib=30.0,
        offload_ram_gib=60.0,
    ),
}

DEFAULT_PROFILE = "wan2.2-ti2v-5b"


@dataclass(frozen=True)
class TierPolicy:
    """Mitä tier tarkoittaa käytännössä.

    Tier ei ole pelkkä nimilappu: se valitsee sekä mallin että lataustavan.
    Ilman tätä tier olisi vain käynnistyksessä logitettu merkkijono, joka ei
    vaikuta mihinkään.

    Huom: speksi kaavaili myös tierkohtaista resoluutiolistaa. Sitä ei ole
    toteutettu, koska se ei nykyisellä mallivalikoimalla tekisi mitään:
    TI2V-5B tarjoaa vain 720p-luokan resoluutiot (1280x704 / 704x1280), joten
    low-tierillä suodatus tyhjentäisi listan kokonaan, eikä A14B koskaan päädy
    low-tierille. Resoluutiolista vaihtelee jo mallin mukana, mikä on se
    havaittava käyttäytyminen jota speksin kohta 5 haki.
    """

    name: str
    profile: str
    # bf16 = painot suoraan GPU:lle | offload = painot RAM:iin, kerrokset GPU:lle
    # | quantized = 4-bit-kvantisoitu transformer
    load_mode: str
    note: str


TIERS: dict[str, TierPolicy] = {
    "low": TierPolicy(
        name="low",
        profile="wan2.2-ti2v-5b",
        load_mode="quantized",
        note="12-20 GB VRAM: 4-bit-kvantisoitu transformer",
    ),
    "mid": TierPolicy(
        name="mid",
        profile="wan2.2-ti2v-5b",
        load_mode="bf16",
        note="20-30 GB VRAM: TI2V-5B täydellä bf16-tarkkuudella",
    ),
    "high": TierPolicy(
        name="high",
        profile="wan2.2-t2v-a14b",
        load_mode="bf16",
        note="30 GB+ VRAM: A14B molemmat asiantuntijat residenttinä",
    ),
    "a14b-offload": TierPolicy(
        name="a14b-offload",
        profile="wan2.2-t2v-a14b",
        load_mode="offload",
        note="24 GB VRAM + ~60 GB RAM: A14B CPU-offloadilla, hidas",
    ),
}


def suggest_tier(vram_gib: float, system_ram_gib: float | None) -> tuple[str, str]:
    """Päättelee VRAM-tierin. Palauttaa (tier, perustelu).

    Perustelu palautetaan, koska hiljainen automaattivalinta on vaikea debugata:
    palvelin logittaa sen käynnistyksessä.

    Järjestelmämuisti on mukana tarkoituksella. A14B:n CPU-offload vaatii
    painojen mahtumisen RAM:iin, joten pelkkä VRAM-tarkistus antaisi liian
    ruusuisen kuvan siitä mitä kone pystyy ajamaan.
    """
    ram_txt = "tuntematon" if system_ram_gib is None else f"{system_ram_gib:.0f} GB"

    if vram_gib >= 30:
        return "high", f"{vram_gib:.0f} GB VRAM riittää A14B:lle bf16-tarkkuudella"
    if vram_gib >= 20:
        if system_ram_gib is not None and system_ram_gib >= 60:
            return "mid", (
                f"{vram_gib:.0f} GB VRAM + {ram_txt} RAM: TI2V-5B natiivisti, "
                "A14B mahdollinen offload-tilassa"
            )
        return "mid", (
            f"{vram_gib:.0f} GB VRAM: TI2V-5B bf16. A14B-offload ei onnistu "
            f"(vaatii ~60 GB RAM, koneessa {ram_txt})"
        )
    if vram_gib >= 11:
        return "low", f"{vram_gib:.0f} GB VRAM: TI2V-5B kvantisoituna, CPU-offload sallittu"
    return "unsupported", f"{vram_gib:.0f} GB VRAM on liian vähän Wan 2.2:lle"


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Muotoa 1280x704 oleva merkkijono -> (1280, 704). ValueError jos väärä muoto."""
    width, _, height = resolution.lower().partition("x")
    return int(width), int(height)
