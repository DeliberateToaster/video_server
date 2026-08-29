# Video Server — tekninen spesifikaatio

> **Revisio 2** (2026-08-28). Ensimmäinen versio oletti, että Wan 2.2 14B ajetaan
> täydellä bfloat16-tarkkuudella 24 GB:n kortilla. Tämä ei pidä paikkaansa (ks.
> Kohdemalli), joten mallivalinta, VRAM-tierit ja niistä riippuvat validoinnit on
> kirjoitettu uusiksi. `## API-rajapinta`-osion kontrakti on säilynyt
> yhteensopivana; siihen on vain lisätty kenttiä.

## Tausta ja tavoite

Rakennetaan kevyt FastAPI-palvelin, joka kääri Wan 2.2 -videogenerointimallin
suoran Python-inferenssin tasaisen REST-rajapinnan taakse. Arkkitehtuurin esikuva on
aiempi `forge_ui` -projekti (Stable Diffusion WebUI Forge -frontti): kaikki
epäolennainen (sampler, scheduler, VAE-käsittely) on hardcodettu palvelimen sisään,
ja rajapinta paljastaa vain sen mitä käyttäjän oikeasti tarvitsee säätää.

Tietoinen valinta: **ei ComfyUI:ta taustalla.** ComfyUIn API vastaanottaa koko
workflow-graafin JSON:na, mikä on tarpeeton monimutkaisuus ja ylimääräinen
riippuvuus muille kehittäjille. Sen sijaan ajetaan mallin virallista
diffusers-pohjaista inferenssiä suoraan, ja rajapinnan skeeman päättää tämä spekti,
ei mallin sisäinen graafiformaatti.

Kohdelaitteisto: yksi RTX 3090 (24 GB VRAM), 32 GB järjestelmämuistia. Yksi malli
residenttinä muistissa kerrallaan, yksi generointi kerrallaan.

---

## Kohdemalli

**Wan 2.2** — Apache 2.0 -lisenssi, T2V + I2V, ei sisäänrakennettua audiota.
Ajetaan diffusers-pipelinen kautta virallisesta inferenssitoteutuksesta,
ei ComfyUI-noodien kautta.

Wan 2.2 ei ole yksi malli vaan perhe, ja variantin valinta on tämän projektin
tärkein yksittäinen tekninen päätös, koska se määrää mahtuuko malli korttiin:

| Variantti | Rakenne | Painot (bf16) | 24 GB:n kortilla |
|---|---|---|---|
| `T2V-A14B` / `I2V-A14B` | MoE: kaksi ~14B-asiantuntijaa (high-noise + low-noise), vaihto boundary-timestepissä, ~27B yhteensä | ~54 GB | **ei mahdu** ilman offloadia tai kvantisointia |
| `TI2V-5B` | yksi tiheä 5B-malli, yhdistetty T2V + I2V | ~10 GB VRAM (31,9 GB levyllä) | mahtuu, tämän projektin oletus |

Seuraukset, jotka on syytä lukea ennen kuin mitään asennetaan:

- **A14B ei aja täydellä tarkkuudella RTX 3090:llä.** Vaihtoehdot ovat
  `enable_model_cpu_offload()` (mahtuu, mutta hidas ja vaatii ~64 GB
  järjestelmämuistia) tai kvantisoidut painot (GGUF / NF4).
- **RTX 3090 on Ampere-sukupolvea eikä sisällä FP8-tensoriytimiä.** FP8 on
  kortilla vain tallennusmuoto, josta puretaan bf16:een laskentaa varten — se
  säästää VRAM:ia mutta ei nopeuta. Tämä koskee suoraan alempaa tier-taulukkoa.
- **Natiivi resoluutio ja fps riippuvat variantista:** A14B on 16 fps, TI2V-5B
  24 fps ja 720p. Tästä syystä sallitut resoluutiot ja fps eivät ole globaaleja
  vakioita vaan malliprofiilin kenttiä (ks. Malliprofiilit).

**Päätös:** oletusprofiili on `TI2V-5B`, koska se ajaa kohdelaitteistolla
sellaisenaan ja antaa toimivan end-to-end-polun. A14B tuetaan samalla
backendillä offload- tai kvantisointiprofiilin takana, ei oletuksena.

Backend-rajapinta suunnitellaan alusta asti niin, että toinen moottori
(esim. LTX-2.3, jos audio-synkronointi tulee tarpeeseen myöhemmin) voidaan
lisätä samaan kontraktiin ilman rajapinnan rikkomista.

> Toteutusvaiheessa varmistettavat yksityiskohdat diffusersin ajantasaisesta
> Wan-dokumentaatiosta: painojen tarkat koot, `boundary_ratio`:n oletus,
> kummankin variantin sallitut resoluutiot ja frame-sääntö. Tämän osion
> *rakenne* pitää, yksittäiset luvut on tarkistettava.

---

## Projektirakenne

```
video_server/
  main.py                  # FastAPI-app, reitit, lifespan
  settings.py              # ajonaikaiset arvot: .env / config.yaml -> Settings
  config.py                # skeema, fallback-oletukset, malliprofiilit
  schemas.py               # Pydantic-mallit pyynnöille/vastauksille
  jobs.py                  # jonorakenne, job-tila, worker-loop
  backends/
    base.py                # abstrakti VideoBackend-rajapinta
    registry.py            # nimi -> backend-luokka, aktiivisen valinta
    mock.py                # GPU:ton testibackend (kehitys + CI)
    wan22.py               # Wan 2.2 -toteutus
  outputs/                 # valmiit mp4-tiedostot, tarjoillaan staattisesti
  tests/                   # ajetaan mock-backendillä, ei vaadi GPU:ta
```

Kaksi lisäystä alkuperäiseen rakenteeseen:

- **`settings.py` erillään `config.py`:stä.** `config.py` määrittää skeeman,
  fallback-oletukset ja malliprofiilit; `settings.py` lukee lopulliset arvot
  ympäristöstä. Ilman tätä jakoa "config on skeema, ei arvoja" -periaate ei
  näy tiedostorakenteessa.
- **`backends/mock.py`.** Ilman GPU:tonta backendiä koko rajapintaa ei voi
  kehittää eikä testata ilman 3090:tä, eikä CI:tä voi ajaa lainkaan.

---

## Arkkitehtuuriperiaatteet

### 1. Malli ladataan kerran, pysyy residenttinä

Toisin kuin kuvageneroinnissa, videomallin lataus VRAM:iin on hidas ja raskas
operaatio. Palvelin lataa oletusmallin käynnistyksessä (lifespan-hookissa) ja pitää
sen muistissa koko prosessin eliniän. Mallin vaihto — jos toinen backend on
rekisteröity — on eksplisiittinen, hidas admin-operaatio, **ei** parametri joka
annetaan per generointipyyntö.

Koska lataus kestää minuutteja, palvelin ottaa yhteyksiä vastaan ennen kuin se on
valmis palvelemaan. Tämä on eksplisiittinen tila: generointi-endpointit palauttavat
`503` kunnes backend on ladattu (ks. `GET /api/v1/health`).

### 2. Yksi työ kerrallaan

24 GB riittää yhteen ajoon kunnolla, ei rinnakkaisiin. Toteutus: **yksi
worker-taski, joka kuluttaa `asyncio.Queue`-jonoa.** Erillistä globaalia lukkoa
ei tarvita — yksi kuluttaja *on* poissulkeminen, ja lukko sen rinnalla olisi vain
toinen paikka, jossa mennä vikaan.

Ajot kestävät minuutteja, joten asiakas ei pidä HTTP-yhteyttä auki koko ajan —
pyyntö palauttaa heti `job_id`:n, ja tilaa pollataan erikseen.

Jonolla on konfiguroitava maksimipituus. Täydessä jonossa `POST` vastaa `429`,
jotta asiakkaan uudelleenyrityssilmukka ei kasvata jonoa rajatta.

### 3. Asynkroninen job-malli, ei streaming-vastausta

`POST`-endpointit palauttavat `202 Accepted` + `job_id` välittömästi. Työ tehdään
taustalla worker-loopissa. Tila haetaan `GET /api/v1/jobs/{id}` -kutsulla.

### 4. Video palautetaan URL:na, ei base64:nä

Kuvat mahtuvat JSON-vastaukseen base64-koodattuna (näin `forge_ui` tekee).
Video-tiedostot ovat kokoluokkaa suurempia — palautetaan sen sijaan URL valmiiseen
mp4-tiedostoon, jota tarjoillaan staattisesti `/outputs/`-polusta.

### 5. Progress on karkeampaa kuin kuvageneroinnissa

Diffusersin `callback_on_step_end` antaa denoising-askeleen etenemän koko
video-tensorille kerralla, ei per-framea.

- **v1: `step / total_steps` + arvioitu ETA** (kuten Forgen progress bar, mutta
  ilman live-preview-kuvaa). ETA lasketaan liukuvana keskiarvona askelten
  kestoista, **ei** kaavalla `kulunut / askel × jäljellä`: ensimmäinen askel
  sisältää lämmittelyn ja vääristäisi arviota pahasti.
- **Ei v1:ssä: VAE-dekoodattu esikatselukuva.** Ylimääräinen VAE-kutsu
  pollauspolulla, ja ETA kattaa tarpeesta valtaosan. Backend-rajapinta
  kirjoitetaan kuitenkin niin, että lisäys myöhemmin ei riko kontraktia
  (progress välitetään oliona, ei kahtena inttinä).

### 6. Kaikki epäolennainen hardcodetaan

Sampler, scheduler, VAE-tarkkuus ja muut Wan 2.2:n sisäiset oletusarvot
kiinnitetään malliprofiiliin. Rajapinta paljastaa vain kentät, joita käyttäjä
oikeasti säätää (ks. alla).

---

## Malliprofiilit

Arvot, jotka alkuperäisessä speksissä olivat globaaleja vakioita, ovat itse asiassa
mallikohtaisia. Ne kootaan yhdeksi profiiliksi, jonka aktiivinen backend julkaisee
ja jota vasten pyynnöt validoidaan:

```python
@dataclass(frozen=True)
class ModelProfile:
    name: str  # "wan2.2-ti2v-5b"
    repo_id: str  # HuggingFace-tunniste
    frame_multiple: int  # num_frames-sääntö: n * k + 1
    native_fps: int  # 24 (5B) / 16 (A14B)
    resolutions: list[str]  # sallitut "LxK"-merkkijonot
    default_steps: int
    default_shift: float
    default_guidance: float
    default_guidance_2: float | None  # A14B: toinen asiantuntija
    supports_i2v: bool
```

`num_frames`-sääntö on erityisen huomionarvoinen: alkuperäinen speksi sanoi
`8n+1`, mutta Wanin temporaalinen VAE pakkaa 4x, joten todellinen rajoite on
`4n+1` - tämä on Vaiheessa 2 varmistettu (oletusarvo 81 = 4·20+1 täyttää molemmat, mikä piilottaa
eron). Sääntö tulee profiilista eikä koodista, koska variantit voivat poiketa
toisistaan.

---

## API-rajapinta

Kaikki polut alkavat prefiksillä `/api/v1`.

### `POST /api/v1/txt2vid`

Aloittaa tekstistä videoksi -generoinnin.

**Pyyntörunko:**

```json
{
  "prompt": "string, pakollinen",
  "negative_prompt": "string, valinnainen, tyhjä oletus",
  "num_frames": 81,
  "fps": 24,
  "resolution": "1280x704",
  "seed": -1,
  "shift": 5.0,
  "guidance_scale": 5.0,
  "guidance_scale_2": null
}
```

Huomioita kenttiin:

- `num_frames` — noudattaa aktiivisen profiilin `frame_multiple`-sääntöä
  (`n × k + 1`). Validoi ja hylkää `400`-vastauksella jos ei täsmää, älä hiljaa
  pyöristä. Virheviestissä on kerrottava mikä sääntö on voimassa ja mitkä
  lähimmät kelvolliset arvot ovat.
- `fps` — **tämä on ulostulotiedoston fps, ei generointiparametri.** Malli
  tuottaa framet omalla natiivinopeudellaan; tämä kenttä kertoo vain millä
  nopeudella ne muxataan mp4:ään. Oletus tulee profiilin `native_fps`:stä.
  Poikkeava arvo tuottaa hidastuksen tai nopeutuksen, ei eri määrää frameja.
- `resolution` — Wan tukee rajattua settiä vapaan leveys×korkeus-yhdistelmän
  sijaan. Sallittu lista tulee aktiivisesta profiilista ja tieristä, joten
  validointi tapahtuu ajonaikaista konfiguraatiota vasten (ei staattisena
  `Literal`-tyyppinä). Kelpaamaton arvo → `400` ja lista sallituista.
- `shift` — Wanin flow-matching-parametri, ei suoraa vastinetta Forge/A1111
  -maailmassa. Oletusarvo profiilista, mutta paljastetaan säädettävänä koska
  vaikuttaa merkittävästi liikkeen laatuun.
- `guidance_scale` / `guidance_scale_2` — Wan 2.2:n A14B-variantilla on **kaksi**
  guidance-arvoa, yksi kummallekin asiantuntijalle (high-noise / low-noise),
  jotka vaihtuvat `boundary_ratio`-kohdassa. Yhden kentän rajapinta romahduttaisi
  kaksi säädintä yhdeksi hiljaisesti. `guidance_scale_2` on valinnainen: `null`
  tarkoittaa "käytä profiilin oletusta", ja yhden asiantuntijan malleilla
  (TI2V-5B) kenttä ohitetaan.
- `seed` — `-1` tarkoittaa satunnaista, kuten Forgessa. Arvottu siemen
  ratkaistaan jobia luotaessa ja **palautetaan job-vastauksessa**, jotta ajo on
  toistettavissa.

**Vastaus (`202`):**

```json
{ "job_id": "uuid" }
```

**Virhevastaukset:** `400` validointivirheestä, `429` jos jono on täynnä,
`503` jos backend ei ole vielä latautunut.

### `POST /api/v1/img2vid`

Sama runko kuin `txt2vid`, plus:

```json
{
  "init_image": "base64-koodattu PNG tai JPEG (ilman data-URI-prefiksiä)"
}
```

**Päätös: base64 JSON-rungossa, ei multipartia.** 832×480 PNG on ~1 MB, base64:nä
~1,4 MB — multipartin tehokkuusetu on tässä kokoluokassa merkityksetön, ja se
pakottaisi toisen, eri muotoisen endpointin ja toisen Pydantic-mallin. Yksi
skeema, yksi malli, curl-attava. Rungon maksimikoko on konfiguroitava
(oletus 20 MB) ja ylitys vastaa `413`.

`prompt` on tässä valinnainen ohjaava teksti. Jos aktiivisen profiilin
`supports_i2v` on `false`, endpoint vastaa `501`.

### `GET /api/v1/jobs/{job_id}`

```json
{
  "job_id": "uuid",
  "status": "queued | running | done | failed | cancelled",
  "progress": {
    "step": 12,
    "total_steps": 30,
    "eta_seconds": 45.2
  },
  "params": { "...": "efektiiviset parametrit, seed ratkaistuna" },
  "created_at": "2026-08-28T12:00:00Z",
  "video_url": "/outputs/{job_id}.mp4",
  "error": "string"
}
```

`video_url` vain kun `status == done`, `error` vain kun `status == failed`.
`params` sisältää arvot sellaisina kuin backend ne todella sai — erityisesti
ratkaistun siemenen — jotta onnistunut ajo on toistettavissa ilman että asiakkaan
täytyy muistaa mitä lähetti.

### `POST /api/v1/jobs/{job_id}/cancel`

Keskeyttää käynnissä olevan tai jonossa olevan työn. Palauttaa `200` ja
päivitetyn job-tilan. Jos työ on jo `done`, `failed` tai `cancelled`, palauta
`409`.

Jonossa oleva työ perutaan pelkällä tilanvaihdolla; worker ohittaa sen. Käynnissä
oleva työ keskeytetään nostamalla poikkeus `callback_on_step_end`-callbackissa
(ks. jonon ja worker-loopin logiikka).

### `GET /api/v1/models`

Listaa rekisteröidyt backendit, aktiivisen ja sen ajonaikaiset rajoitteet. Aluksi
vain yksi rivi, mutta rakenne varautuu useampaan.

```json
{
  "active": "wan2.2-ti2v-5b",
  "available": ["wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "mock"],
  "tier": "mid",
  "constraints": {
    "resolutions": ["1280x704", "704x1280"],
    "frame_multiple": 4,
    "native_fps": 24,
    "supports_i2v": true
  }
}
```

`constraints` on se osa, jonka asiakas tarvitsee rakentaakseen kelvollisen
pyynnön ilman että sen täytyy tietää palvelimen tieristä tai laitteistosta.

### `GET /api/v1/health`

```json
{ "status": "loading | ready | error", "backend": "wan2.2-ti2v-5b", "detail": "string" }
```

Mallin lataus kestää minuutteja, joten "onko palvelin pystyssä" ja "voiko sille
lähettää työtä" ovat eri kysymyksiä. Tämä endpoint vastaa jälkimmäiseen ja
palauttaa aina `200`; generointi-endpointit vastaavat `503` kunnes tila on `ready`.

### `GET /outputs/{filename}`

Staattinen tiedostopalvelu valmiille mp4-tiedostoille. Ei autentikointia (ks.
rajaukset) — tiedostonimi on job-UUID, mikä ei ole pääsynhallintaa vaan
ainoastaan estää nimien arvaamisen.

---

## Backend-abstraktio (`backends/base.py`)

Abstrakti luokka jota `wan22.py` ja `mock.py` toteuttavat, jotta toinen moottori
voidaan lisätä myöhemmin rikkomatta `main.py`:n tai `jobs.py`:n koodia:

```python
@dataclass
class Progress:
    step: int
    total_steps: int
    preview_path: Path | None = None  # varattu; v1 jättää None:ksi


class VideoBackend(ABC):
    @property
    def profile(self) -> ModelProfile:
        """Aktiivisen mallin rajoitteet ja oletukset. Luettavissa ennen
        load()-kutsua, jotta validointi ja /models toimivat latauksen aikana."""

    def load(self) -> None:
        """Lataa mallin VRAM:iin. Kutsutaan kerran käynnistyksessä."""

    def generate(
        self,
        params: GenerationParams,
        on_progress: Callable[[Progress], None],
    ) -> Path:
        """Ajaa generoinnin, kutsuu on_progress per askel, palauttaa
        polun valmiiseen mp4-tiedostoon. Nostaa GenerationCancelled jos
        peruutus havaitaan."""
```

Kaksi muutosta alkuperäiseen:

- **`on_progress` saa olion, ei kahta inttiä.** Esikatselukuvan lisääminen
  myöhemmin on tällöin additiivinen muutos, ei rajapinnan rikkova.
- **Ei erillistä `cancel()`-metodia.** Peruutus kulkee `on_progress`-callbackin
  kautta: callback nostaa poikkeuksen, joka etenee ulos diffusersin pipelinesta.
  Erillinen `cancel()` vaatisi backendiltä oman säietilan ja olisi toinen
  totuuden lähde peruutukselle.

`GenerationParams` on Pydantic-malli joka kattaa sekä `txt2vid`- että
`img2vid`-kentät; backend päättelee moodin siitä onko `init_image` läsnä.

### Mock-backend

`backends/mock.py` toteuttaa saman rajapinnan ilman GPU:ta: nukkuu
`default_steps` kertaa lyhyen hetken, kutsuu `on_progress` jokaisen askeleen
jälkeen ja kirjoittaa pienen synteettisen mp4:n. Sen avulla koko rajapinta,
jono, peruutus ja retention ovat kehitettävissä ja testattavissa sekunneissa
ilman mallin latausta. CI ajaa tätä vasten.

---

## Jonon ja worker-loopin logiikka (`jobs.py`)

- Yksinkertainen `Job`-dataclass: `id`, `status`, `params`, `progress`,
  `output_path`, `error`, `created_at`.
- In-memory `dict[str, Job]` + `asyncio.Queue` riittää alkuun — ei tarvita
  ulkoista queue-systeemiä (Redis, Celery) yhden GPU:n / yhden workerin
  skaalalla. Dokumentoi tämä eksplisiittisenä rajoitteena koodikommenttiin,
  jotta tuleva laajennus (useampi GPU) tietää mistä lähteä.
- Worker-loop: yksi `asyncio`-tausta-taski joka ottaa seuraavan jobin jonosta ja
  ajaa sen `run_in_executor`-säikeessä (Wan-inferenssi on GPU-bound ja blokkaava,
  eikä saa pysäyttää event loopia).
- **Säierajapinta:** `on_progress` suoritetaan executor-säikeessä, mutta job-tila
  elää event loopissa. Päivitykset viedään loopin puolelle
  `loop.call_soon_threadsafe`-kutsulla. Tämä on helppo unohtaa ja tuottaa
  satunnaisesti vanhentunutta progressia.
- **Peruutus:** worker asettaa jobille flagin; `on_progress`-käärö tarkistaa sen
  ja nostaa `GenerationCancelled`, joka etenee ulos diffusersin pipelinesta.
  Käytännössä ajo katkeaa kesken kuluvan askeleen sen sijaan että odottaisi sen
  loppuun — mikä on parempi kuin alkuperäisen speksin "keskeytä seuraavan
  askeleen jälkeen", ilman lisämonimutkaisuutta.
- **Kestävyys uudelleenkäynnistyksen yli:** job-tila on muistissa, mp4:t levyllä.
  Ilman mitään tehtyä `GET /jobs/{id}` vastaa `404` videolle joka on olemassa.
  Ratkaisu: jokaisen valmiin ajon viereen kirjoitetaan `{job_id}.json`-sidecar,
  ja job-hakemisto ladataan siitä käynnistyksessä. ~10 riviä, ei uutta
  riippuvuutta, ei tietokantaa.

---

## Konfiguraatio (`settings.py` + `config.py`)

`config.py` määrittää skeeman, fallback-oletukset ja malliprofiilit.
`settings.py` lukee lopulliset arvot `.env`-tiedostosta tai `config.yaml`:sta
(`pydantic-settings`). Kiinnitettävät arvot:

- Aktiivinen backend / malliprofiili ja HuggingFace-repo-tunniste
- VRAM-tier (tai `auto`)
- Laite (`cuda:0`, `cuda:1`, `cpu`), dtype (esim. `bfloat16`)
- Sallitut resoluutiot (profiilin oletus, ohitettavissa)
- Oletus `shift`, `guidance_scale`, `num_inference_steps`
- `outputs/`-hakemiston polku ja retention: `max_age_days` **ja** `max_total_gb`
- Jonon maksimipituus ja pyyntörungon maksimikoko

---

## VRAM-tierit ja lataustavat

Tier ei ole pelkkä tarkkuusasetus vaan **yhdistelmä mallivarianttia ja
lataustapaa**. Alkuperäisen speksin taulukko lupasi täyden bf16-tarkkuuden
24 GB:llä, mikä ei A14B:llä ole mahdollista.

| Tier | VRAM | Variantti + lataustapa | Huomioita |
|---|---|---|---|
| `low` | 12-20 GB | TI2V-5B 4-bit-kvantisoituna (NF4) | vaatii `quantized`-extran; toteutettu mutta testaamaton |
| `mid` | 20-30 GB | TI2V-5B bf16 - **RTX 3090:n oletus** | mitattu ~23 GB varausta latauksen jälkeen |
| `high` | 30 GB+ | A14B bf16, molemmat asiantuntijat residenttinä | laajin resoluutiolista |
| `a14b-offload` | 24 GB VRAM + ~64 GB RAM | A14B `enable_model_cpu_offload()`-tilassa | **ei toimi kehityskoneella**, ks. alla |

**`a14b-offload` ei ole vaihtoehto tällä laitteistolla.** CPU-offload pitää painot
järjestelmämuistissa ja siirtää kerroksia GPU:lle tarpeen mukaan, joten A14B:n
~54 GB bf16-painot vaativat vähintään sen verran RAM:ia. Kehityskoneessa on 32 GB.
Tier on jätetty taulukkoon dokumentoituna vaihtoehtona sellaiselle koneelle, jossa
RAM riittää — tällä koneella A14B vaatisi kvantisoidut painot levyltä, ei offloadia.
Tämä vahvistaa `TI2V-5B`:n valinnan oletukseksi: se on ainoa variantti, joka ajaa
kohdelaitteistolla ilman kvantisointia.

Huomaa: FP8 ei esiinny taulukossa siinä merkityksessä kuin alkuperäisessä.
Ampere-korteilla (RTX 3090) ei ole FP8-tensoriytimiä, joten FP8-painot
puretaan bf16:een laskentaa varten — VRAM-säästö on todellinen, nopeutus ei.
Ada/Hopper-korteilla tilanne on toinen, ja tier-taulukko saa laajeta sitä varten
kun sellaista laitteistoa vasten testataan.

Automaattitunnistus (`torch.cuda.get_device_properties(0).total_memory`) asettaa
oletustierin, mutta käyttäjä voi ohittaa sen eksplisiittisesti konfiguraatiossa
(esim. jos haluaa tietoisesti ajaa kevyempää profiilia nopeuden vuoksi vaikka
VRAM riittäisi täyteen tarkkuuteen). Valittu tier ja sen perustelu logitetaan
käynnistyksessä — hiljainen automaattivalinta on vaikea debugata.

---

## Riippuvuuksien hallinta ja asennusohje

- `pyproject.toml` pinnatuilla versioilla — erityisesti `torch`, `diffusers`,
  `transformers` ja CUDA-versio, koska video-mallit ovat herkkiä näiden
  yhdistelmille.
- Python 3.11 tai 3.12, **ei 3.13**: torch- ja diffusers-wheel-kattavuus.
- Lyhyt `README.md` jossa: asennuskomento, `.env`-esimerkki eri tiereille,
  ensimmäisen käynnistyksen kulku, tyypilliset virheilmoitukset (CUDA OOM,
  versioristiriita, puuttuvat painot) ja mitä niille tehdä.
- **Painojen lataus: eksplisiittinen.** Erillinen `scripts/download_model.py`
  ajaa `snapshot_download`:n valitulle profiilille, ja palvelin tarkistaa
  käynnistyksessä että painot löytyvät — puuttuessa selkeä virheilmoitus ja
  komento jolla ne haetaan. Kymmenien gigatavujen hiljainen lataus ensimmäisen
  API-kutsun yhteydessä on huono oletus.

---

## Laitteistoabstraktio

- `cuda:0` kiinteänä oletuksena korvataan konfiguroitavalla laite-tunnisteella.
- ROCm (AMD) -tuki jätetään eksplisiittisesti pois v1:n scopesta, mutta
  dokumentoidaan README:ssä tunnettuna rajoitteena, jotta se ei näytä
  unohdukselta.
- Moni-GPU (useampi kortti samassa koneessa, ei rinnakkaisajo vaan valinta
  *kumpaa* käytetään) — konfiguroitava, ei koodattava.

---

## Nimenomaisesti rajauksen ulkopuolella (v1)

- Ei rinnakkaisia ajoja / useampia GPU:ita.
- Ei autentikointia (lisätään erikseen jos palvelin altistetaan verkkoon).
- Ei audiota (Wan 2.2:lla ei ole sitä; jos LTX-2.3 lisätään myöhemmin taustaksi,
  audio-kentät lisätään rajapintaan sen yhteydessä, ei nyt).
- Ei toisen moottorin (LTX) toteutusta — vain rajapinta suunnitellaan sen varalle.
- Ei VAE-dekoodattua esikatselukuvaa progressissa.
- Ei tietokantaa job-historialle; sidecar-JSON riittää.

---

## Ratkaistut avoimet kysymykset

Alkuperäisen speksin kolme avointa kysymystä, päätettyinä:

1. **`init_image`: base64 JSON-rungossa.** Yksi skeema ja yksi Pydantic-malli
   voittaa multipartin tehokkuusedun kokoluokassa ~1 MB. Rungon kokoraja
   konfiguroitava, ylitys → `413`.
2. **Progress-esikatselu: ei v1:ssä.** ETA kattaa tarpeen; backend-rajapinta
   varautuu lisäykseen `Progress.preview_path`-kentällä.
3. **`outputs/`-siivous: molemmat säännöt, konfiguraatiosta.** `max_age_days` ja
   `max_total_gb`, ajetaan käynnistyksessä ja jokaisen valmistuneen työn jälkeen.

Jäljelle jäävät, toteutusvaiheessa varmistettavat:

- Wan-varianttien tarkat resoluutiolistat, `frame_multiple` ja `boundary_ratio`
  diffusersin ajantasaisesta dokumentaatiosta.
- Tuottaako TI2V-5B riittävän laadun käyttötapaukseen, vai onko A14B
  offload-tilassa hitaudestaan huolimatta välttämätön.

---

## Toteutusvaiheet

**Vaihe 0 — ympäristö.** Python 3.11/3.12, `uv`, torch CUDA-wheelit, ffmpeg.
Kohdekone on tällä hetkellä tyhjä: ei Pythonia, ei uv:tä, ei ffmpegiä.

**Vaihe 1 — koko rajapinta mock-backendillä, ilman GPU:ta.** Settings, skeemat,
job-store, worker-loop, kaikki endpointit, staattinen tarjoilu, retention,
testit. Tämä toteuttaa `## API-rajapinta`-osion kokonaan ja on ajettavissa
sekunneissa. Se on myös se, mitä CI ajaa pysyvästi.

**Vaihe 2 — oikea Wan-backend.** Painojen lataus, malli käynnistyksessä,
`generate()`, peruutus, todellinen ETA. Ensin TI2V-5B end-to-end-polun
todistamiseksi, sitten A14B offload-profiililla. **Aikataulu kuluu tähän
vaiheeseen**, ei rajapintaan: riippuvuuksien yhteensovittaminen ja ensimmäinen
onnistunut ajo ovat työn todellinen paino.

**Vaihe 3 — yleistys.** Tierin automaattitunnistus ja ohitus, profiilikohtaiset
resoluutiolistat ja lataustavat, `GET /api/v1/models` kertomaan aktiivisen
tierin, README virhetaulukkoineen, pinnatut riippuvuudet.

**Vaihe 4 — valinnaiset.** Esikatselukuvat, job-historian pysyvyys, autentikointi.

---

## Vaiheessa 2 tarkistetut arvot

Speksi merkitsi joukon arvoja toteutusvaiheessa varmistettaviksi. Lähteenä on
mallien oma konfiguraatio HuggingFacessa (model_index.json, vae/config.json,
scheduler/scheduler_config.json, mallikortit) ja asennetun diffusersin (0.40.0)
lähdekoodi - ei dokumentaatio muistinvaraisesti.

| Asia | Speksin oletus | Todellisuus |
|---|---|---|
| `num_frames`-sääntö | `8n+1` | **`4n+1`**. Pipeline vaatii `num_frames % scale_factor_temporal == 1`, ja molempien varianttien VAE:lla arvo on 4 |
| `shift` | pipelinen parametri | **schedulerin `flow_shift`**, ei kutsuparametri. Scheduler on rakennettava uudelleen per pyyntö |
| `shift`-oletus | 5.0 kaikille | checkpointkohtainen: **5.0** (TI2V-5B), **3.0** (A14B) |
| `boundary_ratio` | tuntematon | **0.875** (T2V-A14B), **0.9** (I2V-A14B), **null** (TI2V-5B) |
| `guidance_scale_2` yhden asiantuntijan mallilla | ohitetaan hiljaisesti | **nostaa ValueErrorin**. Sitä ei saa välittää lainkaan |
| A14B:n guidance-oletukset | arvattu 3.0 / 4.0 | **4.0 / 3.0** (mallikortti) |
| TI2V-5B:n I2V-tuki | oletettu | **vahvistettu**: `image_encoder` ja `image_processor` ovat valinnaisia, eikä Wan 2.2 I2V käytä niitä |
| TI2V-5B:n koko | ~10 GB | **31,9 GB levyllä**, ~10 GB VRAM:issa. Repo sisältää fp32-painot; levytila ja VRAM-tarve ovat eri asia |

Kaksi asiaa, jotka vahvistivat speksin ratkaisut:

- **diffusers pyöristää kelvottoman `num_frames`-arvon hiljaisesti** varoituksen
  kera. Speksin vaatimus hylätä se `400`-vastauksella on siis nimenomaan se,
  mikä estää hiljaisen pyöristyksen - validointi tapahtuu ennen pipelinea.
- **`guidance_scale_2` on aito kaksoissäädin.** Yhden kentän rajapinta olisi
  paitsi hukannut säätimen, myös kaatanut TI2V-5B-ajon, koska arvon
  välittäminen yhden asiantuntijan mallille on virhe eikä no-op.

### Seuraukset toteutukseen

- Profiili kantaa fallback-arvot validointiin ennen latausta; latauksen jälkeen
  `frame_multiple` ja `default_shift` synkronoidaan checkpointista. Checkpoint
  on totuuden lähde niille arvoille jotka se kantaa. Käyttäjän eksplisiittistä
  konfiguraatiota ei ylikirjoiteta.
- `wan2.2-i2v-a14b` lisättiin omana profiilinaan, ja profiiliin tuli
  `supports_t2v`. I2V-checkpoint ei tee tekstistä videota, joten `txt2vid`
  vastaa sillä `501` samalla tavalla kuin `img2vid` T2V-mallilla.
- TI2V-5B ajaa molemmat suunnat samoilla painoilla: I2V-pipeline rakennetaan
  T2V-pipelinen komponenteista, jolloin toista latausta tai lisä-VRAM:ia ei
  tarvita. Tässä on välitettävä myös `expand_timesteps`, jonka TI2V-5B asettaa.

### Vaiheen 2 mittaukset oikealla laitteistolla

Ensimmäinen onnistunut ajo: RTX 3090 (24 GB), TI2V-5B, 1280x704, 25 framea,
8 askelta. Ulostulo h264 1280x704 24 fps, 1,04 s, 985 kt.

| Mitattu | Arvo |
|---|---|
| Mallin lataus (painot paikallisella levyllä) | ~50 s |
| VRAM latauksen jälkeen, muuten vapaa kortti | **~23 GB** |
| Generointi 25 framea / 8 askelta | ~11 min, josta VAE-dekoodaus ~2 min |

Kolme asiaa, jotka mittaus paljasti:

- **VRAM-arvio oli pahasti pielessä.** Profiilin `min_vram_gib` oli 15 GB, mutta
  todellinen varaus on ~23 GB (bf16-transformer + UMT5-XXL-tekstikooderi +
  fp32-VAE + allokaattorin varaukset). 16 GB:n kortilla vanha arvo olisi
  luvannut mahtumisen ja kaatunut vasta ajossa. Korjattu arvoon 20 GB, ja
  `suggest_tier`in mid-rajaa nostettiin vastaavasti.
- **VAE-dekoodaus on ajon suurin muistipiikki, ei denoising.** Ensimmäinen ajo
  kaatui `OutOfMemoryError`:iin vasta kun kaikki 8 denoising-askelta olivat
  valmiit: dekoodaus yritti varata 2,15 GB kun kortilla oli 9 GB muun käytön
  varaamana. Tiilitetty dekoodaus (`vae.enable_tiling()`) on nyt oletuksena
  päällä; ilman sitä ajo on riippuvainen siitä ettei kortilla ole muuta.
- **Progress kattaa vain denoisingin.** `callback_on_step_end` ei tiedä
  dekoodauksesta, joten job näyttää tilaa `8/8` ja `eta_seconds: 0`, vaikka
  ajoa on jäljellä minuutteja. Tämä on rajapinnan tiedossa oleva rajoite, ei
  bugi: dekoodaukselle ei ole askelittaista callbackia. Jos tämä osoittautuu
  häiritseväksi, luonteva korjaus on erillinen `phase`-kenttä
  (`denoising | decoding`) job-vastaukseen - additiivinen muutos, joka ei riko
  nykyistä kontraktia.

Virhepolku toimi kuten suunniteltu: OOM ei kaatanut palvelinta eikä workeria,
job siirtyi tilaan `failed` virheviestin kanssa, ja seuraavat pyynnöt toimivat.

---

## Vaihe 3: tier ohjaa oikeasti

Vaiheen 2 jäljiltä tier oli **pelkkä logirivi**: se tunnistettiin
käynnistyksessä, raportoitiin `GET /api/v1/models`-vastauksessa eikä
vaikuttanut mihinkään. `TIER_PROFILES`-taulu oli koodissa, mutta mikään ei
lukenut sitä. Vaihe 3 kytki sen kiinni.

### Tier valitsee mallin ja lataustavan

`TierPolicy` kertoo, mitä tier käytännössä tarkoittaa: minkä profiilin se
valitsee ja millä lataustavalla (`bf16` / `offload` / `quantized`).

- `VIDEO_SERVER_BACKEND=auto` (uusi oletus) valitsee mallin tunnistetun tierin
  mukaan. Tämä on speksin yleistystavoite konkreettisena: sama konfiguraatio
  toimii eri laitteistoilla ilman koodimuutosta.
- Eksplisiittinen `VIDEO_SERVER_BACKEND` ohittaa automaattivalinnan aina.
- Jos tierille ei ole profiilia (ei CUDAa, tai liian pieni kortti), palvelin
  **kaatuu ohjeen kanssa** eikä arvaa. Hiljainen putoaminen mock-backendiin
  olisi pahin mahdollinen oletus: palvelin näyttäisi toimivan, mutta tuottaisi
  väärennettyä videota.

Lataustavan prioriteetti on: eksplisiittinen `cpu_offload` > tierin politiikka >
VRAM-turvaverkko. Turvaverkko siirtyy offloadiin jos kortti alittaa profiilin
vaatimuksen - mutta **ei** ohita käyttäjän eksplisiittistä asetusta. Sama
periaate kuin shiftin kanssa: konfiguraatio voittaa automatiikan.

### Tierkohtaista resoluutiolistaa ei toteutettu

Speksin kohta 5 kaavaili, että sallittu resoluutiolista riippuisi tieristä.
Sitä ei toteutettu, koska nykyisellä mallivalikoimalla se ei tekisi mitään:

- TI2V-5B tarjoaa vain 720p-luokan resoluutiot (1280x704 / 704x1280). Jos
  low-tier suodattaisi listaa pienemmäksi, listasta tulisi tyhjä.
- A14B:llä on sekä 480p että 720p, mutta A14B ei koskaan päädy low-tierille -
  se vaatii 30 GB.

Resoluutiolista vaihtelee jo aktiivisen mallin mukana, ja `GET /api/v1/models`
raportoi sen. Se on se havaittava käyttäytyminen, jota speksin kohta haki;
tierkohtainen suodatin olisi ollut koodia joka ei koskaan tee mitään.

### Kvantisointi on toteutettu mutta testaamaton

`low`-tier lataa transformerin 4-bit-kvantisoituna (`BitsAndBytesConfig`,
nf4). Polkua ei ole voitu ajaa tällä laitteistolla, koska 24 GB riittää
bf16:een. Se on merkitty testaamattomaksi sekä koodissa että tässä - ei
piilotettu oletukseksi joka toimii. `bitsandbytes` on erillinen extra
(`uv sync --extra gpu --extra quantized`), ja jos se puuttuu, virheilmoitus
kertoo asennuskomennon.

Rajoite: kvantisointi koskee vain `transformer`-moduulia. MoE-malleilla on
lisäksi `transformer_2`, joka jäisi kvantisoimatta - siksi `quantized`-tier
osoittaa yhden asiantuntijan malliin.

### Riippuvuuksien kiinnitys

`gpu`-extra sai ylärajat. `diffusers` on kiinnitetty tiukimmin (`>=0.40,<0.41`),
koska wan22-backend nojaa varmistettuihin pipeline-yksityiskohtiin, jotka voivat
muuttua minor-versiossa: callbackin sopimus, `guidance_scale_2`:n
virhekäyttäytyminen, `boundary_ratio` ja `expand_timesteps` configissa. Tarkat
todennetut versiot ovat `uv.lock`issa; pyprojectin rajat ilmaisevat
yhteensopivuuden.

---

## Vaihe 4: valinnaiset laajennukset

Speksin Vaihe 4 listaa kolme asiaa: esikatselukuvat, job-historian pysyvyys ja
autentikointi. Nämä ovat samat, jotka `## Nimenomaisesti rajauksen ulkopuolella
(v1)` sulkee pois - Vaihe 4 on nimenomaan v1:n jälkeinen joukko, ei ristiriita.

**Kaikki kolme ovat oletuksena pois päältä.** v1:n käyttäytyminen ei siis muutu
lainkaan: rajaukset pitävät edelleen oletusasetuksilla, ja nämä ovat kytkimiä
niille jotka tarvitsevat.

### Job-historian pysyvyys oli jo tehty

Sidecar-JSON toteutettiin Vaiheessa 1, ja speksi sanoo itse ettei tietokantaa
tarvita (`sidecar-JSON riittää`). Tässä ei siis ollut mitään tehtävää: valmiit
ajot säilyvät uudelleenkäynnistyksen yli, ja tietokannan lisääminen olisi
rikkonut speksin rajausta eikä täyttänyt sitä.

### Esikatselukuvat

`VIDEO_SERVER_PREVIEW_EVERY_N_STEPS=N` dekoodaa yhden latenttiframen VAE:lla
joka N:s askel ja tarjoilee sen `preview_url`-kentässä. `0` (oletus) = pois,
kuten speksi edellyttää ("toteutetaan vain jos osoittautuu tarpeelliseksi, ei
oletuksena").

Toteutuksen kannalta olennaista:

- **Normalisointi on sama kuin pipelinen omassa dekoodauksessa**
  (`latents_mean` / `latents_std`). Väärä normalisointi tuottaisi
  uskottavan näköistä roskaa, joka menisi helposti läpi.
- **Esikatselun epäonnistuminen ei kaada ajoa.** Ylimääräinen VAE-kutsu voi
  kaatua muistiin juuri kun kortti on täynnä; silloin logitetaan varoitus ja
  generointi jatkuu. Esikatselu on mukavuus, ei osa lopputulosta.
- Kuva kirjoitetaan valmiin videon viereen (`{job_id}-preview.png`), joten se
  tarjoillaan samaa staattista reittiä eikä vaadi omaa mounttia.

**Todennus:** ajettiin oikealla mallilla ja verrattiin viimeisen askeleen
esikatselua valmiin videon ensimmäiseen frameen - kuvat vastaavat toisiaan,
eli dekoodaus tuottaa saman kuin pipelinen oma dekoodaus. Ajo tehtiin
poikkeuksellisesti pienennetyllä resoluutiolla (`VIDEO_SERVER_RESOLUTIONS`),
jotta se valmistui minuuteissa; sisältö on siksi abstraktia, mutta juuri se
että esikatselu ja lopputulos ovat identtisiä on todiste normalisoinnin
oikeellisuudesta.

### Progress-vaihe (`phase`)

Vaiheen 2 mittaus paljasti, että job näyttää tilaa `n/n` ja `eta_seconds: 0`
minuuttien ajan VAE-dekoodauksen aikana. `ProgressResponse` sai kentän
`phase` (`denoising` | `decoding`), jonka backend raportoi viimeisen
denoising-askeleen jälkeen. Dekoodauksen aikana `eta_seconds` on `null` eikä
`0`, koska kestoa ei voi arvioida - null on rehellisempi kuin nolla.

Muutos on additiivinen: vanhat asiakkaat eivät riko.

### Autentikointi

`VIDEO_SERVER_API_KEY` kytkee `X-API-Key`-otsakevaatimuksen `/api/v1`- ja
`/outputs`-poluille. Ilman asetusta mikään ei muutu, kuten v1-rajaus
edellyttää ("lisätään erikseen jos palvelin altistetaan verkkoon").

Kaksi ratkaisua, jotka on syytä perustella:

- **`/outputs` on suojattu myös.** Videot ovat se varsinainen suojattava
  sisältö; pelkän `/api/v1`:n suojaaminen olisi jättänyt lopputulokset auki
  arvattavan URL:n taakse.
- **`/api/v1/health` jätettiin auki.** Monitoroinnin on toimittava ilman
  avainta, eikä se paljasta muuta kuin latauksen tilan.

Vertailu tehdään `secrets.compare_digest`illa, jottei avain vuoda vastausajan
kautta.
