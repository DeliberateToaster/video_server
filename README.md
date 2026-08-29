# WanFlash

REST-rajapinta Wan 2.2 -videogenerointiin. Yksi malli residenttinä muistissa,
yksi generointi kerrallaan, asynkroninen job-malli.

Tekninen spesifikaatio: [docs/spec.md](docs/spec.md).

## Asennus

Nopein tapa on bootstrap-skripti: se asentaa uv:n jos se puuttuu, asentaa
riippuvuudet, luo `.env`:n, tarkistaa laitteiston ja ajaa testit. Python 3.12
tulee uv:n mukana, joten erillistä Python-asennusta ei tarvita.

```powershell
# Windows. -ExecutionPolicy Bypass tarvitaan jos koneen suorituskäytäntö
# estää allekirjoittamattomat skriptit.
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

```bash
# Linux / macOS
bash scripts/bootstrap.sh
```

Valinnat (molemmissa): `-Weights` / `--weights` lataa myös mallin painot
(~32 GB), `-Minimal` / `--minimal` asentaa vain rajapinnan riippuvuudet ilman
torchia, `-SkipTests` / `--skip-tests` ohittaa testiajon.

Skripti on idempotentti: sen voi ajaa uudelleen turvallisesti, eikä se
ylikirjoita olemassa olevaa `.env`-tiedostoa.

### Käsin

Vaatii [uv](https://docs.astral.sh/uv/):n. Python 3.12 asennetaan automaattisesti.

```powershell
# Rajapinta + kehitystyökalut, ei GPU-riippuvuuksia (~50 MB).
# Tällä ajetaan mock-backend ja koko testisetti.
uv sync --group dev

# GPU-riippuvuudet (torch + diffusers, ~3 GB). Tarvitaan vasta oikeaan
# inferenssiin, ei rajapinnan kehitykseen.
uv sync --extra gpu --group dev

# Vain low-tierille (4-bit-kvantisointi, 12-20 GB VRAM):
# uv sync --extra gpu --extra quantized --group dev

# Tarkista onko kone valmis oikeaan inferenssiin (GPU, CUDA, ffmpeg, tier)
uv run python scripts/check_env.py
```

Riippuvuusjako on tarkoituksellinen: koko rajapinta, jono ja testit toimivat
ilman GPU:ta ja ilman torchia. Vain `backends/wan22.py` tarvitsee `gpu`-extran.

## Mallin painot

Palvelin **ei** lataa painoja itse. Puuttuvista painoista tulee selkeä virhe,
joka kertoo komennon - kymmenien gigatavujen hiljainen lataus ensimmäisen
API-kutsun yhteydessä olisi huono oletus.

```powershell
# Oletusmalli TI2V-5B. Huom: 31,9 GB levytilaa (repo sisältää fp32-painot).
# VRAM-tarve ajossa on ~10 GB - levytila ja VRAM ovat eri asia.
uv run python scripts/download_model.py wan2.2-ti2v-5b

# Onko jo ladattu?
uv run python scripts/download_model.py --check
```

Saatavilla olevat profiilit: `wan2.2-ti2v-5b` (oletus), `wan2.2-t2v-a14b`,
`wan2.2-i2v-a14b`. A14B on MoE-malli, jonka painot ovat bf16:na ~54 GB - se ei
mahdu 24 GB:n kortille edes CPU-offloadilla ilman ~60 GB järjestelmämuistia.

## Laitteisto ja tier

Oletuksena `VIDEO_SERVER_BACKEND=auto`: palvelin tunnistaa VRAM:n ja
järjestelmämuistin, päättelee tierin ja valitsee sen mukaan sekä mallin että
lataustavan. Valinta ja sen perustelu logitetaan käynnistyksessä.

| Tier | VRAM | Malli ja lataustapa |
|---|---|---|
| `low` | 12-20 GB | TI2V-5B 4-bit-kvantisoituna (vaatii `quantized`-extran, testäämaton) |
| `mid` | 20-30 GB | TI2V-5B bf16 |
| `high` | 30 GB+ | A14B bf16 |
| `a14b-offload` | 24 GB + ~60 GB RAM | A14B CPU-offloadilla, hidas |

Eksplisiittinen `VIDEO_SERVER_BACKEND` ohittaa automaattivalinnan aina. Jos
tierille ei löydy mallia (ei CUDAa tai liian pieni kortti), palvelin kaatuu
ohjeen kanssa sen sijaan että putoaisi hiljaa mock-backendiin ja tuottaisi
väärennettyä videota.

RTX 3090 (24 GB) osuu mid-tieriin: TI2V-5B bf16, ~23 GB VRAM ajossa.

## Ajaminen

```powershell
uv run uvicorn video_server.main:app --reload

# Kehitys ilman GPU:ta ja ilman painoja:
$env:VIDEO_SERVER_BACKEND = "mock"; uv run uvicorn video_server.main:app --reload
```

Palvelin avaa portin heti, mutta generointi-endpointit vastaavat `503` kunnes
malli on latautunut. Tila: `GET /api/v1/health`.

Konfiguraatio: kopioi [.env.example](.env.example) nimelle `.env`. Kaikki
asetukset toimivat myös ympäristömuuttujina tai `config.yaml`:n kautta.

## Esimerkki

```powershell
# Aloita generointi
curl -X POST http://127.0.0.1:8000/api/v1/txt2vid `
  -H "Content-Type: application/json" `
  -d '{\"prompt\":\"kissa kävelee rannalla\",\"num_frames\":81}'

# Pollaa tilaa (job_id edellisestä vastauksesta)
curl http://127.0.0.1:8000/api/v1/jobs/<job_id>

# Valmis video löytyy vastauksen video_url-kentästä
```

Mitä palvelin sallii milläkin mallilla, selviää kysymättä arvailua:
`GET /api/v1/models` kertoo aktiivisen mallin sallitut resoluutiot,
frame-säännön ja natiivin fps:n.

## Valinnaiset laajennukset

Molemmat ovat oletuksena pois päältä, eivätkä muuta palvelimen käyttäytymistä
ellei niitä erikseen kytke.

**Esikatselukuvat.** `VIDEO_SERVER_PREVIEW_EVERY_N_STEPS=5` dekoodaa yhden
framen joka viides askel, ja job-vastaus saa `preview_url`-kentän. Maksaa
yhden ylimääräisen VAE-kutsun per kuva. Jos dekoodaus epäonnistuu (esimerkiksi
muistin loppuessa), generointi jatkuu normaalisti ja lokiin tulee varoitus.

**API-avain.** `VIDEO_SERVER_API_KEY=...` vaatii `X-API-Key`-otsakkeen
poluilla `/api/v1` ja `/outputs`. Myös valmiit videot ovat suojattuja, koska ne
ovat se varsinainen suojattava sisältö. `/api/v1/health` jää auki, jotta
monitorointi toimii ilman avainta.

```powershell
curl http://127.0.0.1:8000/api/v1/models -H "X-API-Key: salainen-avain"
```

## Testit

```powershell
uv run pytest          # koko setti, ei vaadi GPU:ta eikä painoja
uv run ruff check .
```

## Tyypilliset virheet

| Oire | Syy | Korjaus |
|---|---|---|
| `WeightsMissingError: mallin ... painoja ei löydy` | Painoja ei ole ladattu | `uv run python scripts/download_model.py` |
| `CAS Client Error: ... error decoding response body` | HuggingFacen Xet-siirto katkesi kesken latauksen | Skripti yrittää nyt automaattisesti uudelleen ja jatkaa siitä mihin jäi. Jos yritykset loppuvat, aja komento uudelleen - valmiit tiedostot säilyvät |
| `ModuleNotFoundError: No module named torch` | GPU-riippuvuuksia ei ole asennettu | `uv sync --extra gpu --group dev` |
| `503` generointipyyntöön | Malli latautuu vielä (kestää minuutteja) | Odota; `GET /api/v1/health` kertoo tilan |
| `400 num_frames ... vaaditaan n * 4 + 1` | Frame-määrä ei kelpaa mallin VAE:lle | Käytä virheviestin ehdottamaa lähintä arvoa |
| `torch.OutOfMemoryError` / CUDA OOM | Malli ei mahdu VRAM:iin | `VIDEO_SERVER_CPU_OFFLOAD=true`, pienempi resoluutio tai vähemmän frameja |
| `ValueError: guidance_scale_2 is only supported when ... boundary_ratio is not None` | Kaksoisguidance yhden asiantuntijan mallille | Ei pitäisi tapahtua: profiilin `has_second_expert` estää tämän. Jos tapahtuu, profiili on väärin |
| Palvelin vastaa `429` | Jono on täynnä | Odota tai kasvata `VIDEO_SERVER_MAX_QUEUE_SIZE` |

## Tila

Vaiheet 0-4 valmiit: ympäristö, rajapinta mock-backendillä, Wan-backend,
tier-automatiikka sekä valinnaiset esikatselukuvat ja API-avain. Todennettu
oikealla ajolla (RTX 3090, 1280x704). Kaikki speksin vaiheet on käyty läpi.

## Tunnetut rajoitteet

- Ei ROCm (AMD) -tukea. Vain CUDA.
- Autentikointi on oletuksena pois. Jos palvelin altistetaan verkkoon, aseta
  `VIDEO_SERVER_API_KEY`.
- Ei rinnakkaisia ajoja: yksi malli ja yksi generointi kerrallaan.
- Etenemistiedon `phase`-kenttä erottaa denoising- ja dekoodausvaiheet.
  Dekoodauksen kestoa ei voi arvioida, joten `eta_seconds` on silloin `null`.
- Job-tila on muistissa; valmiit videot säilyvät levyllä sidecar-metadatan
  kanssa, joten uudelleenkäynnistys ei hukkaa valmiita tuloksia.
