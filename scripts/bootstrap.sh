#!/usr/bin/env bash
# Pystyttää WanFlashin tyhjältä koneelta yhdellä komennolla.
#
# Asentaa uv:n jos se puuttuu, luo virtuaaliympäristön, asentaa riippuvuudet,
# tarkistaa laitteiston ja kertoo miten palvelin käynnistetään. Python 3.12
# asentuu uv:n mukana - erillistä Python-asennusta ei tarvita.
#
# Skripti on idempotentti: sen voi ajaa uudelleen turvallisesti.
#
# Käyttö:
#   bash scripts/bootstrap.sh              # kaikki inferenssiin, ei painoja
#   bash scripts/bootstrap.sh --weights    # myös painot (~32 GB)
#   bash scripts/bootstrap.sh --minimal    # kevyt, ei GPU-riippuvuuksia

set -euo pipefail

MINIMAL=0
WEIGHTS=0
SKIP_TESTS=0
MODEL_PROFILE="wan2.2-ti2v-5b"

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Valinnat:
  --minimal          Vain rajapinnan riippuvuudet (~50 MB), ei torchia.
  --weights          Lataa myös mallin painot (oletusmallilla ~32 GB).
  --profile NIMI     Ladattava malliprofiili (oletus: wan2.2-ti2v-5b).
  --skip-tests       Ohita asennuksen jälkeinen testiajo.
  -h, --help         Tämä ohje.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --minimal) MINIMAL=1 ;;
        --weights) WEIGHTS=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        --profile)
            shift
            [ $# -gt 0 ] || { echo "--profile vaatii arvon" >&2; exit 2; }
            MODEL_PROFILE="$1"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "tuntematon valitsin: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
note() { printf '\033[90m    %s\033[0m\n' "$1"; }

find_uv() {
    # uv voi olla PATH:ssa tai asentajan omassa hakemistossa. Heti asennuksen
    # jälkeen se EI ole vielä nykyisen shellin PATH:ssa - se on se kohta johon
    # käsin asentava kompastuu.
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# --- 1. uv ----------------------------------------------------------------

step "Tarkistetaan uv"
if UV="$(find_uv)"; then
    note "Löytyi: $UV"
else
    step "Asennetaan uv"
    # Virallinen asennusskripti. Kerrotaan ääneen mitä ajetaan, koska
    # etäskriptin suorittaminen on asia josta käyttäjän kuuluu tietää.
    note "Ajetaan virallinen asennusskripti: https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find_uv)" || {
        echo "uv:n asennus ei onnistunut. Asenna käsin: https://docs.astral.sh/uv/" >&2
        exit 1
    }
    note "Asennettu: $UV"
fi

# Lisätään uv:n hakemisto tämän istunnon PATH:iin, jottei shelliä tarvitse
# käynnistää uudelleen kesken asennuksen.
export PATH="$(dirname "$UV"):$PATH"
"$UV" --version

# --- 2. Riippuvuudet ------------------------------------------------------

if [ "$MINIMAL" -eq 0 ] && [ "$(uname -s)" = "Darwin" ]; then
    # GPU-extra hakee torchin CUDA-indeksistä, jossa ei ole macOS-wheelejä.
    # Selkeä viesti on parempi kuin resolverin virhe.
    step "Huom: macOS"
    note "GPU-riippuvuudet on rakennettu CUDA:lle (Linux/Windows)."
    note "macOS:llä ei ole CUDAa, joten asennetaan kevyt versio."
    note "Katso README, kohta Tunnetut rajoitteet."
    MINIMAL=1
fi

if [ "$MINIMAL" -eq 1 ]; then
    step "Asennetaan rajapinnan riippuvuudet (kevyt, ei GPU:ta)"
    "$UV" sync --directory "$REPO_ROOT" --group dev
else
    step "Asennetaan riippuvuudet GPU-tuella (torch + diffusers, ~3 GB)"
    note "Kevyempi vaihtoehto ilman GPU-riippuvuuksia: --minimal"
    "$UV" sync --directory "$REPO_ROOT" --extra gpu --group dev
fi

# --- 3. Konfiguraatio -----------------------------------------------------

if [ ! -f "$REPO_ROOT/.env" ]; then
    step "Luodaan .env"
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    note "Kopioitu .env.example -> .env (oletukset toimivat sellaisenaan)"
else
    step "Konfiguraatio"
    note ".env on jo olemassa, ei ylikirjoiteta"
fi

# --- 4. Laitteistotarkistus ----------------------------------------------

step "Tarkistetaan ympäristö"
# check_env palauttaa 1 jos kone kelpaa vain mock-ajoon. Se on tieto eikä
# bootstrapin virhe, joten se ei saa kaataa skriptiä (set -e).
ENV_STATUS=0
"$UV" run --directory "$REPO_ROOT" python scripts/check_env.py || ENV_STATUS=$?

# --- 5. Testit ------------------------------------------------------------

if [ "$SKIP_TESTS" -eq 0 ]; then
    step "Ajetaan testit (ei vaadi GPU:ta eikä painoja)"
    "$UV" run --directory "$REPO_ROOT" pytest -q
fi

# --- 6. Painot ------------------------------------------------------------

if [ "$WEIGHTS" -eq 1 ]; then
    step "Ladataan mallin painot: $MODEL_PROFILE"
    note "Tämä on kymmeniä gigatavuja ja voi kestää kauan."
    "$UV" run --directory "$REPO_ROOT" python scripts/download_model.py "$MODEL_PROFILE"
fi

# --- 7. Onko painot jo levyllä? -------------------------------------------

# Tarkistetaan tilanne sen sijaan että oletettaisiin: painot ovat voineet olla
# levyllä jo ennen tätä ajoa, jolloin "lataa painot" -ohje olisi väärä.
WEIGHTS_PRESENT=0
if [ "$MINIMAL" -eq 0 ]; then
    step "Tarkistetaan mallin painot"
    if "$UV" run --directory "$REPO_ROOT" python scripts/download_model.py --check "$MODEL_PROFILE"; then
        WEIGHTS_PRESENT=1
    fi
fi

# --- Yhteenveto -----------------------------------------------------------

printf '\n\033[32m======================================================\033[0m\n'
printf '\033[32m Asennus valmis\033[0m\n'
printf '\033[32m======================================================\033[0m\n\n'

if [ "$MINIMAL" -eq 1 ]; then
    echo "Kevyt asennus: käytä mock-backendiä."
    echo "  VIDEO_SERVER_BACKEND=mock uv run uvicorn video_server.main:app"
elif [ "$WEIGHTS_PRESENT" -eq 0 ]; then
    echo "Painoja ei ole vielä ladattu. Lataa ne ennen käynnistystä:"
    echo "  uv run python scripts/download_model.py $MODEL_PROFILE"
    echo
    echo "Tai kokeile rajapintaa heti ilman painoja:"
    echo "  VIDEO_SERVER_BACKEND=mock uv run uvicorn video_server.main:app"
else
    echo "Käynnistä palvelin:"
    echo "  uv run uvicorn video_server.main:app"
    echo
    echo "Malli latautuu taustalla; GET /api/v1/health kertoo koska se on valmis."
fi

if [ "$ENV_STATUS" -ne 0 ] && [ "$MINIMAL" -eq 0 ]; then
    printf '\n\033[33mHuom: ympäristötarkistus ei todennut konetta valmiiksi oikeaan\033[0m\n'
    printf '\033[33minferenssiin. Katso yllä olevat merkinnät.\033[0m\n'
fi

printf '\nDokumentaatio: README.md | Asetukset: .env | API: /docs\n\n'
