#!/usr/bin/env bash
# Sets up WanFlash on a clean machine with a single command.
#
# Installs uv if missing, creates the virtual environment, installs
# dependencies, checks the hardware and prints how to start the server.
# Python 3.12 comes with uv, so no separate Python install is needed.
#
# The script is idempotent: it is safe to run again.
#
# Usage:
#   bash scripts/bootstrap.sh              # everything for inference, no weights
#   bash scripts/bootstrap.sh --weights    # also the weights (~32 GB)
#   bash scripts/bootstrap.sh --minimal    # light, no GPU dependencies

set -euo pipefail

MINIMAL=0
WEIGHTS=0
SKIP_TESTS=0
MODEL_PROFILE="wan2.2-ti2v-5b"

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --minimal          API dependencies only (~50 MB), no torch.
  --weights          Also download the model weights (~32 GB for the default).
  --profile NAME     Model profile to download (default: wan2.2-ti2v-5b).
  --skip-tests       Skip the test run after installation.
  -h, --help         This help.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --minimal) MINIMAL=1 ;;
        --weights) WEIGHTS=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        --profile)
            shift
            [ $# -gt 0 ] || { echo "--profile requires a value" >&2; exit 2; }
            MODEL_PROFILE="$1"
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
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

step "Checking uv"
if UV="$(find_uv)"; then
    note "Found: $UV"
else
    step "Installing uv"
    # Virallinen asennusskripti. Kerrotaan ääneen mitä ajetaan, koska
    # etäskriptin suorittaminen on asia josta käyttäjän kuuluu tietää.
    note "Running the official installer script: https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find_uv)" || {
        echo "uv installation failed. Install it manually: https://docs.astral.sh/uv/" >&2
        exit 1
    }
    note "Installed: $UV"
fi

# Lisätään uv:n hakemisto tämän istunnon PATH:iin, jottei shelliä tarvitse
# käynnistää uudelleen kesken asennuksen.
export PATH="$(dirname "$UV"):$PATH"
"$UV" --version

# --- 2. Dependencies ------------------------------------------------------

if [ "$MINIMAL" -eq 0 ] && [ "$(uname -s)" = "Darwin" ]; then
    # GPU-extra hakee torchin CUDA-indeksistä, jossa ei ole macOS-wheelejä.
    # Selkeä viesti on parempi kuin resolverin virhe.
    step "Note: macOS"
    note "The GPU dependencies are built for CUDA (Linux/Windows)."
    note "macOS has no CUDA, so the light install is used instead."
    note "See README, Known limitations."
    MINIMAL=1
fi

if [ "$MINIMAL" -eq 1 ]; then
    step "Installing API dependencies (light, no GPU)"
    "$UV" sync --directory "$REPO_ROOT" --group dev
else
    step "Installing dependencies with GPU support (torch + diffusers, ~3 GB)"
    note "Lighter option without GPU dependencies: --minimal"
    "$UV" sync --directory "$REPO_ROOT" --extra gpu --group dev
fi

# --- 3. Configuration -----------------------------------------------------

if [ ! -f "$REPO_ROOT/.env" ]; then
    step "Creating .env"
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    note "Copied .env.example -> .env (the defaults work as they are)"
else
    step "Configuration"
    note ".env already exists, leaving it untouched"
fi

# --- 4. Hardware check ----------------------------------------------------

step "Checking the environment"
# check_env palauttaa 1 jos kone kelpaa vain mock-ajoon. Se on tieto eikä
# bootstrapin virhe, joten se ei saa kaataa skriptiä (set -e).
ENV_STATUS=0
"$UV" run --directory "$REPO_ROOT" python scripts/check_env.py || ENV_STATUS=$?

# --- 5. Tests -------------------------------------------------------------

if [ "$SKIP_TESTS" -eq 0 ]; then
    step "Running tests (no GPU or weights required)"
    "$UV" run --directory "$REPO_ROOT" pytest -q
fi

# --- 6. Weights -----------------------------------------------------------

if [ "$WEIGHTS" -eq 1 ]; then
    step "Downloading model weights: $MODEL_PROFILE"
    note "This is tens of gigabytes and may take a long time."
    "$UV" run --directory "$REPO_ROOT" python scripts/download_model.py "$MODEL_PROFILE"
fi

# --- 7. Are the weights already on disk? ----------------------------------

# Tarkistetaan tilanne sen sijaan että oletettaisiin: painot ovat voineet olla
# levyllä jo ennen tätä ajoa, jolloin "lataa painot" -ohje olisi väärä.
WEIGHTS_PRESENT=0
if [ "$MINIMAL" -eq 0 ]; then
    step "Checking model weights"
    if "$UV" run --directory "$REPO_ROOT" python scripts/download_model.py --check "$MODEL_PROFILE"; then
        WEIGHTS_PRESENT=1
    fi
fi

# --- Summary --------------------------------------------------------------

printf '\n\033[32m======================================================\033[0m\n'
printf '\033[32m Installation complete\033[0m\n'
printf '\033[32m======================================================\033[0m\n\n'

if [ "$MINIMAL" -eq 1 ]; then
    echo "Light install: use the mock backend."
    echo "  VIDEO_SERVER_BACKEND=mock uv run uvicorn video_server.main:app"
elif [ "$WEIGHTS_PRESENT" -eq 0 ]; then
    echo "The weights are not downloaded yet. Fetch them before starting:"
    echo "  uv run python scripts/download_model.py $MODEL_PROFILE"
    echo
    echo "Or try the API right away without weights:"
    echo "  VIDEO_SERVER_BACKEND=mock uv run uvicorn video_server.main:app"
else
    echo "Start the server:"
    echo "  uv run uvicorn video_server.main:app"
    echo
    echo "The model loads in the background; GET /api/v1/health reports when it is ready."
fi

if [ "$ENV_STATUS" -ne 0 ] && [ "$MINIMAL" -eq 0 ]; then
    printf '\n\033[33mNote: the environment check did not find this machine ready for\033[0m\n'
    printf '\033[33mreal inference. See the items flagged above.\033[0m\n'
fi

printf '\nDocs: README.md | Settings: .env | API: /docs\n\n'
