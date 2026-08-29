<#
.SYNOPSIS
    Sets up WanFlash on a clean machine with a single command.

.DESCRIPTION
    Installs uv if missing, creates the virtual environment, installs
    dependencies, checks the hardware and prints how to start the server.

    Python 3.12 comes with uv, so no separate Python install is needed.

    The script is idempotent: it is safe to run again.

.PARAMETER Minimal
    Install only the API dependencies (~50 MB) without torch and diffusers.
    Enough for the mock backend and the test suite, not for real inference.

.PARAMETER Weights
    Also download the model weights. For the default model this is ~32 GB,
    so it never happens without this switch.

.PARAMETER ModelProfile
    Which model profile to download when -Weights is given.

.PARAMETER SkipTests
    Skip the test run after installation.

.EXAMPLE
    .\scripts\bootstrap.ps1
    Installs everything needed for inference, without the weights.

.EXAMPLE
    .\scripts\bootstrap.ps1 -Weights
    Installs everything and downloads the weights (~32 GB).

.EXAMPLE
    .\scripts\bootstrap.ps1 -Minimal
    Light install for API development, no GPU dependencies.
#>
[CmdletBinding()]
param(
    [switch]$Minimal,
    [switch]$Weights,
    [string]$ModelProfile = "wan2.2-ti2v-5b",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Note {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Invoke-Native {
    <#
        Ajaa natiiviohjelman ja palauttaa sen paluukoodin.

        Windows PowerShell tulkitsee natiiviohjelman stderr-tulosteen
        virheeksi, ja $ErrorActionPreference = "Stop" muuttaisi sen
        poikkeukseksi. uv kirjoittaa edistymisensä nimenomaan stderriin, joten
        ilman tätä kääriä koko asennus kaatuisi onnistuneeseen komentoon.
        Onnistuminen luetaan paluukoodista, ei tulostevirrasta.
    #>
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(ValueFromRemainingArguments)][string[]]$Arguments
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Out-Host: tuloste menee konsoliin eikä putkeen. Ilman tätä
        # "$code = Invoke-Native ..." nielaisisi komennon tulosteen muuttujaan,
        # jolloin käyttäjä ei näkisi mitään ja paluukoodi olisi taulukko.
        & $Exe @Arguments | Out-Host
    }
    finally {
        $ErrorActionPreference = $previous
    }
    return $LASTEXITCODE
}

function Find-Uv {
    <#
        uv voi olla kolmessa paikassa: PATH:ssa, wingetin linkkihakemistossa
        tai standalone-asentajan hakemistossa. Näitä etsitään erikseen, koska
        heti asennuksen jälkeen uv EI ole vielä nykyisen shellin PATH:ssa -
        se on se kohta johon käsin asentava kompastuu.
    #>
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Install-Uv {
    Write-Step "Installing uv"

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Note "Source: winget (astral-sh.uv)"
        $code = Invoke-Native winget install --id astral-sh.uv -e --source winget `
            --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($code -ne 0) { Write-Note "winget returned $code, checking anyway" }
    }
    else {
        # Virallinen asennusskripti. Kerrotaan ääneen mitä ajetaan, koska
        # etäskriptin suorittaminen on asia josta käyttäjän kuuluu tietää.
        Write-Note "winget not found, using the official installer script:"
        Write-Note "  https://astral.sh/uv/install.ps1"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }

    $found = Find-Uv
    if (-not $found) {
        throw "uv installation failed. Install it manually: https://docs.astral.sh/uv/"
    }
    return $found
}

# --- 1. uv ----------------------------------------------------------------

Write-Step "Checking uv"
$Uv = Find-Uv
if ($Uv) {
    Write-Note "Found: $Uv"
}
else {
    $Uv = Install-Uv
    Write-Note "Installed: $Uv"
}

# Lisätään uv:n hakemisto tämän istunnon PATH:iin, jotta myöhemmät komennot
# toimivat ilman shellin uudelleenkäynnistystä.
$UvDir = Split-Path -Parent $Uv
if ($env:PATH -notlike "*$UvDir*") {
    $env:PATH = "$UvDir;$env:PATH"
}

$null = Invoke-Native $Uv --version

# --- 2. Dependencies ------------------------------------------------------

if ($Minimal) {
    Write-Step "Installing API dependencies (light, no GPU)"
    $code = Invoke-Native $Uv sync --directory $RepoRoot --group dev
}
else {
    Write-Step "Installing dependencies with GPU support (torch + diffusers, ~3 GB)"
    Write-Note "Lighter option without GPU dependencies: -Minimal"
    $code = Invoke-Native $Uv sync --directory $RepoRoot --extra gpu --group dev
}
if ($code -ne 0) { throw "dependency installation failed (exit code $code)" }

# --- 3. Configuration -----------------------------------------------------

$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"
if (-not (Test-Path $EnvFile)) {
    Write-Step "Creating .env"
    Copy-Item $EnvExample $EnvFile
    Write-Note "Copied .env.example -> .env (the defaults work as they are)"
}
else {
    Write-Step "Configuration"
    Write-Note ".env already exists, leaving it untouched"
}

# --- 4. Hardware check ----------------------------------------------------

Write-Step "Checking the environment"
# check_env palauttaa 1 jos kone kelpaa vain mock-ajoon. Se ei ole
# bootstrapin virhe vaan tieto, joten sitä ei käsitellä kaatumisena.
$EnvStatus = Invoke-Native $Uv run --directory $RepoRoot python scripts/check_env.py

# --- 5. Tests -------------------------------------------------------------

if (-not $SkipTests) {
    Write-Step "Running tests (no GPU or weights required)"
    $code = Invoke-Native $Uv run --directory $RepoRoot pytest -q
    if ($code -ne 0) { throw "tests failed - the installation is not sound" }
}

# --- 6. Weights -----------------------------------------------------------

if ($Weights) {
    Write-Step "Downloading model weights: $ModelProfile"
    Write-Note "This is tens of gigabytes and may take a long time."
    $code = Invoke-Native $Uv run --directory $RepoRoot python scripts/download_model.py $ModelProfile
    if ($code -ne 0) { throw "weight download failed (exit code $code)" }
}

# --- 7. Are the weights already on disk? ----------------------------------

# Tarkistetaan tilanne sen sijaan että oletettaisiin. Painot ovat voineet olla
# levyllä jo ennen tätä ajoa, jolloin "lataa painot" -ohje olisi väärä.
$WeightsPresent = $false
if (-not $Minimal) {
    Write-Step "Checking model weights"
    $WeightsPresent = (Invoke-Native $Uv run --directory $RepoRoot `
            python scripts/download_model.py --check $ModelProfile) -eq 0
}

# --- Summary --------------------------------------------------------------

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host " Installation complete" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

if ($Minimal) {
    Write-Host "Light install: use the mock backend." -ForegroundColor Yellow
    Write-Host '  $env:VIDEO_SERVER_BACKEND = "mock"'
    Write-Host "  uv run uvicorn video_server.main:app"
}
elseif (-not $WeightsPresent) {
    Write-Host "The weights are not downloaded yet. Fetch them before starting:"
    Write-Host "  uv run python scripts/download_model.py $ModelProfile"
    Write-Host ""
    Write-Host "Or try the API right away without weights:"
    Write-Host '  $env:VIDEO_SERVER_BACKEND = "mock"'
    Write-Host "  uv run uvicorn video_server.main:app"
}
else {
    Write-Host "Start the server:"
    Write-Host "  uv run uvicorn video_server.main:app"
    Write-Host ""
    Write-Host "The model loads in the background; GET /api/v1/health reports when it is ready."
}

if ($EnvStatus -ne 0 -and -not $Minimal) {
    Write-Host ""
    Write-Host "Note: the environment check did not find this machine ready for" -ForegroundColor Yellow
    Write-Host "real inference. See the items flagged above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Docs: README.md | Settings: .env | API: /docs"
Write-Host ""
