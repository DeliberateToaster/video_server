<#
.SYNOPSIS
    Pystyttää WanFlashin tyhjältä koneelta yhdellä komennolla.

.DESCRIPTION
    Asentaa uv:n jos se puuttuu, luo virtuaaliympäristön, asentaa riippuvuudet,
    tarkistaa laitteiston ja kertoo miten palvelin käynnistetään.

    Python 3.12 asentuu uv:n mukana - erillistä Python-asennusta ei tarvita.

    Skripti on idempotentti: sen voi ajaa uudelleen turvallisesti.

.PARAMETER Minimal
    Asenna vain rajapinnan riippuvuudet (~50 MB) ilman torchia ja diffusersia.
    Tällä ajetaan mock-backend ja testit, ei oikeaa inferenssiä.

.PARAMETER Weights
    Lataa myös mallin painot. Oletusmallilla tämä on ~32 GB, joten se ei
    tapahdu ilman tätä valintaa.

.PARAMETER ModelProfile
    Ladattava malliprofiili, kun -Weights on annettu.

.PARAMETER SkipTests
    Ohita asennuksen jälkeinen testiajo.

.EXAMPLE
    .\scripts\bootstrap.ps1
    Asentaa kaiken inferenssiin tarvittavan, ei painoja.

.EXAMPLE
    .\scripts\bootstrap.ps1 -Weights
    Asentaa kaiken ja lataa painot (~32 GB).

.EXAMPLE
    .\scripts\bootstrap.ps1 -Minimal
    Kevyt asennus rajapintakehitykseen, ei GPU-riippuvuuksia.
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
    Write-Step "Asennetaan uv"

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Note "Lähde: winget (astral-sh.uv)"
        $code = Invoke-Native winget install --id astral-sh.uv -e --source winget `
            --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($code -ne 0) { Write-Note "winget palautti koodin $code, tarkistetaan silti" }
    }
    else {
        # Virallinen asennusskripti. Kerrotaan ääneen mitä ajetaan, koska
        # etäskriptin suorittaminen on asia josta käyttäjän kuuluu tietää.
        Write-Note "winget puuttuu, käytetään virallista asennusskriptiä:"
        Write-Note "  https://astral.sh/uv/install.ps1"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }

    $found = Find-Uv
    if (-not $found) {
        throw "uv:n asennus ei onnistunut. Asenna se käsin: https://docs.astral.sh/uv/"
    }
    return $found
}

# --- 1. uv ----------------------------------------------------------------

Write-Step "Tarkistetaan uv"
$Uv = Find-Uv
if ($Uv) {
    Write-Note "Löytyi: $Uv"
}
else {
    $Uv = Install-Uv
    Write-Note "Asennettu: $Uv"
}

# Lisätään uv:n hakemisto tämän istunnon PATH:iin, jotta myöhemmät komennot
# toimivat ilman shellin uudelleenkäynnistystä.
$UvDir = Split-Path -Parent $Uv
if ($env:PATH -notlike "*$UvDir*") {
    $env:PATH = "$UvDir;$env:PATH"
}

$null = Invoke-Native $Uv --version

# --- 2. Riippuvuudet ------------------------------------------------------

if ($Minimal) {
    Write-Step "Asennetaan rajapinnan riippuvuudet (kevyt, ei GPU:ta)"
    $code = Invoke-Native $Uv sync --directory $RepoRoot --group dev
}
else {
    Write-Step "Asennetaan riippuvuudet GPU-tuella (torch + diffusers, ~3 GB)"
    Write-Note "Kevyempi vaihtoehto ilman GPU-riippuvuuksia: -Minimal"
    $code = Invoke-Native $Uv sync --directory $RepoRoot --extra gpu --group dev
}
if ($code -ne 0) { throw "riippuvuuksien asennus epäonnistui (koodi $code)" }

# --- 3. Konfiguraatio -----------------------------------------------------

$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"
if (-not (Test-Path $EnvFile)) {
    Write-Step "Luodaan .env"
    Copy-Item $EnvExample $EnvFile
    Write-Note "Kopioitu .env.example -> .env (oletukset toimivat sellaisenaan)"
}
else {
    Write-Step "Konfiguraatio"
    Write-Note ".env on jo olemassa, ei ylikirjoiteta"
}

# --- 4. Laitteistotarkistus ----------------------------------------------

Write-Step "Tarkistetaan ympäristö"
# check_env palauttaa 1 jos kone kelpaa vain mock-ajoon. Se ei ole
# bootstrapin virhe vaan tieto, joten sitä ei käsitellä kaatumisena.
$EnvStatus = Invoke-Native $Uv run --directory $RepoRoot python scripts/check_env.py

# --- 5. Testit ------------------------------------------------------------

if (-not $SkipTests) {
    Write-Step "Ajetaan testit (ei vaadi GPU:ta eikä painoja)"
    $code = Invoke-Native $Uv run --directory $RepoRoot pytest -q
    if ($code -ne 0) { throw "testit epäonnistuivat - asennus ei ole kunnossa" }
}

# --- 6. Painot ------------------------------------------------------------

if ($Weights) {
    Write-Step "Ladataan mallin painot: $ModelProfile"
    Write-Note "Tämä on kymmeniä gigatavuja ja voi kestää kauan."
    $code = Invoke-Native $Uv run --directory $RepoRoot python scripts/download_model.py $ModelProfile
    if ($code -ne 0) { throw "painojen lataus epäonnistui (koodi $code)" }
}

# --- 7. Onko painot jo levyllä? -------------------------------------------

# Tarkistetaan tilanne sen sijaan että oletettaisiin. Painot ovat voineet olla
# levyllä jo ennen tätä ajoa, jolloin "lataa painot" -ohje olisi väärä.
$WeightsPresent = $false
if (-not $Minimal) {
    Write-Step "Tarkistetaan mallin painot"
    $WeightsPresent = (Invoke-Native $Uv run --directory $RepoRoot `
            python scripts/download_model.py --check $ModelProfile) -eq 0
}

# --- Yhteenveto -----------------------------------------------------------

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host " Asennus valmis" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

if ($Minimal) {
    Write-Host "Kevyt asennus: käytä mock-backendiä." -ForegroundColor Yellow
    Write-Host '  $env:VIDEO_SERVER_BACKEND = "mock"'
    Write-Host "  uv run uvicorn video_server.main:app"
}
elseif (-not $WeightsPresent) {
    Write-Host "Painoja ei ole vielä ladattu. Lataa ne ennen käynnistystä:"
    Write-Host "  uv run python scripts/download_model.py $ModelProfile"
    Write-Host ""
    Write-Host "Tai kokeile rajapintaa heti ilman painoja:"
    Write-Host '  $env:VIDEO_SERVER_BACKEND = "mock"'
    Write-Host "  uv run uvicorn video_server.main:app"
}
else {
    Write-Host "Käynnistä palvelin:"
    Write-Host "  uv run uvicorn video_server.main:app"
    Write-Host ""
    Write-Host "Malli latautuu taustalla; GET /api/v1/health kertoo koska se on valmis."
}

if ($EnvStatus -ne 0 -and -not $Minimal) {
    Write-Host ""
    Write-Host "Huom: ympäristötarkistus ei todennut konetta valmiiksi oikeaan" -ForegroundColor Yellow
    Write-Host "inferenssiin. Katso yllä olevat merkinnät." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Dokumentaatio: README.md | Asetukset: .env | API: /docs"
Write-Host ""
