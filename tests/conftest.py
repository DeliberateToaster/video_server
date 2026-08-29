"""Testien yhteiset apurit.

Kaikki testit ajetaan mock-backendillä: ei GPU:ta, ei torchia, ei mallin
latausta. Tämä on koko mock-backendin olemassaolon syy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from video_server.main import create_app
from video_server.settings import Settings

TERMINAL = {"done", "failed", "cancelled"}


def wait_ready(client: TestClient, timeout: float = 10.0) -> None:
    """Backend latautuu taustataskissa, joten valmius on odotettava.

    Testit käyttävät samaa reittiä kuin oikea asiakas: /health kertoo koska
    työtä voi lähettää.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get("/api/v1/health").json()
        if payload["status"] == "ready":
            return
        if payload["status"] == "error":
            raise AssertionError(f"backendin lataus epäonnistui: {payload['detail']}")
        time.sleep(0.02)
    raise AssertionError("backend ei tullut valmiiksi ajoissa")


def wait_job(
    client: TestClient,
    job_id: str,
    until: set[str] = TERMINAL,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in until:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} jäi tilaan {payload.get('status')!r}")


@pytest.fixture
def make_client(tmp_path) -> Iterator[Callable[..., TestClient]]:
    """Luo TestClientin annetulla konfiguraatiolla.

    Tehdasfunktio eikä valmis client, koska osa testeistä tarvitsee eri
    asetukset (hidas mock peruutustesteihin, pieni jono 429-testiin).
    """
    clients: list[TestClient] = []

    def _make(**overrides) -> TestClient:
        options = {
            "backend": "mock",
            "outputs_dir": tmp_path,
            "mock_step_seconds": 0.0,
            "max_queue_size": 4,
            "retention_max_age_days": None,
            "retention_max_total_gb": None,
            # Eksplisiittinen tier ohittaa laitteistotunnistuksen, jottei
            # testeissä tuoda torchia sisään.
            "tier": "test",
        }
        options.update(overrides)
        client = TestClient(create_app(Settings(**options)))
        client.__enter__()
        clients.append(client)
        wait_ready(client)
        return client

    yield _make

    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def small_request() -> dict:
    """Pieni pyyntö, jotta testit eivät enkoodaa 81 framea 1280x704:ssä."""
    return {
        "prompt": "testivideo",
        "num_frames": 5,
        "resolution": "256x144",
        "num_inference_steps": 3,
    }
