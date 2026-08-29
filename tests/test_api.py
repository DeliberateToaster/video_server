"""Rajapintatestit mock-backendiä vasten."""

from __future__ import annotations

import base64
import io
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import wait_job, wait_ready
from video_server.main import create_app
from video_server.settings import Settings


def _tiny_png() -> str:
    """Oikea PNG base64:nä. Mock ei dekoodaa kuvaa, mutta Vaiheen 2 Wan-backend
    dekoodaa - testidatan on syytä olla aitoa jo nyt."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 80, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


TINY_PNG = _tiny_png()


def test_health_reports_ready(client: TestClient) -> None:
    payload = client.get("/api/v1/health").json()
    assert payload["status"] == "ready"
    assert payload["backend"] == "mock"


def test_models_exposes_constraints(client: TestClient) -> None:
    payload = client.get("/api/v1/models").json()
    assert payload["active"] == "mock"
    assert "mock" in payload["available"]
    assert payload["tier"] == "test"
    constraints = payload["constraints"]
    # Asiakkaan on saatava tästä kaikki mitä kelvollisen pyynnön rakentaminen
    # vaatii, ilman tietoa palvelimen laitteistosta.
    assert constraints["frame_multiple"] == 4
    assert constraints["native_fps"] == 24
    assert constraints["supports_i2v"] is True
    assert "256x144" in constraints["resolutions"]


def test_txt2vid_completes_and_serves_video(client: TestClient, small_request: dict) -> None:
    response = client.post("/api/v1/txt2vid", json=small_request)
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    payload = wait_job(client, job_id)
    assert payload["status"] == "done", payload
    assert payload["error"] is None
    assert payload["video_url"] == f"/outputs/{job_id}.mp4"
    assert payload["progress"]["step"] == payload["progress"]["total_steps"] == 3

    video = client.get(payload["video_url"])
    assert video.status_code == 200
    assert len(video.content) > 0


def test_seed_is_resolved_and_echoed(client: TestClient, small_request: dict) -> None:
    """seed=-1 tarkoittaa satunnaista, mutta ajon on oltava toistettavissa:
    vastauksen on kerrottava mikä siemen oikeasti käytettiin."""
    job_id = client.post("/api/v1/txt2vid", json={**small_request, "seed": -1}).json()["job_id"]
    payload = wait_job(client, job_id)
    assert payload["params"]["seed"] >= 0


def test_params_echo_omits_init_image(client: TestClient, small_request: dict) -> None:
    job_id = client.post("/api/v1/img2vid", json={**small_request, "init_image": TINY_PNG}).json()[
        "job_id"
    ]
    payload = wait_job(client, job_id)
    assert payload["status"] == "done"
    # Base64-kuva ei kuulu vastaukseen eikä sidecar-metadataan.
    assert "init_image" not in payload["params"]


def test_defaults_come_from_profile(client: TestClient) -> None:
    job_id = client.post("/api/v1/txt2vid", json={"prompt": "oletukset"}).json()["job_id"]
    payload = client.get(f"/api/v1/jobs/{job_id}").json()
    assert payload["params"]["fps"] == 24  # profiilin native_fps
    assert payload["params"]["resolution"] == "256x144"  # profiilin ensimmäinen
    assert payload["params"]["num_frames"] == 81
    client.post(f"/api/v1/jobs/{job_id}/cancel")


def test_invalid_frame_count_is_rejected_with_guidance(
    client: TestClient, small_request: dict
) -> None:
    response = client.post("/api/v1/txt2vid", json={**small_request, "num_frames": 50})
    assert response.status_code == 400
    detail = response.json()["detail"]
    # Virheviestin on kerrottava sääntö ja lähimmät kelvolliset arvot, jottei
    # asiakkaan tarvitse arvata.
    assert "49" in detail and "53" in detail


def test_invalid_resolution_is_rejected(client: TestClient, small_request: dict) -> None:
    response = client.post("/api/v1/txt2vid", json={**small_request, "resolution": "1920x1080"})
    assert response.status_code == 400
    assert "256x144" in response.json()["detail"]


def test_empty_prompt_is_rejected(client: TestClient) -> None:
    assert client.post("/api/v1/txt2vid", json={"prompt": ""}).status_code == 422


def test_unknown_job_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/jobs/ei-ole").status_code == 404
    assert client.post("/api/v1/jobs/ei-ole/cancel").status_code == 404


def test_cancel_running_job(make_client, small_request: dict) -> None:
    client = make_client(mock_step_seconds=0.05)
    job_id = client.post(
        "/api/v1/txt2vid", json={**small_request, "num_inference_steps": 100}
    ).json()["job_id"]

    wait_job(client, job_id, until={"running"}, timeout=10)
    assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200

    payload = wait_job(client, job_id, timeout=10)
    assert payload["status"] == "cancelled"
    assert payload["video_url"] is None


def test_cancel_finished_job_conflicts(client: TestClient, small_request: dict) -> None:
    job_id = client.post("/api/v1/txt2vid", json=small_request).json()["job_id"]
    wait_job(client, job_id)
    response = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert response.status_code == 409


def test_queue_full_returns_429(make_client, small_request: dict) -> None:
    client = make_client(mock_step_seconds=0.05, max_queue_size=2)
    slow = {**small_request, "num_inference_steps": 100}
    codes = [client.post("/api/v1/txt2vid", json=slow).status_code for _ in range(6)]
    assert 429 in codes, codes
    # Jonon täyttyminen ei saa rikkoa palvelinta.
    assert client.get("/api/v1/health").json()["status"] == "ready"


def test_oversized_body_returns_413(make_client, small_request: dict) -> None:
    client = make_client(max_request_bytes=2048)
    response = client.post("/api/v1/img2vid", json={**small_request, "init_image": "A" * 5000})
    assert response.status_code == 413


def test_jobs_survive_restart(tmp_path, small_request: dict) -> None:
    """Job-tila on muistissa, video levyllä. Ilman sidecar-metadataa
    uudelleenkäynnistys jättäisi valmiin videon orvoksi."""
    options = {
        "backend": "mock",
        "outputs_dir": tmp_path,
        "mock_step_seconds": 0.0,
        "tier": "test",
        "retention_max_age_days": None,
        "retention_max_total_gb": None,
    }

    with TestClient(create_app(Settings(**options))) as first:
        wait_ready(first)
        job_id = first.post("/api/v1/txt2vid", json=small_request).json()["job_id"]
        assert wait_job(first, job_id)["status"] == "done"

    with TestClient(create_app(Settings(**options))) as second:
        wait_ready(second)
        payload = second.get(f"/api/v1/jobs/{job_id}").json()
        assert payload["status"] == "done"
        assert payload["video_url"] == f"/outputs/{job_id}.mp4"
        assert second.get(payload["video_url"]).status_code == 200


def test_generation_blocked_until_backend_ready(tmp_path) -> None:
    """Malli latautuu minuutteja; siihen asti generointi vastaa 503 eikä
    hyväksy työtä hiljaisesti jonoon."""
    app = create_app(
        Settings(backend="mock", outputs_dir=tmp_path, tier="test", mock_step_seconds=0.0)
    )

    with TestClient(app) as client:
        wait_ready(client)
        # Mock latautuu välittömästi, joten latauksen aikainen tila asetetaan
        # käsin: testattava asia on portti, ei latauksen kesto.
        app.state.backend_state.status = "loading"
        assert client.post("/api/v1/txt2vid", json={"prompt": "x"}).status_code == 503
        # /health vastaa aina 200, myös latauksen aikana.
        assert client.get("/api/v1/health").status_code == 200

        app.state.backend_state.status = "ready"
        assert client.post("/api/v1/txt2vid", json={"prompt": "x"}).status_code == 202


def test_unsupported_mode_returns_501(client: TestClient, small_request: dict) -> None:
    """501 eikä 400: pyyntö on kelvollinen, mutta aktiivinen malli ei tarjoa
    tätä suuntaa. A14B on erikseen T2V- ja I2V-checkpointeina."""
    backend = client.app.state.backend
    backend._profile = replace(backend.profile, supports_i2v=False)

    response = client.post("/api/v1/img2vid", json={**small_request, "init_image": TINY_PNG})
    assert response.status_code == 501
    assert "i2v" in response.json()["detail"]

    # T2V-suunta toimii edelleen samalla mallilla.
    assert client.post("/api/v1/txt2vid", json=small_request).status_code == 202


# --- Vaihe 4: esikatselu, vaiheet ja autentikointi ------------------------


def test_preview_is_off_by_default(client: TestClient, small_request: dict) -> None:
    """Speksi: esikatselu on ylimääräinen VAE-kutsu, ei oletus."""
    job_id = client.post("/api/v1/txt2vid", json=small_request).json()["job_id"]
    assert wait_job(client, job_id)["preview_url"] is None


def test_preview_is_served_when_enabled(make_client, small_request: dict) -> None:
    client = make_client(preview_every_n_steps=1)
    job_id = client.post("/api/v1/txt2vid", json=small_request).json()["job_id"]

    payload = wait_job(client, job_id)
    assert payload["preview_url"] == f"/outputs/{job_id}-preview.png"

    # Esikatselu tarjoillaan samaa staattista reittiä kuin valmis video.
    response = client.get(payload["preview_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_decoding_phase_is_visible_in_job_status(make_client, small_request: dict) -> None:
    """Ilman vaihetietoa job näyttäisi tilaa n/n vaikka dekoodaus on kesken."""
    client = make_client(mock_step_seconds=0.0)
    job_id = client.post("/api/v1/txt2vid", json=small_request).json()["job_id"]
    payload = wait_job(client, job_id)

    # Valmiin jobin viimeisin raportoitu vaihe on dekoodaus.
    assert payload["progress"]["phase"] == "decoding"


def test_api_key_disabled_by_default(client: TestClient) -> None:
    """v1-rajaus: ei autentikointia ellei sitä erikseen kytketä."""
    assert client.get("/api/v1/models").status_code == 200


def test_api_key_required_when_configured(make_client, small_request: dict) -> None:
    client = make_client(api_key="salainen-avain")

    assert client.get("/api/v1/models").status_code == 401
    assert client.post("/api/v1/txt2vid", json=small_request).status_code == 401
    assert client.get("/api/v1/models", headers={"X-API-Key": "vaara"}).status_code == 401

    ok = client.get("/api/v1/models", headers={"X-API-Key": "salainen-avain"})
    assert ok.status_code == 200


def test_health_stays_open_for_monitoring(make_client) -> None:
    """Monitoroinnin on toimittava ilman avainta; /health ei paljasta muuta
    kuin latauksen tilan."""
    client = make_client(api_key="salainen-avain")
    assert client.get("/api/v1/health").status_code == 200


def test_api_key_also_guards_generated_videos(make_client, small_request: dict) -> None:
    """Videot ovat se varsinainen suojattava sisältö; pelkkä /api/v1:n
    suojaaminen jättäisi ne auki."""
    client = make_client(api_key="salainen-avain")
    headers = {"X-API-Key": "salainen-avain"}
    job_id = client.post("/api/v1/txt2vid", json=small_request, headers=headers).json()["job_id"]
    payload = wait_job(client, job_id, headers=headers)

    assert client.get(payload["video_url"]).status_code == 401
    assert client.get(payload["video_url"], headers=headers).status_code == 200
