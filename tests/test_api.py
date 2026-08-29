from fastapi.testclient import TestClient

from app.main import app
from app.database import save_site
from app.models import (
    ExtractionResult,
    FieldValue,
    JobRequest,
    LocationVerification,
    Metrics,
    Provenance,
    RunMetadata,
    SiteRecord,
)


client = TestClient(app)


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job():
    response = client.post(
        "/api/jobs",
        json={
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    assert response.status_code == 202

    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"


def test_get_job():
    create_response = client.post(
        "/api/jobs",
        json={
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    job_id = create_response.json()["job_id"]

    response = client.get(
        f"/api/jobs/{job_id}"
    )

    assert response.status_code == 200
    assert response.json()["job_id"] == job_id


def test_get_missing_job():
    response = client.get(
        "/api/jobs/does-not-exist"
    )

    assert response.status_code == 404


def test_invalid_job_request():
    response = client.post(
        "/api/jobs",
        json={
            "latitude": 100,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    assert response.status_code == 422


def test_sites():
    response = client.get("/api/sites")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_missing_site():
    response = client.get(
        "/api/sites/does-not-exist"
    )

    assert response.status_code == 404 

def test_get_saved_site():
    site = SiteRecord(
        site_id="api-site-1",
        input=JobRequest(
            latitude=48.8566,
            longitude=2.3522,
            radius_km=10,
        ),
        extraction=ExtractionResult(
            official_name=FieldValue(
                value="API Test Quarry",
                confidence=0.95,
            ),
            site_type=FieldValue(
                value="Quarry",
                confidence=0.95,
            ),
            description=FieldValue(
                value="Test limestone quarry.",
                confidence=0.90,
            ),
            materials_produced=[
                FieldValue(
                    value="limestone",
                    confidence=0.90,
                )
            ],
            certifications=[],
            operational_status=FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="not_found",
            ),
            location_verification=LocationVerification(
                is_verified=False,
                confidence=0.0,
                method="none",
            ),
        ),
        provenance=Provenance(),
        metrics=Metrics(),
        run_metadata=RunMetadata(
            run_id="run-api-1",
            prompt_hash="test-hash",
            scraper_version="1.0.0",
        ),
    )

    save_site(site)

    response = client.get("/api/sites/api-site-1")

    assert response.status_code == 200

    data = response.json()

    assert data["site_id"] == "api-site-1"
    assert data["extraction"]["official_name"]["value"] == "API Test Quarry"