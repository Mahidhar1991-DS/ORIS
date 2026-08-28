from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search(monkeypatch):
    async def fake_pipeline(
        latitude,
        longitude,
        radius_km,
    ):
        return {
            "count": 1,
            "results": [],
        }

    monkeypatch.setattr(
        "app.main.run_pipeline",
        fake_pipeline,
    )

    response = client.post(
        "/search",
        json={
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_search_validates_coordinates():
    response = client.post(
        "/search",
        json={
            "latitude": 100,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    assert response.status_code == 422