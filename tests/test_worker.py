import pytest

from app.models import Job
from app.worker import process_job


@pytest.mark.asyncio
async def test_process_job(monkeypatch):
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
        "app.worker.run_pipeline",
        fake_pipeline,
    )

    job = Job(
        job_id="job-1",
        result={
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    await process_job(job)

    assert job.status == "completed"
    assert job.progress == 1.0
    assert job.result["count"] == 1


@pytest.mark.asyncio
async def test_process_job_failure(monkeypatch):
    async def fake_pipeline(
        latitude,
        longitude,
        radius_km,
    ):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(
        "app.worker.run_pipeline",
        fake_pipeline,
    )

    job = Job(
        job_id="job-2",
        result={
            "latitude": 48.8566,
            "longitude": 2.3522,
            "radius_km": 50,
        },
    )

    await process_job(job)

    assert job.status == "failed"
    assert job.error == "pipeline failed"