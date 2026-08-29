import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse

from app.database import (
    get_job,
    get_site,
    list_sites,
)
from app.models import Job, JobRequest
from app.worker import submit_job, worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(worker())
    app.state.worker_task = worker_task

    try:
        yield
    finally:
        worker_task.cancel()

        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ORIS Quarry Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def home():
    return FileResponse(
        Path(__file__).resolve().parent.parent
        / "static"
        / "index.html"
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/api/jobs",
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(request: JobRequest):
    job_id = str(uuid4())

    job = Job(
        job_id=job_id,
        result={
            "latitude": request.latitude,
            "longitude": request.longitude,
            "radius_km": request.radius_km,
        },
    )

    await submit_job(job)

    return {
        "job_id": job_id,
        "status": job.status,
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job


@app.get("/api/sites")
async def get_sites():
    return list_sites()


@app.get("/api/sites/{site_id}")
async def get_site_by_id(site_id: str):
    site = get_site(site_id)

    if site is None:
        raise HTTPException(
            status_code=404,
            detail="Site not found",
        )

    return site