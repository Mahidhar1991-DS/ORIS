import asyncio

from app.database import save_job
from app.models import Job
from app.pipeline import run_pipeline


queue: asyncio.Queue[Job] = asyncio.Queue()


async def submit_job(job: Job) -> None:
    """Add a job to the worker queue."""

    save_job(job)
    await queue.put(job)


async def process_job(job: Job) -> None:
    """Process one queued job."""

    print(
        f"PROCESSING JOB: {job.job_id}",
        flush=True,
    )

    job.status = "running"
    job.progress = 0.05
    job.error = None
    save_job(job)

    try:
        request = job.result or {}

        latitude = request["latitude"]
        longitude = request["longitude"]
        radius_km = request["radius_km"]

        print(
            f"JOB {job.job_id}: starting pipeline",
            flush=True,
        )

        print(
            f"JOB {job.job_id}: "
            f"coordinates={latitude},{longitude} "
            f"radius={radius_km}km",
            flush=True,
        )

        # ---------------------------------------------------------
        # Run the complete quarry pipeline.
        # ---------------------------------------------------------
        result = await run_pipeline(
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
        )

        print(
            f"JOB {job.job_id}: pipeline completed",
            flush=True,
        )

        # ---------------------------------------------------------
        # Store final result.
        # ---------------------------------------------------------
        job.status = "completed"
        job.progress = 1.0
        job.result = result
        job.error = None

        save_job(job)

        print(
            f"JOB {job.job_id}: completed successfully",
            flush=True,
        )

    except Exception as exc:
        # ---------------------------------------------------------
        # Store failure.
        # ---------------------------------------------------------
        job.status = "failed"
        job.error = str(exc)

        save_job(job)

        print(
            f"JOB {job.job_id}: FAILED: {exc}",
            flush=True,
        )


async def worker() -> None:
    """Continuously process jobs from the queue."""

    print(
        "ORIS WORKER STARTED",
        flush=True,
    )

    while True:
        job = await queue.get()

        print(
            f"WORKER PICKED UP JOB: {job.job_id}",
            flush=True,
        )

        try:
            await process_job(job)

        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)

            save_job(job)

            print(
                f"WORKER ERROR: {exc}",
                flush=True,
            )

        finally:
            queue.task_done()