from fastapi import FastAPI, HTTPException

from app.models import JobRequest
from app.pipeline import run_pipeline


app = FastAPI(
    title="ORIS Quarry Intelligence",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search")
async def search(request: JobRequest):
    try:
        return await run_pipeline(
            latitude=request.latitude,
            longitude=request.longitude,
            radius_km=request.radius_km,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc