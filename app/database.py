import json
import os
import sqlite3
from pathlib import Path

from app.models import Job, SiteRecord


DB_PATH = Path(
    os.getenv(
        "ORIS_DB_PATH",
        str(
            Path(__file__).resolve().parent.parent
            / "oris.db"
        ),
    )
)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sites (
                site_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )

        connection.commit()


_init_db()


def save_job(job: Job) -> Job:
    data = job.model_dump_json()

    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO jobs (
                job_id,
                data
            )
            VALUES (?, ?)
            """,
            (
                job.job_id,
                data,
            ),
        )

        connection.commit()

    return job


def get_job(job_id: str) -> Job | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT data
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    return Job.model_validate_json(row["data"])


def save_site(site: SiteRecord) -> SiteRecord:
    data = site.model_dump_json()

    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO sites (
                site_id,
                data
            )
            VALUES (?, ?)
            """,
            (
                site.site_id,
                data,
            ),
        )

        connection.commit()

    return site


def get_site(site_id: str) -> SiteRecord | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT data
            FROM sites
            WHERE site_id = ?
            """,
            (site_id,),
        ).fetchone()

    if row is None:
        return None

    return SiteRecord.model_validate_json(
        row["data"]
    )


def list_sites() -> list[SiteRecord]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT data
            FROM sites
            ORDER BY site_id
            """
        ).fetchall()

    return [
        SiteRecord.model_validate_json(
            row["data"]
        )
        for row in rows
    ]