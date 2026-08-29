import pytest


@pytest.fixture(autouse=True)
def test_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_oris.db"

    monkeypatch.setenv(
        "ORIS_DB_PATH",
        str(db_path),
    )

    import app.database as database

    database.DB_PATH = db_path
    database._init_db()


def test_save_and_get_job():
    from app.database import get_job, save_job
    from app.models import Job

    job = Job(
        job_id="job-1"
    )

    save_job(job)

    result = get_job("job-1")

    assert result is not None
    assert result.job_id == "job-1"
    assert result.status == "queued"


def test_get_missing_job():
    from app.database import get_job

    result = get_job("missing")

    assert result is None


def test_save_and_get_site():
    from app.database import get_site, save_site
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

    site = SiteRecord(
        site_id="site-1",
        input=JobRequest(
            latitude=48.8566,
            longitude=2.3522,
            radius_km=10,
        ),
        extraction=ExtractionResult(
            official_name=FieldValue(
                value="Test Quarry",
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
            run_id="run-test-1",
            prompt_hash="test-hash",
            scraper_version="1.0.0",
        ),
    )

    save_site(site)

    result = get_site("site-1")

    assert result is not None
    assert result.site_id == "site-1"

    assert (
        result.extraction.official_name.value
        == "Test Quarry"
    )

    assert (
        result.extraction.site_type.value
        == "Quarry"
    )

    assert (
        result.extraction.description.value
        == "Test limestone quarry."
    )

    assert (
        result.extraction.materials_produced[0].value
        == "limestone"
    )

    assert (
        result.input.latitude
        == 48.8566
    )

    assert (
        result.input.longitude
        == 2.3522
    )


def test_list_sites():
    from app.database import list_sites, save_site
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

    site = SiteRecord(
        site_id="site-1",
        input=JobRequest(
            latitude=48.8566,
            longitude=2.3522,
            radius_km=10,
        ),
        extraction=ExtractionResult(
            official_name=FieldValue(
                value="Test Quarry",
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
            materials_produced=[],
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
            run_id="run-test-2",
            prompt_hash="test-hash",
            scraper_version="1.0.0",
        ),
    )

    save_site(site)

    sites = list_sites()

    assert len(sites) == 1
    assert sites[0].site_id == "site-1"

    assert (
        sites[0].extraction.official_name.value
        == "Test Quarry"
    )