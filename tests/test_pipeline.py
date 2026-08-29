import fitz
import httpx
import pytest

from app.crawler import (
    html_to_text,
    pdf_to_text,
    validate_url,
)
from app.discovery import (
    deduplicate_quarries,
    discover_quarries,
)
from app.models import (
    Document,
    Evidence,
    FieldValue,
    LocationVerification,
    Quarry,
)
from app.extraction import (
    create_field,
    extract,
    find_evidence,
)
from app.pipeline import (
    process_quarry,
    run_pipeline,
)
from app.reconciliation import reconcile


class FakeResponse:
    def __init__(
        self,
        data,
        status_code=200,
    ):
        self.data = data
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeClient:
    response_data = {
        "elements": []
    }

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        *args,
    ):
        pass

    async def post(
        self,
        *args,
        **kwargs,
    ):
        return FakeResponse(
            self.response_data
        )

    async def get(
        self,
        *args,
        **kwargs,
    ):
        return FakeResponse(
            {
                "address": {
                    "town": "Paris",
                }
            }
        )


# ============================================================
# DISCOVERY TESTS
# ============================================================

@pytest.mark.asyncio
async def test_reverse_geocode_success(monkeypatch):
    from app import geocoding

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "address": {
                    "town": "Cormeilles-en-Parisis",
                    "municipality": "Argenteuil",
                }
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            pass

        async def get(
            self,
            *args,
            **kwargs,
        ):
            return FakeResponse()

    geocoding.clear_cache()

    monkeypatch.setattr(
        "app.geocoding.httpx.AsyncClient",
        FakeClient,
    )

    result = await geocoding.reverse_geocode(
        48.9702088,
        2.2186234,
    )

    assert result == "Cormeilles-en-Parisis"

@pytest.mark.asyncio
async def test_reverse_geocode_failure_returns_none(
    monkeypatch,
):
    from app import geocoding

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc,
            tb,
        ):
            pass

        async def get(
            self,
            *args,
            **kwargs,
        ):
            raise httpx.ConnectError(
                "Nominatim unavailable"
            )

    geocoding.clear_cache()

    monkeypatch.setattr(
        "app.geocoding.httpx.AsyncClient",
        FakeClient,
    )

    result = await geocoding.reverse_geocode(
        48.9702088,
        2.2186234,
    )

    assert result is None


@pytest.mark.asyncio
async def test_quarry_municipality_enrichment(
    monkeypatch,
):
    from app.discovery import (
        enrich_quarry_municipalities,
    )

    async def fake_reverse_geocode(
        latitude,
        longitude,
    ):
        return "Cormeilles-en-Parisis"

    monkeypatch.setattr(
        "app.discovery.reverse_geocode",
        fake_reverse_geocode,
    )

    quarry = Quarry(
        site_id="test-quarry",
        name="Carrière Lambert",
        municipality=None,
        latitude=48.9702088,
        longitude=2.2186234,
        distance_km=15.965,
        website=None,
    )

    result = await enrich_quarry_municipalities(
        [quarry]
    )

    assert len(result) == 1
    assert (
        result[0].municipality
        == "Cormeilles-en-Parisis"
    )

@pytest.mark.asyncio
async def test_discover_node_quarry(monkeypatch):
    FakeClient.response_data = {
        "elements": [
            {
                "type": "node",
                "id": 123,
                "lat": 48.857,
                "lon": 2.353,
                "tags": {
                    "name": "Test Quarry",
                    "website": "https://example.com",
                },
            }
        ]
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert len(result) == 1
    assert result[0].name == "Test Quarry"
    assert result[0].website == "https://example.com"
    assert result[0].municipality == "Paris"

@pytest.mark.asyncio
async def test_discover_uses_second_overpass_endpoint(
    monkeypatch,
):
    calls = []

    async def fake_query(
        client,
        endpoint,
        query,
    ):
        calls.append(endpoint)

        if len(calls) == 1:
            raise httpx.ConnectError(
                "first endpoint unavailable"
            )

        return {
            "elements": [
                {
                    "type": "node",
                    "id": 999,
                    "lat": 48.857,
                    "lon": 2.353,
                    "tags": {
                        "name": "Fallback Quarry",
                    },
                }
            ]
        }

    async def fake_reverse_geocode(
        latitude,
        longitude,
    ):
        return "Paris"

    monkeypatch.setattr(
        "app.discovery._query_overpass",
        fake_query,
    )

    monkeypatch.setattr(
        "app.discovery.reverse_geocode",
        fake_reverse_geocode,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert len(calls) == 2
    assert result[0].name == "Fallback Quarry"
    assert result[0].municipality == "Paris"


def test_verify_extracted_location_match():
    from app.models import (
        ExtractionResult,
        FieldValue,
        LocationVerification,
    )
    from app.pipeline import verify_extracted_location

    extraction = ExtractionResult(
        official_name=FieldValue(
            value=None,
            confidence=0.0,
        ),
        site_type=FieldValue(
            value=None,
            confidence=0.0,
        ),
        description=FieldValue(
            value=None,
            confidence=0.0,
        ),
        materials_produced=[],
        certifications=[],
        operational_status=FieldValue(
            value=None,
            confidence=0.0,
        ),
        location_verification=LocationVerification(
            is_verified=False,
            confidence=0.5,
            extracted_city="Cormeilles-en-Parisis",
            method="llm_inference",
        ),
    )

    result = verify_extracted_location(
        extraction,
        "Cormeilles-en-Parisis",
    )

    assert result.location_verification.is_verified is True
    assert result.location_verification.confidence == 1.0
    assert (
        result.location_verification.method
        == "string_match"
    )

def test_verify_extracted_location_mismatch():
    from app.models import (
        ExtractionResult,
        FieldValue,
        LocationVerification,
    )
    from app.pipeline import verify_extracted_location

    extraction = ExtractionResult(
        official_name=FieldValue(
            value=None,
            confidence=0.0,
        ),
        site_type=FieldValue(
            value=None,
            confidence=0.0,
        ),
        description=FieldValue(
            value=None,
            confidence=0.0,
        ),
        materials_produced=[],
        certifications=[],
        operational_status=FieldValue(
            value=None,
            confidence=0.0,
        ),
        location_verification=LocationVerification(
            is_verified=True,
            confidence=0.95,
            extracted_city="Trehorenteuc",
            method="llm_inference",
        ),
    )

    result = verify_extracted_location(
        extraction,
        "Saclay",
    )

    assert result.location_verification.is_verified is False
    assert result.location_verification.confidence == 0.0
    assert (
        result.location_verification.method
        == "none"
    )
    assert (
        result.location_verification.extracted_city
        == "Trehorenteuc"
    )

def test_verify_extracted_location_missing_city():
    from app.models import (
        ExtractionResult,
        FieldValue,
        LocationVerification,
    )
    from app.pipeline import verify_extracted_location

    extraction = ExtractionResult(
        official_name=FieldValue(
            value=None,
            confidence=0.0,
        ),
        site_type=FieldValue(
            value=None,
            confidence=0.0,
        ),
        description=FieldValue(
            value=None,
            confidence=0.0,
        ),
        materials_produced=[],
        certifications=[],
        operational_status=FieldValue(
            value=None,
            confidence=0.0,
        ),
        location_verification=LocationVerification(
            is_verified=False,
            confidence=0.0,
            extracted_city=None,
            method="none",
        ),
    )

    result = verify_extracted_location(
        extraction,
        "Saclay",
    )

    assert result.location_verification.is_verified is False
    assert (
        result.location_verification.extracted_city
        is None
    )


@pytest.mark.asyncio
async def test_discover_returns_empty_when_provider_succeeds(
    monkeypatch,
):
    async def fake_query(
        client,
        endpoint,
        query,
    ):
        return {
            "elements": []
        }

    monkeypatch.setattr(
        "app.discovery._query_overpass",
        fake_query,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert result == []


@pytest.mark.asyncio
async def test_discover_raises_when_all_overpass_endpoints_fail(
    monkeypatch,
):
    async def fake_query(
        client,
        endpoint,
        query,
    ):
        raise httpx.ConnectError(
            "Overpass unavailable"
        )

    monkeypatch.setattr(
        "app.discovery._query_overpass",
        fake_query,
    )

    with pytest.raises(
        RuntimeError,
        match="All Overpass endpoints failed",
    ):
        await discover_quarries(
            48.8566,
            2.3522,
            50,
        )


@pytest.mark.asyncio
async def test_discover_way_quarry(monkeypatch):
    FakeClient.response_data = {
        "elements": [
            {
                "type": "way",
                "id": 456,
                "center": {
                    "lat": 48.857,
                    "lon": 2.353,
                },
                "tags": {
                    "name": "Way Quarry",
                },
            }
        ]
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert len(result) == 1
    assert result[0].name == "Way Quarry"


@pytest.mark.asyncio
async def test_skip_way_without_center(monkeypatch):
    FakeClient.response_data = {
        "elements": [
            {
                "type": "way",
                "id": 456,
                "tags": {
                    "name": "Unknown Quarry",
                },
            }
        ]
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert result == []


@pytest.mark.asyncio
async def test_skip_quarry_outside_radius(monkeypatch):
    FakeClient.response_data = {
        "elements": [
            {
                "type": "node",
                "id": 789,
                "lat": 50.8503,
                "lon": 4.3517,
                "tags": {
                    "name": "Far Quarry",
                },
            }
        ]
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        10,
    )

    assert result == []


@pytest.mark.asyncio
async def test_no_quarries(monkeypatch):
    FakeClient.response_data = {
        "elements": []
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(
        48.8566,
        2.3522,
        50,
    )

    assert result == []


def test_deduplicate_quarries():
    quarries = [
        Quarry(
            site_id="1",
            name="Test Quarry",
            latitude=48.857,
            longitude=2.353,
            distance_km=1.0,
        ),
        Quarry(
            site_id="2",
            name="Test Quarry",
            latitude=48.8571,
            longitude=2.3531,
            distance_km=1.1,
        ),
        Quarry(
            site_id="3",
            name="Another Quarry",
            latitude=48.86,
            longitude=2.36,
            distance_km=2.0,
        ),
    ]

    result = deduplicate_quarries(
        quarries
    )

    assert len(result) == 2
    assert result[0].name == "Test Quarry"
    assert result[1].name == "Another Quarry"


# ============================================================
# CRAWLER TESTS
# ============================================================


def test_validate_url():
    assert validate_url(
        "https://example.com"
    )

    assert validate_url(
        "http://example.com"
    )

    assert not validate_url(
        "ftp://example.com"
    )

    assert not validate_url(
        "example.com"
    )


def test_validate_url_rejects_private_ip():
    assert not validate_url(
        "http://127.0.0.1"
    )

    assert not validate_url(
        "http://192.168.1.1"
    )

    assert not validate_url(
        "http://10.0.0.1"
    )


def test_html_to_text():
    html = """
    <html>
        <head>
            <style>
                .hidden {
                    display: none;
                }
            </style>
        </head>

        <body>
            <h1>Test Quarry</h1>

            <p>
                Limestone producer
            </p>

            <script>
                alert("remove me")
            </script>
        </body>
    </html>
    """

    text = html_to_text(
        html
    )

    assert "Test Quarry" in text
    assert "Limestone producer" in text
    assert "remove me" not in text


def test_pdf_to_text():
    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (50, 50),
        "Test Quarry produces limestone.",
    )

    pdf_bytes = document.tobytes()

    document.close()

    text = pdf_to_text(
        pdf_bytes
    )

    assert "Test Quarry" in text
    assert "limestone" in text


# ============================================================
# EXTRACTION TESTS
# ============================================================


def test_find_evidence():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Our quarry produces "
            "limestone aggregates."
        ),
    )

    evidence = find_evidence(
        document,
        "limestone aggregates",
    )

    assert evidence is not None
    assert evidence.source_id == "source-1"
    assert evidence.char_start == 20
    assert evidence.char_end == 40
    assert evidence.quote == (
        "limestone aggregates"
    )


def test_find_evidence_not_found():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Our quarry produces limestone."
        ),
    )

    evidence = find_evidence(
        document,
        "granite",
    )

    assert evidence is None


def test_create_grounded_field():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Our quarry produces "
            "limestone aggregates."
        ),
    )

    field = create_field(
        document=document,
        value="Limestone",
        quote="limestone aggregates",
        confidence=0.9,
    )

    assert field.value == "Limestone"
    assert field.confidence == 0.9
    assert len(field.evidence) == 1


def test_create_field_abstains_without_evidence():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Our quarry produces limestone."
        ),
    )

    field = create_field(
        document=document,
        value="Granite",
        quote="granite",
        confidence=0.9,
    )

    assert field.value is None
    assert (
        field.abstain_reason
        == "evidence_not_found"
    )


@pytest.mark.asyncio
async def test_extract_without_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Test Quarry produces limestone."
        ),
    )

    result = await extract(
        document
    )

    assert (
        result.official_name.value
        is None
    )

    assert (
        result.site_type.value
        is None
    )

    assert (
        result.description.value
        is None
    )

    assert (
        result.operational_status.value
        is None
    )

    assert (
        result.location_verification
        .is_verified
        is False
    )


@pytest.mark.asyncio
async def test_extract_with_mock_llm(
    monkeypatch,
):
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Test Quarry is operated by "
            "ABC Materials. "
            "The quarry produces limestone."
        ),
    )

    class FakeMessage:
        content = """
        {
            "official_name": {
                "value": "Test Quarry",
                "confidence": 0.95,
                "quote": "Test Quarry"
            },
            "site_type": {
                "value": "Quarry",
                "confidence": 0.95,
                "quote": "The quarry"
            },
            "description": {
                "value": "The quarry produces limestone.",
                "confidence": 0.90,
                "quote": "The quarry produces limestone."
            },
            "materials_produced": [
                {
                    "value": "limestone",
                    "confidence": 0.90,
                    "quote": "The quarry produces limestone."
                }
            ],
            "certifications": [],
            "operational_status": {
                "value": null,
                "confidence": 0.0,
                "quote": null
            },
            "location_verification": {
                "is_verified": false,
                "confidence": 0.0,
                "extracted_city": null,
                "method": "none"
            }
        }
        """

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [
            FakeChoice()
        ]

    class FakeCompletions:
        async def create(
            self,
            **kwargs,
        ):
            return FakeResponse()

    class FakeClient:
        def __init__(
            self,
            **kwargs,
        ):
            self.chat = type(
                "Chat",
                (),
                {
                    "completions":
                        FakeCompletions()
                },
            )()

    monkeypatch.setattr(
        "app.extraction.AsyncOpenAI",
        FakeClient,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    result = await extract(
        document
    )

    assert (
        result.official_name.value
        == "Test Quarry"
    )

    assert (
        result.site_type.value
        == "Quarry"
    )

    assert (
        result.description.value
        == "The quarry produces limestone."
    )

    assert (
        result.materials_produced[0].value
        == "limestone"
    )


def test_create_field_rejects_unsupported_value():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Our quarry produces limestone."
        ),
    )

    field = create_field(
        document=document,
        value="Granite",
        quote=(
            "Our quarry produces limestone."
        ),
        confidence=0.95,
    )

    assert field.value is None

    assert (
        field.abstain_reason
        == "value_not_supported_by_evidence"
    )


# ============================================================
# RECONCILIATION TESTS
# ============================================================


def test_reconcile_selects_best_value():
    first = FieldValue(
        value="Limestone",
        confidence=0.7,
        evidence=[
            Evidence(
                source_id="source-1",
                char_start=0,
                char_end=9,
                quote="Limestone",
            )
        ],
    )

    second = FieldValue(
        value="Limestone",
        confidence=0.95,
        evidence=[
            Evidence(
                source_id="source-2",
                char_start=10,
                char_end=19,
                quote="Limestone",
            )
        ],
    )

    result = reconcile(
        [first, second]
    )

    assert result.value == "Limestone"
    assert result.confidence == 0.95
    assert (
        result.evidence[0].source_id
        == "source-2"
    )


def test_reconcile_abstains_when_no_supported_values():
    result = reconcile(
        [
            FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="no_value_found",
            )
        ]
    )

    assert result.value is None

    assert (
        result.abstain_reason
        == "no_supported_value"
    )


# ============================================================
# PIPELINE TESTS
# ============================================================


@pytest.mark.asyncio
async def test_process_quarry(
    monkeypatch,
):
    async def fake_crawl(url):
        return (
            "Test Quarry produces limestone."
        )

    async def fake_extract(
    document,
    cost_tracker=None,
):
        from app.models import (
            Evidence,
            ExtractionResult,
            FieldValue,
        )

        evidence = Evidence(
            source_id=document.source_id,
            char_start=0,
            char_end=11,
            quote="Test Quarry",
        )

        return ExtractionResult(
            official_name=FieldValue(
                value="Test Quarry",
                confidence=0.95,
                evidence=[evidence],
            ),
            site_type=FieldValue(
                value="Quarry",
                confidence=0.95,
                evidence=[evidence],
            ),
            description=FieldValue(
                value=(
                    "Test Quarry produces "
                    "limestone."
                ),
                confidence=0.90,
                evidence=[evidence],
            ),
            materials_produced=[
                FieldValue(
                    value="limestone",
                    confidence=0.90,
                    evidence=[evidence],
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
        )

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    monkeypatch.setattr(
        "app.pipeline.extract",
        fake_extract,
    )

    result = await process_quarry(
        site_id="quarry-1",
        website="https://example.com",
    )

    assert (
        result.official_name.value
        == "Test Quarry"
    )

    assert (
        result.official_name.confidence
        == 0.95
    )

    assert len(
        result.official_name.evidence
    ) == 1


@pytest.mark.asyncio
async def test_process_quarry_when_crawl_fails(
    monkeypatch,
):
    async def fake_crawl(url):
        return None

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    result = await process_quarry(
        site_id="quarry-1",
        website="https://example.com",
    )

    assert result is None


@pytest.mark.asyncio
async def test_process_quarry_crawls_and_extracts(
    monkeypatch,
):
    crawled_urls = []

    async def fake_crawl(url):
        crawled_urls.append(url)

        return (
            "Test Quarry produces limestone."
        )

    async def fake_extract(
    document,
    cost_tracker=None,
):
        from app.models import (
            Evidence,
            ExtractionResult,
            FieldValue,
            LocationVerification,
        )

        evidence = Evidence(
            source_id=document.source_id,
            char_start=0,
            char_end=11,
            quote="Test Quarry",
        )

        return ExtractionResult(
            official_name=FieldValue(
                value="Test Quarry",
                confidence=0.95,
                evidence=[evidence],
            ),
            site_type=FieldValue(
                value="Quarry",
                confidence=0.95,
                evidence=[evidence],
            ),
            description=FieldValue(
                value=(
                    "Test Quarry produces "
                    "limestone."
                ),
                confidence=0.90,
                evidence=[evidence],
            ),
            materials_produced=[
                FieldValue(
                    value="limestone",
                    confidence=0.90,
                    evidence=[evidence],
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
        )

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    monkeypatch.setattr(
        "app.pipeline.extract",
        fake_extract,
    )

    result = await process_quarry(
        site_id="quarry-1",
        website="https://example.com",
    )

    assert crawled_urls == [
        "https://example.com"
    ]

    assert result is not None

    assert (
        result.official_name.value
        == "Test Quarry"
    )

    assert (
        result.materials_produced[0].value
        == "limestone"
    )


@pytest.mark.asyncio
async def test_process_quarry_crawl_failure(
    monkeypatch,
):
    extract_called = False

    async def fake_crawl(url):
        return None

    async def fake_extract(
    document,
    cost_tracker=None,
):
        nonlocal extract_called

        extract_called = True

        raise AssertionError(
            "extract should not be called"
        )

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    monkeypatch.setattr(
        "app.pipeline.extract",
        fake_extract,
    )

    result = await process_quarry(
        site_id="quarry-1",
        website="https://example.com",
    )

    assert result is None
    assert extract_called is False


@pytest.mark.asyncio
async def test_run_pipeline_saves_site(
    monkeypatch,
):
    async def fake_discover(
        latitude,
        longitude,
        radius_km,
    ):
        return [
            Quarry(
                site_id="site-1",
                name="Test Quarry",
                latitude=48.857,
                longitude=2.353,
                distance_km=1.0,
                website="https://example.com",
            )
        ]

    async def fake_crawl(url):
        return (
            "Test Quarry produces limestone."
        )

    async def fake_extract(
    document,
    cost_tracker=None,
):
        from app.models import (
            ExtractionResult,
            FieldValue,
            LocationVerification,
        )

        return ExtractionResult(
            official_name=FieldValue(
                value="Test Quarry",
                confidence=0.95,
            ),
            site_type=FieldValue(
                value="Quarry",
                confidence=0.95,
            ),
            description=FieldValue(
                value=(
                    "Test Quarry produces "
                    "limestone."
                ),
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
        )

    monkeypatch.setattr(
        "app.pipeline.discover_quarries",
        fake_discover,
    )

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    monkeypatch.setattr(
        "app.pipeline.extract",
        fake_extract,
    )

    result = await run_pipeline(
        latitude=48.8566,
        longitude=2.3522,
        radius_km=10,
    )

    assert result["count"] == 1

    assert (
        result["results"][0]["status"]
        == "extracted"
    )


@pytest.mark.asyncio
async def test_run_pipeline_discovers_missing_website(
    monkeypatch,
):
    async def fake_discover(
        latitude,
        longitude,
        radius_km,
    ):
        return [
            Quarry(
                site_id="site-ai-1",
                name="AI Test Quarry",
                latitude=48.857,
                longitude=2.353,
                distance_km=1.0,
                website=None,
            )
        ]

    async def fake_website_discovery(
    quarry_name,
    latitude,
    longitude,
    municipality=None,
    cost_tracker=None,
):
        assert quarry_name == "AI Test Quarry"

        return "https://example.com"

    async def fake_crawl(url):
        assert url == "https://example.com"

        return (
            "AI Test Quarry produces limestone."
        )

    async def fake_extract(
    document,
    cost_tracker=None,
):
        from app.models import (
            ExtractionResult,
            FieldValue,
            LocationVerification,
        )

        return ExtractionResult(
            official_name=FieldValue(
                value="AI Test Quarry",
                confidence=0.95,
            ),
            site_type=FieldValue(
                value="Quarry",
                confidence=0.95,
            ),
            description=FieldValue(
                value=(
                    "AI Test Quarry produces "
                    "limestone."
                ),
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
        )

    monkeypatch.setattr(
        "app.pipeline.discover_quarries",
        fake_discover,
    )

    monkeypatch.setattr(
        "app.pipeline.discover_website",
        fake_website_discovery,
    )

    monkeypatch.setattr(
        "app.pipeline.crawl",
        fake_crawl,
    )

    monkeypatch.setattr(
        "app.pipeline.extract",
        fake_extract,
    )

    result = await run_pipeline(
        latitude=48.8566,
        longitude=2.3522,
        radius_km=10,
    )

    assert result["count"] == 1

    assert (
        result["results"][0]["status"]
        == "extracted"
    )


@pytest.mark.asyncio
async def test_run_pipeline_when_website_discovery_fails(
    monkeypatch,
):
    async def fake_discover(
        latitude,
        longitude,
        radius_km,
    ):
        return [
            Quarry(
                site_id="site-no-web-1",
                name="Unknown Quarry",
                latitude=48.857,
                longitude=2.353,
                distance_km=1.0,
                website=None,
            )
        ]

    async def fake_website_discovery(
        quarry_name,
        latitude,
        longitude,
        municipality=None,
        cost_tracker=None,
    ):
        return None

    monkeypatch.setattr(
        "app.pipeline.discover_quarries",
        fake_discover,
    )

    monkeypatch.setattr(
        "app.pipeline.discover_website",
        fake_website_discovery,
    )

    result = await run_pipeline(
        latitude=48.8566,
        longitude=2.3522,
        radius_km=10,
    )

    assert result["count"] == 1

    assert (
        result["results"][0]["status"]
        == "no_website"
    )

    assert (
        result["results"][0]["extraction"]
        is None
    )

@pytest.mark.asyncio
async def test_run_pipeline_passes_cost_tracker_to_website_discovery(
    monkeypatch,
):
    captured = {}

    async def fake_discover(
        latitude,
        longitude,
        radius_km,
    ):
        return [
            Quarry(
                site_id="site-cost-1",
                name="Unknown Quarry",
                latitude=48.857,
                longitude=2.353,
                distance_km=1.0,
                website=None,
            )
        ]

    async def fake_website_discovery(
        quarry_name,
        latitude,
        longitude,
        municipality=None,
        cost_tracker=None,
    ):
        captured["cost_tracker"] = cost_tracker
        return None

    monkeypatch.setattr(
        "app.pipeline.discover_quarries",
        fake_discover,
    )

    monkeypatch.setattr(
        "app.pipeline.discover_website",
        fake_website_discovery,
    )

    result = await run_pipeline(
        latitude=48.8566,
        longitude=2.3522,
        radius_km=10,
        max_usd_cost=1.0,
    )

    assert result["count"] == 1
    assert result["results"][0]["status"] == "no_website"

    assert captured["cost_tracker"] is not None
    assert captured["cost_tracker"].max_usd_cost == 1.0