import fitz
import pytest

from app.crawler import html_to_text, pdf_to_text, validate_url
from app.discovery import deduplicate_quarries, discover_quarries
from app.models import Quarry
from app.extraction import create_field, find_evidence
from app.models import Document
from app.extraction import extract
from app.models import Document
from app.reconciliation import reconcile
from app.models import Evidence, FieldValue
from app.pipeline import process_quarry


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeClient:
    response_data = {"elements": []}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, *args, **kwargs):
        return FakeResponse(self.response_data)


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

    result = await discover_quarries(48.8566, 2.3522, 50)

    assert len(result) == 1
    assert result[0].name == "Test Quarry"
    assert result[0].website == "https://example.com"


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

    result = await discover_quarries(48.8566, 2.3522, 50)

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

    result = await discover_quarries(48.8566, 2.3522, 50)

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

    result = await discover_quarries(48.8566, 2.3522, 10)

    assert result == []


@pytest.mark.asyncio
async def test_no_quarries(monkeypatch):
    FakeClient.response_data = {
        "elements": [],
    }

    monkeypatch.setattr(
        "app.discovery.httpx.AsyncClient",
        FakeClient,
    )

    result = await discover_quarries(48.8566, 2.3522, 50)

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

    result = deduplicate_quarries(quarries)

    assert len(result) == 2
    assert result[0].name == "Test Quarry"
    assert result[1].name == "Another Quarry"


def test_validate_url():
    assert validate_url("https://example.com")
    assert validate_url("http://example.com")
    assert not validate_url("ftp://example.com")
    assert not validate_url("example.com")  
def test_validate_url_rejects_private_ip():
    assert not validate_url("http://127.0.0.1")
    assert not validate_url("http://192.168.1.1")
    assert not validate_url("http://10.0.0.1")

def test_html_to_text():
    html = """
    <html>
        <head>
            <style>.hidden { display: none; }</style>
        </head>
        <body>
            <h1>Test Quarry</h1>
            <p>Limestone producer</p>
            <script>alert("remove me")</script>
        </body>
    </html>
    """

    text = html_to_text(html)

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

    text = pdf_to_text(pdf_bytes)

    assert "Test Quarry" in text
    assert "limestone" in text
    
def test_find_evidence():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text="Our quarry produces limestone aggregates.",
    )

    evidence = find_evidence(
        document,
        "limestone aggregates",
    )

    assert evidence is not None
    assert evidence.source_id == "source-1"
    assert evidence.char_start == 20
    assert evidence.char_end == 40
    assert evidence.quote == "limestone aggregates"


def test_find_evidence_not_found():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text="Our quarry produces limestone.",
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
        text="Our quarry produces limestone aggregates.",
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
        text="Our quarry produces limestone.",
    )

    field = create_field(
        document=document,
        value="Granite",
        quote="granite",
        confidence=0.9,
    )

    assert field.value is None
    assert field.abstain_reason == "evidence_not_found"

@pytest.mark.asyncio
async def test_extract_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    document = Document(
        source_id="source-1",
        url="https://example.com",
        text="Test Quarry produces limestone.",
    )

    result = await extract(document)

    assert result.name.value is None
    assert result.name.abstain_reason == "llm_not_configured"
    assert result.material.value is None
    assert result.operator.value is None
    assert result.operational_status.value is None

@pytest.mark.asyncio
async def test_extract_with_mock_llm(monkeypatch):
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text=(
            "Test Quarry is operated by ABC Materials. "
            "The quarry produces limestone."
        ),
    )

    class FakeMessage:
        content = """
        {
            "name": {
                "value": "Test Quarry",
                "confidence": 0.95,
                "quote": "Test Quarry"
            },
            "material": {
                "value": "limestone",
                "confidence": 0.90,
                "quote": "The quarry produces limestone."
            },
            "operator": {
                "value": "ABC Materials",
                "confidence": 0.95,
                "quote": "operated by ABC Materials"
            },
            "operational_status": {
                "value": null,
                "confidence": 0.0,
                "quote": null
            }
        }
        """

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type(
                "Chat",
                (),
                {"completions": FakeCompletions()},
            )()

    monkeypatch.setattr(
        "app.extraction.AsyncOpenAI",
        FakeClient,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    result = await extract(document)

    assert result.name.value == "Test Quarry"
    assert result.name.confidence == 0.95
    assert len(result.name.evidence) == 1

    assert result.material.value == "limestone"
    assert len(result.material.evidence) == 1

    assert result.operator.value == "ABC Materials"
    assert len(result.operator.evidence) == 1

    assert result.operational_status.value is None
    
    

def test_create_field_rejects_unsupported_value():
    document = Document(
        source_id="source-1",
        url="https://example.com",
        text="Our quarry produces limestone.",
    )

    field = create_field(
        document=document,
        value="Granite",
        quote="Our quarry produces limestone.",
        confidence=0.95,
    )

    assert field.value is None
    assert field.abstain_reason == "value_not_supported_by_evidence"
    
    
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

    result = reconcile([first, second])

    assert result.value == "Limestone"
    assert result.confidence == 0.95
    assert result.evidence[0].source_id == "source-2"


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
    assert result.abstain_reason == "no_supported_value"

@pytest.mark.asyncio
async def test_process_quarry(monkeypatch):
    async def fake_crawl(url):
        return "Test Quarry produces limestone."

    async def fake_extract(document):
        from app.models import Evidence, ExtractionResult, FieldValue

        evidence = Evidence(
            source_id=document.source_id,
            char_start=0,
            char_end=11,
            quote="Test Quarry",
        )

        return ExtractionResult(
            name=FieldValue(
                value="Test Quarry",
                confidence=0.95,
                evidence=[evidence],
            ),
            material=FieldValue(
                value=None,
                confidence=0,
                abstain_reason="not_found",
            ),
            operator=FieldValue(
                value=None,
                confidence=0,
                abstain_reason="not_found",
            ),
            operational_status=FieldValue(
                value=None,
                confidence=0,
                abstain_reason="not_found",
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

    assert result.name.value == "Test Quarry"
    assert result.name.confidence == 0.95
    assert len(result.name.evidence) == 1

@pytest.mark.asyncio
async def test_process_quarry_when_crawl_fails(monkeypatch):
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