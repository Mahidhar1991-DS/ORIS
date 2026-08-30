import hashlib
import unicodedata
from uuid import uuid4

from app.llm_cost import CostTracker
from app.crawler import crawl, crawl_with_status
from app.database import save_site
from app.discovery import deduplicate_quarries, discover_quarries
from app.extraction import extract
from app.models import (
    Document,
    ExtractionResult,
    JobRequest,
    Metrics,
    Provenance,
    RunMetadata,
    SiteRecord,
    Source,
)
from app.website_discovery import discover_website


SCRAPER_VERSION = "1.0.0"


async def process_quarry(
    site_id: str,
    website: str,
) -> ExtractionResult | None:
    """Crawl and extract one quarry website."""

    text = await crawl(website)

    if not text:
        return None

    document = Document(
        source_id=site_id,
        url=website,
        text=text,
    )

    return await extract(document)


def verify_extracted_location(
    extraction: ExtractionResult,
    quarry_municipality: str | None,
) -> ExtractionResult:
    """Verify the extracted city against the quarry municipality."""

    if not quarry_municipality:
        return extraction

    extracted_city = extraction.location_verification.extracted_city

    if not extracted_city:
        return extraction

    def normalize(value: str) -> str:
        value = unicodedata.normalize("NFKD", value)

        value = "".join(
            character
            for character in value
            if not unicodedata.combining(character)
        )

        return " ".join(
            value.lower()
            .replace("-", " ")
            .replace("'", " ")
            .split()
        )

    expected = normalize(quarry_municipality)
    actual = normalize(extracted_city)

    if expected == actual:
        extraction.location_verification.is_verified = True
        extraction.location_verification.confidence = 1.0
        extraction.location_verification.method = "string_match"
        return extraction

    extraction.location_verification.is_verified = False
    extraction.location_verification.confidence = 0.0
    extraction.location_verification.method = "none"

    return extraction


def build_run_metadata(text: str) -> RunMetadata:
    """Build metadata for an extraction run."""

    prompt_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return RunMetadata(
        run_id=str(uuid4()),
        prompt_hash=prompt_hash,
        scraper_version=SCRAPER_VERSION,
    )


async def run_pipeline(
    latitude: float,
    longitude: float,
    radius_km: float,
    max_usd_cost: float | None = None,
) -> dict:
    """Run quarry discovery and enrichment."""

    cost_tracker = CostTracker(
        max_usd_cost=max_usd_cost,
    )

    quarries = await discover_quarries(
        latitude,
        longitude,
        radius_km,
    )

    quarries = deduplicate_quarries(quarries)

    request = JobRequest(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        max_usd_cost=max_usd_cost,
    )

    results = []

    for quarry in quarries:
        website = quarry.website

        if not website:
            website = await discover_website(
                quarry_name=quarry.name,
                municipality=quarry.municipality,
                latitude=quarry.latitude,
                longitude=quarry.longitude,
                cost_tracker=cost_tracker,
            )

            if website:
                quarry.website = website

        if not website:
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": "no_website",
                }
            )
            continue

        crawl_result = await crawl_with_status(website)

        if crawl_result.status != "success":
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": crawl_result.status,
                }
            )
            continue

        text = crawl_result.text

        if not text:
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": "empty_content",
                }
            )
            continue

        document = Document(
            source_id=quarry.site_id,
            url=website,
            text=text,
        )

        extraction = await extract(
            document,
            cost_tracker=cost_tracker,
        )

        extraction = verify_extracted_location(
            extraction=extraction,
            quarry_municipality=quarry.municipality,
        )

        content_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        source = Source(
            source_id=quarry.site_id,
            url=website,
            source_type="web",
            content_hash=content_hash,
        )

        site_record = SiteRecord(
            site_id=quarry.site_id,
            input=request,
            extraction=extraction,
            provenance=Provenance(
                sources=[source],
            ),
            metrics=Metrics(
                llm_tokens_in=cost_tracker.tokens_in,
                llm_tokens_out=cost_tracker.tokens_out,
                usd_cost=cost_tracker.usd_cost,
                model_calls=cost_tracker.model_calls,
            ),
            run_metadata=build_run_metadata(text),
        )

        save_site(site_record)

        results.append(
            {
                "quarry": quarry,
                "extraction": extraction,
                "status": "extracted",
            }
        )

    return {
        "count": len(results),
        "results": results,
    }