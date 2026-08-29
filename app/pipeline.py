import hashlib
from uuid import uuid4

from app.crawler import crawl, crawl_with_status
from app.database import save_site
from app.discovery import (
    deduplicate_quarries,
    discover_quarries,
)
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
    """
    Crawl one quarry website and extract supported facts.

    This function intentionally uses the original `crawl()`
    interface so existing tests and callers remain compatible.
    """

    text = await crawl(
        website
    )

    if not text:
        return None

    document = Document(
        source_id=site_id,
        url=website,
        text=text,
    )

    return await extract(
        document
    )


def build_run_metadata(
    text: str,
) -> RunMetadata:
    """
    Build metadata for one extraction run.
    """

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
) -> dict:
    """
    Run quarry discovery and enrichment.

    Pipeline:

    1. Discover quarries from OpenStreetMap.
    2. Deduplicate quarry records.
    3. Use OSM website when available.
    4. If no website exists, use AI website discovery.
    5. Pass quarry name, municipality, and coordinates
       to website discovery.
    6. Crawl the discovered website.
    7. Preserve the detailed crawler status.
    8. Extract grounded information.
    9. Persist the resulting SiteRecord.
    """

    # ---------------------------------------------------------
    # Step 1: Discover quarries.
    # ---------------------------------------------------------
    quarries = await discover_quarries(
        latitude,
        longitude,
        radius_km,
    )

    # ---------------------------------------------------------
    # Step 2: Deduplicate quarry records.
    # ---------------------------------------------------------
    quarries = deduplicate_quarries(
        quarries
    )

    # ---------------------------------------------------------
    # Request information stored with each SiteRecord.
    # ---------------------------------------------------------
    request = JobRequest(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
    )

    results = []

    # ---------------------------------------------------------
    # Process each quarry independently.
    # ---------------------------------------------------------
    for quarry in quarries:

        # -----------------------------------------------------
        # Step 3: Use website already provided by OSM.
        # -----------------------------------------------------
        website = quarry.website

        # -----------------------------------------------------
        # Step 4: If OSM has no website, use AI discovery.
        #
        # We pass:
        #   - quarry name
        #   - municipality
        #   - latitude
        #   - longitude
        #
        # This allows website discovery to work with both
        # named and unnamed OSM quarry records.
        # -----------------------------------------------------
        if not website:
            website = await discover_website(
                quarry_name=quarry.name,
                municipality=quarry.municipality,
                latitude=quarry.latitude,
                longitude=quarry.longitude,
            )

            if website:
                quarry.website = website

        # -----------------------------------------------------
        # Step 5: No website found.
        # -----------------------------------------------------
        if not website:
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": "no_website",
                }
            )

            continue

        # -----------------------------------------------------
        # Step 6: Crawl website with detailed status.
        # -----------------------------------------------------
        crawl_result = await crawl_with_status(
            website
        )

        # -----------------------------------------------------
        # Step 7: Preserve crawler failure reason.
        #
        # Examples:
        #   http_403
        #   http_404
        #   http_429
        #   robots_blocked
        #   network_error
        #   unsupported_content_type
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Protect against a successful status with no text.
        # -----------------------------------------------------
        if not text:
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": "empty_content",
                }
            )

            continue

        # -----------------------------------------------------
        # Step 8: Create source document.
        # -----------------------------------------------------
        document = Document(
            source_id=quarry.site_id,
            url=website,
            text=text,
        )

        # -----------------------------------------------------
        # Step 9: Extract grounded information.
        # -----------------------------------------------------
        extraction = await extract(
            document
        )

        # -----------------------------------------------------
        # Step 10: Create provenance source.
        # -----------------------------------------------------
        content_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        source = Source(
            source_id=quarry.site_id,
            url=website,
            source_type="web",
            content_hash=content_hash,
        )

        # -----------------------------------------------------
        # Step 11: Create SiteRecord.
        # -----------------------------------------------------
        site_record = SiteRecord(
            site_id=quarry.site_id,
            input=request,
            extraction=extraction,
            provenance=Provenance(
                sources=[source],
            ),
            metrics=Metrics(),
            run_metadata=build_run_metadata(
                text
            ),
        )

        # -----------------------------------------------------
        # Step 12: Persist SiteRecord.
        # -----------------------------------------------------
        save_site(
            site_record
        )

        # -----------------------------------------------------
        # Step 13: Add successful result.
        # -----------------------------------------------------
        results.append(
            {
                "quarry": quarry,
                "extraction": extraction,
                "status": "extracted",
            }
        )

    # ---------------------------------------------------------
    # Final pipeline result.
    # ---------------------------------------------------------
    return {
        "count": len(results),
        "results": results,
    }