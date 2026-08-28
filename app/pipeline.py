from app.crawler import crawl
from app.discovery import deduplicate_quarries, discover_quarries
from app.extraction import extract
from app.models import Document, ExtractionResult


async def process_quarry(
    site_id: str,
    website: str,
) -> ExtractionResult | None:
    """Crawl one quarry website and extract supported facts."""

    text = await crawl(website)

    if not text:
        return None

    document = Document(
        source_id=site_id,
        url=website,
        text=text,
    )

    return await extract(document)


async def run_pipeline(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> dict:
    """Run quarry discovery and enrichment."""

    quarries = await discover_quarries(
        latitude,
        longitude,
        radius_km,
    )

    quarries = deduplicate_quarries(quarries)

    results = []

    for quarry in quarries:
        if not quarry.website:
            results.append(
                {
                    "quarry": quarry,
                    "extraction": None,
                    "status": "no_website",
                }
            )
            continue

        extraction = await process_quarry(
            quarry.site_id,
            quarry.website,
        )

        results.append(
            {
                "quarry": quarry,
                "extraction": extraction,
                "status": (
                    "extracted"
                    if extraction
                    else "crawl_failed"
                ),
            }
        )

    return {
        "count": len(results),
        "results": results,
    }