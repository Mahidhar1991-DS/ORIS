import json
import logging
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv(".env")


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You identify the official website of a quarry or quarry operator.

Your job is entity resolution, not website guessing.

Rules:

1. Only return a website when there is strong evidence that it belongs
   to the identified quarry or its operator.

2. Prefer the quarry operator's official website.

3. Do not return:
   - directories
   - social media
   - review websites
   - Google Maps
   - map websites
   - search-result pages
   - Wikipedia
   - generic business databases

4. Never guess a website.

5. Coordinates are approximate supporting evidence.
   They are NOT sufficient by themselves to establish identity.

6. When the quarry name is UNKNOWN, first use web search to identify:
   - the likely municipality/location
   - the quarry name
   - the operator/company
   - the official website

7. For French locations, search using French terminology such as:
   - carrière
   - carrière de granulats
   - carrière de sable
   - carrière de pierre
   - carrière de gypse
   - exploitation de carrière
   - site d'extraction

8. Search using multiple combinations when the name is unknown:
   - exact coordinates
   - latitude and longitude
   - nearby municipality
   - "carrière" + municipality
   - "carrière" + coordinates
   - quarry + municipality
   - quarry operator + municipality

9. A website should only be accepted when there is a reasonable
   connection between:
   - quarry/site identity
   - operator
   - geographic location
   - official website

10. When a municipality is provided, use it as geographic evidence.
    The municipality was resolved from the administrative boundary
    containing the quarry coordinates.

11. If search results identify only a nearby quarry but do not establish
    that it corresponds to the supplied coordinates, return null.

12. Do not infer that a company operates the quarry merely because
    the company operates somewhere in the same region.

13. If the official website cannot be established confidently,
    return null.

14. Return valid JSON only.

Return exactly:

{
  "website": null,
  "confidence": 0.0,
  "reason": "..."
}

or:

{
  "website": "https://example.com",
  "confidence": 0.0,
  "reason": "..."
}
"""


# Domains that should never be accepted as the official quarry website.
BLOCKED_DOMAINS = {
    "google.com",
    "www.google.com",
    "google.fr",
    "www.google.fr",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "linkedin.com",
    "www.linkedin.com",
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "youtube.com",
    "www.youtube.com",
    "wikipedia.org",
    "www.wikipedia.org",
    "yelp.com",
    "www.yelp.com",
    "tripadvisor.com",
    "www.tripadvisor.com",
}


def normalize_website(
    url: str | None,
) -> str | None:
    """
    Normalize and validate a candidate website.

    This function performs application-level validation in addition
    to the AI's website selection rules.
    """

    if not url:
        return None

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url:
        return None

    # Remove common wrapping characters that an LLM might return.
    url = url.strip(
        "\"'<>"
    )

    if not url:
        return None

    if not url.startswith(
        ("http://", "https://")
    ):
        url = f"https://{url}"

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        return None

    hostname = parsed.hostname

    if not hostname:
        return None

    hostname = hostname.lower().rstrip(".")

    # Reject obvious non-official destinations.
    if hostname in BLOCKED_DOMAINS:
        return None

    return url


async def discover_website(
    quarry_name: str | None,
    latitude: float,
    longitude: float,
    municipality: str | None = None,
) -> str | None:
    """
    Find a likely official website for a quarry.

    The quarry name may be missing. In that case,
    coordinate and municipality-based web discovery
    is used.

    The function deliberately returns None when identity
    cannot be established with sufficient confidence.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY is not configured; "
            "website discovery disabled."
        )
        return None

    # ---------------------------------------------------------
    # Quarry identity context.
    # ---------------------------------------------------------
    if quarry_name:
        quarry_context = f"""
Known quarry name:
{quarry_name}

The quarry name comes from OpenStreetMap.

Use it as the primary identity clue, but still verify
the geographic relationship.
"""
    else:
        quarry_context = """
Known quarry name:
UNKNOWN

OpenStreetMap did not provide a quarry name.

You must first identify the likely quarry/site
and its operator from web search.

Do not infer a quarry name simply because a company
operates somewhere in the same region.
"""

    # ---------------------------------------------------------
    # Municipality context.
    # ---------------------------------------------------------
    if municipality:
        municipality_context = f"""
Known municipality:
{municipality}

This municipality was resolved from the OpenStreetMap
administrative boundary containing the quarry coordinates.

Treat this as strong geographic evidence.

The municipality is useful for narrowing the search,
but it does not by itself prove the identity of the quarry.
"""
    else:
        municipality_context = """
Known municipality:
UNKNOWN

Use the supplied coordinates to determine the likely
geographic context during web search.
"""

    # ---------------------------------------------------------
    # AI discovery prompt.
    # ---------------------------------------------------------
    prompt = f"""
Identify the official website of the quarry represented
by the following OpenStreetMap location.

{quarry_context}

{municipality_context}

Approximate coordinates:
latitude={latitude}
longitude={longitude}

IMPORTANT:
The coordinates are approximate.

Use web search systematically.

First determine the geographic context around the
coordinates.

Then search for quarry-related sites in that area.

For a French location, prioritize searches involving:

"carrière"
"carrière de granulats"
"carrière de sable"
"carrière de pierre"
"carrière de gypse"
"exploitation de carrière"
"site d'extraction"

Also search using combinations of:

- quarry name
- municipality
- operator
- coordinates
- latitude and longitude
- French quarry terminology
- quarry operator + municipality
- carrière + municipality

For every candidate website, verify:

1. Does the website belong to the quarry operator?
2. Does the operator operate the identified quarry?
3. Is the quarry/site geographically consistent
   with the supplied coordinates?
4. Is the website an official operator/company website?
5. Is there evidence connecting the quarry to the
   municipality?

Only after those checks should you return a website.

Do NOT return:

- directories
- maps
- Google Maps
- social networks
- review sites
- Wikipedia
- search-result pages
- generic business databases

If there is insufficient evidence to connect the
candidate quarry to the supplied coordinates,
return null.

If multiple candidate quarries are found and none can
be reliably matched to the supplied coordinates,
return null.

Return only this JSON:

{{
  "website": null,
  "confidence": 0.0,
  "reason": "..."
}}

or:

{{
  "website": "https://example.com",
  "confidence": 0.0,
  "reason": "..."
}}
"""

    logger.info(
        "Starting website discovery: "
        "quarry=%r municipality=%r latitude=%s longitude=%s",
        quarry_name,
        municipality,
        latitude,
        longitude,
    )

    # ---------------------------------------------------------
    # Create OpenAI client.
    # ---------------------------------------------------------
    client = AsyncOpenAI(
        api_key=api_key
    )

    # ---------------------------------------------------------
    # Call OpenAI web search.
    # ---------------------------------------------------------
    try:
        response = await client.responses.create(
            model=os.getenv(
                "OPENAI_MODEL",
                "gpt-5.4-mini",
            ),
            tools=[
                {
                    "type": "web_search",
                }
            ],
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    except Exception as exc:
        logger.exception(
            "Website discovery failed for "
            "quarry=%r municipality=%r coordinates=(%s,%s): %s",
            quarry_name,
            municipality,
            latitude,
            longitude,
            exc,
        )
        return None

    # ---------------------------------------------------------
    # Read model output.
    # ---------------------------------------------------------
    text = response.output_text

    if not text:
        logger.warning(
            "Website discovery returned empty output: "
            "quarry=%r municipality=%r",
            quarry_name,
            municipality,
        )
        return None

    text = text.strip()

    # ---------------------------------------------------------
    # Parse JSON.
    # ---------------------------------------------------------
    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        logger.warning(
            "Website discovery returned invalid JSON: %r",
            text[:500],
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "Website discovery returned non-object JSON: %r",
            data,
        )
        return None

    # ---------------------------------------------------------
    # Extract candidate website.
    # ---------------------------------------------------------
    website = normalize_website(
        data.get("website")
    )

    if not website:
        logger.info(
            "No reliable website discovered: "
            "quarry=%r municipality=%r reason=%r",
            quarry_name,
            municipality,
            data.get("reason"),
        )
        return None

    # ---------------------------------------------------------
    # Validate confidence.
    # ---------------------------------------------------------
    try:
        confidence = float(
            data.get(
                "confidence",
                0,
            )
        )

    except (TypeError, ValueError):
        confidence = 0.0

    # Reject invalid confidence values.
    if not 0.0 <= confidence <= 1.0:
        logger.warning(
            "Invalid confidence returned by website discovery: %r",
            confidence,
        )
        return None

    # ---------------------------------------------------------
    # Conservative acceptance threshold.
    # ---------------------------------------------------------
    if confidence < 0.80:
        logger.info(
            "Website rejected because confidence is too low: "
            "website=%s confidence=%s reason=%r",
            website,
            confidence,
            data.get("reason"),
        )
        return None

    logger.info(
        "Website discovered successfully: "
        "website=%s confidence=%s quarry=%r municipality=%r",
        website,
        confidence,
        quarry_name,
        municipality,
    )

    return website