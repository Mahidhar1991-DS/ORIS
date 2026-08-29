import json
import logging
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.llm_cost import CostTracker, calculate_cost, create_model_call

load_dotenv(".env")


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = SYSTEM_PROMPT = """
Identify the official website of the quarry or quarry operator.

This is entity resolution, not website guessing.

Rules:
1. Return a website only when strong evidence connects it to the quarry/site and operator.
2. Prefer the official operator/company website.
3. Never guess.
4. Coordinates are supporting evidence only, not sufficient proof.
5. Use the municipality as geographic evidence when provided.
6. If the quarry name is UNKNOWN, identify the quarry, operator, location, and official website from web search.
7. For French locations, use relevant terms such as "carrière", "carrière de granulats", "site d'extraction", and "exploitation de carrière".
8. Reject directories, maps, Google Maps, social media, reviews, Wikipedia, search-result pages, and generic business databases.
9. Do not assume an operator is responsible merely because it operates elsewhere in the region.
10. If the candidate cannot be confidently connected to the supplied quarry/location, return null.
11. Return JSON only with a short reason.

Output:
{
  "website": "https://example.com" or null,
  "confidence": 0.0,
  "reason": "Short explanation."
}
"""


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
    """

    if not url:
        return None

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url:
        return None

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

    if hostname in BLOCKED_DOMAINS:
        return None

    return url


def _parse_website_response(
    text: str,
) -> dict | None:
    """
    Parse the model response safely.

    Handles common cases such as:
    - normal JSON
    - JSON wrapped in markdown fences
    - JSON with surrounding whitespace
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences if the model adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    return data


async def _request_website_discovery(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    cost_tracker: CostTracker | None = None,
    purpose: str = "website_discovery",
) -> str | None:
    """
    Execute one website discovery request.

    Uses structured JSON output so that the model is strongly
    constrained to return the expected object.

    When a CostTracker is supplied, actual token usage and
    calculated USD cost are recorded after the request.
    """

    # ---------------------------------------------------------
    # Check whether another LLM call is allowed.
    #
    # We cannot know the exact cost before the API request,
    # because the final token usage is returned by OpenAI.
    # Therefore the tracker prevents calls after the already
    # accumulated budget has been exhausted.
    # ---------------------------------------------------------
    if cost_tracker is not None:
        estimated_input_tokens = (
            len(SYSTEM_PROMPT) + len(prompt)
        ) // 4

        estimated_output_tokens = 300

        estimated_cost = calculate_cost(
            model=model,
            tokens_in=estimated_input_tokens,
            tokens_out=estimated_output_tokens,
        )

        if not cost_tracker.can_call(
            estimated_cost=estimated_cost
        ):
            logger.warning(
                "Website discovery skipped because "
                "estimated cost would exceed budget: "
                "used=$%.8f estimated=$%.8f remaining=$%s",
                cost_tracker.usd_cost,
                estimated_cost,
                cost_tracker.remaining_usd,
            )
            return None

    try:
        response = await client.responses.create(
            model=model,
            tools=[
                {
                    "type": "web_search",
                }
            ],
            max_output_tokens=150,
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
            text={
                "format": {
                    "type": "json_schema",
                    "name": "quarry_website_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "website": {
                                "type": [
                                    "string",
                                    "null",
                                ]
                            },
                            "confidence": {
                                "type": "number",
                            },
                            "reason": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "website",
                            "confidence",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )

    except Exception as exc:
        logger.exception(
            "Website discovery request failed: %s",
            exc,
        )
        return None

    # ---------------------------------------------------------
    # Record actual token usage.
    # ---------------------------------------------------------
    if cost_tracker is not None:
        usage = getattr(
            response,
            "usage",
            None,
        )

        tokens_in = int(
            getattr(
                usage,
                "input_tokens",
                0,
            )
            or 0
        )

        tokens_out = int(
            getattr(
                usage,
                "output_tokens",
                0,
            )
            or 0
        )

        model_call = create_model_call(
            model=model,
            purpose=purpose,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        cost_tracker.record(
            model_call
        )

        logger.info(
            "Website discovery usage: "
            "purpose=%s input_tokens=%d output_tokens=%d "
            "cost=$%.8f total=$%.8f",
            purpose,
            tokens_in,
            tokens_out,
            model_call.usd_cost,
            cost_tracker.usd_cost,
        )

    text = response.output_text

    if not text:
        logger.warning(
            "Website discovery returned empty output."
        )
        return None

    return text


async def discover_website(
    quarry_name: str | None,
    latitude: float,
    longitude: float,
    municipality: str | None = None,
    cost_tracker: CostTracker | None = None,
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

Keep the reason short.

Return only the required JSON object.
"""

    logger.info(
        "Starting website discovery: "
        "quarry=%r municipality=%r latitude=%s longitude=%s",
        quarry_name,
        municipality,
        latitude,
        longitude,
    )

    client = AsyncOpenAI(
        api_key=api_key
    )

    model = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.4-mini",
    )

    # ---------------------------------------------------------
    # First attempt.
    # ---------------------------------------------------------
    text = await _request_website_discovery(
    client,
    model,
    prompt,
    cost_tracker=cost_tracker,
    purpose="website_discovery",
)

    if not text:
        return None

    data = _parse_website_response(
        text
    )

    # ---------------------------------------------------------
    # Retry once if the response is malformed.
    # ---------------------------------------------------------
    if data is None:
        logger.warning(
            "Website discovery returned invalid JSON. "
            "Retrying with a shorter response requirement."
        )

        retry_prompt = f"""
Identify the official quarry operator website.

Quarry name:
{quarry_name or "UNKNOWN"}

Municipality:
{municipality or "UNKNOWN"}

Coordinates:
{latitude}, {longitude}

Use web search.

Only return a website if you can establish a credible
connection between the quarry, its operator, the location,
and the official company website.

Never guess.

Return ONLY this JSON object:

{{
  "website": null,
  "confidence": 0.0,
  "reason": "short reason"
}}

The reason must be fewer than 20 words.
"""

        text = await _request_website_discovery(
            client,
            model,
            retry_prompt,
            cost_tracker=cost_tracker,
            purpose="website_discovery_retry",
        )

        if not text:
            return None

        data = _parse_website_response(
            text
        )

    if data is None:
        logger.warning(
            "Website discovery returned invalid JSON after retry: %r",
            text[:500] if text else "",
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