import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.models import (
    Document,
    Evidence,
    ExtractionResult,
    FieldValue,
    LocationVerification,
)

load_dotenv(".env")

SYSTEM_PROMPT = """
You extract factual information from quarry documents.

Rules:
1. Only use information explicitly present in the document.
2. Never guess.
3. Every non-null value must have an exact supporting quote.
4. If a value cannot be established, return null.
5. Keep quotes short and copy them exactly.
6. Do not generate character offsets.
7. site_type must be "Quarry" when explicitly supported, otherwise null.
8. operational_status must be supported by recent evidence. If unclear, return null.
9. For arrays, return an empty array when no value is supported.
10. location_verification must use one of:
    "string_match", "geocode", "llm_inference", "none".

Return JSON with these fields:

{
  "official_name": {
    "value": null,
    "confidence": 0.0,
    "quote": null
  },
  "site_type": {
    "value": null,
    "confidence": 0.0,
    "quote": null
  },
  "description": {
    "value": null,
    "confidence": 0.0,
    "quote": null
  },
  "materials_produced": [],
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

Each array item must have:

{
  "value": "...",
  "confidence": 0.0,
  "quote": "..."
}
"""


def find_evidence(
    document: Document,
    quote: str,
) -> Evidence | None:
    """Find an exact quote in the original document."""

    if not quote:
        return None

    start = document.text.find(quote)

    if start == -1:
        return None

    return Evidence(
        source_id=document.source_id,
        char_start=start,
        char_end=start + len(quote),
        quote=quote,
    )


def create_field(
    document: Document,
    value: str | None,
    quote: str | None,
    confidence: float,
) -> FieldValue:
    """Create a field only when evidence supports the value."""

    if value is None:
        return FieldValue(
            value=None,
            confidence=0.0,
            abstain_reason="no_value_found",
        )

    if not quote:
        return FieldValue(
            value=None,
            confidence=0.0,
            abstain_reason="no_evidence",
        )

    evidence = find_evidence(document, quote)

    if evidence is None:
        return FieldValue(
            value=None,
            confidence=0.0,
            abstain_reason="evidence_not_found",
        )

    if str(value).lower() not in quote.lower():
        return FieldValue(
            value=None,
            confidence=0.0,
            abstain_reason="value_not_supported_by_evidence",
        )

    return FieldValue(
        value=value,
        confidence=confidence,
        evidence=[evidence],
    )


def _field_from_response(
    document: Document,
    data: dict,
) -> FieldValue:
    return create_field(
        document=document,
        value=data.get("value"),
        quote=data.get("quote"),
        confidence=float(data.get("confidence", 0)),
    )


def _list_from_response(
    document: Document,
    data: list,
) -> list[FieldValue]:
    return [
        _field_from_response(document, item)
        for item in data
        if isinstance(item, dict)
    ]


def _empty_field(reason: str) -> FieldValue:
    return FieldValue(
        value=None,
        confidence=0.0,
        abstain_reason=reason,
    )


async def extract(document: Document) -> ExtractionResult:
    """Extract quarry information using the LLM."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return ExtractionResult(
            official_name=_empty_field("llm_not_configured"),
            site_type=_empty_field("llm_not_configured"),
            description=_empty_field("llm_not_configured"),
            materials_produced=[],
            certifications=[],
            operational_status=_empty_field("llm_not_configured"),
            location_verification=LocationVerification(
                is_verified=False,
                confidence=0.0,
                extracted_city=None,
                method="none",
            ),
        )

    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": document.text[:12000],
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned empty response")

    data = json.loads(content)

    location_data = data.get(
        "location_verification",
        {},
    )

    return ExtractionResult(
        official_name=_field_from_response(
            document,
            data.get("official_name", {}),
        ),
        site_type=_field_from_response(
            document,
            data.get("site_type", {}),
        ),
        description=_field_from_response(
            document,
            data.get("description", {}),
        ),
        materials_produced=_list_from_response(
            document,
            data.get("materials_produced", []),
        ),
        certifications=_list_from_response(
            document,
            data.get("certifications", []),
        ),
        operational_status=_field_from_response(
            document,
            data.get("operational_status", {}),
        ),
        location_verification=LocationVerification(
            is_verified=bool(
                location_data.get("is_verified", False)
            ),
            confidence=float(
                location_data.get("confidence", 0)
            ),
            extracted_city=location_data.get(
                "extracted_city"
            ),
            method=location_data.get(
                "method",
                "none",
            ),
        ),
    )