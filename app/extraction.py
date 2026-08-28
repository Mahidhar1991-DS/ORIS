import json
import os

from openai import AsyncOpenAI

from app.models import Document, Evidence, ExtractionResult, FieldValue


SYSTEM_PROMPT = """
You extract factual information from quarry documents.

Rules:
1. Only use information explicitly present in the document.
2. Never guess.
3. Every non-null value must have an exact supporting quote.
4. If a value cannot be established, return null.
5. Keep quotes short and copy them exactly.
6. Do not generate character offsets.

Return JSON with these fields:

{
  "name": {
    "value": "...",
    "confidence": 0.0,
    "quote": "..."
  },
  "material": {
    "value": "...",
    "confidence": 0.0,
    "quote": "..."
  },
  "operator": {
    "value": "...",
    "confidence": 0.0,
    "quote": "..."
  },
  "operational_status": {
    "value": "...",
    "confidence": 0.0,
    "quote": "..."
  }
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

    if value.lower() not in quote.lower():
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


async def extract(document: Document) -> ExtractionResult:
    """Extract quarry information using the LLM."""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return ExtractionResult(
            name=FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="llm_not_configured",
            ),
            material=FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="llm_not_configured",
            ),
            operator=FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="llm_not_configured",
            ),
            operational_status=FieldValue(
                value=None,
                confidence=0.0,
                abstain_reason="llm_not_configured",
            ),
        )

    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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

    return ExtractionResult(
        name=_field_from_response(
            document,
            data.get("name", {}),
        ),
        material=_field_from_response(
            document,
            data.get("material", {}),
        ),
        operator=_field_from_response(
            document,
            data.get("operator", {}),
        ),
        operational_status=_field_from_response(
            document,
            data.get("operational_status", {}),
        ),
    )