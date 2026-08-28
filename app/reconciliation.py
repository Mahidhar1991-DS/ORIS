from app.models import FieldValue


def reconcile(values: list[FieldValue]) -> FieldValue:
    """Select the best supported value from multiple sources."""

    valid = [
        value
        for value in values
        if value.value is not None and value.evidence
    ]

    if not valid:
        return FieldValue(
            value=None,
            confidence=0.0,
            abstain_reason="no_supported_value",
        )

    return max(
        valid,
        key=lambda value: value.confidence,
    )