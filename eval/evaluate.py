import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_PATH = ROOT / "eval" / "ground_truth.json"


def normalize(value: Any) -> str | None:
    if value is None:
        return None

    return " ".join(
        str(value).lower().strip().split()
    )


def get_value(result: dict, field: str) -> Any:
    field_data = result.get(field)

    if isinstance(field_data, dict):
        return field_data.get("value")

    return field_data


def has_valid_evidence(field: dict) -> bool:
    value = field.get("value")

    if value is None:
        return True

    evidence = field.get("evidence", [])

    if not evidence:
        return False

    return all(
        evidence_item.get("quote")
        for evidence_item in evidence
    )


def evaluate_result(
    actual: dict,
    expected: dict,
) -> dict:

    field_scores = {}

    scalar_fields = [
        "official_name",
        "site_type",
        "description",
        "operational_status",
    ]

    for field in scalar_fields:
        actual_value = get_value(
            actual,
            field,
        )

        expected_value = expected.get(field)

        field_scores[field] = (
            normalize(actual_value)
            == normalize(expected_value)
        )

    expected_materials = {
        normalize(value)
        for value in expected.get(
            "materials_produced",
            [],
        )
    }

    actual_materials = {
        normalize(item.get("value"))
        for item in actual.get(
            "materials_produced",
            [],
        )
        if isinstance(item, dict)
        and item.get("value") is not None
    }

    field_scores["materials_produced"] = (
        expected_materials == actual_materials
    )

    expected_certifications = {
        normalize(value)
        for value in expected.get(
            "certifications",
            [],
        )
    }

    actual_certifications = {
        normalize(item.get("value"))
        for item in actual.get(
            "certifications",
            [],
        )
        if isinstance(item, dict)
        and item.get("value") is not None
    }

    field_scores["certifications"] = (
        expected_certifications
        == actual_certifications
    )

    expected_location = expected.get(
        "location_verification",
        {},
    )

    actual_location = actual.get(
        "location_verification",
        {},
    )

    field_scores["location_verified"] = (
        actual_location.get("is_verified")
        == expected_location.get("is_verified")
    )

    field_scores["location_city"] = (
        normalize(
            actual_location.get(
                "extracted_city"
            )
        )
        == normalize(
            expected_location.get(
                "extracted_city"
            )
        )
    )

    evidence_total = 0
    evidence_passed = 0

    for field in scalar_fields:
        actual_field = actual.get(field, {})

        if (
            isinstance(actual_field, dict)
            and actual_field.get("value") is not None
        ):
            evidence_total += 1

            if has_valid_evidence(actual_field):
                evidence_passed += 1

    for field in [
        "materials_produced",
        "certifications",
    ]:
        for item in actual.get(field, []):
            if (
                isinstance(item, dict)
                and item.get("value") is not None
            ):
                evidence_total += 1

                if has_valid_evidence(item):
                    evidence_passed += 1

    accuracy = (
        sum(field_scores.values())
        / len(field_scores)
    )

    evidence_score = (
        evidence_passed / evidence_total
        if evidence_total
        else 1.0
    )

    return {
        "field_scores": field_scores,
        "accuracy": round(accuracy, 3),
        "evidence_score": round(
            evidence_score,
            3,
        ),
        "evidence_passed": evidence_passed,
        "evidence_total": evidence_total,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python eval/evaluate.py "
            "<result.json>"
        )
        return 1

    result_path = Path(sys.argv[1])

    if not result_path.exists():
        print(
            f"Result file not found: "
            f"{result_path}"
        )
        return 1

    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        ground_truth = json.load(file)

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        actual = json.load(file)

    if not ground_truth:
        print("Ground truth is empty.")
        return 1

    expected = ground_truth[0]["expected"]

    evaluation = evaluate_result(
        actual,
        expected,
    )

    print("ORIS EVALUATION")
    print("=" * 50)

    print(
        f"Accuracy:       "
        f"{evaluation['accuracy']:.1%}"
    )

    print(
        f"Evidence score: "
        f"{evaluation['evidence_score']:.1%}"
    )

    print()

    for field, passed in evaluation[
        "field_scores"
    ].items():
        status = "PASS" if passed else "FAIL"

        print(
            f"{status:4}  {field}"
        )

    print()

    print(
        "Evidence:",
        f"{evaluation['evidence_passed']}/"
        f"{evaluation['evidence_total']}",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
