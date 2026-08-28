import pytest

from app.utils import (
    haversine_distance_km,
    normalize_name,
    stable_id,
    validate_coordinates,
)


def test_valid_coordinates():
    validate_coordinates(48.8566, 2.3522)


def test_invalid_latitude():
    with pytest.raises(ValueError):
        validate_coordinates(100, 2.3522)


def test_invalid_longitude():
    with pytest.raises(ValueError):
        validate_coordinates(48.8566, 200)


def test_haversine_same_point():
    distance = haversine_distance_km(
        48.8566,
        2.3522,
        48.8566,
        2.3522,
    )

    assert distance == pytest.approx(0, abs=0.001)


def test_haversine_distance():
    distance = haversine_distance_km(
        48.8566,
        2.3522,
        48.8576,
        2.3522,
    )

    assert 0.05 < distance < 0.2


def test_normalize_name():
    assert normalize_name("  Quarry  ABC! ") == "quarry abc"


def test_stable_id():
    first = stable_id("quarry-abc")
    second = stable_id("quarry-abc")

    assert first == second
    assert len(first) == 16