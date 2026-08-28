import hashlib
import math
import re


def validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split())


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]