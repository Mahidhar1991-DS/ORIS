import httpx

from app.models import Quarry
from app.utils import haversine_distance_km, stable_id


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


async def discover_quarries(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[Quarry]:
    radius_m = radius_km * 1000

    query = f"""
    [out:json];
    (
      nwr["landuse"="quarry"](around:{radius_m},{latitude},{longitude});
    );
    out center;
    """

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            OVERPASS_URL,
            data={"data": query},
        )
        response.raise_for_status()
        data = response.json()

    quarries = []

    for element in data.get("elements", []):
        tags = element.get("tags", {})

        if element["type"] == "node":
            lat = element["lat"]
            lon = element["lon"]
        else:
            center = element.get("center")

            if not center:
                continue

            lat = center["lat"]
            lon = center["lon"]

        distance = haversine_distance_km(
            latitude,
            longitude,
            lat,
            lon,
        )

        if distance > radius_km:
            continue

        site_id = stable_id(
            f"{element['type']}:{element['id']}"
        )

        quarries.append(
            Quarry(
                site_id=site_id,
                name=tags.get("name"),
                latitude=lat,
                longitude=lon,
                distance_km=round(distance, 3),
                website=tags.get("website") or tags.get("contact:website"),
            )
        )

    return quarries


def deduplicate_quarries(
    quarries: list[Quarry],
) -> list[Quarry]:
    unique = {}

    for quarry in quarries:
        key = (
            quarry.name.lower().strip()
            if quarry.name
            else f"{quarry.latitude:.5f}:{quarry.longitude:.5f}"
        )

        if key not in unique:
            unique[key] = quarry

    return list(unique.values())


def get_website(tags: dict) -> str | None:
    """Return a website from OSM tags if available."""
    website = tags.get("website")

    if website:
        return website

    return tags.get("contact:website")