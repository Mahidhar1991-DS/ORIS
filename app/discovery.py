import httpx

from app.models import Quarry
from app.utils import haversine_distance_km, stable_id


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

USER_AGENT = "ORIS-QuarryResearch/1.0"

CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 45.0
WRITE_TIMEOUT = 45.0
POOL_TIMEOUT = 10.0


def _build_quarry_query(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> str:
    """Build the Overpass quarry discovery query."""

    radius_m = radius_km * 1000

    return f"""
    [out:json][timeout:30];

    (
      node["landuse"="quarry"](
        around:{radius_m},{latitude},{longitude}
      );

      way["landuse"="quarry"](
        around:{radius_m},{latitude},{longitude}
      );

      relation["landuse"="quarry"](
        around:{radius_m},{latitude},{longitude}
      );
    );

    out center;
    """


async def _query_overpass(
    client: httpx.AsyncClient,
    endpoint: str,
    query: str,
) -> dict:
    """
    Execute one Overpass request.

    HTTP/network errors are allowed to propagate so the caller
    can try the next provider.
    """

    response = await client.post(
        endpoint,
        data={"data": query},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": (
                "application/x-www-form-urlencoded"
            ),
        },
    )

    response.raise_for_status()

    return response.json()


def _parse_quarries(
    data: dict,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[Quarry]:
    """Convert an Overpass response into Quarry objects."""

    quarries: list[Quarry] = []

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        element_type = element.get("type")

        if element_type == "node":
            lat = element.get("lat")
            lon = element.get("lon")

            if lat is None or lon is None:
                continue

        elif element_type in {"way", "relation"}:
            center = element.get("center")

            if not center:
                continue

            lat = center.get("lat")
            lon = center.get("lon")

            if lat is None or lon is None:
                continue

        else:
            continue

        distance = haversine_distance_km(
            latitude,
            longitude,
            lat,
            lon,
        )

        if distance > radius_km:
            continue

        element_id = element.get("id")

        if element_id is None:
            continue

        site_id = stable_id(
            f"{element_type}:{element_id}"
        )

        website = (
            tags.get("website")
            or tags.get("contact:website")
        )

        quarries.append(
            Quarry(
                site_id=site_id,
                name=tags.get("name"),
                municipality=None,
                latitude=lat,
                longitude=lon,
                distance_km=round(distance, 3),
                website=website,
            )
        )

    return quarries


async def discover_quarries(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[Quarry]:
    """
    Discover quarry sites around a coordinate.

    The function tries each configured Overpass provider
    independently.

    Important distinction:

    - Successful response with zero elements means a valid
      empty discovery result.
    - Network/HTTP/provider failure means the provider is
      unavailable and the next provider is tried.

    If every provider fails, RuntimeError is raised.
    """

    query = _build_quarry_query(
        latitude,
        longitude,
        radius_km,
    )

    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    )

    last_error: Exception | None = None

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    ) as client:

        for endpoint in OVERPASS_ENDPOINTS:
            print(
                f"QUERYING OVERPASS: {endpoint}",
                flush=True,
            )

            try:
                data = await _query_overpass(
                    client,
                    endpoint,
                    query,
                )

                quarries = _parse_quarries(
                    data,
                    latitude,
                    longitude,
                    radius_km,
                )

                print(
                    f"OVERPASS SUCCESS: {endpoint}",
                    flush=True,
                )

                print(
                    f"QUARRIES DISCOVERED: {len(quarries)}",
                    flush=True,
                )

                return quarries

            except httpx.HTTPStatusError as exc:
                last_error = exc

                print(
                    f"OVERPASS HTTP ERROR: "
                    f"{endpoint}: "
                    f"{exc.response.status_code}",
                    flush=True,
                )

            except httpx.RequestError as exc:
                last_error = exc

                print(
                    f"OVERPASS CONNECTION ERROR: "
                    f"{endpoint}: {exc}",
                    flush=True,
                )

            except (ValueError, TypeError) as exc:
                last_error = exc

                print(
                    f"OVERPASS INVALID RESPONSE: "
                    f"{endpoint}: {exc}",
                    flush=True,
                )

            except Exception as exc:
                last_error = exc

                print(
                    f"OVERPASS FAILED: "
                    f"{endpoint}: {exc}",
                    flush=True,
                )

    raise RuntimeError(
        "All Overpass endpoints failed: "
        f"{last_error}"
    )


def deduplicate_quarries(
    quarries: list[Quarry],
) -> list[Quarry]:
    """Remove duplicate quarry records."""

    unique: dict[str, Quarry] = {}

    for quarry in quarries:
        if quarry.name:
            key = quarry.name.lower().strip()
        else:
            key = (
                f"{quarry.latitude:.5f}:"
                f"{quarry.longitude:.5f}"
            )

        if key not in unique:
            unique[key] = quarry

    return list(unique.values())


def get_website(
    tags: dict,
) -> str | None:
    """Return a website from OSM tags."""

    return (
        tags.get("website")
        or tags.get("contact:website")
    )