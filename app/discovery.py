import httpx

from app.models import Quarry
from app.utils import haversine_distance_km, stable_id


OVERPASS_ENDPOINTS = [

    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

USER_AGENT = "ORIS-QuarryResearch/1.0"


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
    """Execute one Overpass query."""

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
    """Convert Overpass response into Quarry objects."""

    quarries = []

    for element in data.get(
        "elements",
        [],
    ):
        tags = element.get(
            "tags",
            {},
        )

        element_type = element.get(
            "type"
        )

        # ---------------------------------------------------------
        # Node.
        # ---------------------------------------------------------
        if element_type == "node":
            lat = element.get("lat")
            lon = element.get("lon")

            if lat is None or lon is None:
                continue

        # ---------------------------------------------------------
        # Way / relation.
        # ---------------------------------------------------------
        else:
            center = element.get(
                "center"
            )

            if not center:
                continue

            lat = center.get("lat")
            lon = center.get("lon")

            if lat is None or lon is None:
                continue

        # ---------------------------------------------------------
        # Distance.
        # ---------------------------------------------------------
        distance = haversine_distance_km(
            latitude,
            longitude,
            lat,
            lon,
        )

        if distance > radius_km:
            continue

        # ---------------------------------------------------------
        # Stable site ID.
        # ---------------------------------------------------------
        site_id = stable_id(
            f"{element_type}:{element['id']}"
        )

        # ---------------------------------------------------------
        # Website from OSM.
        # ---------------------------------------------------------
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
                distance_km=round(
                    distance,
                    3,
                ),
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

    Municipality resolution is intentionally excluded
    from this stage.

    This function only discovers OSM quarry records.
    """

    query = _build_quarry_query(
        latitude,
        longitude,
        radius_km,
    )

    timeout = httpx.Timeout(
        connect=10.0,
        read=45.0,
        write=45.0,
        pool=10.0,
    )

    last_error = None

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
                    f"QUARRIES DISCOVERED: "
                    f"{len(quarries)}",
                    flush=True,
                )

                return quarries

            except httpx.HTTPError as exc:
                last_error = exc

                print(
                    f"OVERPASS FAILED: "
                    f"{endpoint}: {exc}",
                    flush=True,
                )

                continue

    raise RuntimeError(
        "All Overpass endpoints failed: "
        f"{last_error}"
    )


def deduplicate_quarries(
    quarries: list[Quarry],
) -> list[Quarry]:
    """Remove duplicate quarry records."""

    unique = {}

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