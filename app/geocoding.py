import asyncio
import time

import httpx


NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/reverse"
)

USER_AGENT = (
    "ORIS-QuarryResearch/1.0 "
    "(quarry research application)"
)

REQUEST_INTERVAL_SECONDS = 1.05
REQUEST_TIMEOUT = 15.0

_cache: dict[tuple[float, float], str | None] = {}
_last_request_time = 0.0


async def _respect_rate_limit() -> None:
    """Respect the public Nominatim request-rate limit."""

    global _last_request_time

    now = time.monotonic()
    elapsed = now - _last_request_time

    if elapsed < REQUEST_INTERVAL_SECONDS:
        await asyncio.sleep(
            REQUEST_INTERVAL_SECONDS - elapsed
        )

    _last_request_time = time.monotonic()


async def reverse_geocode(
    latitude: float,
    longitude: float,
) -> str | None:
    """Resolve coordinates to the nearest town or locality."""

    key = (
        round(latitude, 5),
        round(longitude, 5),
    )

    if key in _cache:
        return _cache[key]

    await _respect_rate_limit()

    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "zoom": 10,
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                NOMINATIM_URL,
                params=params,
                headers=headers,
            )

            response.raise_for_status()
            data = response.json()

    except (
        httpx.HTTPError,
        ValueError,
        TypeError,
    ) as exc:
        print(
            f"NOMINATIM FAILED: "
            f"{latitude},{longitude}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        _cache[key] = None
        return None

    if not isinstance(data, dict):
        _cache[key] = None
        return None

    address = data.get("address", {})

    if not isinstance(address, dict):
        _cache[key] = None
        return None

    municipality = (
        address.get("town")
        or address.get("city")
        or address.get("village")
        or address.get("municipality")
        or address.get("commune")
    )

    if municipality:
        municipality = str(municipality).strip()

    _cache[key] = municipality

    return municipality


async def enrich_coordinates(
    coordinates: list[tuple[float, float]],
) -> dict[tuple[float, float], str | None]:
    """Resolve multiple coordinates sequentially."""

    results = {}

    for latitude, longitude in coordinates:
        key = (
            round(latitude, 5),
            round(longitude, 5),
        )

        results[key] = await reverse_geocode(
            latitude,
            longitude,
        )

    return results


def clear_cache() -> None:
    """Clear the in-memory geocoding cache."""

    _cache.clear()


def cache_size() -> int:
    """Return the number of cached coordinate lookups."""

    return len(_cache)