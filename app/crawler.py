import ipaddress
import socket
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import fitz
import httpx
from bs4 import BeautifulSoup


USER_AGENT = "ORIS-QuarryResearch/1.0"
MAX_CONTENT_SIZE = 2_000_000


def validate_url(url: str) -> bool:
    """Validate URL and reject unsafe/private destinations."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if not parsed.hostname:
        return False

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            None,
        )

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return False

    except socket.gaierror:
        return False

    return True


async def allowed_by_robots(url: str) -> bool:
    """Check whether our crawler is allowed to access the URL."""
    parsed = urlparse(url)

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
            )

        if response.status_code == 404:
            return True

        if response.status_code >= 400:
            return False

        parser.parse(response.text.splitlines())

        return parser.can_fetch(USER_AGENT, url)

    except httpx.HTTPError:
        return False


def html_to_text(html: str) -> str:
    """Convert HTML into clean text."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    return " ".join(soup.stripped_strings)


def pdf_to_text(content: bytes) -> str:
    """Extract text from a PDF."""
    document = fitz.open(
        stream=content,
        filetype="pdf",
    )

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages).strip()


async def crawl(url: str) -> str | None:
    """Fetch a public HTML/PDF page and return its text."""

    if not validate_url(url):
        return None

    if not await allowed_by_robots(url):
        return None

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=False,
        ) as client:
            response = await client.get(
                url,
                headers={"User-Agent": USER_AGENT},
            )

        if response.status_code != 200:
            return None

        if len(response.content) > MAX_CONTENT_SIZE:
            return None

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "text/html" in content_type:
            return html_to_text(response.text)

        if "application/pdf" in content_type:
            return pdf_to_text(response.content)

        return None

    except (httpx.HTTPError, ValueError):
        return None