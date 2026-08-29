from dataclasses import dataclass
import ipaddress
import socket
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pymupdf
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 "
    "ORIS-QuarryResearch/1.0"
)

MAX_CONTENT_SIZE = 2_000_000


DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "image/avif,"
        "image/webp,"
        "*/*;q=0.8"
    ),
    "Accept-Language": (
        "fr-FR,fr;q=0.9,"
        "en-US;q=0.8,en;q=0.7"
    ),
    "Accept-Encoding": (
        "gzip, deflate, br"
    ),
    "Connection": "keep-alive",
}


@dataclass
class CrawlResult:
    """
    Result returned by crawl_with_status().
    """

    status: str
    text: str | None = None


def validate_url(url: str) -> bool:
    """
    Validate URL and reject unsafe/private destinations.

    This protects the crawler from SSRF attempts and
    private/local network destinations.
    """

    if not url:
        return False

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in {
        "http",
        "https",
    }:
        return False

    if not parsed.hostname:
        return False

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            None,
        )

        for address in addresses:
            ip = ipaddress.ip_address(
                address[4][0]
            )

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return False

    except (
        socket.gaierror,
        ValueError,
    ):
        return False

    return True


async def allowed_by_robots(
    url: str,
) -> bool:
    """
    Check whether the crawler is allowed to
    access the URL according to robots.txt.
    """

    parsed = urlparse(url)

    robots_url = (
        f"{parsed.scheme}://"
        f"{parsed.netloc}/robots.txt"
    )

    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
        ) as client:

            response = await client.get(
                robots_url,
                headers=DEFAULT_HEADERS,
            )

        if response.status_code == 404:
            return True

        if response.status_code >= 400:
            return False

        parser.parse(
            response.text.splitlines()
        )

        return parser.can_fetch(
            USER_AGENT,
            url,
        )

    except (
        httpx.HTTPError,
        ValueError,
    ):
        return False


def html_to_text(
    html: str,
) -> str:
    """
    Convert HTML into clean readable text.
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
        ]
    ):
        element.decompose()

    return " ".join(
        soup.stripped_strings
    )


def pdf_to_text(
    content: bytes,
) -> str:
    """
    Extract text from PDF content.
    """

    document = pymupdf.open(
        stream=content,
        filetype="pdf",
    )

    pages = []

    try:
        for page in document:
            pages.append(
                page.get_text()
            )
    finally:
        document.close()

    return "\n".join(
        pages
    ).strip()


async def crawl_with_status(
    url: str,
) -> CrawlResult:
    """
    Crawl a public HTML/PDF page and return
    both the content and a machine-readable status.

    Possible statuses include:

        success
        invalid_url
        robots_blocked
        http_403
        http_404
        http_429
        http_500
        http_error
        too_large
        unsupported_content_type
        empty_content
        network_error
        invalid_redirect
    """

    # ---------------------------------------------------------
    # Step 1: Validate URL.
    # ---------------------------------------------------------

    if not validate_url(url):
        return CrawlResult(
            status="invalid_url",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 2: Check robots.txt.
    # ---------------------------------------------------------

    if not await allowed_by_robots(url):
        return CrawlResult(
            status="robots_blocked",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 3: Fetch page.
    # ---------------------------------------------------------

    try:
        async with httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers=DEFAULT_HEADERS,
        ) as client:

            response = await client.get(
                url
            )

    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.NetworkError,
    ):
        return CrawlResult(
            status="network_error",
            text=None,
        )

    except (
        httpx.HTTPError,
        ValueError,
    ):
        return CrawlResult(
            status="http_error",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 4: Validate redirect destination.
    # ---------------------------------------------------------

    final_url = str(
        response.url
    )

    if not validate_url(final_url):
        return CrawlResult(
            status="invalid_redirect",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 5: Handle HTTP status codes.
    # ---------------------------------------------------------

    if response.status_code == 403:
        return CrawlResult(
            status="http_403",
            text=None,
        )

    if response.status_code == 404:
        return CrawlResult(
            status="http_404",
            text=None,
        )

    if response.status_code == 429:
        return CrawlResult(
            status="http_429",
            text=None,
        )

    if response.status_code >= 500:
        return CrawlResult(
            status=f"http_{response.status_code}",
            text=None,
        )

    if response.status_code != 200:
        return CrawlResult(
            status=f"http_{response.status_code}",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 6: Protect against oversized responses.
    # ---------------------------------------------------------

    if len(response.content) > MAX_CONTENT_SIZE:
        return CrawlResult(
            status="too_large",
            text=None,
        )

    # ---------------------------------------------------------
    # Step 7: Determine content type.
    # ---------------------------------------------------------

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    # ---------------------------------------------------------
    # Step 8: HTML.
    # ---------------------------------------------------------

    if "text/html" in content_type:

        text = html_to_text(
            response.text
        )

        if not text:
            return CrawlResult(
                status="empty_content",
                text=None,
            )

        return CrawlResult(
            status="success",
            text=text,
        )

    # ---------------------------------------------------------
    # Step 9: PDF.
    # ---------------------------------------------------------

    if (
        "application/pdf"
        in content_type
        or "application/x-pdf"
        in content_type
    ):

        try:
            text = pdf_to_text(
                response.content
            )

        except Exception:
            return CrawlResult(
                status="pdf_parse_error",
                text=None,
            )

        if not text:
            return CrawlResult(
                status="empty_content",
                text=None,
            )

        return CrawlResult(
            status="success",
            text=text,
        )

    # ---------------------------------------------------------
    # Step 10: Unsupported content.
    # ---------------------------------------------------------

    return CrawlResult(
        status="unsupported_content_type",
        text=None,
    )


async def crawl(
    url: str,
) -> str | None:
    """
    Backwards-compatible crawler.

    The existing ORIS pipeline expects:

        str | None

    Therefore this function wraps crawl_with_status()
    and returns only the extracted text.
    """

    result = await crawl_with_status(
        url
    )

    if result.status != "success":
        return None

    return result.text
