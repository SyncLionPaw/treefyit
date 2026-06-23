"""Fetch and parse remote URLs into Markdown or plain text.

LangChain loads URLs via document loaders (``WebBaseLoader`` fetches one page;
``RecursiveUrlLoader`` crawls child links).  We follow the same shape:

1. resolve the URL
2. fetch (or delegate to MinerU for remote PDFs)
3. detect format from path extension + ``Content-Type``
4. convert to Markdown / text for the existing ``parse_md`` pipeline

No LangChain dependency — just ``requests``, ``markitdown``, and existing parsers.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; treefyit/0.1; +https://github.com/treefyit)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}
_BINARY_EXTENSIONS = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".epub"}
_SKIP_LINK_PREFIXES = ("mailto:", "javascript:", "tel:", "#")
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.goog",
    }
)


class SSRFError(ValueError):
    """URL targets a non-public or blocked host."""


def assert_public_url(url: str) -> None:
    """Reject URLs that resolve to private/link-local/reserved addresses."""
    parsed = urlparse(url.strip())
    host = parsed.hostname
    if not host:
        raise SSRFError(f"URL has no hostname: {url!r}")

    lowered = host.lower().rstrip(".")
    if lowered in _BLOCKED_HOSTNAMES:
        raise SSRFError(f"Blocked hostname: {host}")

    def check_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        ):
            raise SSRFError(f"URL resolves to non-public address: {addr}")

    try:
        check_ip(ipaddress.ip_address(lowered))
        return
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SSRFError(f"Cannot resolve hostname {host!r}") from exc

    if not infos:
        raise SSRFError(f"Cannot resolve hostname {host!r}")

    for info in infos:
        check_ip(ipaddress.ip_address(info[4][0]))


@dataclass(frozen=True)
class UrlDocument:
    """Parsed URL payload — LangChain ``Document``-like, without the dependency."""

    url: str
    text: str
    content_type: str
    filename: str
    is_html: bool = False


def is_url(value: str) -> bool:
    return urlparse(value.strip()).scheme in ("http", "https")


def parse_url(
    url: str,
    *,
    recursive: bool = False,
    max_depth: int = 2,
    prevent_outside: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict | None = None,
    continue_on_failure: bool = True,
    **kwargs,
) -> str:
    """Fetch *url* and return Markdown or plain text.

    Args:
        url: ``http(s)://`` link to a document or web page.
        recursive: When ``True``, crawl child links (LangChain ``RecursiveUrlLoader``).
        max_depth: Maximum link depth for recursive mode (root = 0).
        prevent_outside: In recursive mode, stay under the root URL's netloc + path prefix.
        timeout: HTTP timeout in seconds.
        headers: Optional extra request headers.
        continue_on_failure: Skip failed child URLs in recursive mode.
        **kwargs: Forwarded to :func:`src.parser.pdf.parse_pdf` for PDF URLs.

    Returns:
        Document text suitable for :func:`src.tree.structure.build_tree_structure`.
    """
    url = url.strip()
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url!r}")

    if recursive:
        return _parse_recursive(
            url,
            max_depth=max_depth,
            prevent_outside=prevent_outside,
            timeout=timeout,
            headers=headers,
            continue_on_failure=continue_on_failure,
            pdf_kwargs=kwargs,
        )

    doc = fetch_url(url, timeout=timeout, headers=headers, **kwargs)
    return doc.text


def fetch_url(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict | None = None,
    **kwargs,
) -> UrlDocument:
    """Fetch a single URL and return structured text + metadata."""
    url = url.strip()
    if not is_url(url):
        raise ValueError(f"Not an http(s) URL: {url!r}")
    assert_public_url(url)

    ext = _path_extension(url)
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}

    if ext == ".pdf":
        text = _parse_pdf_url(url, **kwargs)
        return UrlDocument(
            url=url,
            text=text,
            content_type="application/pdf",
            filename=_filename_from_url(url, ".pdf"),
        )

    if ext in _TEXT_EXTENSIONS:
        resp = _get(url, timeout=timeout, headers=merged_headers)
        text = _decode_bytes(resp.content, resp)
        return UrlDocument(
            url=url,
            text=text,
            content_type=resp.headers.get("Content-Type", "text/plain"),
            filename=_filename_from_url(url, ext or ".txt"),
        )

    if ext in _BINARY_EXTENSIONS:
        resp = _get(url, timeout=timeout, headers=merged_headers)
        text = _markitdown_bytes(resp.content, suffix=ext)
        return UrlDocument(
            url=url,
            text=text,
            content_type=resp.headers.get("Content-Type", "application/octet-stream"),
            filename=_filename_from_url(url, ext),
        )

    md = _try_markitdown_url(url)
    if md:
        is_page = ext in ("", ".html", ".htm")
        return UrlDocument(
            url=url,
            text=md,
            content_type="text/markdown",
            filename=_filename_from_url(url, ".md"),
            is_html=is_page,
        )

    resp = _get(url, timeout=timeout, headers=merged_headers)
    content_type = _content_type(resp)

    if content_type.startswith("text/html") or content_type.startswith(
        "application/xhtml"
    ):
        text = _html_to_markdown(resp.content, resp)
        return UrlDocument(
            url=url,
            text=text,
            content_type=content_type,
            filename=_filename_from_url(url, ".html"),
            is_html=True,
        )

    text = _parse_response(url, resp, content_type=content_type, pdf_kwargs=kwargs)
    return UrlDocument(
        url=url,
        text=text,
        content_type=content_type,
        filename=_filename_from_url(url, _guess_suffix(content_type, url)),
    )


def _parse_recursive(
    root: str,
    *,
    max_depth: int,
    prevent_outside: bool,
    timeout: int,
    headers: dict | None,
    continue_on_failure: bool,
    pdf_kwargs: dict,
) -> str:
    """Breadth-first crawl, one section per page (RecursiveUrlLoader-style)."""
    root = urldefrag(root)[0]
    root_parts = urlparse(root)
    root_prefix = root if root.endswith("/") else root.rsplit("/", 1)[0] + "/"

    visited: set[str] = set()
    sections: list[str] = []
    queue: list[tuple[str, int]] = [(root, 0)]

    while queue:
        url, depth = queue.pop(0)
        url = urldefrag(url)[0]
        if url in visited:
            continue
        visited.add(url)

        try:
            doc = fetch_url(
                url, timeout=timeout, headers=headers, **pdf_kwargs
            )
        except Exception as exc:
            logger.warning("[url] skip %s: %s", url, exc)
            if not continue_on_failure:
                raise
            continue

        title = _title_from_text(doc.text) or url
        sections.append(f"# {title}\n\nSource: {url}\n\n{doc.text.strip()}")

        if depth >= max_depth or not doc.is_html:
            continue

        for link in _extract_links_from_html(url, timeout=timeout, headers=headers):
            if prevent_outside and not _is_under_root(link, root_parts, root_prefix):
                continue
            if link not in visited:
                queue.append((link, depth + 1))

    if not sections:
        raise RuntimeError(f"No content fetched from {root}")
    return "\n\n---\n\n".join(sections)


def _parse_pdf_url(url: str, **kwargs) -> str:
    assert_public_url(url)
    from src.parser.pdf import parse_pdf

    return parse_pdf(url, **kwargs)


def _parse_response(
    url: str,
    resp: requests.Response,
    *,
    content_type: str,
    pdf_kwargs: dict,
) -> str:
    if content_type == "application/pdf" or _path_extension(url) == ".pdf":
        return _parse_pdf_url(url, **pdf_kwargs)

    if content_type.startswith("text/html") or content_type.startswith(
        "application/xhtml"
    ):
        return _html_to_markdown(resp.content, resp)

    if content_type.startswith("text/"):
        return _decode_bytes(resp.content, resp)

    suffix = _guess_suffix(content_type, url)
    if suffix in _BINARY_EXTENSIONS:
        return _markitdown_bytes(resp.content, suffix=suffix)

    decoded = _decode_bytes(resp.content, resp)
    if decoded.strip():
        return decoded
    return _markitdown_bytes(resp.content, suffix=suffix or ".bin")


def _get(url: str, *, timeout: int, headers: dict) -> requests.Response:
    assert_public_url(url)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


def _try_markitdown_url(url: str) -> str | None:
    assert_public_url(url)
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None

    try:
        result = MarkItDown().convert(url)
        text = result.text_content if hasattr(result, "text_content") else str(result)
        return text.strip() or None
    except Exception as exc:
        logger.debug("[url] markitdown.convert(%r) failed: %s", url, exc)
        return None


def _markitdown_bytes(content: bytes, *, suffix: str) -> str:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "markitdown is required to parse this file type; pip install markitdown"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        path = tmp.name

    try:
        result = MarkItDown().convert(path)
        text = result.text_content if hasattr(result, "text_content") else str(result)
        if not text.strip():
            raise RuntimeError("markitdown returned empty text")
        return text
    finally:
        Path(path).unlink(missing_ok=True)


def _html_to_markdown(content: bytes, resp: requests.Response) -> str:
    md = _markitdown_bytes(content, suffix=".html")
    if md.strip():
        return md

    html = _decode_bytes(content, resp)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    if not text.strip():
        raise RuntimeError("Could not extract text from HTML")
    return text


def _decode_bytes(content: bytes, resp: requests.Response) -> str:
    encoding = resp.encoding
    if not encoding or encoding.lower() == "iso-8859-1":
        encoding = resp.apparent_encoding or "utf-8"
    for candidate in (encoding, "utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(candidate)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _extract_links_from_html(
    page_url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict | None = None,
) -> list[str]:
    """Pull http(s) links from a live HTML page."""
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    resp = _get(page_url, timeout=timeout, headers=merged_headers)
    soup = BeautifulSoup(resp.content, "lxml")
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(_SKIP_LINK_PREFIXES):
            continue
        absolute = urldefrag(urljoin(page_url, href))[0]
        if is_url(absolute):
            links.append(absolute)
    return links


def _is_under_root(link: str, root_parts, root_prefix: str) -> bool:
    parts = urlparse(link)
    if parts.netloc != root_parts.netloc:
        return False
    return link.startswith(root_prefix) or link == root_prefix.rstrip("/")


def _path_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _content_type(resp: requests.Response) -> str:
    return resp.headers.get("Content-Type", "").split(";")[0].strip().lower()


def _guess_suffix(content_type: str, url: str) -> str:
    ext = _path_extension(url)
    if ext:
        return ext
    mapping = {
        "application/pdf": ".pdf",
        "text/html": ".html",
        "text/markdown": ".md",
        "text/plain": ".txt",
    }
    return mapping.get(content_type, ".bin")


def _filename_from_url(url: str, suffix: str) -> str:
    name = Path(urlparse(url).path).name
    if name:
        return name
    host = urlparse(url).netloc.replace(":", "_")
    return f"{host}{suffix}"


def _title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            return stripped[:120]
    return ""
