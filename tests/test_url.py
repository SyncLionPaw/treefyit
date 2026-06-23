"""Tests for URL fetching and routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.parser.url import SSRFError, assert_public_url, fetch_url, is_url, parse_url


def test_is_url():
    assert is_url("https://example.com/doc.md")
    assert not is_url("/local/path.pdf")
    assert not is_url("ftp://example.com/x")


def test_ssrf_blocks_localhost():
    with pytest.raises(SSRFError):
        assert_public_url("http://localhost/secret")


def test_ssrf_blocks_private_ip():
    with pytest.raises(SSRFError):
        assert_public_url("http://192.168.1.1/internal")


def test_ssrf_blocks_loopback_ip():
    with pytest.raises(SSRFError):
        assert_public_url("http://127.0.0.1/admin")


@patch("src.parser.url._get")
def test_fetch_rejects_localhost_before_request(get):
    with pytest.raises(SSRFError):
        fetch_url("http://127.0.0.1/x")
    get.assert_not_called()


@patch("src.parser.url._get")
def test_fetch_plain_markdown(get):
    resp = MagicMock()
    resp.content = b"# Hello\n\nworld"
    resp.encoding = "utf-8"
    resp.headers = {"Content-Type": "text/markdown; charset=utf-8"}
    get.return_value = resp

    doc = fetch_url("https://example.com/readme.md")
    assert doc.text == "# Hello\n\nworld"
    assert doc.filename == "readme.md"
    assert not doc.is_html


@patch("src.parser.url._parse_pdf_url")
def test_fetch_pdf_by_extension(parse_pdf):
    parse_pdf.return_value = "# PDF\n\ncontent"
    doc = fetch_url("https://example.com/paper.pdf")
    assert doc.text.startswith("# PDF")
    parse_pdf.assert_called_once()


@patch("src.parser.url._try_markitdown_url", return_value=None)
@patch("src.parser.url._get")
def test_fetch_html_page(get, _md):
    resp = MagicMock()
    resp.content = b"<html><body><h1>Title</h1><p>Body</p></body></html>"
    resp.encoding = "utf-8"
    resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    get.return_value = resp

    with patch("src.parser.url._html_to_markdown", return_value="# Title\n\nBody"):
        doc = fetch_url("https://example.com/docs/intro")

    assert doc.is_html
    assert "Title" in doc.text


@patch("src.parser.url.fetch_url")
def test_parse_url_single(fetch):
    fetch.return_value = MagicMock(text="hello", is_html=False)
    assert parse_url("https://example.com/a.txt") == "hello"


@patch("src.parser.url.fetch_url")
@patch("src.parser.url._extract_links_from_html")
def test_parse_url_recursive(extract_links, fetch):
    fetch.side_effect = [
        MagicMock(
            text="# Home\n\nindex",
            is_html=True,
            content_type="text/html",
        ),
        MagicMock(
            text="# Child\n\npage",
            is_html=False,
            content_type="text/markdown",
        ),
    ]
    extract_links.return_value = ["https://example.com/docs/child"]

    text = parse_url(
        "https://example.com/docs/",
        recursive=True,
        max_depth=1,
    )
    assert "Home" in text
    assert "Child" in text
    assert "---" in text


if __name__ == "__main__":
    test_is_url()
    test_fetch_plain_markdown()
    test_fetch_pdf_by_extension()
    test_fetch_html_page()
    test_parse_url_single()
    test_parse_url_recursive()
    print("ok")
