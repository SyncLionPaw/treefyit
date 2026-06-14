"""PDF parsing via MinerU API — converts PDF to Markdown, then flows into the MD pipeline.

Agent API (lightweight, no auth): ≤ 10 MB, ≤ 20 pages
Precision API (token required): ≤ 200 MB, ≤ 200 pages

Set ``MINERU_TOKEN`` in your ``.env`` for the Precision API.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENT_BASE = "https://mineru.net/api/v1/agent"
PRECISION_BASE = "https://mineru.net/api/v4"

MINERU_TOKEN = os.getenv("MINERU_TOKEN", "")
POLL_INTERVAL = 3  # seconds
POLL_TIMEOUT = 300  # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_pdf(source: str | Path, **kwargs) -> str:
    """Convert a PDF file (local path or remote URL) to Markdown via MinerU.

    Args:
        source: Local file path or a ``https://`` URL to a PDF.
        **kwargs: Passed through to the API.  Common keys:
            language (str)     — default ``"ch"``
            enable_table (bool) — default ``True``
            is_ocr (bool)      — default ``False``
            enable_formula (bool) — default ``True``
            page_range (str)   — e.g. ``"1-10"``
            model_version (str) — Precision API only: ``"pipeline"``, ``"vlm"``

    Returns:
        Markdown text.
    """
    source = str(source)
    is_remote = _is_url(source)
    logger.info(
        "[pdf] parse_pdf source=%s remote=%s token=%s",
        source,
        is_remote,
        bool(MINERU_TOKEN),
    )

    try:
        if is_remote:
            result = _parse_url(source, **kwargs)
        else:
            result = _parse_file(source, **kwargs)
        logger.info("[pdf] parse_pdf done (%d chars)", len(result))
        return result
    except Exception:
        logger.error("[pdf] parse_pdf failed for %s", source, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Agent API (no auth, lightweight)
# ---------------------------------------------------------------------------


def _parse_url(url: str, **kwargs) -> str:
    """Submit a remote PDF URL to MinerU Agent API, poll for Markdown result."""
    logger.info("[pdf] _parse_url %s", url)
    body = {
        "url": url,
        "language": kwargs.get("language", "ch"),
        "enable_table": kwargs.get("enable_table", True),
        "is_ocr": kwargs.get("is_ocr", False),
        "enable_formula": kwargs.get("enable_formula", True),
    }
    if kwargs.get("page_range"):
        body["page_range"] = kwargs["page_range"]

    # Try Agent API first; fall back to Precision if token is set
    if MINERU_TOKEN:
        logger.info("[pdf] using Precision API (token present)")
        return _precision_parse(body, **kwargs)

    logger.info("[pdf] Agent API /parse/url request")
    resp = requests.post(f"{AGENT_BASE}/parse/url", json=body, timeout=30)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU: {data.get('msg', 'unknown error')}")

    task_id = data["data"]["task_id"]
    logger.info("[pdf] Agent API task_id=%s", task_id)
    md_url = _poll_agent(task_id)
    logger.info("[pdf] Agent API markdown_url fetched")
    return requests.get(md_url, timeout=60).text


def _parse_file(path: str, **kwargs) -> str:
    """Upload a local PDF file to MinerU, poll for Markdown result."""
    file_name = Path(path).name
    logger.info("[pdf] _parse_file %s (%d bytes)", file_name, Path(path).stat().st_size)
    body = {
        "file_name": file_name,
        "language": kwargs.get("language", "ch"),
        "enable_table": kwargs.get("enable_table", True),
        "is_ocr": kwargs.get("is_ocr", False),
        "enable_formula": kwargs.get("enable_formula", True),
    }
    if kwargs.get("page_range"):
        body["page_range"] = kwargs["page_range"]

    # Try Agent API first; fall back to Precision if token is set
    if MINERU_TOKEN:
        logger.info("[pdf] using Precision API (token present)")
        with open(path, "rb") as f:
            return _precision_parse_file(f.read(), file_name, **kwargs)

    # Step 1: get signed upload URL
    logger.info("[pdf] Agent API /parse/file request")
    resp = requests.post(f"{AGENT_BASE}/parse/file", json=body, timeout=30)
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU: {data.get('msg', 'unknown error')}")

    task_id = data["data"]["task_id"]
    file_url = data["data"]["file_url"]
    logger.info("[pdf] Agent API task_id=%s upload_url=%s", task_id, file_url)

    # Step 2: PUT file
    with open(path, "rb") as f:
        put = requests.put(file_url, data=f, timeout=120)
        logger.info("[pdf] upload HTTP %d", put.status_code)
        if put.status_code not in (200, 201):
            raise RuntimeError(f"MinerU upload failed: HTTP {put.status_code}")

    # Step 3: poll
    md_url = _poll_agent(task_id)
    return requests.get(md_url, timeout=60).text


def _poll_agent(task_id: str) -> str:
    """Poll the Agent API until the task is done. Returns the markdown URL."""
    logger.info("[pdf] polling Agent API task_id=%s", task_id)
    deadline = time.time() + POLL_TIMEOUT
    polls = 0
    while time.time() < deadline:
        resp = requests.get(f"{AGENT_BASE}/parse/{task_id}", timeout=15)
        data = resp.json()
        state = data["data"]["state"]
        polls += 1
        if state == "done":
            logger.info("[pdf] poll done after %d attempt(s)", polls)
            return data["data"]["markdown_url"]
        if state == "failed":
            raise RuntimeError(
                f"MinerU failed: {data['data'].get('err_msg', 'unknown')}"
            )
        logger.debug("[pdf] poll %d state=%s", polls, state)
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"MinerU task {task_id} timed out after {POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Precision API (token required, higher limits)
# ---------------------------------------------------------------------------


def _precision_parse(body: dict, **kwargs) -> str:
    """Submit via Precision API (URL mode)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }
    req = {
        "url": body["url"],
        "model_version": kwargs.get("model_version", "vlm"),
        "is_ocr": body.get("is_ocr", False),
        "enable_formula": body.get("enable_formula", True),
        "enable_table": body.get("enable_table", True),
        "language": body.get("language", "ch"),
    }
    if body.get("page_range"):
        req["page_ranges"] = body["page_range"]

    resp = requests.post(
        f"{PRECISION_BASE}/extract/task", headers=headers, json=req, timeout=30
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU: {data.get('msg', 'unknown error')}")

    task_id = data["data"]["task_id"]
    zip_url = _poll_precision(task_id)
    return _fetch_md_from_zip(zip_url)


def _precision_parse_file(content: bytes, file_name: str, **kwargs) -> str:
    """Upload a file via Precision batch API and return the Markdown."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}",
    }

    # Step 1: request upload URL
    batch_req = {
        "files": [{"name": file_name}],
        "model_version": kwargs.get("model_version", "vlm"),
        "is_ocr": kwargs.get("is_ocr", False),
        "enable_formula": kwargs.get("enable_formula", True),
        "enable_table": kwargs.get("enable_table", True),
        "language": kwargs.get("language", "ch"),
    }
    resp = requests.post(
        f"{PRECISION_BASE}/file-urls/batch", headers=headers, json=batch_req, timeout=30
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"MinerU batch: {data.get('msg', 'unknown error')}")

    batch_id = data["data"]["batch_id"]
    file_urls = data["data"]["file_urls"]

    # Step 2: upload file
    put = requests.put(file_urls[0], data=content, timeout=120)
    if put.status_code not in (200, 201):
        raise RuntimeError(f"MinerU upload failed: HTTP {put.status_code}")

    # Step 3: poll batch
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(
            f"{PRECISION_BASE}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            time.sleep(POLL_INTERVAL)
            continue
        results = data.get("data", {}).get("extract_result", [])
        if results:
            r = results[0]
            state = r.get("state", "")
            if state == "done":
                zip_url = r.get("full_zip_url", "")
                return _fetch_md_from_zip(zip_url)
            if state == "failed":
                raise RuntimeError(f"MinerU failed: {r.get('err_msg', 'unknown')}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"MinerU batch {batch_id} timed out")


def _poll_precision(task_id: str) -> str:
    """Poll Precision API, return the ZIP download URL."""
    headers = {"Authorization": f"Bearer {MINERU_TOKEN}"}
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(
            f"{PRECISION_BASE}/extract/task/{task_id}", headers=headers, timeout=15
        )
        data = resp.json()
        state = data["data"]["state"]
        if state == "done":
            return data["data"]["full_zip_url"]
        if state == "failed":
            raise RuntimeError(
                f"MinerU failed: {data['data'].get('err_msg', 'unknown')}"
            )
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"MinerU task {task_id} timed out")


def _fetch_md_from_zip(zip_url: str) -> str:
    """Download the result ZIP and extract ``full.md`` from it."""
    import io
    import zipfile

    resp = requests.get(zip_url, timeout=120)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Find the markdown file (full.md or similar)
        for name in zf.namelist():
            if name.endswith(".md") and "full" in name.lower():
                return zf.read(name).decode("utf-8")
        # Fallback: take the first .md file
        for name in zf.namelist():
            if name.endswith(".md"):
                return zf.read(name).decode("utf-8")
    raise RuntimeError("No .md file found in MinerU result ZIP")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_url(s: str) -> bool:
    return urlparse(s).scheme in ("http", "https")
