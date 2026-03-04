"""Bright Data Datasets (v3) helper.

Purpose: run a dataset scrape, poll for completion, then download results in JSON.

Why this exists:
- It avoids putting Bearer tokens into shell history/logs.
- It prints snapshot IDs and error payloads so "0 results" is diagnosable.

Usage:
    export BRIGHTDATA_TOKEN='...'
  python scripts/brightdata_dataset_scrape.py \
    --dataset-id gd_l88xvdka1uao86xvlb \
    --url https://g2.com/products/gong/reviews \
    --sort-filter "Most Recent" \
    --start-date 2025-03-02T00:00:00.000Z

    # Or store the token in a local file (recommended; avoids env + shell history):
    #   mkdir -p .secrets && chmod 700 .secrets
    #   printf '%s' 'YOUR_TOKEN_HERE' > .secrets/brightdata_token && chmod 600 .secrets/brightdata_token
    python scripts/brightdata_dataset_scrape.py \
        --token-file .secrets/brightdata_token \
        --dataset-id gd_l88xvdka1uao86xvlb \
        --url https://g2.com/products/gong/reviews

  # Or provide full request body (must include top-level "input": [...])
  python scripts/brightdata_dataset_scrape.py --dataset-id ... --input-file /path/to/input.json

Notes:
- This script assumes Bright Data returns a snapshot ID.
- Endpoints vary slightly by plan/version; the downloader tries several known URLs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import ssl
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.brightdata.com"


@dataclass(frozen=True)
class ScrapeStart:
    snapshot_id: str
    raw: dict[str, Any]


def _normalize_g2_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    if url.startswith("https://g2.com/"):
        url = "https://www.g2.com/" + url[len("https://g2.com/") :]
    return url


def _load_input(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            body = json.load(f)
        if not isinstance(body, dict) or "input" not in body:
            raise ValueError("input-file must be a JSON object with top-level 'input' array")
        return body

    if not args.url:
        raise ValueError("Provide --url at least once, or --input-file")

    rows: list[dict[str, Any]] = []
    for raw_url in args.url:
        url = _normalize_g2_url(raw_url)
        row: dict[str, Any] = {"url": url}
        if args.sort_filter is not None:
            row["sort_filter"] = args.sort_filter
        if args.start_date is not None:
            row["start_date"] = args.start_date
        rows.append(row)

    return {"input": rows}


def _get_token(*, token_file: str | None = None) -> str:
    if token_file:
        with open(token_file, "r", encoding="utf-8") as f:
            token = f.read().strip()
        if not token:
            raise RuntimeError(f"Token file is empty: {token_file}")
        return token

    token = os.environ.get("BRIGHTDATA_TOKEN", "").strip()
    if not token:
        token = os.environ.get("ATLAS_BRIGHTDATA_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing token. Provide --token-file, or set BRIGHTDATA_TOKEN / ATLAS_BRIGHTDATA_TOKEN in your environment."
        )
    return token


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _safe_json_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"_non_object_json": data}
    except Exception:
        return {"_non_json_body": text[:2000]}


@dataclass(frozen=True)
class _HttpResponse:
    status: int
    headers: dict[str, str]
    text: str


def _http_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: int = 60,
) -> _HttpResponse:
    headers = dict(headers or {})
    if params:
        qs = urllib.parse.urlencode(params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{qs}"

    data: bytes | None = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())

    # Bright Data uses valid TLS certs; default context is fine.
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
            raw = resp.read()
            return _HttpResponse(
                status=getattr(resp, "status", 200),
                headers={k.lower(): v for k, v in resp.headers.items()},
                text=raw.decode("utf-8", errors="replace"),
            )
    except urllib.error.HTTPError as e:
        raw = e.read() if hasattr(e, "read") else b""
        return _HttpResponse(
            status=int(getattr(e, "code", 0) or 0),
            headers={k.lower(): v for k, v in (e.headers.items() if e.headers else [])},
            text=raw.decode("utf-8", errors="replace"),
        )


def start_scrape(*, dataset_id: str, token: str, body: dict[str, Any]) -> ScrapeStart:
    url = f"{API_BASE}/datasets/v3/scrape"
    params = {
        "dataset_id": dataset_id,
        "notify": "false",
        "include_errors": "true",
    }

    resp = _http_request(
        "POST",
        url,
        params=params,
        headers=_auth_headers(token),
        json_body=body,
        timeout_s=60,
    )
    if resp.status >= 400:
        raise RuntimeError(f"Start scrape failed HTTP {resp.status}: {_safe_json_text(resp.text)}")

    data = _safe_json_text(resp.text)

    # Observed variants:
    # - {"snapshot_id": "..."}
    # - {"id": "..."}
    # - {"snapshotId": "..."}
    snapshot_id = (
        data.get("snapshot_id")
        or data.get("snapshotId")
        or data.get("id")
        or ""
    )
    if not snapshot_id:
        raise RuntimeError(f"Could not find snapshot id in response: {data}")

    return ScrapeStart(snapshot_id=snapshot_id, raw=data)


def poll_progress(*, snapshot_id: str, token: str, timeout_s: int) -> dict[str, Any]:
    # Bright Data progress endpoints vary; try common ones.
    candidates = [
        f"{API_BASE}/datasets/v3/progress/{snapshot_id}",
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}/progress",
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}",
    ]

    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] = {}

    while time.time() < deadline:
        for u in candidates:
            r = _http_request("GET", u, headers=_auth_headers(token), timeout_s=60)

            if r.status == 404:
                continue
            if r.status >= 400:
                last_payload = {"_url": u, "http": r.status, "body": _safe_json_text(r.text)}
                continue

            payload = _safe_json_text(r.text)
            last_payload = {"_url": u, **payload}

            # Heuristics: these show up in different versions.
            status = (
                str(payload.get("status", ""))
                or str(payload.get("state", ""))
                or str(payload.get("snapshot_status", ""))
            ).lower()

            if status in {"ready", "completed", "complete", "done", "finished"}:
                return last_payload
            if status in {"failed", "error"}:
                return last_payload

        time.sleep(3)

    return {"_timeout": True, **last_payload}


def download_results(*, snapshot_id: str, token: str, max_bytes: int = 3_000_000) -> dict[str, Any]:
    # Try common download locations. Prefer JSON.
    candidates = [
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}?format=json",
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}/download?format=json",
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}.json",
        f"{API_BASE}/datasets/v3/snapshot/{snapshot_id}",
    ]

    last_err: dict[str, Any] = {}
    for u in candidates:
        r = _http_request("GET", u, headers=_auth_headers(token), timeout_s=60)
        if r.status == 404:
            last_err = {"_url": u, "http": 404}
            continue
        if r.status >= 400:
            last_err = {"_url": u, "http": r.status, "body": _safe_json_text(r.text)}
            continue

        # Limit output size in case user accidentally downloads huge datasets.
        content = r.text.encode("utf-8")[:max_bytes]
        try:
            return json.loads(content.decode("utf-8"))
        except Exception:
            return {"_url": u, "_non_json_preview": content[:2000].decode("utf-8", errors="replace")}

    return {"_download_failed": True, **last_err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--token-file",
        help="Path to a file containing ONLY the Bright Data API Bearer token (whitespace trimmed).",
    )
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--input-file", help="JSON file containing {input: [...]}.")
    ap.add_argument(
        "--url",
        action="append",
        help="URL to scrape. Repeat --url multiple times to pass multiple inputs.",
    )
    ap.add_argument("--sort-filter", dest="sort_filter")
    ap.add_argument("--start-date", dest="start_date")
    ap.add_argument("--timeout", type=int, default=300, help="Poll timeout seconds")
    ap.add_argument("--no-download", action="store_true", help="Only start + poll; skip result download")

    args = ap.parse_args()

    token = _get_token(token_file=args.token_file)
    body = _load_input(args)

    started = start_scrape(dataset_id=args.dataset_id, token=token, body=body)
    print(json.dumps({"snapshot_id": started.snapshot_id, "start": started.raw}, indent=2))

    progress = poll_progress(snapshot_id=started.snapshot_id, token=token, timeout_s=args.timeout)
    print(json.dumps({"progress": progress}, indent=2))

    if not args.no_download:
        results = download_results(snapshot_id=started.snapshot_id, token=token)
        # Helpful summary if it looks like a list.
        if isinstance(results, list):
            print(json.dumps({"items": len(results)}, indent=2))
        print(json.dumps({"results": results}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
