#!/usr/bin/env python3
"""Verify the hosted Content Ops FAQ search route contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any


DEFAULT_ROUTE = "/api/v1/content-ops/faq-deflection-search"
DEFAULT_QUERY = "mortgage payment dispute"
RESULT_FIELDS = (
    "question",
    "answer_summary",
    "topic",
    "source_ids",
    "ticket_count",
    "score",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the hosted Content Ops FAQ search route contract."
    )
    parser.add_argument("--base-url", default=os.environ.get("ATLAS_API_BASE_URL", ""))
    parser.add_argument(
        "--token",
        default=os.environ.get("ATLAS_B2B_JWT") or os.environ.get("ATLAS_TOKEN", ""),
    )
    parser.add_argument("--query", default=os.environ.get("ATLAS_FAQ_SEARCH_QUERY", DEFAULT_QUERY))
    parser.add_argument("--corpus-id", default=os.environ.get("ATLAS_FAQ_SEARCH_CORPUS_ID", ""))
    parser.add_argument("--status", default=os.environ.get("ATLAS_FAQ_SEARCH_STATUS", ""))
    parser.add_argument("--limit", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_LIMIT", "5"))
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--timeout", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_TIMEOUT", "10"))
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--output-result", type=Path)
    return parser


def _clean_url(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _build_url(
    *,
    base_url: str,
    route: str,
    query: str,
    corpus_id: str = "",
    status: str = "",
    limit: int = 5,
) -> str:
    path = "/" + route.strip("/")
    params: dict[str, str] = {
        "q": query,
        "limit": str(limit),
    }
    if corpus_id.strip():
        params["corpus_id"] = corpus_id.strip()
    if status.strip():
        params["status"] = status.strip()
    return f"{_clean_url(base_url)}{path}?{urllib.parse.urlencode(params)}"


def _fetch_json(url: str, *, token: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"route returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"route request failed: {exc.reason}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("route did not return JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("route returned non-object JSON")
    return data


def _validate_envelope(data: Mapping[str, Any], *, require_results: bool) -> list[str]:
    errors: list[str] = []
    query = data.get("query")
    results = data.get("results")
    count = data.get("count")
    if not isinstance(query, str):
        errors.append("query must be a string")
    if not isinstance(results, list):
        errors.append("results must be a list")
    if type(count) is not int:
        errors.append("count must be an integer")
    if isinstance(results, list) and isinstance(count, int) and count != len(results):
        errors.append("count must match len(results)")
    if require_results:
        if not isinstance(results, list) or not results:
            errors.append("results must include at least one item")
        elif not isinstance(results[0], Mapping):
            errors.append("results[0] must be an object")
        else:
            # The demo handoff maps results[0]; this smoke validates that surface.
            errors.extend(_validate_first_result(results[0]))
    return errors


def _validate_first_result(result: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in RESULT_FIELDS:
        if field not in result:
            errors.append(f"results[0].{field} is required")
    if "source_ids" in result and not isinstance(result["source_ids"], list):
        errors.append("results[0].source_ids must be a list")
    if "ticket_count" in result and type(result["ticket_count"]) is not int:
        errors.append("results[0].ticket_count must be an integer")
    if "score" in result and type(result["score"]) is not int:
        errors.append("results[0].score must be an integer")
    for field in ("question", "answer_summary", "topic"):
        if field in result and not isinstance(result[field], str):
            errors.append(f"results[0].{field} must be a string")
    return errors


def _result_payload(
    *,
    ok: bool,
    phase: str,
    base_url: str,
    route: str,
    query: str,
    corpus_id: str,
    status: str,
    limit: int,
    require_results: bool,
    count: Any = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "phase": phase,
        "route": route,
        "base_url": base_url,
        "query": query,
        "corpus_id": corpus_id,
        "status": status,
        "limit": limit,
        "require_results": require_results,
        "errors": list(errors or ()),
    }
    if type(count) is int:
        payload["count"] = count
    return payload


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _build_parser().parse_args()
    base_url = _clean_url(args.base_url)
    token = str(args.token or "").strip()
    query = str(args.query or "").strip()
    corpus_id = str(args.corpus_id or "").strip()
    status = str(args.status or "").strip()
    route = str(args.route or "").strip()
    limit = int(args.limit)
    if not base_url:
        print("ATLAS_API_BASE_URL or --base-url is required.")
        _write_result(
            args.output_result,
            _result_payload(
                ok=False,
                phase="preflight",
                base_url=base_url,
                route=route,
                query=query,
                corpus_id=corpus_id,
                status=status,
                limit=limit,
                require_results=bool(args.require_results),
                errors=["ATLAS_API_BASE_URL or --base-url is required"],
            ),
        )
        return 2
    if not token:
        print("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required.")
        _write_result(
            args.output_result,
            _result_payload(
                ok=False,
                phase="preflight",
                base_url=base_url,
                route=route,
                query=query,
                corpus_id=corpus_id,
                status=status,
                limit=limit,
                require_results=bool(args.require_results),
                errors=["ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required"],
            ),
        )
        return 2
    if not query:
        print("ATLAS_FAQ_SEARCH_QUERY or --query is required.")
        _write_result(
            args.output_result,
            _result_payload(
                ok=False,
                phase="preflight",
                base_url=base_url,
                route=route,
                query=query,
                corpus_id=corpus_id,
                status=status,
                limit=limit,
                require_results=bool(args.require_results),
                errors=["ATLAS_FAQ_SEARCH_QUERY or --query is required"],
            ),
        )
        return 2
    if limit <= 0:
        print("--limit must be positive.")
        _write_result(
            args.output_result,
            _result_payload(
                ok=False,
                phase="preflight",
                base_url=base_url,
                route=route,
                query=query,
                corpus_id=corpus_id,
                status=status,
                limit=limit,
                require_results=bool(args.require_results),
                errors=["--limit must be positive"],
            ),
        )
        return 2

    url = _build_url(
        base_url=base_url,
        route=route,
        query=query,
        corpus_id=corpus_id,
        status=status,
        limit=limit,
    )
    try:
        data = _fetch_json(url, token=token, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"FAQ search route check failed: {exc}")
        _write_result(
            args.output_result,
            _result_payload(
                ok=False,
                phase="request",
                base_url=base_url,
                route=route,
                query=query,
                corpus_id=corpus_id,
                status=status,
                limit=limit,
                require_results=bool(args.require_results),
                errors=[str(exc)],
            ),
        )
        return 1

    errors = _validate_envelope(data, require_results=args.require_results)
    payload = _result_payload(
        ok=not errors,
        phase="contract",
        base_url=base_url,
        route=route,
        query=query,
        corpus_id=corpus_id,
        status=status,
        limit=limit,
        require_results=bool(args.require_results),
        count=data.get("count"),
        errors=errors,
    )
    _write_result(args.output_result, payload)
    if errors:
        print("FAQ search route contract failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print(
            "FAQ search route contract passed: "
            f"query={data.get('query')!r}, count={data.get('count')}"
        )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
