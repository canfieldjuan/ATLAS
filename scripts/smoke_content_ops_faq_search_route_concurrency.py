#!/usr/bin/env python3
"""Smoke-test concurrent hosted FAQ search route reads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_content_ops_faq_search_route_contract as contract  # noqa: E402


def _case_snapshot(case: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = {
        "query": str(case["query"]),
        "corpus_id": str(case.get("corpus_id") or ""),
        "status": str(case.get("status") or ""),
        "limit": int(case["limit"]),
        "require_results": bool(case["require_results"]),
    }
    for key in (
        "expected_count",
        "expected_first_account_id",
        "expected_first_corpus_id",
        "expected_first_faq_id",
    ):
        if key in case:
            snapshot[key] = case[key]
    return snapshot


def _default_case(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "query": str(args.query),
        "corpus_id": str(args.corpus_id or ""),
        "status": str(args.status or ""),
        "limit": int(args.limit),
        "require_results": bool(args.require_results),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run concurrent hosted FAQ search route reads."
    )
    parser.add_argument("--base-url", default=os.environ.get("ATLAS_API_BASE_URL", ""))
    parser.add_argument(
        "--token",
        default=os.environ.get("ATLAS_B2B_JWT") or os.environ.get("ATLAS_TOKEN", ""),
    )
    parser.add_argument(
        "--query",
        default=os.environ.get("ATLAS_FAQ_SEARCH_QUERY", contract.DEFAULT_QUERY),
    )
    parser.add_argument("--corpus-id", default=os.environ.get("ATLAS_FAQ_SEARCH_CORPUS_ID", ""))
    parser.add_argument("--status", default=os.environ.get("ATLAS_FAQ_SEARCH_STATUS", ""))
    parser.add_argument("--limit", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_LIMIT", "5"))
    parser.add_argument("--route", default=contract.DEFAULT_ROUTE)
    parser.add_argument("--timeout", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_TIMEOUT", "10"))
    parser.add_argument("--requests", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_REQUESTS", "12"))
    parser.add_argument("--concurrency", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_CONCURRENCY", "4"))
    parser.add_argument("--max-error-rate", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_ERROR_RATE", "0"))
    parser.add_argument("--max-p95-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_P95_MS") or None)
    parser.add_argument("--max-single-request-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_SINGLE_REQUEST_MS") or None)
    parser.set_defaults(require_results=True)
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--allow-empty-results", action="store_false", dest="require_results")
    parser.add_argument("--case-file", type=Path)
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not contract._clean_url(args.base_url):
        errors.append("ATLAS_API_BASE_URL or --base-url is required")
    if not str(args.token or "").strip():
        errors.append("ATLAS_B2B_JWT, ATLAS_TOKEN, or --token is required")
    if args.case_file is None and not str(args.query or "").strip():
        errors.append("ATLAS_FAQ_SEARCH_QUERY or --query is required")
    for name in ("limit", "requests", "concurrency"):
        if int(getattr(args, name)) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    for name in ("timeout", "max_error_rate", "max_p95_ms", "max_single_request_ms"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            errors.append(f"--{name.replace('_', '-')} must be finite")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    if not 0 <= float(args.max_error_rate) <= 1:
        errors.append("--max-error-rate must be between 0 and 1")
    for name in ("max_p95_ms", "max_single_request_ms"):
        value = getattr(args, name)
        if value is not None and float(value) <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive")
    return errors


def _load_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    if args.case_file is None:
        return [_default_case(args)], []

    try:
        raw = json.loads(args.case_file.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], [f"--case-file could not be read: {exc}"]
    except json.JSONDecodeError as exc:
        return [], [f"--case-file must contain JSON: {exc.msg}"]

    if not isinstance(raw, list) or not raw:
        return [], ["--case-file must contain a non-empty JSON list"]

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"case[{index}] must be an object")
            continue

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"case[{index}].query must be a non-empty string")

        corpus_id = item.get("corpus_id", args.corpus_id or "")
        if not isinstance(corpus_id, str):
            errors.append(f"case[{index}].corpus_id must be a string")

        status = item.get("status", args.status or "")
        if not isinstance(status, str):
            errors.append(f"case[{index}].status must be a string")

        limit = item.get("limit", int(args.limit))
        if type(limit) is not int or limit <= 0:
            errors.append(f"case[{index}].limit must be a positive integer")

        require_results = item.get("require_results", bool(args.require_results))
        if type(require_results) is not bool:
            errors.append(f"case[{index}].require_results must be a boolean")

        expected_count = item.get("expected_count")
        if "expected_count" in item and (type(expected_count) is not int or expected_count < 0):
            errors.append(f"case[{index}].expected_count must be a non-negative integer")

        expected_first: dict[str, str] = {}
        for key in (
            "expected_first_account_id",
            "expected_first_corpus_id",
            "expected_first_faq_id",
        ):
            if key not in item:
                continue
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"case[{index}].{key} must be a non-empty string")
            else:
                expected_first[key] = value.strip()

        if errors and any(error.startswith(f"case[{index}].") for error in errors):
            continue

        case = {
            "query": query.strip(),
            "corpus_id": corpus_id.strip(),
            "status": status.strip(),
            "limit": limit,
            "require_results": require_results,
        }
        if "expected_count" in item:
            case["expected_count"] = expected_count
        case.update(expected_first)
        cases.append(case)
    return cases, errors


def _expected_case_errors(payload: Mapping[str, Any], case: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    count = payload.get("count")
    if "expected_count" in case and count != case["expected_count"]:
        errors.append(f"expected count {case['expected_count']} but got {count}")

    first_expectations = {
        "expected_first_account_id": "account_id",
        "expected_first_corpus_id": "corpus_id",
        "expected_first_faq_id": "faq_id",
    }
    expected_keys = [key for key in first_expectations if key in case]
    if not expected_keys:
        return errors

    results = payload.get("results")
    first = results[0] if isinstance(results, list) and results else None
    if not isinstance(first, Mapping):
        errors.append("expected first result but none was returned")
        return errors
    for expected_key in expected_keys:
        field = first_expectations[expected_key]
        expected = case[expected_key]
        actual = first.get(field)
        if actual != expected:
            errors.append(f"expected first {field} {expected!r} but got {actual!r}")
    return errors


def _run_one(
    index: int,
    args: argparse.Namespace,
    case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    errors: list[str] = []
    count: int | None = None
    active_case = _default_case(args) if case is None else case
    try:
        url = contract._build_url(
            base_url=str(args.base_url),
            route=str(args.route),
            query=str(active_case["query"]),
            corpus_id=str(active_case.get("corpus_id") or ""),
            status=str(active_case.get("status") or ""),
            limit=int(active_case["limit"]),
        )
        payload = contract._fetch_json(url, token=str(args.token).strip(), timeout=float(args.timeout))
        errors.extend(
            contract._validate_envelope(
                payload,
                require_results=bool(active_case["require_results"]),
            )
        )
        errors.extend(_expected_case_errors(payload, active_case))
        if type(payload.get("count")) is int:
            count = int(payload["count"])
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000)
    return {
        "index": index,
        "ok": not errors,
        "count": count,
        "elapsed_ms": round(elapsed_ms, 6),
        "errors": errors,
        "case_index": int(active_case.get("case_index", 0)),
        "case": _case_snapshot(active_case),
    }


def _run_concurrent(
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    active_cases = list(cases) if cases is not None else [_default_case(args)]
    workers = min(int(args.concurrency), int(args.requests))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one,
                index,
                args,
                {
                    **dict(active_cases[index % len(active_cases)]),
                    "case_index": index % len(active_cases),
                },
            )
            for index in range(int(args.requests))
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return sorted(results, key=lambda row: int(row["index"]))


def _latency_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row.get("elapsed_ms") or 0.0) for row in results)
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    p95_index = min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))
    return {
        "count": len(values),
        "p50_ms": round(float(statistics.median(values)), 6),
        "p95_ms": round(values[p95_index], 6),
        "max_ms": round(values[-1], 6),
    }


def _error_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [
        {
            "index": row.get("index"),
            **({"case_index": row.get("case_index")} if "case_index" in row else {}),
            **({"case": row.get("case")} if "case" in row else {}),
            "errors": row.get("errors"),
        }
        for row in results
        if row.get("errors")
    ]
    total = len(results)
    return {
        "count": len(failures),
        "rate": round((len(failures) / total) if total else 0.0, 6),
        "items": failures[:20],
        "truncated": len(failures) > 20,
    }


def _budget_summary(
    *,
    latency: Mapping[str, Any],
    errors: Mapping[str, Any],
    max_error_rate: float,
    max_p95_ms: float | None,
    max_single_request_ms: float | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for metric, actual, limit in (
        ("error_rate", float(errors["rate"]), float(max_error_rate)),
        ("p95_ms", float(latency["p95_ms"]), max_p95_ms),
        ("max_ms", float(latency["max_ms"]), max_single_request_ms),
    ):
        if limit is None:
            continue
        limit_value = round(float(limit), 6)
        ok = actual <= limit_value
        checks.append({"metric": metric, "actual": actual, "max": limit_value, "ok": ok})
        if not ok:
            failures.append(f"{metric} exceeded {limit_value}")
    return {"ok": not failures, "checks": checks, "failures": failures}


def _summary_payload(
    *,
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, Any]] | None = None,
    results: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    preflight_errors: Sequence[str] = (),
) -> dict[str, Any]:
    active_cases = list(cases) if cases is not None else [_default_case(args)]
    errors = _error_summary(results)
    latency = _latency_summary(results)
    budgets = _budget_summary(
        latency=latency,
        errors=errors,
        max_error_rate=float(args.max_error_rate),
        max_p95_ms=args.max_p95_ms,
        max_single_request_ms=args.max_single_request_ms,
    )
    return {
        "ok": not preflight_errors and budgets["ok"],
        "phase": "preflight" if preflight_errors else "complete",
        "route": str(args.route),
        "base_url": contract._clean_url(str(args.base_url)),
        "query": str(args.query),
        "corpus_id": str(args.corpus_id or ""),
        "status": str(args.status or ""),
        "limit": int(args.limit),
        "require_results": bool(args.require_results),
        "cases": {
            "total": len(active_cases),
            "case_file": str(args.case_file or ""),
            "items": [_case_snapshot(case) for case in active_cases[:20]],
            "truncated": len(active_cases) > 20,
        },
        "requests": {
            "total": len(results),
            "configured": int(args.requests),
            "concurrency": int(args.concurrency),
        },
        "latency": latency,
        "errors": errors,
        "budgets": budgets,
        "preflight_errors": list(preflight_errors),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _write_result(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    latency = summary["latency"]
    print(
        "FAQ search hosted concurrency smoke: "
        f"ok={summary['ok']} requests={summary['requests']['total']} "
        f"errors={summary['errors']['count']} p95_ms={latency['p95_ms']} "
        f"max_ms={latency['max_ms']} budget_failures={len(summary['budgets']['failures'])}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.perf_counter()
    preflight_errors = _validate_args(args)
    cases, case_errors = _load_cases(args)
    preflight_errors.extend(case_errors)
    results: list[dict[str, Any]] = []
    if not preflight_errors:
        results = _run_concurrent(args, cases)
    summary = _summary_payload(
        args=args,
        cases=cases,
        results=results,
        elapsed_seconds=time.perf_counter() - started,
        preflight_errors=preflight_errors,
    )
    _write_result(args.output_result, summary)
    _print_summary(summary, as_json=bool(args.json))
    if preflight_errors:
        return 2
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
