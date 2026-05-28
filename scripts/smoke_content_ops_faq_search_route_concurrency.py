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
    parser.add_argument("--detail-route", default=os.environ.get("ATLAS_FAQ_DETAIL_ROUTE", ""))
    parser.add_argument("--timeout", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_TIMEOUT", "10"))
    parser.add_argument("--requests", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_REQUESTS", "12"))
    parser.add_argument("--concurrency", type=int, default=os.environ.get("ATLAS_FAQ_SEARCH_CONCURRENCY", "4"))
    parser.add_argument("--max-error-rate", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_ERROR_RATE", "0"))
    parser.add_argument("--max-p95-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_P95_MS") or None)
    parser.add_argument("--max-single-request-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_SINGLE_REQUEST_MS") or None)
    parser.add_argument("--max-detail-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_DETAIL_MS") or None)
    parser.add_argument("--max-case-error-rate", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_CASE_ERROR_RATE") or None)
    parser.add_argument("--max-case-p95-ms", type=float, default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_CASE_P95_MS") or None)
    parser.add_argument(
        "--max-case-single-request-ms",
        type=float,
        default=os.environ.get("ATLAS_FAQ_SEARCH_MAX_CASE_SINGLE_REQUEST_MS") or None,
    )
    parser.set_defaults(require_results=True)
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--allow-empty-results", action="store_false", dest="require_results")
    parser.add_argument("--require-detail", action="store_true")
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
    for name in (
        "timeout",
        "max_error_rate",
        "max_p95_ms",
        "max_single_request_ms",
        "max_detail_ms",
        "max_case_error_rate",
        "max_case_p95_ms",
        "max_case_single_request_ms",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            errors.append(f"--{name.replace('_', '-')} must be finite")
    if float(args.timeout) <= 0:
        errors.append("--timeout must be positive")
    if not 0 <= float(args.max_error_rate) <= 1:
        errors.append("--max-error-rate must be between 0 and 1")
    if args.max_case_error_rate is not None and not 0 <= float(args.max_case_error_rate) <= 1:
        errors.append("--max-case-error-rate must be between 0 and 1")
    if bool(args.require_detail) and not bool(args.require_results):
        errors.append("--require-detail requires result rows; remove --allow-empty-results")
    if args.max_detail_ms is not None and not bool(args.require_detail):
        errors.append("--max-detail-ms requires --require-detail")
    for name in (
        "max_p95_ms",
        "max_single_request_ms",
        "max_detail_ms",
        "max_case_p95_ms",
        "max_case_single_request_ms",
    ):
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
        elif bool(args.require_detail) and not require_results:
            errors.append(
                f"case[{index}].require_results cannot be false when --require-detail is set"
            )

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
    detail_checked = False
    detail_elapsed_ms: float | None = None
    detail_errors: list[str] = []
    detail_faq_id: str | None = None
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
        if bool(args.require_detail) and not errors:
            detail_faq_id = contract._first_result_faq_id(payload)
            if detail_faq_id is None:
                detail_errors.append("results[0].faq_id is required when --require-detail is set")
            else:
                detail_url = contract._build_detail_url(
                    base_url=str(args.base_url),
                    route=str(args.route),
                    detail_route=str(args.detail_route or ""),
                    faq_id=detail_faq_id,
                )
                try:
                    detail_payload, detail_elapsed_ms = contract._timed_fetch_json(
                        detail_url,
                        token=str(args.token).strip(),
                        timeout=float(args.timeout),
                    )
                    detail_checked = True
                    detail_errors.extend(
                        contract._validate_detail(detail_payload, faq_id=detail_faq_id)
                    )
                except (RuntimeError, OSError, TypeError, ValueError) as exc:
                    detail_errors.append(f"{type(exc).__name__}: {exc}")
            errors.extend(detail_errors)
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000)
    row = {
        "index": index,
        "ok": not errors,
        "count": count,
        "elapsed_ms": round(elapsed_ms, 6),
        "errors": errors,
        "case_index": int(active_case.get("case_index", 0)),
        "case": _case_snapshot(active_case),
    }
    if bool(args.require_detail):
        row.update(
            {
                "detail_checked": detail_checked,
                "detail_faq_id": detail_faq_id,
                "detail_elapsed_ms": (
                    round(detail_elapsed_ms, 6) if detail_elapsed_ms is not None else None
                ),
                "detail_errors": detail_errors,
            }
        )
    return row


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


def _detail_latency_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    detail_results = [
        {"elapsed_ms": row.get("detail_elapsed_ms")}
        for row in results
        if row.get("detail_checked") is True and row.get("detail_elapsed_ms") is not None
    ]
    return _latency_summary(detail_results)


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


def _detail_summary(
    results: Sequence[Mapping[str, Any]], *, required: bool
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "checked": 0,
            "failures": 0,
            "items": [],
            "latency": _latency_summary(()),
        }

    latency = _detail_latency_summary(results)
    failures = [
        {
            "index": row.get("index"),
            "faq_id": row.get("detail_faq_id"),
            "errors": row.get("detail_errors"),
        }
        for row in results
        if row.get("detail_errors")
    ]
    return {
        "required": True,
        "checked": sum(1 for row in results if row.get("detail_checked") is True),
        "failures": len(failures),
        "items": failures[:20],
        "truncated": len(failures) > 20,
        "latency": latency,
    }


def _case_result_summaries(
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    detail_required: bool,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {index: [] for index in range(len(cases))}
    for row in results:
        case_index = row.get("case_index")
        if type(case_index) is int and 0 <= case_index < len(cases):
            grouped.setdefault(case_index, []).append(row)

    summaries: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        rows = grouped.get(case_index, [])
        errors = _error_summary(rows)
        detail = _detail_summary(rows, required=detail_required)
        summaries.append(
            {
                "case_index": case_index,
                "case": _case_snapshot(case),
                "requests": len(rows),
                "errors": {
                    "count": errors["count"],
                    "rate": errors["rate"],
                },
                "latency": _latency_summary(rows),
                "detail": {
                    "checked": detail["checked"],
                    "failures": detail["failures"],
                    "latency": detail["latency"],
                },
            }
        )
    return summaries


def _worst_case_summary(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    cases = summary.get("cases")
    if not isinstance(cases, Mapping):
        return None
    case_summaries = cases.get("summaries")
    if not isinstance(case_summaries, Sequence) or isinstance(case_summaries, (str, bytes)):
        return None

    best: Mapping[str, Any] | None = None
    best_key: tuple[float, float, float, int] | None = None
    for item in case_summaries:
        if not isinstance(item, Mapping):
            continue
        errors = item.get("errors")
        latency = item.get("latency")
        if not isinstance(errors, Mapping) or not isinstance(latency, Mapping):
            continue
        case_index = int(item.get("case_index") or 0)
        key = (
            float(errors.get("count") or 0.0),
            float(latency.get("p95_ms") or 0.0),
            float(latency.get("max_ms") or 0.0),
            -case_index,
        )
        if best_key is None or key > best_key:
            best = item
            best_key = key
    return best


def _budget_summary(
    *,
    latency: Mapping[str, Any],
    detail_latency: Mapping[str, Any],
    case_summaries: Sequence[Mapping[str, Any]],
    errors: Mapping[str, Any],
    max_error_rate: float,
    max_case_error_rate: float | None,
    max_p95_ms: float | None,
    max_single_request_ms: float | None,
    max_detail_ms: float | None,
    max_case_p95_ms: float | None,
    max_case_single_request_ms: float | None,
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
    if max_detail_ms is not None:
        limit_value = round(float(max_detail_ms), 6)
        if int(detail_latency.get("count") or 0) <= 0:
            checks.append(
                {
                    "metric": "detail_max_ms",
                    "actual": None,
                    "max": limit_value,
                    "ok": False,
                }
            )
            failures.append("detail_max_ms had no checked detail rows")
        else:
            actual = float(detail_latency["max_ms"])
            ok = actual <= limit_value
            checks.append(
                {
                    "metric": "detail_max_ms",
                    "actual": actual,
                    "max": limit_value,
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(f"detail_max_ms exceeded {limit_value}")
    if max_case_error_rate is not None:
        limit_value = round(float(max_case_error_rate), 6)
        for case_summary in case_summaries:
            case_index = int(case_summary.get("case_index") or 0)
            if int(case_summary.get("requests") or 0) <= 0:
                checks.append(
                    {
                        "metric": "case_error_rate",
                        "case_index": case_index,
                        "actual": None,
                        "max": limit_value,
                        "ok": False,
                    }
                )
                failures.append(f"case_error_rate had no request samples for case {case_index}")
                continue
            case_errors = case_summary.get("errors")
            actual = float(case_errors.get("rate") or 0.0) if isinstance(case_errors, Mapping) else 0.0
            ok = actual <= limit_value
            checks.append(
                {
                    "metric": "case_error_rate",
                    "case_index": case_index,
                    "actual": actual,
                    "max": limit_value,
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(f"case_error_rate exceeded {limit_value} for case {case_index}")
    for metric, latency_key, limit in (
        ("case_p95_ms", "p95_ms", max_case_p95_ms),
        ("case_max_ms", "max_ms", max_case_single_request_ms),
    ):
        if limit is None:
            continue
        limit_value = round(float(limit), 6)
        for case_summary in case_summaries:
            case_index = int(case_summary.get("case_index") or 0)
            case_latency = case_summary.get("latency")
            if not isinstance(case_latency, Mapping) or int(case_latency.get("count") or 0) <= 0:
                checks.append(
                    {
                        "metric": metric,
                        "case_index": case_index,
                        "actual": None,
                        "max": limit_value,
                        "ok": False,
                    }
                )
                failures.append(f"{metric} had no latency samples for case {case_index}")
                continue
            actual = float(case_latency.get(latency_key) or 0.0)
            ok = actual <= limit_value
            checks.append(
                {
                    "metric": metric,
                    "case_index": case_index,
                    "actual": actual,
                    "max": limit_value,
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(f"{metric} exceeded {limit_value} for case {case_index}")
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
    detail = _detail_summary(results, required=bool(args.require_detail))
    case_summaries = _case_result_summaries(
        active_cases,
        results,
        detail_required=bool(args.require_detail),
    )
    budgets = _budget_summary(
        latency=latency,
        detail_latency=detail["latency"],
        case_summaries=case_summaries,
        errors=errors,
        max_error_rate=float(args.max_error_rate),
        max_case_error_rate=args.max_case_error_rate,
        max_p95_ms=args.max_p95_ms,
        max_single_request_ms=args.max_single_request_ms,
        max_detail_ms=args.max_detail_ms,
        max_case_p95_ms=args.max_case_p95_ms,
        max_case_single_request_ms=args.max_case_single_request_ms,
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
        "require_detail": bool(args.require_detail),
        "cases": {
            "total": len(active_cases),
            "case_file": str(args.case_file or ""),
            "items": [_case_snapshot(case) for case in active_cases[:20]],
            "summaries": case_summaries,
            "truncated": len(active_cases) > 20,
        },
        "requests": {
            "total": len(results),
            "configured": int(args.requests),
            "concurrency": int(args.concurrency),
        },
        "latency": latency,
        "errors": errors,
        "detail": detail,
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
    worst_case = _worst_case_summary(summary)
    worst_case_text = ""
    if worst_case is not None:
        worst_latency = worst_case["latency"]
        worst_errors = worst_case["errors"]
        worst_case_text = (
            f" worst_case_index={worst_case['case_index']}"
            f" worst_case_errors={worst_errors['count']}"
            f" worst_case_p95_ms={worst_latency['p95_ms']}"
            f" worst_case_max_ms={worst_latency['max_ms']}"
        )
    print(
        "FAQ search hosted concurrency smoke: "
        f"ok={summary['ok']} requests={summary['requests']['total']} "
        f"errors={summary['errors']['count']} p95_ms={latency['p95_ms']} "
        f"max_ms={latency['max_ms']} detail_checked={summary['detail']['checked']} "
        f"detail_failures={summary['detail']['failures']} "
        f"budget_failures={len(summary['budgets']['failures'])}"
        f"{worst_case_text}"
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
