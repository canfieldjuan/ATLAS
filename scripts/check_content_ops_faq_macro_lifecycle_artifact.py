#!/usr/bin/env python3
"""Validate and summarize one FAQ macro writeback lifecycle artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for operator consistency; output is always JSON.",
    )
    return parser.parse_args(argv)


def validate_lifecycle_artifact(payload: Any) -> dict[str, Any]:
    """Return a sanitized validation result for one lifecycle artifact."""

    if not isinstance(payload, Mapping):
        return _result(errors=["artifact_not_object"])

    write = payload.get("write")
    cleanup = payload.get("cleanup")
    write_map = write if isinstance(write, Mapping) else {}
    cleanup_map = cleanup if isinstance(cleanup, Mapping) else {}
    live_smoke = write_map.get("live_smoke")
    live_smoke_map = live_smoke if isinstance(live_smoke, Mapping) else {}

    errors: list[str] = []
    faq_id = _clean(payload.get("faq_id"))
    write_faq_id = _clean(write_map.get("faq_id"))
    cleanup_faq_id = _clean(cleanup_map.get("faq_id"))
    live_smoke_faq_id = _clean(live_smoke_map.get("faq_id"))
    external_deletes = _external_delete_summaries(cleanup_map.get("external_deletes"))

    _require(errors, payload.get("ok") is True, "lifecycle_not_ok")
    _require(errors, payload.get("skipped") is False, "lifecycle_skipped")
    _require(errors, payload.get("cleanup_skipped") is False, "cleanup_skipped")
    _require(errors, _clean(payload.get("stage")) == "complete", "lifecycle_incomplete")
    _require(errors, bool(faq_id), "missing_faq_id")
    _require(errors, isinstance(write, Mapping), "missing_write_payload")
    _require(errors, isinstance(cleanup, Mapping), "missing_cleanup_payload")
    _require(errors, write_map.get("ok") is True, "write_not_ok")
    _require(errors, write_map.get("skipped") is False, "write_skipped")
    _require(errors, _clean(write_map.get("stage")) == "complete", "write_incomplete")
    _require(errors, cleanup_map.get("ok") is True, "cleanup_not_ok")
    _require(errors, cleanup_map.get("skipped") is False, "cleanup_skipped")
    _require(errors, _clean(cleanup_map.get("stage")) == "complete", "cleanup_incomplete")
    _require(errors, bool(write_faq_id), "missing_write_faq_id")
    _require(errors, bool(cleanup_faq_id), "missing_cleanup_faq_id")
    _require(errors, isinstance(live_smoke, Mapping), "missing_live_smoke_payload")
    if isinstance(live_smoke, Mapping):
        _require(errors, bool(live_smoke_faq_id), "missing_live_smoke_faq_id")

    if faq_id and write_faq_id and faq_id != write_faq_id:
        errors.append("write_faq_id_mismatch")
    if faq_id and cleanup_faq_id and faq_id != cleanup_faq_id:
        errors.append("cleanup_faq_id_mismatch")
    if faq_id and live_smoke_faq_id and faq_id != live_smoke_faq_id:
        errors.append("live_smoke_faq_id_mismatch")

    if live_smoke_map:
        _require(errors, live_smoke_map.get("ok") is True, "live_smoke_not_ok")
        _require(
            errors,
            live_smoke_map.get("skipped") is False,
            "live_smoke_skipped",
        )

    deleted_faq_count = _int(cleanup_map.get("deleted_faq_count"))
    _require(errors, deleted_faq_count == 1, "cleanup_deleted_faq_count_not_one")
    _require(errors, bool(external_deletes), "missing_external_delete_proof")
    for item in external_deletes:
        if not item["external_id"]:
            errors.append("missing_external_delete_id")
            break
        if not item["ok"]:
            errors.append("external_delete_not_ok")
            break
    zendesk_base_url = _clean(
        payload.get("zendesk_base_url")
        or cleanup_map.get("zendesk_base_url")
        or write_map.get("zendesk_base_url")
        or live_smoke_map.get("zendesk_base_url")
    )
    if _base_url_mismatched(payload, write_map, cleanup_map, live_smoke_map):
        errors.append("zendesk_base_url_mismatch")

    stage_errors = _stage_errors(payload, write_map, cleanup_map, live_smoke_map)
    for error in stage_errors:
        if error not in errors:
            errors.append(error)

    return _result(
        errors=errors,
        account_id=_clean(payload.get("account_id")),
        faq_id=faq_id,
        zendesk_base_url=zendesk_base_url,
        lifecycle_stage=_clean(payload.get("stage")),
        write_stage=_clean(write_map.get("stage")),
        cleanup_stage=_clean(cleanup_map.get("stage")),
        publishable_count=_int(
            write_map.get("publishable_count") or live_smoke_map.get("publishable_count")
        ),
        deleted_faq_count=deleted_faq_count,
        external_deletes=external_deletes,
    )


def render_summary(result: Mapping[str, Any]) -> str:
    """Render a sanitized Markdown proof summary from a validation result."""

    lines = [
        "# FAQ macro writeback sandbox lifecycle proof",
        "",
        f"- Status: {'PASS' if result.get('ok') else 'FAIL'}",
        f"- Account id: {_clean(result.get('account_id')) or '(missing)'}",
        f"- FAQ id: {_clean(result.get('faq_id')) or '(missing)'}",
        f"- Zendesk base URL: {_clean(result.get('zendesk_base_url')) or '(missing)'}",
        f"- Lifecycle stage: {_clean(result.get('lifecycle_stage')) or '(missing)'}",
        f"- Write stage: {_clean(result.get('write_stage')) or '(missing)'}",
        f"- Cleanup stage: {_clean(result.get('cleanup_stage')) or '(missing)'}",
        f"- Publishable macro count: {_int(result.get('publishable_count'))}",
        f"- Deleted FAQ rows: {_int(result.get('deleted_faq_count'))}",
        "",
        "## External Deletes",
    ]
    external_deletes = result.get("external_deletes")
    if isinstance(external_deletes, list) and external_deletes:
        for item in external_deletes:
            if not isinstance(item, Mapping):
                continue
            status = "ok" if item.get("ok") else "failed"
            suffix = " already deleted" if item.get("already_deleted") else ""
            lines.append(
                f"- {_clean(item.get('external_id')) or '(missing)'}: {status}{suffix}"
            )
    else:
        lines.append("- None recorded.")

    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {_clean(error)}" for error in errors)
    return "\n".join(lines).rstrip() + "\n"


def _result(
    *,
    errors: list[str],
    account_id: str = "",
    faq_id: str = "",
    zendesk_base_url: str = "",
    lifecycle_stage: str = "",
    write_stage: str = "",
    cleanup_stage: str = "",
    publishable_count: int = 0,
    deleted_faq_count: int = 0,
    external_deletes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "errors": errors,
        "account_id": account_id,
        "faq_id": faq_id,
        "zendesk_base_url": zendesk_base_url,
        "lifecycle_stage": lifecycle_stage,
        "write_stage": write_stage,
        "cleanup_stage": cleanup_stage,
        "publishable_count": publishable_count,
        "deleted_faq_count": deleted_faq_count,
        "external_deletes": list(external_deletes or ()),
    }


def _external_delete_summaries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        rows.append({
            "external_id": _clean(item.get("external_id")),
            "ok": item.get("ok") is True,
            "already_deleted": bool(item.get("already_deleted")),
        })
    return rows


def _base_url_mismatched(*payloads: Mapping[str, Any]) -> bool:
    values = {
        _clean(payload.get("zendesk_base_url")).rstrip("/")
        for payload in payloads
        if _clean(payload.get("zendesk_base_url"))
    }
    return len(values) > 1


def _stage_errors(*payloads: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for payload in payloads:
        value = payload.get("errors")
        if not isinstance(value, list):
            continue
        for item in value:
            error = _clean(item)
            if error:
                errors.append(error)
    return errors


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition and code not in errors:
        errors.append(code)


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _read_artifact(path: Path) -> tuple[Any | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return None, ["artifact_not_found"]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, ["artifact_json_invalid"]
    except OSError:
        return None, ["artifact_read_failed"]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload, read_errors = _read_artifact(args.artifact)
    result = _result(errors=read_errors) if read_errors else validate_lifecycle_artifact(payload)
    if args.summary_output:
        args.summary_output.write_text(render_summary(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
