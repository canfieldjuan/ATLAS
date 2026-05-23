#!/usr/bin/env python3
"""Fan out uploaded-file import load across child processes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    parse_default_fields_or_exit,
)
from smoke_content_ops_ingestion_file_route_inprocess_load import (  # noqa: E402
    _default_database_url,
)


_CHILD_SCRIPT = SCRIPTS / "smoke_content_ops_ingestion_file_route_inprocess_load.py"
_REQUIRED_DEFAULT_FIELDS = ("company_name", "vendor_name", "contact_email")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple in-process uploaded-file import load probes as child "
            "processes to expose process-local admission behavior."
        )
    )
    parser.add_argument("path", type=Path, help="Source-row JSON, JSONL, or CSV file.")
    parser.add_argument("--source-format", choices=("auto", "json", "jsonl", "csv"), default="auto")
    parser.add_argument("--source", default="content-ops-file-route-multiprocess-load")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--min-source-rows", type=int, default=1)
    parser.add_argument("--max-source-text-chars", type=int, default=1200)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help=(
            "Fallback metadata for public/source-row files. Repeat as key=value. "
            "The multiprocess probe requires company_name, vendor_name, and contact_email."
        ),
    )
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--opportunity-table", default="campaign_opportunities")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--database-url", default=_default_database_url())
    parser.add_argument(
        "--admission-provider",
        choices=("local", "postgres"),
        default="local",
        help=(
            "Admission backend passed to each child. 'postgres' exercises the "
            "Atlas advisory-lock provider across processes."
        ),
    )
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--child-concurrency", type=int, default=2)
    parser.add_argument("--child-import-max-concurrency", type=int, default=1)
    parser.add_argument("--child-min-successes", type=int, default=1)
    parser.add_argument("--child-expect-at-capacity-min", type=int, default=1)
    parser.add_argument("--min-total-successes", type=int, default=1)
    parser.add_argument("--expect-total-at-capacity-min", type=int, default=1)
    parser.add_argument(
        "--allow-capacity-only-children",
        action="store_true",
        help=(
            "Treat child processes that only hit admission capacity as expected. "
            "Use with --admission-provider postgres when global slots are fewer "
            "than child processes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / "content_ops_file_route_multiprocess_load")
    parser.add_argument("--output-result", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    code, payload = asyncio.run(run_multiprocess_load(args))
    if args.output_result is not None:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload["summary"]
        print(
            "ok={ok} processes={processes} successes={successes} "
            "at_capacity={at_capacity} failures={failures} inserted={inserted}".format(
                ok=str(payload["ok"]).lower(),
                processes=summary["processes"],
                successes=summary["successes"],
                at_capacity=summary["at_capacity"],
                failures=summary["unexpected_failures"],
                inserted=summary["inserted"],
            )
        )
    return code


def _validate_args(args: argparse.Namespace) -> None:
    if not args.path.exists():
        raise SystemExit(f"source file not found: {args.path}")
    if int(args.min_source_rows) < 1:
        raise SystemExit("--min-source-rows must be positive")
    if int(args.max_source_text_chars) < 1:
        raise SystemExit("--max-source-text-chars must be positive")
    if int(args.sample_limit) < 0:
        raise SystemExit("--sample-limit must be non-negative")
    if int(args.processes) < 1:
        raise SystemExit("--processes must be positive")
    if int(args.child_concurrency) < 1:
        raise SystemExit("--child-concurrency must be positive")
    if int(args.child_import_max_concurrency) < 1:
        raise SystemExit("--child-import-max-concurrency must be positive")
    if int(args.child_min_successes) < 0:
        raise SystemExit("--child-min-successes must be non-negative")
    if int(args.child_expect_at_capacity_min) < 0:
        raise SystemExit("--child-expect-at-capacity-min must be non-negative")
    if int(args.min_total_successes) < 0:
        raise SystemExit("--min-total-successes must be non-negative")
    if int(args.expect_total_at_capacity_min) < 0:
        raise SystemExit("--expect-total-at-capacity-min must be non-negative")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if not str(args.database_url or "").strip():
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")
    if not str(args.opportunity_table or "").strip():
        raise SystemExit("--opportunity-table is required")
    defaults = parse_default_fields_or_exit(args.default_field)
    missing = [field for field in _REQUIRED_DEFAULT_FIELDS if not str(defaults.get(field) or "").strip()]
    if missing:
        raise SystemExit(
            "uploaded source-row multiprocess probe requires default fields: "
            + ", ".join(missing)
        )


async def run_multiprocess_load(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_specs = [
        _child_spec(args, index=index)
        for index in range(int(args.processes))
    ]
    child_results = await asyncio.gather(
        *[
            _run_child_process(spec["command"], result_path=spec["result_path"])
            for spec in child_specs
        ]
    )
    summary = _summarize_children(
        child_results,
        allow_capacity_only_children=bool(args.allow_capacity_only_children),
    )
    errors = _load_errors(
        summary,
        min_total_successes=int(args.min_total_successes),
        expect_total_at_capacity_min=int(args.expect_total_at_capacity_min),
    )
    payload = {
        "ok": not errors,
        "source_path": str(args.path),
        "source": str(args.source),
        "processes": int(args.processes),
        "child_concurrency": int(args.child_concurrency),
        "child_import_max_concurrency": int(args.child_import_max_concurrency),
        "admission_provider": str(args.admission_provider),
        "allow_capacity_only_children": bool(args.allow_capacity_only_children),
        "account_id": str(args.account_id).strip(),
        "output_dir": str(args.output_dir),
        "summary": summary,
        "children": child_results,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    return (0 if payload["ok"] else 1), payload


def _child_spec(args: argparse.Namespace, *, index: int) -> dict[str, Any]:
    result_path = args.output_dir / f"child_{index + 1}.json"
    account_id = f"{str(args.account_id).strip()}-p{index + 1}"
    source = f"{str(args.source).strip()}-p{index + 1}"
    command = _child_command(
        args,
        index=index,
        result_path=result_path,
        account_id=account_id,
        source=source,
    )
    return {
        "index": index,
        "account_id": account_id,
        "source": source,
        "result_path": result_path,
        "command": command,
    }


def _child_command(
    args: argparse.Namespace,
    *,
    index: int,
    result_path: Path,
    account_id: str,
    source: str,
) -> list[str]:
    del index
    command = [
        sys.executable,
        str(_CHILD_SCRIPT),
        str(args.path),
        "--source-format",
        str(args.source_format),
        "--source",
        source,
        "--target-mode",
        str(args.target_mode),
        "--min-source-rows",
        str(int(args.min_source_rows)),
        "--max-source-text-chars",
        str(int(args.max_source_text_chars)),
        "--sample-limit",
        str(int(args.sample_limit)),
        "--account-id",
        account_id,
        "--opportunity-table",
        str(args.opportunity_table),
        "--database-url",
        str(args.database_url),
        "--admission-provider",
        str(args.admission_provider),
        "--concurrency",
        str(int(args.child_concurrency)),
        "--import-max-concurrency",
        str(int(args.child_import_max_concurrency)),
        "--min-successes",
        str(int(args.child_min_successes)),
        "--expect-at-capacity-min",
        str(int(args.child_expect_at_capacity_min)),
        "--output-result",
        str(result_path),
    ]
    if str(args.user_id or "").strip():
        command.extend(["--user-id", str(args.user_id).strip()])
    if bool(args.replace_existing):
        command.append("--replace-existing")
    for item in args.default_field:
        command.extend(["--default-field", str(item)])
    return command


async def _run_child_process(command: list[str], *, result_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    payload = _load_child_payload(result_path)
    summary = payload.get("summary") if isinstance(payload, Mapping) else None
    return {
        "result_path": str(result_path),
        "returncode": int(process.returncode or 0),
        "ok": bool(payload.get("ok")) if isinstance(payload, Mapping) else False,
        "summary": dict(summary or {}),
        "errors": list(payload.get("errors") or []) if isinstance(payload, Mapping) else [],
        "stdout_tail": _tail(stdout.decode("utf-8", errors="replace")),
        "stderr_tail": _tail(stderr.decode("utf-8", errors="replace")),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _load_child_payload(result_path: Path) -> Mapping[str, Any]:
    if not result_path.exists():
        return {
            "ok": False,
            "summary": {},
            "errors": [f"missing child result: {result_path}"],
        }
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "summary": {},
            "errors": [f"invalid child result JSON: {exc}"],
        }
    if not isinstance(value, Mapping):
        return {
            "ok": False,
            "summary": {},
            "errors": ["child result JSON was not an object"],
        }
    return value


def _summarize_children(
    children: list[Mapping[str, Any]],
    *,
    allow_capacity_only_children: bool = False,
) -> dict[str, Any]:
    accepted_children = [
        child
        for child in children
        if child.get("ok")
        or (allow_capacity_only_children and _is_capacity_only_child(child))
    ]
    return {
        "processes": len(children),
        "successful_processes": len(accepted_children),
        "capacity_only_processes": sum(
            1 for child in children if _is_capacity_only_child(child)
        ),
        "failed_processes": len(children) - len(accepted_children),
        "successes": sum(_summary_int(child, "successes") for child in children),
        "at_capacity": sum(_summary_int(child, "at_capacity") for child in children),
        "unexpected_failures": sum(
            _summary_int(child, "unexpected_failures") for child in children
        ),
        "inserted": sum(_summary_int(child, "inserted") for child in children),
        "returncode_counts": _count_by(children, "returncode"),
    }


def _is_capacity_only_child(child: Mapping[str, Any]) -> bool:
    return (
        not child.get("ok")
        and _summary_int(child, "successes") == 0
        and _summary_int(child, "at_capacity") > 0
        and _summary_int(child, "unexpected_failures") == 0
    )


def _summary_int(child: Mapping[str, Any], key: str) -> int:
    summary = child.get("summary")
    if not isinstance(summary, Mapping):
        return 0
    return int(summary.get(key) or 0)


def _load_errors(
    summary: Mapping[str, Any],
    *,
    min_total_successes: int,
    expect_total_at_capacity_min: int,
) -> list[str]:
    errors: list[str] = []
    failed_processes = int(summary.get("failed_processes") or 0)
    unexpected_failures = int(summary.get("unexpected_failures") or 0)
    successes = int(summary.get("successes") or 0)
    at_capacity = int(summary.get("at_capacity") or 0)
    if failed_processes:
        errors.append(f"failed child process count: {failed_processes}")
    if unexpected_failures:
        errors.append(f"unexpected child failure count: {unexpected_failures}")
    if successes < min_total_successes:
        errors.append(
            f"expected at least {min_total_successes} total success(es), got {successes}"
        )
    if at_capacity < expect_total_at_capacity_min:
        errors.append(
            "expected at least "
            f"{expect_total_at_capacity_min} total admission 429 response(s), got {at_capacity}"
        )
    return errors


def _count_by(items: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        raw = item.get(key)
        value = "" if raw is None else str(raw).strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _tail(text: str, *, max_chars: int = 800) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())
