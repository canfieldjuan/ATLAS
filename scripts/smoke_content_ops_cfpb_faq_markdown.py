#!/usr/bin/env python3
"""Smoke-test live CFPB source rows through FAQ Markdown generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_source_adapters import load_source_campaign_opportunities_from_file  # noqa: E402
from extracted_content_pipeline.ticket_faq_markdown import DEFAULT_TITLE, build_ticket_faq_markdown  # noqa: E402


def _load_cfpb_exporter_module():
    path = ROOT / "scripts" / "export_content_ops_cfpb_sources.py"
    spec = importlib.util.spec_from_file_location("content_ops_cfpb_exporter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CFPB source exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cfpb = _load_cfpb_exporter_module()
fetch_cfpb_source_rows = _cfpb.fetch_cfpb_source_rows
render_jsonl = _cfpb.render_jsonl


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test live CFPB rows through FAQ Markdown generation."
    )
    parser.add_argument("--company", default=None)
    parser.add_argument("--product", default=None)
    parser.add_argument("--issue", default=None)
    parser.add_argument("--search-term", default=None)
    parser.add_argument("--api-url", default=_cfpb.DEFAULT_API_URL)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-rows-scanned", type=int, default=_cfpb.DEFAULT_MAX_ROWS_SCANNED)
    parser.add_argument("--timeout", type=float, default=_cfpb.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--source-system", default=_cfpb.DEFAULT_SOURCE_SYSTEM)
    parser.add_argument("--source-type", default=_cfpb.DEFAULT_SOURCE_TYPE)
    parser.add_argument("--user-agent", default=_cfpb.DEFAULT_USER_AGENT)
    parser.add_argument("--referer", default=_cfpb.DEFAULT_REFERER)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-evidence-per-item", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--support-contact", default=None)
    parser.add_argument("--output-source-rows", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("limit", "max_items", "max_evidence_per_item", "max_text_chars"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if int(args.max_rows_scanned) < int(args.limit):
        raise SystemExit("--max-rows-scanned must be >= --limit")
    if float(args.timeout) <= 0:
        raise SystemExit("--timeout must be positive")


def run_cfpb_faq_markdown_smoke(
    args: argparse.Namespace,
    *,
    source_rows_path: Path,
) -> tuple[int, dict[str, Any]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    faq: dict[str, Any] | None = None
    try:
        rows = fetch_cfpb_source_rows(
            api_url=str(args.api_url),
            company=args.company,
            product=args.product,
            issue=args.issue,
            search_term=args.search_term,
            limit=int(args.limit),
            max_rows_scanned=int(args.max_rows_scanned),
            timeout=float(args.timeout),
            source_system=str(args.source_system),
            source_type=str(args.source_type),
            user_agent=str(args.user_agent),
            referer=str(args.referer),
            require_narrative=True,
        )
        if len(rows) < int(args.limit):
            errors.append(f"expected {int(args.limit)} CFPB source row(s), got {len(rows)}")
        _write_jsonl(rows, source_rows_path)
        if not errors:
            loaded = load_source_campaign_opportunities_from_file(
                source_rows_path,
                file_format="jsonl",
                max_text_chars=int(args.max_text_chars),
            )
            result = build_ticket_faq_markdown(
                loaded.opportunities,
                title=str(args.title),
                max_items=int(args.max_items),
                max_evidence_per_item=int(args.max_evidence_per_item),
                support_contact=args.support_contact,
            )
            faq = result.as_dict()
            failed = _failed_output_checks(result.output_checks)
            if not result.items:
                errors.append("FAQ Markdown generated no items")
            if failed:
                errors.append(f"FAQ output checks failed: {', '.join(failed)}")
            if args.output_markdown and not errors:
                args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
                args.output_markdown.write_text(result.markdown, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - exercised by live hosts
        errors.append(f"{type(exc).__name__}: {exc}")
    return (0 if not errors else 1), {
        "ok": not errors,
        "source": "cfpb",
        "source_rows": len(rows),
        "source_rows_path": str(source_rows_path),
        "markdown_path": str(args.output_markdown) if args.output_markdown else None,
        "faq": faq,
        "errors": errors,
    }


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_jsonl(rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _failed_output_checks(output_checks: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in sorted(output_checks.items()) if passed is not True]


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True))
    elif payload.get("ok") and payload.get("markdown_path"):
        print(f"Content Ops CFPB FAQ Markdown smoke passed: markdown={payload.get('markdown_path')}")
    elif payload.get("ok"):
        faq = payload.get("faq") if isinstance(payload.get("faq"), Mapping) else {}
        print(str(faq.get("markdown") or ""), end="")
    else:
        print("Content Ops CFPB FAQ Markdown smoke failed:", file=sys.stderr)
        for error in payload.get("errors") or []:
            print(f"- {error}", file=sys.stderr)


def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if args.output_source_rows:
        code, payload = run_cfpb_faq_markdown_smoke(args, source_rows_path=args.output_source_rows)
    else:
        with tempfile.TemporaryDirectory(prefix="content-ops-cfpb-faq-") as tmpdir:
            code, payload = run_cfpb_faq_markdown_smoke(args, source_rows_path=Path(tmpdir) / "cfpb_sources.jsonl")
    _print_payload(payload, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
