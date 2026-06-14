#!/usr/bin/env python3
"""Smoke-test live CFPB source rows through FAQ Markdown generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

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
fetch_cfpb_source_rows_with_profile = _cfpb.fetch_cfpb_source_rows_with_profile
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
    parser.add_argument(
        "--compare-embedding-booster",
        action="store_true",
        help=(
            "Run the baseline FAQ path and the host mxbai embedding-boosted "
            "path from the same fetched CFPB rows."
        ),
    )
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
    embedding_port_factory: Callable[[], Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    source_profile: dict[str, Any] = {"status": "not_started"}
    faq: dict[str, Any] | None = None
    embedding_comparison: dict[str, Any] | None = None
    try:
        rows, source_profile = fetch_cfpb_source_rows_with_profile(
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
            baseline_result = build_ticket_faq_markdown(
                loaded.opportunities,
                title=str(args.title),
                max_items=int(args.max_items),
                max_evidence_per_item=int(args.max_evidence_per_item),
                support_contact=args.support_contact,
            )
            result = baseline_result
            if bool(getattr(args, "compare_embedding_booster", False)):
                embedding_comparison = {
                    "enabled": True,
                    "primary": "boosted",
                }
                try:
                    factory = embedding_port_factory or _build_default_embedding_port
                    embedding_port = _EmbeddingPortProbe(factory())
                except Exception as exc:
                    result = baseline_result
                    embedding_comparison.update({
                        "primary": "baseline",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    errors.append(
                        f"embedding booster unavailable: {type(exc).__name__}: {exc}"
                    )
                else:
                    boosted_result = build_ticket_faq_markdown(
                        loaded.opportunities,
                        title=str(args.title),
                        max_items=int(args.max_items),
                        max_evidence_per_item=int(args.max_evidence_per_item),
                        support_contact=args.support_contact,
                        embedding_port=embedding_port,
                    )
                    embedding_error = embedding_port.error
                    if embedding_error is None and embedding_port.valid_batches < 1:
                        embedding_error = "embedding booster applied no valid embedding batches"
                    if embedding_error is not None:
                        result = baseline_result
                        embedding_comparison.update({
                            "primary": "baseline",
                            "error": embedding_error,
                        })
                        errors.append(f"embedding booster unavailable: {embedding_error}")
                    else:
                        result = boosted_result
                        embedding_comparison.update(
                            _embedding_comparison_payload(
                                baseline=baseline_result,
                                boosted=boosted_result,
                            )
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
        if source_profile.get("status") == "not_started":
            source_profile = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        errors.append(f"{type(exc).__name__}: {exc}")
    return (0 if not errors else 1), {
        "ok": not errors,
        "source": "cfpb",
        "source_rows": len(rows),
        "source_profile": source_profile,
        "source_rows_path": str(source_rows_path),
        "markdown_path": str(args.output_markdown) if args.output_markdown else None,
        "faq": faq,
        "embedding_comparison": embedding_comparison,
        "errors": errors,
    }


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_jsonl(rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _failed_output_checks(output_checks: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in sorted(output_checks.items()) if passed is not True]


def _build_default_embedding_port() -> Any:
    from atlas_brain._content_ops_infrastructure import (
        build_content_ops_faq_embedding_port,
    )

    return build_content_ops_faq_embedding_port()


class _EmbeddingPortProbe:
    def __init__(self, port: Any) -> None:
        self._port = port
        self.calls = 0
        self.valid_batches = 0
        self.error: str | None = None

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls += 1
        try:
            embedded = self._port.embed_texts(texts)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        if (
            isinstance(embedded, Sequence)
            and not isinstance(embedded, (str, bytes, bytearray))
            and len(embedded) == len(texts)
        ):
            self.valid_batches += 1
        else:
            self.error = "embedding port returned an invalid batch"
        return embedded


def _embedding_comparison_payload(*, baseline: Any, boosted: Any) -> dict[str, Any]:
    baseline_summary = _faq_summary(baseline)
    boosted_summary = _faq_summary(boosted)
    return {
        "baseline": baseline_summary,
        "boosted": boosted_summary,
        "delta": {
            "generated": boosted_summary["generated"] - baseline_summary["generated"],
            "ticket_source_count": (
                boosted_summary["ticket_source_count"]
                - baseline_summary["ticket_source_count"]
            ),
            "non_repeat_ticket_count": (
                boosted_summary["non_repeat_ticket_count"]
                - baseline_summary["non_repeat_ticket_count"]
            ),
            "non_repeat_question_count": (
                boosted_summary["non_repeat_question_count"]
                - baseline_summary["non_repeat_question_count"]
            ),
            "changed_top_question": (
                boosted_summary["top_question"] != baseline_summary["top_question"]
            ),
            "added_questions": [
                question
                for question in boosted_summary["questions"]
                if question not in baseline_summary["questions"]
            ],
            "removed_questions": [
                question
                for question in baseline_summary["questions"]
                if question not in boosted_summary["questions"]
            ],
        },
    }


def _faq_summary(result: Any) -> dict[str, Any]:
    questions = [
        str(item.get("question") or "")
        for item in getattr(result, "items", ())
        if str(item.get("question") or "")
    ]
    return {
        "generated": len(getattr(result, "items", ())),
        "ticket_source_count": int(getattr(result, "ticket_source_count", 0)),
        "non_repeat_ticket_count": int(getattr(result, "non_repeat_ticket_count", 0)),
        "non_repeat_question_count": int(getattr(result, "non_repeat_question_count", 0)),
        "top_question": questions[0] if questions else None,
        "questions": questions,
        "output_checks": dict(getattr(result, "output_checks", {})),
    }


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
