#!/usr/bin/env python3
"""Validate hosted deflection result-page observations against the QA scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    build_deflection_full_report_qa_deterministic_harness,
)


DEFAULT_REQUIRED_SURFACES = ("result_page", "evidence_export")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} must be readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload


def build_hosted_smoke_scorecard(
    *,
    report_model: dict[str, Any],
    evidence_export: dict[str, Any],
    surface_observations: dict[str, Any],
    surface_caps: dict[str, Any] | None = None,
    required_surfaces: tuple[str, ...] = DEFAULT_REQUIRED_SURFACES,
) -> dict[str, Any]:
    """Return the sanitized scorecard for hosted result-page smoke inputs."""

    return build_deflection_full_report_qa_deterministic_harness(
        report_model,
        evidence_export=evidence_export,
        surface_observations=surface_observations,
        surface_caps=surface_caps,
        required_surfaces=required_surfaces,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-model", type=Path, required=True)
    parser.add_argument("--evidence-export", type=Path, required=True)
    parser.add_argument("--surface-observations", type=Path, required=True)
    parser.add_argument("--surface-caps", type=Path)
    parser.add_argument(
        "--required-surface",
        action="append",
        default=[],
        help="Required hosted surface. Repeat for multiple surfaces.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    scorecard = build_hosted_smoke_scorecard(
        report_model=_load_object(args.report_model, "--report-model"),
        evidence_export=_load_object(args.evidence_export, "--evidence-export"),
        surface_observations=_load_object(
            args.surface_observations,
            "--surface-observations",
        ),
        surface_caps=(
            _load_object(args.surface_caps, "--surface-caps")
            if args.surface_caps
            else None
        ),
        required_surfaces=tuple(args.required_surface) or DEFAULT_REQUIRED_SURFACES,
    )
    text = json.dumps(scorecard, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if scorecard.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
