#!/usr/bin/env python3
"""Build a surrogate-only deflection PII eval corpus from labeled local JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.deflection_pii_eval_corpus import (  # noqa: E402
    SCHEMA_VERSION,
    build_surrogate_eval_corpus,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _write_error({"ok": False, "schema_version": SCHEMA_VERSION, "errors": [{
            "code": "input_json_unreadable",
            "message": str(exc.__class__.__name__),
        }]})
        return 1
    result = build_surrogate_eval_corpus(source)
    if not result.ok:
        _write_error({
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "errors": list(result.errors),
        })
        return 1
    assert result.artifact is not None
    text = json.dumps(result.artifact, indent=2 if args.pretty else None, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "output": str(args.output),
        "ticket_count": result.artifact["summary"]["ticket_count"],
        "label_count": result.artifact["summary"]["label_count"],
    }, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local labeled-source JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Surrogate artifact path.")
    parser.add_argument("--pretty", action="store_true", help="Write pretty JSON.")
    return parser.parse_args(argv)


def _write_error(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
