#!/usr/bin/env python3
"""Build Stage-1 sources.json for an Evidence-to-Story package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_evidence_to_story.sources import (  # noqa: E402
    load_evidence_story_sources,
    write_evidence_story_sources,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an Evidence-to-Story v0 manifest and write Stage-1 "
            "sources.json."
        )
    )
    parser.add_argument("manifest", type=Path, help="Stage-1 manifest JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where sources.json is written. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output_dir:
        output_path = write_evidence_story_sources(args.manifest, args.output_dir)
        print(f"Wrote {output_path}")
        return 0

    loaded = load_evidence_story_sources(args.manifest)
    print(json.dumps(loaded.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
