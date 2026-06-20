#!/usr/bin/env python3
"""Summarize maturity-sweep baseline debt across lanes."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LaneSummary:
    lane: str
    path: str
    files: int
    total_score: int
    top_score: int
    counts: dict[str, int]


class BaselineError(ValueError):
    """Raised when a baseline cannot be summarized safely."""


def lane_name(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("baseline_")


def _int_field(value: object, *, path: Path, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineError(f"{path}: {field} must be an integer")
    return value


def summarize_baseline(path: Path) -> LaneSummary:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineError(f"{path}: baseline must be a JSON object")

    total_score = 0
    top_score = 0
    counts: dict[str, int] = {}
    for file_path, entry in payload.items():
        if not isinstance(file_path, str):
            raise BaselineError(f"{path}: baseline file keys must be strings")
        if not isinstance(entry, dict):
            raise BaselineError(f"{path}: {file_path} entry must be an object")
        if "score" not in entry:
            raise BaselineError(f"{path}: {file_path}.score is required")
        score = _int_field(entry["score"], path=path, field=f"{file_path}.score")
        total_score += score
        top_score = max(top_score, score)
        raw_counts = entry.get("counts", {})
        if not isinstance(raw_counts, dict):
            raise BaselineError(f"{path}: {file_path}.counts must be an object")
        for code, count in raw_counts.items():
            if not isinstance(code, str):
                raise BaselineError(f"{path}: {file_path}.counts keys must be strings")
            counts[code] = counts.get(code, 0) + _int_field(
                count, path=path, field=f"{file_path}.counts.{code}"
            )

    return LaneSummary(
        lane=lane_name(path),
        path=str(path),
        files=len(payload),
        total_score=total_score,
        top_score=top_score,
        counts=dict(sorted(counts.items())),
    )


def collect_summaries(patterns: Iterable[str]) -> list[LaneSummary]:
    paths = sorted({Path(match) for pattern in patterns for match in glob.glob(pattern)})
    if not paths:
        raise BaselineError("no baseline files matched")
    summaries = [summarize_baseline(path) for path in paths]
    return sorted(summaries, key=lambda item: (-item.total_score, item.lane))


def render_text(summaries: list[LaneSummary], *, top_counts: int) -> str:
    lines = ["lane files total_score top_score top_findings"]
    for item in summaries:
        top = sorted(item.counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top_counts]
        findings = ",".join(f"{code}:{count}" for code, count in top) or "-"
        lines.append(
            f"{item.lane} {item.files} {item.total_score} {item.top_score} {findings}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["tests/maturity_sweep/baseline_*.json"],
        help="baseline JSON glob(s) to summarize",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON summaries")
    parser.add_argument(
        "--top-counts",
        type=int,
        default=5,
        help="number of finding classes to show in text output",
    )
    args = parser.parse_args(argv)

    try:
        if args.top_counts < 0:
            raise BaselineError("--top-counts must be greater than or equal to 0")
        summaries = collect_summaries(args.patterns)
    except BaselineError as exc:
        print(f"maturity baseline summary: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(item) for item in summaries], indent=2, sort_keys=True))
    else:
        print(render_text(summaries, top_counts=args.top_counts), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
