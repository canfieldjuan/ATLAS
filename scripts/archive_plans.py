#!/usr/bin/env python3
"""Archive merged plan docs and keep a queryable index of them.

Plan docs ship *with* their PR, so every ``plans/PR-*.md`` on ``main`` is from a
merged PR -- the directory is an unstructured, ever-growing archive (877+ files),
which is a slow orientation tax: a fresh or compacted session that lists ``plans/``
faces the whole history with no navigation aid.

This tool gives the directory a lifecycle:

- ``archive``  move every ``plans/PR-*.md`` in the root into ``plans/archive/`` and
               regenerate ``plans/INDEX.md``. Run it on a branch off ``main`` (where
               all plans are merged); do not run it on an in-flight feature branch
               whose unmerged plan still lives in the root.
- ``index``    regenerate ``plans/INDEX.md`` from the archive without moving anything.
- ``check``    print a non-blocking warning when the root holds more than
               ``--threshold`` plan docs (a nudge to run ``archive``). Always exits 0
               so it never adds friction to a PR.

The moves are plain filesystem renames; commit with ``git add -A`` and git records
them as renames, preserving history.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIRNAME = "archive"
INDEX_FILENAME = "INDEX.md"
DEFAULT_THRESHOLD = 50

_LANE = re.compile(r"^Ownership lane:\s*(?P<lane>.+?)\s*$", re.MULTILINE)
_PHASE = re.compile(r"^Slice phase:\s*(?P<phase>.+?)\s*$", re.MULTILINE)
_TITLE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)


def root_plan_files(plans_dir: Path) -> list[Path]:
    """Return ``PR-*.md`` plan docs in the root (not under ``archive/``)."""
    return sorted(
        p for p in plans_dir.glob("PR-*.md") if p.is_file()
    )


def archived_plan_files(plans_dir: Path) -> list[Path]:
    """Return ``PR-*.md`` plan docs already under ``plans/archive/``."""
    archive = plans_dir / ARCHIVE_DIRNAME
    if not archive.is_dir():
        return []
    return sorted(p for p in archive.glob("PR-*.md") if p.is_file())


def plan_metadata(text: str) -> dict[str, str]:
    """Extract the title, ownership lane, and slice phase from a plan doc."""
    meta: dict[str, str] = {}
    title = _TITLE.search(text)
    if title:
        meta["title"] = title.group("title")
    lane = _LANE.search(text)
    if lane:
        meta["lane"] = lane.group("lane")
    phase = _PHASE.search(text)
    if phase:
        meta["phase"] = phase.group("phase")
    return meta


def _index_line(plan_path: Path) -> str:
    name = plan_path.stem
    meta = plan_metadata(plan_path.read_text(encoding="utf-8"))
    parts = [f"- [{name}]({ARCHIVE_DIRNAME}/{plan_path.name})"]
    tail: list[str] = []
    if meta.get("lane"):
        tail.append(f"lane: {meta['lane']}")
    if meta.get("phase"):
        tail.append(f"phase: {meta['phase']}")
    if tail:
        parts.append(" - " + " | ".join(tail))
    return "".join(parts)


def build_index(plans_dir: Path) -> str:
    """Build the markdown index of archived plan docs."""
    archived = archived_plan_files(plans_dir)
    lines = [
        "# Plan archive index",
        "",
        (
            f"{len(archived)} archived plan doc(s). Merged plan docs are moved under "
            f"`{ARCHIVE_DIRNAME}/` and listed below; once the archive sweep has run, "
            "the `plans/` root holds only in-flight slices."
        ),
        "",
    ]
    lines.extend(_index_line(p) for p in archived)
    return "\n".join(lines).rstrip() + "\n"


def write_index(plans_dir: Path) -> Path:
    index_path = plans_dir / INDEX_FILENAME
    index_path.write_text(build_index(plans_dir), encoding="utf-8")
    return index_path


def archive_plans(plans_dir: Path) -> list[Path]:
    """Move root ``PR-*.md`` plans into ``plans/archive/``. Returns moved paths.

    Refuses to move anything if a root plan's filename already exists under
    ``plans/archive/`` -- plan filenames are human-chosen slice names, not
    immutable PR numbers, so a reused name would otherwise silently overwrite
    (and destroy) the older archived plan. Raises ``FileExistsError`` listing
    the collisions; nothing is moved.
    """
    archive = plans_dir / ARCHIVE_DIRNAME
    archive.mkdir(exist_ok=True)
    plans = root_plan_files(plans_dir)
    collisions = sorted(p.name for p in plans if (archive / p.name).exists())
    if collisions:
        raise FileExistsError(
            "refusing to overwrite archived plan(s) with reused slice name(s): "
            + ", ".join(collisions)
        )
    moved: list[Path] = []
    for plan in plans:
        target = archive / plan.name
        plan.rename(target)
        moved.append(target)
    return moved


def over_threshold(plans_dir: Path, threshold: int) -> tuple[int, bool]:
    count = len(root_plan_files(plans_dir))
    return count, count > threshold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("archive", "index", "check"), help="action to perform"
    )
    parser.add_argument(
        "--plans-dir",
        type=Path,
        default=REPO_ROOT / "plans",
        help="plans directory (default: <repo>/plans)",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    plans_dir: Path = args.plans_dir
    if not plans_dir.is_dir():
        print(f"plans dir not found: {plans_dir}", file=sys.stderr)
        return 2

    if args.command == "archive":
        try:
            moved = archive_plans(plans_dir)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        index_path = write_index(plans_dir)
        print(f"archived {len(moved)} plan doc(s) -> {plans_dir / ARCHIVE_DIRNAME}")
        print(f"wrote {index_path}")
        return 0

    if args.command == "index":
        index_path = write_index(plans_dir)
        print(f"wrote {index_path} ({len(archived_plan_files(plans_dir))} plans)")
        return 0

    # check: non-blocking nudge, always exits 0
    count, over = over_threshold(plans_dir, args.threshold)
    if over:
        print(
            f"WARNING: {count} plan doc(s) in {plans_dir} root exceeds threshold "
            f"{args.threshold}; run `python scripts/archive_plans.py archive` to "
            "move merged plans into the archive."
        )
    else:
        print(f"OK: {count} plan doc(s) in root (threshold {args.threshold}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
