#!/usr/bin/env python3
"""Sync repository labels from the committed Atlas label manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


COLOR_RE = re.compile(r"^[0-9a-f]{6}$")
MAX_DESCRIPTION_LENGTH = 100


@dataclass(frozen=True)
class LabelSpec:
    name: str
    color: str
    description: str


@dataclass(frozen=True)
class LabelUpdate:
    current_name: str
    wanted: LabelSpec


@dataclass(frozen=True)
class SyncPlan:
    create: tuple[LabelSpec, ...]
    update: tuple[LabelUpdate, ...]
    unchanged: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.create or self.update)


def normalize_color(value: object, *, label_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label_name}: color must be a string")
    color = value.strip().lstrip("#").lower()
    if not COLOR_RE.fullmatch(color):
        raise ValueError(f"{label_name}: color must be a 6-digit hex value")
    return color


def label_spec_from_mapping(raw: object, *, source: str) -> LabelSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: label entries must be objects")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source}: label name must be a non-empty string")
    label_name = name.strip()
    description = raw.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ValueError(f"{label_name}: description must be a string")
    label_description = description.strip()
    if len(label_description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"{label_name}: description must be {MAX_DESCRIPTION_LENGTH} characters or fewer"
        )
    return LabelSpec(
        name=label_name,
        color=normalize_color(raw.get("color"), label_name=label_name),
        description=label_description,
    )


def load_manifest_text(text: str) -> tuple[LabelSpec, ...]:
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("label manifest must be a JSON array")

    labels = tuple(
        label_spec_from_mapping(entry, source=f"label manifest entry {index}")
        for index, entry in enumerate(parsed, start=1)
    )
    if not labels:
        raise ValueError("label manifest must define at least one label")

    seen: set[str] = set()
    duplicates: list[str] = []
    for label in labels:
        key = label.name.lower()
        if key in seen:
            duplicates.append(label.name)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate repo label definitions: {', '.join(sorted(set(duplicates)))}")
    return labels


def load_manifest(path: Path) -> tuple[LabelSpec, ...]:
    return load_manifest_text(path.read_text(encoding="utf-8"))


def load_repo_labels_text(text: str) -> dict[str, LabelSpec]:
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("repo labels JSON must be an array")

    labels: dict[str, LabelSpec] = {}
    for index, entry in enumerate(parsed, start=1):
        label = label_spec_from_mapping(entry, source=f"repo label entry {index}")
        labels[label.name.lower()] = label
    return labels


def plan_label_sync(
    manifest_labels: tuple[LabelSpec, ...],
    repo_labels: dict[str, LabelSpec],
) -> SyncPlan:
    create: list[LabelSpec] = []
    update: list[LabelSpec] = []
    unchanged: list[str] = []
    for wanted in manifest_labels:
        current = repo_labels.get(wanted.name.lower())
        if current is None:
            create.append(wanted)
            continue
        if (
            current.name != wanted.name
            or current.color != wanted.color
            or current.description != wanted.description
        ):
            update.append(LabelUpdate(current.name, wanted))
            continue
        unchanged.append(wanted.name)
    return SyncPlan(tuple(create), tuple(update), tuple(unchanged))


def run_gh_json(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "gh command failed"
        raise RuntimeError(detail)
    return proc.stdout


def load_live_repo_labels(repo_labels_json: Path | None) -> dict[str, LabelSpec]:
    if repo_labels_json is not None:
        return load_repo_labels_text(repo_labels_json.read_text(encoding="utf-8"))
    return load_repo_labels_text(
        run_gh_json(["label", "list", "--limit", "1000", "--json", "name,description,color"])
    )


def apply_plan(plan: SyncPlan) -> None:
    for label in plan.create:
        run_gh_json(
            [
                "label",
                "create",
                label.name,
                "--color",
                label.color,
                "--description",
                label.description,
            ]
        )
    for update in plan.update:
        label = update.wanted
        run_gh_json(
            [
                "label",
                "edit",
                update.current_name,
                "--name",
                label.name,
                "--color",
                label.color,
                "--description",
                label.description,
            ]
        )


def print_plan(plan: SyncPlan) -> None:
    if not plan.has_changes:
        print("GitHub labels already match .github/labels.json.")
        return
    if plan.create:
        print("Labels to create:")
        for label in plan.create:
            print(f"- {label.name}")
    if plan.update:
        print("Labels to update:")
        for update in plan.update:
            if update.current_name == update.wanted.name:
                print(f"- {update.wanted.name}")
            else:
                print(f"- {update.current_name} -> {update.wanted.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(".github/labels.json"))
    parser.add_argument(
        "--repo-labels-json",
        type=Path,
        default=None,
        help="Optional gh label list JSON fixture. Defaults to live gh label list.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail if live labels drift from the manifest.")
    mode.add_argument("--apply", action="store_true", help="Create or update live labels to match the manifest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_labels = load_manifest(args.manifest)
        repo_labels = load_live_repo_labels(args.repo_labels_json)
        plan = plan_label_sync(manifest_labels, repo_labels)
        print_plan(plan)
        if args.check:
            return 1 if plan.has_changes else 0
        apply_plan(plan)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
