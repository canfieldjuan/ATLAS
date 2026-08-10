#!/usr/bin/env python3
"""Shared root-trace parsing for fix-mode hook and PR-body audit."""
from __future__ import annotations

import os
import re
from pathlib import Path

PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`|([^,\s]+)")
PLACEHOLDER_TOKENS = {
    "",
    "none",
    "n/a",
    "na",
    "tbd",
    "to" "do",
    "unknown",
    "?",
    "-",
    "--",
    "...",
}
TEMPLATE_ENDPOINTS = {
    "symptom",
    "intermediate cause",
    "root cause",
    "upstream source",
    "source",
}


def is_placeholder_text(value: object) -> bool:
    return not isinstance(value, str) or value.strip().lower() in PLACEHOLDER_TOKENS


def trace_endpoint_is_valid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    endpoint = value.strip()
    normalized = " ".join(endpoint.lower().strip("<>{}[]()").split())
    if normalized in PLACEHOLDER_TOKENS or normalized in TEMPLATE_ENDPOINTS:
        return False
    tokens = re.findall(r"\w+", normalized)
    if tokens and tokens[0] in {"tbd", "todo"}:
        return False
    if endpoint.startswith(("<", "{")) or endpoint.endswith((">", "}")):
        return False
    return any(ch.isalnum() for ch in endpoint)


def source_trace_is_valid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parts = [part.strip() for part in value.split("->")]
    if len(parts) < 2:
        return False
    return all(trace_endpoint_is_valid(part) for part in parts)


def normalize_repo_path(path: str, project_dir: str | os.PathLike[str] | None = None) -> str:
    candidate = path.replace("\\", "/")
    if project_dir is not None and (os.path.isabs(path) or os.path.isabs(candidate)):
        candidate = os.path.relpath(path, project_dir)
    return os.path.normpath(candidate).replace("\\", "/")


def parse_repo_path_tokens(value: str) -> set[str]:
    paths: set[str] = set()
    for match in PATH_TOKEN_RE.finditer(value.replace(",", " ")):
        raw = (match.group(1) or match.group(2) or "").strip().strip("`")
        if is_placeholder_text(raw):
            continue
        normalized = normalize_repo_path(raw)
        parts = Path(normalized).parts
        if (
            not normalized
            or normalized.startswith("/")
            or normalized in {".", ".."}
            or ".." in parts
            or normalized.startswith("../")
        ):
            continue
        paths.add(normalized)
    return paths
