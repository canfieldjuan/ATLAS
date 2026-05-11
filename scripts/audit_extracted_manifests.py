#!/usr/bin/env python3
"""Verify every extracted_*/manifest.json is consistent with the filesystem.

For each manifest:
  - Every mappings[i].source path exists under atlas_brain/.
  - Every mappings[i].target path exists in the package.
  - Every owned[i].target path exists in the package.
  - Synced pairs (mappings entries) are byte-identical between
    source and target. Drift here means sync_extracted.sh has not
    been run since the source was edited.

Exits 0 if all manifests are consistent. Exits 1 on any drift.

Usage:
    python scripts/audit_extracted_manifests.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _validate_path(
    rel: str, must_start_with: str, kind: str, idx: int
) -> str | None:
    """Return an error string if rel is unsafe or misplaced, else None.

    Rejects:
      - absolute paths
      - paths containing '..' segments (parent-dir traversal)
      - paths not anchored under `must_start_with`
    """
    if rel.startswith("/"):
        return f"{kind}[{idx}]: absolute path rejected ({rel!r})"
    parts = Path(rel).parts
    if ".." in parts:
        return f"{kind}[{idx}]: parent-dir traversal rejected ({rel!r})"
    if not rel.startswith(must_start_with):
        return (
            f"{kind}[{idx}]: path not under expected tree "
            f"(expected to start with {must_start_with!r}, got {rel!r})"
        )
    return None


def check_manifest(manifest_path: Path) -> list[str]:
    """Return a list of failure descriptions for this manifest."""
    failures: list[str] = []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [f"{manifest_path}: failed to parse JSON ({exc})"]

    package_dir = manifest_path.parent.name + "/"
    mappings = data.get("mappings", [])
    owned = data.get("owned", [])

    for i, entry in enumerate(mappings):
        src_rel = entry.get("source")
        tgt_rel = entry.get("target")
        if not src_rel or not tgt_rel:
            failures.append(
                f"mappings[{i}]: missing source or target ({entry!r})"
            )
            continue
        # Reject malformed / unsafe paths before any disk I/O.
        src_err = _validate_path(src_rel, "atlas_brain/", "mappings.source", i)
        tgt_err = _validate_path(tgt_rel, package_dir, "mappings.target", i)
        if src_err:
            failures.append(src_err)
        if tgt_err:
            failures.append(tgt_err)
        if src_err or tgt_err:
            continue
        src = REPO_ROOT / src_rel
        tgt = REPO_ROOT / tgt_rel
        if not src.exists():
            failures.append(f"mappings[{i}].source missing: {src_rel}")
        if not tgt.exists():
            failures.append(f"mappings[{i}].target missing: {tgt_rel}")
        if src.exists() and tgt.exists():
            try:
                if src.read_bytes() != tgt.read_bytes():
                    failures.append(
                        f"sync drift: {src_rel} != {tgt_rel} "
                        "(run extracted/_shared/scripts/sync_extracted.sh)"
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"mappings[{i}]: could not compare ({exc})"
                )

    for i, entry in enumerate(owned):
        tgt_rel = entry.get("target")
        if not tgt_rel:
            failures.append(f"owned[{i}]: missing target ({entry!r})")
            continue
        tgt_err = _validate_path(tgt_rel, package_dir, "owned.target", i)
        if tgt_err:
            failures.append(tgt_err)
            continue
        tgt = REPO_ROOT / tgt_rel
        if not tgt.exists():
            failures.append(f"owned[{i}].target missing: {tgt_rel}")

    return failures


def main() -> int:
    manifests = sorted(REPO_ROOT.glob("extracted_*/manifest.json"))
    if not manifests:
        print("No extracted_*/manifest.json files found.")
        return 1

    drift = False
    for mf in manifests:
        rel = mf.relative_to(REPO_ROOT)
        failures = check_manifest(mf)
        if not failures:
            data = json.loads(mf.read_text(encoding="utf-8"))
            n_map = len(data.get("mappings", []))
            n_own = len(data.get("owned", []))
            print(f"OK     {rel}  (mappings={n_map}, owned={n_own})")
        else:
            drift = True
            print(f"DRIFT  {rel}")
            for f in failures:
                print(f"  - {f}")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
