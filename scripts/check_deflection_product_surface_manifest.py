#!/usr/bin/env python3
"""Validate the non-Python deflection product surface manifest."""

import argparse
import json
from pathlib import Path


DEFAULT_MANIFEST = "tests/maturity_sweep/deflection_product_surface_manifest.json"


def _posix(path):
    return Path(path).as_posix()


def _load_manifest(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit("manifest not found: %s" % path)
    except json.JSONDecodeError as exc:
        raise SystemExit("manifest is not valid JSON: %s: %s" % (path, exc))
    patterns = raw.get("patterns")
    files = raw.get("files")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise SystemExit("manifest patterns must be a string array")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        raise SystemExit("manifest files must be a string array")
    return patterns, files


def _discover(patterns):
    discovered = set()
    for pattern in patterns:
        matches = [path for path in Path(".").glob(pattern) if path.is_file()]
        discovered.update(_posix(path) for path in matches)
    return sorted(discovered)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the non-Python deflection product surface manifest."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    patterns, expected = _load_manifest(args.manifest)
    discovered = _discover(patterns)
    expected_set = set(expected)
    discovered_set = set(discovered)

    missing = sorted(expected_set - discovered_set)
    untracked = sorted(discovered_set - expected_set)
    if missing or untracked:
        if missing:
            print("manifest lists missing file(s):")
            for path in missing:
                print("- %s" % path)
        if untracked:
            print("manifest is missing discovered file(s):")
            for path in untracked:
                print("- %s" % path)
        print()
        print("Update %s if this product surface change is intentional." % args.manifest)
        return 1

    print("deflection product surface manifest ok: %d file(s)" % len(discovered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
