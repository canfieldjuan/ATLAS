"""Regression test for the manifest-driven compintel-standalone smoke.

Pins three contracts:

1. ``scripts/smoke_extracted_competitive_intelligence_standalone.py``
   exits 0 in the current state (all 4 phases green).
2. ``_load_modules`` returns the union of manifest entries (mappings +
   owned, filtering migrations + ``__init__.py``) and the
   ``_EXTRA_STANDALONE_SHIMS`` constant.
3. ``_load_owned_files`` returns only entries from manifest ``owned``
   (mappings are byte-synced from atlas_brain and may legitimately
   contain ``atlas_brain.`` text).

See plans/PR-Audit-ManifestDrivenSmokes-2.md.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = (
    ROOT / "scripts" / "smoke_extracted_competitive_intelligence_standalone.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("_compintel_smoke_under_test", SMOKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_script_exists():
    assert SMOKE_SCRIPT.exists(), f"missing: {SMOKE_SCRIPT}"
    first_line = SMOKE_SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), "missing shebang"


def test_smoke_passes_in_current_state():
    """All 4 phases (import, owner-verify, fallback-probe, atlas-scan)
    pass under the standalone toggle."""
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"compintel standalone smoke exited {result.returncode}; "
        f"see stdout/stderr above for the failing phase."
    )


def test_load_modules_includes_manifest_and_extras():
    smoke = _load_smoke_module()
    manifest = smoke._load_manifest()
    modules = smoke._load_modules(manifest)

    # Every extras entry must show up in the union.
    for shim in smoke._EXTRA_STANDALONE_SHIMS:
        assert shim in modules, f"extras shim missing from _load_modules: {shim}"

    # Every manifest .py target (excl migrations + __init__) shows up.
    expected_manifest_modules = {
        smoke._target_to_module(e["target"])
        for e in manifest.get("mappings", []) + manifest.get("owned", [])
        if e["target"].endswith(".py")
        and "/migrations/" not in e["target"]
        and not e["target"].endswith("/__init__.py")
    }
    for m in expected_manifest_modules:
        assert m in modules, f"manifest module missing from _load_modules: {m}"


def test_load_owned_files_excludes_mappings():
    smoke = _load_smoke_module()
    manifest = smoke._load_manifest()
    owned_files = smoke._load_owned_files(manifest)

    expected_owned = {
        ROOT / e["target"]
        for e in manifest.get("owned", [])
        if e["target"].endswith(".py")
    }
    assert set(owned_files) == expected_owned

    # And concretely: vendor_briefing_delivery is a mapping and must
    # NOT be in owned-files (this was the conceptual bug in the
    # pre-PR hardcoded list).
    mapping_path = ROOT / "extracted_competitive_intelligence/services/b2b/vendor_briefing_delivery.py"
    assert mapping_path not in set(owned_files), (
        "owned-files scan must not include manifest mappings"
    )


def test_target_to_module_handles_init_files():
    smoke = _load_smoke_module()
    pkg = smoke.PACKAGE
    assert smoke._target_to_module(f"{pkg}/foo/bar.py") == f"{pkg}.foo.bar"
    assert smoke._target_to_module(f"{pkg}/foo/__init__.py") == f"{pkg}.foo"


def test_extras_constant_is_documented_as_methodology_gap():
    """The _EXTRA_STANDALONE_SHIMS constant exists because some
    standalone shims aren't yet on the manifest. This is a known
    methodology gap. If the constant changes, the plan should
    reflect the new state."""
    smoke = _load_smoke_module()
    # Just assert the constant is non-empty -- any change to its
    # contents should be deliberate and reviewed.
    assert len(smoke._EXTRA_STANDALONE_SHIMS) >= 1
    # And every entry must be a string starting with the package name.
    for entry in smoke._EXTRA_STANDALONE_SHIMS:
        assert isinstance(entry, str)
        assert entry.startswith(smoke.PACKAGE + ".")
