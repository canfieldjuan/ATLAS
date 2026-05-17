from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "extracted_reasoning_core" / "manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _owned_targets() -> set[str]:
    return {entry["target"] for entry in _manifest().get("owned", [])}


def test_manifest_marks_reasoning_core_as_product_owned() -> None:
    manifest = _manifest()

    assert manifest["mappings"] == []
    assert "extracted_reasoning_core/api.py" in _owned_targets()
    assert "extracted_reasoning_core/types.py" in _owned_targets()
    assert "extracted_reasoning_core/ports.py" in _owned_targets()
    assert "extracted_reasoning_core/evidence_map.yaml" in _owned_targets()
    assert "extracted_reasoning_core/skills/registry.py" in _owned_targets()


def test_manifest_targets_exist() -> None:
    for target in _owned_targets():
        assert (ROOT / target).exists(), target


def test_manifest_python_targets_do_not_import_atlas_reasoning() -> None:
    for target in _owned_targets():
        path = ROOT / target
        if path.suffix != ".py":
            continue
        text = path.read_text()
        assert "from atlas_brain.reasoning" not in text
        assert "import atlas_brain.reasoning" not in text


def test_standalone_smoke_imports_manifest_modules_and_runs_reasoning() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_extracted_reasoning_core_standalone.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["claims"] == 1
    assert payload["imported_modules"] >= 18
    assert payload["summary"] == "Reasoning core standalone smoke passed."
