"""Regression test for the manifest-driven standalone smoke script.

Pins two contracts:

1. ``scripts/smoke_extracted_pipeline_standalone.py`` exits 0 in the
   current state -- every manifest-tracked Python target imports
   cleanly under ``EXTRACTED_PIPELINE_STANDALONE=1``.
2. The smoke's failure detection actually distinguishes a real
   decoupling failure from a 3rd-party env failure -- a synthetic
   missing-module pointing at ``extracted_*`` causes a non-zero
   exit; a synthetic pointing at a 3rd-party name does not.

See plans/PR-Audit-PipelineStandaloneSmoke-1.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke_extracted_pipeline_standalone.py"


def test_smoke_script_exists_and_is_executable():
    assert SMOKE_SCRIPT.exists(), f"missing: {SMOKE_SCRIPT}"
    # Should be readable Python source -- shebang line first.
    first_line = SMOKE_SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line!r}"


def test_smoke_passes_in_current_state():
    """Every manifest-tracked Python target imports cleanly under
    EXTRACTED_PIPELINE_STANDALONE=1."""
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        # Surface the smoke's own report on failure for debuggability.
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"smoke exited {result.returncode}; "
        f"see stdout above for the gate-breaking decoupling failures."
    )


def test_decoupling_failure_classifier():
    """Direct unit check on the classifier helper -- avoids relying
    on the smoke's full manifest walk for testing the gate logic."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        # Import the smoke module by file path since scripts/ is not a package.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_smoke_under_test", SMOKE_SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        is_decoupling = module._is_decoupling_failure

        assert is_decoupling(
            "ModuleNotFoundError: No module named "
            "'extracted_content_pipeline.services.brand_registry'"
        ), "should flag missing extracted_* module as decoupling failure"
        assert is_decoupling(
            "ModuleNotFoundError: No module named 'atlas_brain.services.foo'"
        ), "should flag missing atlas_brain.* module as decoupling failure"
        assert not is_decoupling(
            "ModuleNotFoundError: No module named 'httpx'"
        ), "should NOT flag missing 3rd-party package as decoupling failure"
        assert not is_decoupling(
            "ModuleNotFoundError: No module named 'pydantic'"
        ), "should NOT flag missing 3rd-party package as decoupling failure"
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_target_to_module_handles_init_files():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_smoke_under_test", SMOKE_SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert (
            module._target_to_module("extracted_content_pipeline/foo/bar.py")
            == "extracted_content_pipeline.foo.bar"
        )
        assert (
            module._target_to_module("extracted_content_pipeline/foo/__init__.py")
            == "extracted_content_pipeline.foo"
        )
    finally:
        sys.path.remove(str(ROOT / "scripts"))
