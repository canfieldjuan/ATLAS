"""Regression tests for the manifest-driven default-mode smokes.

Pins three contracts:

1. Both refactored smokes
   (``smoke_extracted_pipeline_imports.py`` and
   ``smoke_extracted_competitive_intelligence_imports.py``)
   exit 0 in the current state -- no decoupling failures across
   either package's manifest.
2. The smoke's failure classifier correctly distinguishes a real
   decoupling failure (``ModuleNotFoundError`` for ``extracted_*``
   or ``atlas_brain``) from an env failure (any other exception).
3. The ``_target_to_module`` helper handles ``__init__.py`` correctly.

See plans/PR-Audit-ManifestDrivenSmokes-1.md.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


SMOKES = {
    "pipeline": ROOT / "scripts" / "smoke_extracted_pipeline_imports.py",
    "compintel": ROOT / "scripts" / "smoke_extracted_competitive_intelligence_imports.py",
}


def _load_smoke_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_smoke_under_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("smoke_name", list(SMOKES.keys()))
def test_smoke_script_exists(smoke_name):
    path = SMOKES[smoke_name]
    assert path.exists(), f"missing: {path}"
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang in {path.name}"


@pytest.mark.parametrize("smoke_name", list(SMOKES.keys()))
def test_smoke_passes_in_current_state(smoke_name):
    """No decoupling failures across the manifest. Env failures are
    permitted (warning only) and don't break the gate."""
    path = SMOKES[smoke_name]
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"{smoke_name} smoke exited {result.returncode}; "
        f"see stdout/stderr above for the gate-breaking decoupling failures."
    )


@pytest.mark.parametrize("smoke_name", list(SMOKES.keys()))
def test_decoupling_failure_classifier(smoke_name):
    module = _load_smoke_module(SMOKES[smoke_name])
    is_decoupling = module._is_decoupling_failure

    # Construct ModuleNotFoundError with a `name` attribute -- Python's
    # real ImportError populates `name` on relative-import failures.
    extracted_failure = ModuleNotFoundError(
        "No module named 'extracted_content_pipeline.services.brand_registry'",
        name="extracted_content_pipeline.services.brand_registry",
    )
    atlas_failure = ModuleNotFoundError(
        "No module named 'atlas_brain.services.foo'",
        name="atlas_brain.services.foo",
    )
    third_party_failure = ModuleNotFoundError("No module named 'httpx'", name="httpx")
    other_exception = ValueError("not an import error at all")

    assert is_decoupling(extracted_failure), (
        "missing extracted_* module should be a decoupling failure"
    )
    assert is_decoupling(atlas_failure), (
        "missing atlas_brain.* module should be a decoupling failure"
    )
    assert not is_decoupling(third_party_failure), (
        "missing 3rd-party module should NOT be a decoupling failure"
    )
    assert not is_decoupling(other_exception), (
        "non-ImportError exception should NOT be a decoupling failure"
    )


@pytest.mark.parametrize("smoke_name", list(SMOKES.keys()))
def test_target_to_module_handles_init_files(smoke_name):
    module = _load_smoke_module(SMOKES[smoke_name])
    pkg = module.PACKAGE
    assert module._target_to_module(f"{pkg}/foo/bar.py") == f"{pkg}.foo.bar"
    assert module._target_to_module(f"{pkg}/foo/__init__.py") == f"{pkg}.foo"


@pytest.mark.parametrize("smoke_name", list(SMOKES.keys()))
def test_load_modules_skips_migrations_and_init(smoke_name):
    module = _load_smoke_module(SMOKES[smoke_name])
    modules = module._load_modules()
    assert modules, "expected at least one Python target from the manifest"
    for name in modules:
        assert "migrations" not in name, f"migrations should be filtered: {name}"
        assert not name.endswith(".__init__"), f"__init__ should be filtered: {name}"
        # Every module name should start with the package name
        assert name.startswith(module.PACKAGE + "."), (
            f"{name} does not belong to {module.PACKAGE}"
        )
