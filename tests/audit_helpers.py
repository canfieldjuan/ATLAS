"""Shared loader for audit scripts under scripts/."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_auditor(name: str) -> ModuleType:
    """Return the auditor module named `<name>` from scripts/."""
    if name in sys.modules:
        return sys.modules[name]

    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():
        raise AssertionError(f"required auditor missing: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not build import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
