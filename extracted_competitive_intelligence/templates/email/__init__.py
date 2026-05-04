"""Phase 1 package bridge: lazily exposes names from atlas_brain.templates.email.

PEP 562 ``__getattr__`` resolves ``from PACKAGE import some_name`` at
runtime. Product-owned submodules resolve from this package first; other
names delegate to the atlas_brain peer package's __init__ namespace.
This avoids triggering the atlas_brain peer's heavy import chain at
scaffold-load time -- e.g., importing
``extracted_competitive_intelligence.services.vendor_registry`` no
longer eagerly loads ``atlas_brain.services`` (which pulls in the
torch/llm chain) just to satisfy a hypothetical
``from ...services import llm_registry`` runtime fallback.

Phase 2 replaces this with a standalone implementation gated on
EXTRACTED_COMP_INTEL_STANDALONE=1.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


_PACKAGE_DIR = Path(__file__).resolve().parent
_PRODUCT_ROOT = _PACKAGE_DIR.parents[1]
_PACKAGE_TARGET_PREFIX = __name__.replace(".", "/") + "/"


@lru_cache(maxsize=1)
def _owned_submodule_names() -> frozenset[str]:
    manifest_path = _PRODUCT_ROOT / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return frozenset()

    submodules: set[str] = set()
    for entry in manifest.get("owned", []):
        if not isinstance(entry, dict):
            continue
        target = entry.get("target")
        if not isinstance(target, str):
            continue
        if not target.startswith(_PACKAGE_TARGET_PREFIX):
            continue
        path = Path(target)
        if path.suffix == ".py" and path.stem != "__init__":
            submodules.add(path.stem)
    return frozenset(submodules)


def _load_local_submodule(name: str) -> Any | None:
    module_name = f"{__name__}.{name}"
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


def _load_owned_submodule_attr(name: str) -> Any | None:
    for submodule_name in sorted(_owned_submodule_names()):
        submodule = importlib.import_module(f"{__name__}.{submodule_name}")
        if hasattr(submodule, name):
            return getattr(submodule, name)
    return None


def __getattr__(name: str) -> Any:
    local_submodule = _load_local_submodule(name)
    if local_submodule is not None:
        return local_submodule
    owned_attr = _load_owned_submodule_attr(name)
    if owned_attr is not None:
        return owned_attr
    if os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
        raise AttributeError(
            f"module {__name__!r} has no standalone attribute {name!r}; "
            "import an extracted template module explicitly"
        )
    src = importlib.import_module("atlas_brain.templates.email")
    try:
        return getattr(src, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
