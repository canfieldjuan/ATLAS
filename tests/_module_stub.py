"""Importability-guarded stand-ins for optional heavy dependencies.

Several tests stub a dependency (notably ``asyncpg``) into ``sys.modules`` so
the production module they import does not require the real package. The
historic idiom --

    if "asyncpg" not in sys.modules:
        sys.modules["asyncpg"] = types.ModuleType("asyncpg")   # or setdefault

plants the fake whenever the dependency is not *currently imported*. In CI,
where the dependency IS installed, that shadows the real module for the rest
of the session and poisons sibling tests that need it. Worse, a bare
``ModuleType`` has ``__spec__ = None``, so a later
``importlib.util.find_spec(name)`` raises ``'<name>.__spec__ is not set'`` at
collection (this is exactly what broke test_atlas_content_ops_input_provider,
test_competitive_intelligence, and test_b2b_phase4_causality_gate under the
repo-wide unit backstop).

``stub_missing_module`` stubs only when the dependency genuinely cannot be
imported, and gives any stub it creates a real ``__spec__`` -- so it never
shadows an installed package and never trips ``find_spec``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Mapping, Optional


def stub_missing_module(
    name: str,
    *,
    attributes: Optional[Mapping[str, object]] = None,
) -> Optional[types.ModuleType]:
    """Ensure ``name`` is importable, stubbing a stand-in only if it is not.

    Returns the freshly-created stub module, or ``None`` when no stub was
    needed (the dependency is already present or genuinely importable). The
    ``None`` return lets callers stub a submodule (e.g. ``asyncpg.exceptions``)
    only in the same branch where the parent was stubbed.
    """
    if name in sys.modules:
        return None  # already present (real import or a prior stub); leave it
    try:
        if importlib.util.find_spec(name) is not None:
            return None  # real package importable; let the caller import it
    except (ImportError, ValueError):
        pass  # parent missing / unset spec -> treat as absent and stub it
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attr, value in (attributes or {}).items():
        setattr(module, attr, value)
    sys.modules[name] = module
    return module
