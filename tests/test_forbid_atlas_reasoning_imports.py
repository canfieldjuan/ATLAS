"""Unit tests for ``extracted/_shared/scripts/forbid_atlas_reasoning_imports``.

PR-D7a's CI guard fails closed on any ``atlas_brain.reasoning`` import
in an extracted product. These tests pin:

- positive: every import form is detected
  (``from``, ``import``, ``importlib.import_module``, sub-paths,
   forms inside try/except / if/else / def -- the guard is *stricter*
   than ``forbid_hard_atlas_imports.py`` and rejects gated escapes too)
- negative: clean files emit no violations
- negative: documentation references in docstrings/comments don't fire
  (only real ``import`` syntax is flagged)
- negative: imports of unrelated atlas modules (e.g.
  ``atlas_brain.services``) don't fire -- only the
  ``atlas_brain.reasoning`` prefix is forbidden by this guard

The guard module's path makes it a non-package script; we load it via
:mod:`importlib.util` so the tests work regardless of how
``sys.path`` is configured in the standalone-CI runner.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest


_GUARD_PATH = (
    Path(__file__).resolve().parent.parent
    / "extracted"
    / "_shared"
    / "scripts"
    / "forbid_atlas_reasoning_imports.py"
)


def _load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "forbid_atlas_reasoning_imports", _GUARD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_guard = _load_guard()


def _scan(source: str, tmp_path: Path) -> list[tuple[int, str]]:
    """Helper: scan a snippet and return ``(lineno, target)`` pairs."""
    fp = tmp_path / "snippet.py"
    fp.write_text(textwrap.dedent(source).lstrip())
    return _guard.scan_file(fp)


# ----------------------------------------------------------------------
# Positive cases (each form must be detected)
# ----------------------------------------------------------------------


def test_from_atlas_reasoning_root_is_violation(tmp_path: Path) -> None:
    violations = _scan("from atlas_brain.reasoning import foo\n", tmp_path)
    assert len(violations) == 1
    assert "from atlas_brain.reasoning import" in violations[0][1]


def test_from_atlas_reasoning_subpath_is_violation(tmp_path: Path) -> None:
    violations = _scan(
        "from atlas_brain.reasoning.semantic_cache import compute_evidence_hash\n",
        tmp_path,
    )
    assert len(violations) == 1
    assert "from atlas_brain.reasoning.semantic_cache" in violations[0][1]


def test_import_atlas_reasoning_is_violation(tmp_path: Path) -> None:
    violations = _scan("import atlas_brain.reasoning\n", tmp_path)
    assert len(violations) == 1
    assert "import atlas_brain.reasoning" in violations[0][1]


def test_import_atlas_reasoning_subpath_is_violation(tmp_path: Path) -> None:
    violations = _scan("import atlas_brain.reasoning.archetypes\n", tmp_path)
    assert len(violations) == 1
    assert "atlas_brain.reasoning.archetypes" in violations[0][1]


def test_importlib_import_module_atlas_reasoning_is_violation(tmp_path: Path) -> None:
    violations = _scan(
        """
        import importlib
        src = importlib.import_module("atlas_brain.reasoning")
        """,
        tmp_path,
    )
    assert len(violations) == 1
    assert "atlas_brain.reasoning" in violations[0][1]


def test_importlib_import_module_subpath_is_violation(tmp_path: Path) -> None:
    violations = _scan(
        """
        import importlib
        x = importlib.import_module("atlas_brain.reasoning.graph")
        """,
        tmp_path,
    )
    assert len(violations) == 1


def test_bare_import_module_call_is_violation(tmp_path: Path) -> None:
    # ``from importlib import import_module`` then bare call.
    violations = _scan(
        """
        from importlib import import_module
        x = import_module("atlas_brain.reasoning.evidence_engine")
        """,
        tmp_path,
    )
    assert len(violations) == 1


def test_atlas_reasoning_inside_try_except_still_violates(tmp_path: Path) -> None:
    # Stricter than forbid_hard_atlas_imports.py -- gated atlas_brain
    # reads are allowed for OTHER atlas modules, but the reasoning
    # prefix has a ready core replacement so even gated escapes here
    # are out of contract.
    violations = _scan(
        """
        try:
            from atlas_brain.reasoning import foo
        except ImportError:
            foo = None
        """,
        tmp_path,
    )
    assert len(violations) == 1


def test_atlas_reasoning_inside_function_still_violates(tmp_path: Path) -> None:
    violations = _scan(
        """
        def loader():
            from atlas_brain.reasoning.archetypes import scoring
            return scoring
        """,
        tmp_path,
    )
    assert len(violations) == 1


def test_multiple_violations_same_file_all_reported(tmp_path: Path) -> None:
    violations = _scan(
        """
        from atlas_brain.reasoning import a
        import atlas_brain.reasoning.b
        """,
        tmp_path,
    )
    assert len(violations) == 2


# ----------------------------------------------------------------------
# Negative cases (must not fire)
# ----------------------------------------------------------------------


def test_clean_module_has_no_violations(tmp_path: Path) -> None:
    violations = _scan(
        """
        from extracted_reasoning_core.types import ArchetypeMatch
        from extracted_reasoning_core.api import score_archetypes
        """,
        tmp_path,
    )
    assert violations == []


def test_docstring_reference_to_atlas_reasoning_does_not_fire(tmp_path: Path) -> None:
    # The audit's PR 7 acceptance is about *runtime imports*, not
    # documentation references. The bridge module PR-D7a removes had
    # docstrings that mentioned the path -- those should not have
    # tripped the guard.
    violations = _scan(
        '''
        """This module used to delegate to atlas_brain.reasoning."""
        from extracted_reasoning_core.types import ArchetypeMatch
        # historical: from atlas_brain.reasoning import foo
        ''',
        tmp_path,
    )
    assert violations == []


def test_other_atlas_modules_do_not_fire(tmp_path: Path) -> None:
    # The guard is specific to atlas_brain.reasoning. The broader
    # forbid_hard_atlas_imports.py covers everything-else atlas; this
    # one only owns the reasoning prefix.
    violations = _scan(
        """
        from atlas_brain.services import foo
        import atlas_brain.config
        """,
        tmp_path,
    )
    assert violations == []


def test_atlas_reason_substring_does_not_fire(tmp_path: Path) -> None:
    # Defensive: a module named e.g. ``atlas_brain.reason_helpers``
    # (no dot before the ``_``) must not match. The guard checks for
    # exact equality or ``atlas_brain.reasoning.`` prefix.
    violations = _scan(
        "from atlas_brain.reason_helpers import x\n",
        tmp_path,
    )
    assert violations == []


def test_importlib_import_module_with_dynamic_arg_does_not_fire(tmp_path: Path) -> None:
    # When the import-module argument is a variable (not a string
    # literal), we can't statically tell what's being loaded. Skip --
    # this is a knowable limitation; runtime-import-checking is out of
    # scope. forbid_hard_atlas_imports.py has the same limitation.
    violations = _scan(
        """
        import importlib
        name = "atlas_brain.reasoning"
        x = importlib.import_module(name)
        """,
        tmp_path,
    )
    assert violations == []


# ----------------------------------------------------------------------
# scan_package + main entry-point sanity
# ----------------------------------------------------------------------


def test_scan_package_collects_violations_per_file(tmp_path: Path) -> None:
    pkg = tmp_path / "fake_product"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "clean.py").write_text(
        "from extracted_reasoning_core.api import score_archetypes\n"
    )
    (pkg / "dirty.py").write_text(
        "from atlas_brain.reasoning.archetypes import s\n"
    )

    findings = _guard.scan_package(pkg)
    assert (pkg / "dirty.py") in findings
    assert (pkg / "clean.py") not in findings
    assert (pkg / "__init__.py") not in findings


def test_main_returns_zero_on_clean_package(tmp_path: Path, capsys) -> None:
    pkg = tmp_path / "clean_product"
    pkg.mkdir()
    (pkg / "ok.py").write_text("from extracted_reasoning_core.api import x\n")

    rc = _guard_main([str(pkg)])
    assert rc == 0


def test_main_returns_one_on_dirty_package(tmp_path: Path) -> None:
    pkg = tmp_path / "dirty_product"
    pkg.mkdir()
    (pkg / "bad.py").write_text(
        "from atlas_brain.reasoning import x\n"
    )

    rc = _guard_main([str(pkg)])
    assert rc == 1


def test_main_returns_two_when_path_is_not_a_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "missing"
    rc = _guard_main([str(not_a_dir)])
    assert rc == 2


def _guard_main(argv: list[str]) -> int:
    """Invoke the guard's ``main()`` with a controlled argv."""
    saved = sys.argv
    sys.argv = ["forbid_atlas_reasoning_imports.py", *argv]
    try:
        return _guard.main()
    finally:
        sys.argv = saved
