"""Fixture tests for scripts/audit_extracted_manifests.py."""
from __future__ import annotations

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_extracted_manifests")


def test_posix_absolute_path_rejected(auditor):
    err = auditor._validate_path(
        "/etc/passwd", "atlas_brain/", "mappings.source", 0
    )

    assert err is not None
    assert "absolute path rejected" in err


def test_windows_absolute_path_rejected(auditor):
    err = auditor._validate_path(
        "C:\\Windows\\System32", "atlas_brain/", "mappings.source", 0
    )

    assert err is not None
    assert "absolute path rejected" in err


def test_unc_absolute_path_rejected(auditor):
    err = auditor._validate_path(
        "\\\\srv\\share\\foo", "atlas_brain/", "mappings.source", 0
    )

    assert err is not None
    assert "absolute path rejected" in err


def test_parent_dir_traversal_rejected(auditor):
    err = auditor._validate_path(
        "../etc/passwd", "atlas_brain/", "mappings.source", 0
    )

    assert err is not None
    assert "parent-dir traversal rejected" in err


def test_off_tree_path_rejected(auditor):
    err = auditor._validate_path(
        "elsewhere/x.py", "atlas_brain/", "mappings.source", 0
    )

    assert err is not None
    assert "not under expected tree" in err


def test_happy_path_accepted(auditor):
    err = auditor._validate_path(
        "atlas_brain/services/b2b/legit.py",
        "atlas_brain/",
        "mappings.source",
        0,
    )

    assert err is None
