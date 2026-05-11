"""Fixture tests for scripts/audit_extracted_manifests.py.

Locks down the regression Copilot caught on PR #484: _validate_path
originally used `rel.startswith("/")` which only detects POSIX
absolute paths. Windows drive-letter ("C:\\...") and UNC
("\\\\srv\\share\\...") absolute paths were accepted, which could
let a malformed manifest escape the repo subtree during byte-compare.

The fix was PurePosixPath(rel).is_absolute() OR
PureWindowsPath(rel).is_absolute() for OS-agnostic detection. This
test continues to pin that contract.
"""
from __future__ import annotations

import pytest

from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_extracted_manifests")


def test_posix_absolute_path_rejected(auditor):
    """Happy negative: POSIX-style absolute path is rejected."""
    err = auditor._validate_path(
        "/etc/passwd", "atlas_brain/", "mappings.source", 0
    )
    assert err is not None
    assert "absolute path rejected" in err


def test_windows_absolute_path_rejected(auditor):
    """Regression for PR #484 Copilot catch: Windows drive-letter
    absolute paths must be rejected. The previous startswith("/")
    check missed these."""
    err = auditor._validate_path(
        "C:\\Windows\\System32", "atlas_brain/", "mappings.source", 0
    )
    assert err is not None
    assert "absolute path rejected" in err


def test_unc_absolute_path_rejected(auditor):
    """Regression for PR #484 Copilot catch: UNC paths
    (\\\\server\\share\\...) are absolute on Windows and must be
    rejected."""
    err = auditor._validate_path(
        "\\\\srv\\share\\foo", "atlas_brain/", "mappings.source", 0
    )
    assert err is not None
    assert "absolute path rejected" in err


def test_parent_dir_traversal_rejected(auditor):
    """Pathological input: `..` segments must be rejected."""
    err = auditor._validate_path(
        "../etc/passwd", "atlas_brain/", "mappings.source", 0
    )
    assert err is not None
    assert "parent-dir traversal rejected" in err


def test_off_tree_path_rejected(auditor):
    """A path that doesn't start with the required tree prefix
    must be flagged."""
    err = auditor._validate_path(
        "elsewhere/x.py", "atlas_brain/", "mappings.source", 0
    )
    assert err is not None
    assert "not under expected tree" in err


def test_happy_path_accepted(auditor):
    """A legit relative path under the expected tree returns None."""
    err = auditor._validate_path(
        "atlas_brain/services/b2b/legit.py",
        "atlas_brain/",
        "mappings.source",
        0,
    )
    assert err is None
