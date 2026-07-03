"""Shared repository-root resolution for mechanical audit scripts."""

from __future__ import annotations

import os
from pathlib import Path


def audit_repo_root(script_file: str | Path) -> Path:
    """Return the tree inspected by audit scripts.

    Normal local execution inspects the checkout that owns the script. Trusted
    CI execution can run base-owned scripts against a separate PR checkout by
    setting ``ATLAS_AUDIT_REPO_ROOT``.
    """

    override = os.environ.get("ATLAS_AUDIT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(script_file).resolve().parent.parent
