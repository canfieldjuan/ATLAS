"""Host factory for the Content Ops reasoning context provider.

PR #402 shipped the route-level ``reasoning_context_provider``
seam in ``extracted_content_pipeline/api/control_surfaces.py``
plus the bundle's per-request ``with_reasoning_context()``
derivation. This module is the host adapter the route mount
calls to obtain a configured provider (or ``None``).

Operators opt in by setting
``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
readable by the package's ``FileCampaignReasoningContextProvider``.
When unset the factory returns ``None`` -- no behavior change
for hosts that haven't opted in.

Loader exceptions resolve to ``None`` with a WARN log so a
malformed file never crashes the route mount.

Lives at ``atlas_brain/`` root with underscore prefix to dodge
the heavy ``atlas_brain.services`` import chain (same pattern as
``_content_ops_infrastructure.py`` / ``_content_ops_scope.py``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


_ENV_VAR = "ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH"


def build_content_ops_reasoning_context_provider(
    *,
    path_factory: Callable[[], str | None] | None = None,
    loader_factory: Callable[[str], Any] | None = None,
) -> Any | None:
    """Return the configured reasoning context provider, or
    ``None`` when the host hasn't opted in.

    Hosts opt in by setting
    ``ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH`` to a JSON file
    readable by ``FileCampaignReasoningContextProvider``. Failures
    (missing file, parse errors) resolve to ``None`` with a
    warning logged -- the reasoning provider is enrichment, a
    bad file must not block the entire Content Ops surface for
    all tenants.

    DI kwargs let tests stub the env-var read and the loader
    without touching the filesystem or the heavy
    ``extracted_content_pipeline.campaign_reasoning_data`` module.
    """

    path = (path_factory or _read_env_path)()
    if not path:
        return None
    if not Path(path).is_file():
        logger.warning(
            "Content Ops reasoning context path %s does not exist; "
            "provider stays unwired.",
            path,
        )
        return None

    loader = loader_factory or _default_loader
    try:
        return loader(path)
    except Exception as exc:
        logger.warning(
            "Failed to load Content Ops reasoning context from %s: %s",
            path,
            exc,
        )
        return None


def _read_env_path() -> str | None:
    """Read the env var; return ``None`` for unset / empty."""

    value = os.environ.get(_ENV_VAR)
    return value or None


def _default_loader(path: str) -> Any:
    """Lazy import the package's file-backed loader.

    Keeps this module light enough to import in dependency-light
    dev envs; the heavier
    ``extracted_content_pipeline.campaign_reasoning_data`` only
    loads when the env var is actually set.
    """

    from extracted_content_pipeline.campaign_reasoning_data import (
        load_campaign_reasoning_context_provider,
    )

    return load_campaign_reasoning_context_provider(path)


__all__ = ["build_content_ops_reasoning_context_provider"]
