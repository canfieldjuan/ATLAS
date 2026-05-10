"""Pin the host's Content Ops reasoning context provider factory.

`atlas_brain/_content_ops_reasoning.py` is the host adapter the
route mount calls to obtain a configured
`CampaignReasoningContextProvider` (or `None`).

Test inventory (7 tests):

1. Env var unset returns `None` (default unwired path).
2. Env var set but file missing returns `None` and logs WARN
   (defensive against typo'd paths or unmounted volumes).
3. Loader exception (malformed file) returns `None` and logs
   WARN (must not crash route mount).
4. Valid path returns whatever the loader produces.
5. `path_factory` DI kwarg short-circuits the env-var read.
6. `path_factory` returning empty string equals returning
   `None` (the env-var coercion treats both as unset).
7. `loader_factory` DI kwarg short-circuits the lazy import
   of the package's file-backed loader.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from atlas_brain._content_ops_reasoning import (
    build_content_ops_reasoning_context_provider,
)


def test_env_var_unset_returns_none() -> None:
    """Default behavior when the host hasn't opted in -- no
    factory call should reach the loader."""

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: None,
    )
    assert provider is None


def test_missing_file_returns_none_and_logs_warn(
    tmp_path: Any, caplog: Any,
) -> None:
    """If an operator typos the path or mounts the wrong
    volume, we must not crash the route mount."""

    missing = tmp_path / "nope.json"
    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: str(missing),
    )

    assert provider is None
    assert any("does not exist" in rec.message for rec in caplog.records)


def test_loader_exception_returns_none_and_logs_warn(
    tmp_path: Any, caplog: Any,
) -> None:
    """Malformed JSON / loader-internal exceptions resolve to
    None with WARN, never propagate."""

    real_file = tmp_path / "bad.json"
    real_file.write_text("not valid json", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    def _raising_loader(_path: str) -> Any:
        raise RuntimeError("simulated parse failure")

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: str(real_file),
        loader_factory=_raising_loader,
    )

    assert provider is None
    assert any("Failed to load" in rec.message for rec in caplog.records)


def test_valid_path_returns_loaded_provider(tmp_path: Any) -> None:
    """The factory returns whatever the loader hands back when
    the path exists and the loader succeeds."""

    real_file = tmp_path / "ok.json"
    real_file.write_text("[]", encoding="utf-8")

    sentinel = object()

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: str(real_file),
        loader_factory=lambda _path: sentinel,
    )

    assert provider is sentinel


def test_path_factory_short_circuits_env_var_read(
    tmp_path: Any, monkeypatch: Any,
) -> None:
    """A test-supplied path_factory must take precedence over
    the env var so tests don't need to set environment state."""

    real_file = tmp_path / "ok.json"
    real_file.write_text("[]", encoding="utf-8")
    sentinel = object()

    monkeypatch.setenv(
        "ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH",
        "/should/never/be/read.json",
    )

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: str(real_file),
        loader_factory=lambda _path: sentinel,
    )
    assert provider is sentinel


def test_path_factory_empty_string_equals_unset() -> None:
    """The env-var read coerces empty -> None; the explicit
    factory mirror must do the same."""

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: "",
    )
    assert provider is None


def test_loader_factory_short_circuits_default_loader(
    tmp_path: Any,
) -> None:
    """The DI loader_factory replaces the lazy import of
    `extracted_content_pipeline.campaign_reasoning_data` so
    tests don't trigger the module load chain."""

    real_file = tmp_path / "ok.json"
    real_file.write_text("[]", encoding="utf-8")
    captured: dict[str, str] = {}

    def _capturing_loader(path: str) -> Any:
        captured["path"] = path
        return "loader-result"

    provider = build_content_ops_reasoning_context_provider(
        path_factory=lambda: str(real_file),
        loader_factory=_capturing_loader,
    )

    assert provider == "loader-result"
    assert captured["path"] == str(real_file)
