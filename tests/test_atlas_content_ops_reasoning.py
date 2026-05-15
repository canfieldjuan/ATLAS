"""Pin the host's Content Ops reasoning context provider factories.

`atlas_brain/_content_ops_reasoning.py` is the host adapter the
route mount calls to obtain a configured
`CampaignReasoningContextProvider` (or `None`).

Test inventory:

File-backed factory (PR #462, 7 tests):

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

DB-backed factory (7 tests):

8.  Env var unset returns `None` (default unwired path).
9.  Settings-backed DB opt-in is honored when the legacy env var
    is unset.
10. Legacy env var overrides settings-backed opt-in.
11. B2B campaign config exposes the DB opt-in flag.
12. Env var set + pool present returns whatever the
    repository factory produces.
13. Env var set but pool factory returns `None` (DB not
    initialized yet) returns `None` with WARN.
14. Repository factory exception resolves to `None` with WARN
    (defensive against an upstream init bug).

Intervention-report factory (7 tests):

15. Env var unset returns `None`.
16. Env var set + pool present returns whatever the provider
    factory produces.
17. Env var set but pool factory returns `None` returns `None`
    with WARN.
18. Env var set but pool is uninitialized returns `None` with
    WARN.
19. Provider factory exception resolves to `None` with WARN.
20. Matching report normalizes into campaign reasoning context.
21. Missing selectors return `None`.

Chooser (4 tests):

22. DB > file > `None` priority: when DB factory returns a
    provider, the file factory is never called.
23. DB returns `None`, file factory returns a provider --
    chooser returns the file provider.
24. DB returns `None`, intervention returns a provider --
    chooser returns the intervention provider before file.
25. All factories return `None` -- chooser returns `None`.

Status (4 tests):

26. DB provider reports configured with source=db.
27. Intervention fallback reports configured with source=intervention.
28. File fallback reports configured with source=file.
29. Neither provider reports configured=false with source=none.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from atlas_brain._content_ops_reasoning import (
    InterventionReportCampaignReasoningProvider,
    build_content_ops_reasoning_context_provider,
    build_intervention_content_ops_reasoning_context_provider,
    build_postgres_content_ops_reasoning_context_provider,
    describe_content_ops_reasoning_context_provider,
    _read_db_enabled,
    select_content_ops_reasoning_context_provider,
)
from atlas_brain.config import B2BCampaignConfig
from extracted_content_pipeline.campaign_ports import TenantScope


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


# ---------- DB-backed factory ------------------------------------------------


def test_db_factory_disabled_returns_none() -> None:
    """Default behavior when the host hasn't opted in -- no
    pool acquire, no repository construction."""

    provider = build_postgres_content_ops_reasoning_context_provider(
        enabled_factory=lambda: False,
    )
    assert provider is None


def test_db_enabled_reads_settings_when_legacy_env_unset(
    monkeypatch: Any,
) -> None:
    """Settings integration: the host can enable the DB provider
    through b2b_campaign config when the legacy top-level env var
    is absent."""

    monkeypatch.delenv("ATLAS_CONTENT_OPS_REASONING_DB_ENABLED", raising=False)
    monkeypatch.setattr(
        "atlas_brain._content_ops_reasoning._settings_db_enabled",
        lambda: True,
    )

    assert _read_db_enabled() is True


def test_db_enabled_legacy_env_overrides_settings(monkeypatch: Any) -> None:
    """The old env var remains the explicit operator override."""

    monkeypatch.setenv("ATLAS_CONTENT_OPS_REASONING_DB_ENABLED", "false")
    monkeypatch.setattr(
        "atlas_brain._content_ops_reasoning._settings_db_enabled",
        lambda: True,
    )

    assert _read_db_enabled() is False


def test_b2b_campaign_config_exposes_content_ops_reasoning_db_flag() -> None:
    cfg = B2BCampaignConfig(content_ops_reasoning_db_enabled=True)
    assert cfg.content_ops_reasoning_db_enabled is True


def test_db_factory_enabled_returns_repository() -> None:
    """When opted in + a pool is available, the factory returns
    whatever the repository factory hands back."""

    sentinel = object()

    provider = build_postgres_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: "fake-pool",
        repository_factory=lambda pool: sentinel,
    )
    assert provider is sentinel


def test_db_factory_missing_pool_returns_none_and_warns(
    caplog: Any,
) -> None:
    """Enabled but the host's pool isn't initialized yet --
    must not crash; resolves to `None` with WARN so the route
    mount stays unwired."""

    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    provider = build_postgres_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: None,
    )

    assert provider is None
    assert any("pool is not" in rec.message for rec in caplog.records)


def test_db_factory_uninitialized_pool_returns_none_and_warns(
    caplog: Any,
) -> None:
    """Codex P2: ``DatabasePool`` is a wrapper that always
    returns non-None from ``get_db_pool()`` even before
    ``initialize()``. Without an ``is_initialized`` check the
    chooser would treat DB as usable, skip the file fallback,
    and the first request would raise ``RuntimeError`` from
    ``DatabasePool.fetchrow``. Verify the guard."""

    class _UninitializedPool:
        is_initialized = False

    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    provider = build_postgres_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: _UninitializedPool(),
    )

    assert provider is None
    assert any("not initialized" in rec.message for rec in caplog.records)


def test_db_factory_repository_exception_returns_none_and_warns(
    caplog: Any,
) -> None:
    """Repository construction failures (an upstream init bug,
    a schema mismatch in dev) resolve to `None` with WARN
    rather than propagating into the route mount."""

    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    def _raising_factory(_pool: Any) -> Any:
        raise RuntimeError("simulated repo init failure")

    provider = build_postgres_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: "fake-pool",
        repository_factory=_raising_factory,
    )

    assert provider is None
    assert any("Failed to construct" in rec.message for rec in caplog.records)


# ---------- Intervention-report factory --------------------------------------


def test_intervention_factory_disabled_returns_none() -> None:
    provider = build_intervention_content_ops_reasoning_context_provider(
        enabled_factory=lambda: False,
    )
    assert provider is None


def test_intervention_factory_enabled_returns_provider() -> None:
    sentinel = object()

    provider = build_intervention_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: "fake-pool",
        provider_factory=lambda pool: sentinel,
    )

    assert provider is sentinel


def test_intervention_factory_missing_pool_returns_none_and_warns(
    caplog: Any,
) -> None:
    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    provider = build_intervention_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: None,
    )

    assert provider is None
    assert any("pool is not" in rec.message for rec in caplog.records)


def test_intervention_factory_uninitialized_pool_returns_none_and_warns(
    caplog: Any,
) -> None:
    class _UninitializedPool:
        is_initialized = False

    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    provider = build_intervention_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: _UninitializedPool(),
    )

    assert provider is None
    assert any("not initialized" in rec.message for rec in caplog.records)


def test_intervention_factory_provider_exception_returns_none_and_warns(
    caplog: Any,
) -> None:
    caplog.set_level(logging.WARNING, logger="atlas_brain._content_ops_reasoning")

    def _raise(_pool: Any) -> object:
        raise RuntimeError("boom")

    provider = build_intervention_content_ops_reasoning_context_provider(
        enabled_factory=lambda: True,
        pool_factory=lambda: "fake-pool",
        provider_factory=_raise,
    )

    assert provider is None
    assert any("Failed to construct" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_intervention_provider_reads_matching_report_as_reasoning_context() -> None:
    class _Pool:
        def __init__(self) -> None:
            self.args: tuple[Any, ...] | None = None

        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
            self.args = args
            assert "FROM intelligence_reports" in query
            return {
                "id": "report-1",
                "entity_name": "Acme",
                "entity_type": "company",
                "report_type": "intervention",
                "time_window_days": 7,
                "report_text": "Use a proof-led migration narrative.",
                "structured_data": {
                    "pipeline_id": "pipe-1",
                    "stages_completed": 2,
                    "stages_total": 3,
                    "stage_statuses": {
                        "playbook": "completed",
                    },
                    "safety_warnings": ["Narrative stage requires approval."],
                },
                "pressure_snapshot": {"pressure_score": 7},
                "requested_by": "api",
                "created_at": "2026-05-14T19:00:00Z",
            }

    pool = _Pool()
    provider = InterventionReportCampaignReasoningProvider(pool)

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="target-1",
        target_mode="vendor_retention",
        opportunity={"company_name": "Acme"},
    )

    assert context is not None
    assert pool.args is not None
    assert pool.args[0] == "intervention"
    assert "acme" in pool.args[1]
    assert context.canonical_reasoning["wedge"] == "intervention"
    assert context.top_theses[0]["claim"] == "Latest intervention analysis for Acme"
    assert context.proof_points[0]["label"] == "playbook status"
    assert context.proof_points[-1]["label"] == "pressure_score"
    assert context.reference_ids["intelligence_reports"] == ("report-1",)
    assert context.reference_ids["intervention_pipeline"] == ("pipe-1",)
    assert context.coverage_limits == ("Narrative stage requires approval.",)


@pytest.mark.asyncio
async def test_intervention_provider_returns_none_without_selector_match() -> None:
    class _Pool:
        async def fetchrow(self, _query: str, *_args: Any) -> None:
            return None

    provider = InterventionReportCampaignReasoningProvider(_Pool())

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="",
        target_mode="vendor_retention",
        opportunity={"company_name": ""},
    )

    assert context is None


# ---------- Chooser ----------------------------------------------------------


def test_chooser_prefers_db_provider() -> None:
    """DB > file priority: when the DB factory returns a
    provider, the file factory must not be called."""

    db_sentinel = object()
    file_called = {"hit": False}

    def _file_factory() -> Any:
        file_called["hit"] = True
        return "file-provider"

    provider = select_content_ops_reasoning_context_provider(
        db_factory=lambda: db_sentinel,
        intervention_factory=lambda: "intervention-provider",
        file_factory=_file_factory,
    )

    assert provider is db_sentinel
    assert file_called["hit"] is False


def test_chooser_falls_back_to_file_when_db_none() -> None:
    """DB returns `None` -- chooser advances to the file
    factory and returns whatever it produces."""

    file_sentinel = object()

    provider = select_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: None,
        file_factory=lambda: file_sentinel,
    )

    assert provider is file_sentinel


def test_chooser_uses_intervention_before_file() -> None:
    file_called = {"hit": False}
    intervention_sentinel = object()

    def _file_factory() -> Any:
        file_called["hit"] = True
        return "file-provider"

    provider = select_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: intervention_sentinel,
        file_factory=_file_factory,
    )

    assert provider is intervention_sentinel
    assert file_called["hit"] is False


def test_chooser_returns_none_when_neither_configured() -> None:
    """No DB, no file -- chooser returns `None` and the
    bundle's `with_reasoning_context()` derivation falls back
    to zero-context defaults."""

    provider = select_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: None,
        file_factory=lambda: None,
    )

    assert provider is None


# ---------- Status -----------------------------------------------------------


def test_reasoning_status_reports_db_provider() -> None:
    file_called = {"hit": False}

    def _file_factory() -> Any:
        file_called["hit"] = True
        return "file-provider"

    status = describe_content_ops_reasoning_context_provider(
        db_factory=lambda: object(),
        intervention_factory=lambda: "intervention-provider",
        file_factory=_file_factory,
    )

    assert status == {"configured": True, "source": "db"}
    assert file_called["hit"] is False


def test_reasoning_status_reports_file_fallback() -> None:
    status = describe_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: None,
        file_factory=lambda: object(),
    )

    assert status == {"configured": True, "source": "file"}


def test_reasoning_status_reports_intervention_provider() -> None:
    file_called = {"hit": False}

    def _file_factory() -> Any:
        file_called["hit"] = True
        return "file-provider"

    status = describe_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: object(),
        file_factory=_file_factory,
    )

    assert status == {"configured": True, "source": "intervention"}
    assert file_called["hit"] is False


def test_reasoning_status_reports_none_when_unconfigured() -> None:
    status = describe_content_ops_reasoning_context_provider(
        db_factory=lambda: None,
        intervention_factory=lambda: None,
        file_factory=lambda: None,
    )

    assert status == {"configured": False, "source": "none"}
