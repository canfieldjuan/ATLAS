"""Tests for the standalone substrate's settings surface (PR-A6a).

Pins the contract that ``settings.provider_cost`` is fully populated
when the standalone toggle is on, so ``services/provider_cost_sync.py``
and ``services/cost/openai_billing.py`` resolve their settings reads
without falling back to atlas_brain.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Iterator

import pytest


_STANDALONE_ENV_VAR = "EXTRACTED_LLM_INFRA_STANDALONE"


def _reload_settings_in_standalone_mode(monkeypatch, **env_overrides) -> object:
    """Force a fresh import of the standalone settings module under
    ``EXTRACTED_LLM_INFRA_STANDALONE=1`` plus the given env overrides.

    Returns the freshly-imported ``LLMInfraSettings`` instance
    (``settings`` global on the module).
    """
    monkeypatch.setenv(_STANDALONE_ENV_VAR, "1")
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    for mod_name in (
        "extracted_llm_infrastructure._standalone.config",
        "extracted_llm_infrastructure.config",
    ):
        sys.modules.pop(mod_name, None)

    config_mod = importlib.import_module("extracted_llm_infrastructure.config")
    return config_mod.settings


@pytest.fixture
def fresh_standalone_settings(monkeypatch) -> Iterator[object]:
    """Yields the standalone settings global. Cleans up sys.modules
    afterward so other tests get a fresh import too."""
    settings = _reload_settings_in_standalone_mode(monkeypatch)
    try:
        yield settings
    finally:
        for mod_name in (
            "extracted_llm_infrastructure._standalone.config",
            "extracted_llm_infrastructure.config",
        ):
            sys.modules.pop(mod_name, None)


# ---- Default values (mirror atlas_brain.config.ProviderCostConfig) ----


def test_provider_cost_namespace_exists(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert hasattr(settings, "provider_cost"), (
        "LLMInfraSettings is missing the provider_cost field"
    )


def test_provider_cost_enabled_default_false(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.enabled is False


def test_provider_cost_interval_seconds_default(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.interval_seconds == 3600


def test_provider_cost_sync_timeout_seconds_default(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.sync_timeout_seconds == 20


def test_provider_cost_snapshot_retention_days_default(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.snapshot_retention_days == 90


def test_provider_cost_daily_retention_days_default(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.daily_retention_days == 365


def test_provider_cost_openrouter_enabled_default_true(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.openrouter_enabled is True


def test_provider_cost_openrouter_api_key_default_empty(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.openrouter_api_key == ""


def test_provider_cost_anthropic_enabled_default_false(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.anthropic_enabled is False


def test_provider_cost_anthropic_admin_api_key_default_empty(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.anthropic_admin_api_key == ""


def test_provider_cost_anthropic_lookback_days_default(fresh_standalone_settings):
    settings = fresh_standalone_settings
    assert settings.provider_cost.anthropic_lookback_days == 7


# ---- Env-var overrides (ATLAS_PROVIDER_COST_*) ----


def test_provider_cost_enabled_env_override(monkeypatch):
    settings = _reload_settings_in_standalone_mode(
        monkeypatch, ATLAS_PROVIDER_COST_ENABLED="true"
    )
    assert settings.provider_cost.enabled is True


def test_provider_cost_anthropic_lookback_days_env_override(monkeypatch):
    settings = _reload_settings_in_standalone_mode(
        monkeypatch, ATLAS_PROVIDER_COST_ANTHROPIC_LOOKBACK_DAYS="14"
    )
    assert settings.provider_cost.anthropic_lookback_days == 14


def test_provider_cost_openrouter_api_key_env_override(monkeypatch):
    settings = _reload_settings_in_standalone_mode(
        monkeypatch, ATLAS_PROVIDER_COST_OPENROUTER_API_KEY="sk-xxx"
    )
    assert settings.provider_cost.openrouter_api_key == "sk-xxx"


# ---- provider_cost_sync.py call sites resolve in standalone mode ----


def test_provider_cost_sync_settings_reads_resolve(monkeypatch):
    """``services/provider_cost_sync.py`` reads
    ``settings.provider_cost.snapshot_retention_days`` and
    ``...daily_retention_days``. PR-A6a's whole point is that those
    two reads no longer fall back to atlas mode."""
    settings = _reload_settings_in_standalone_mode(monkeypatch)
    snapshot_days = int(settings.provider_cost.snapshot_retention_days)
    daily_days = int(settings.provider_cost.daily_retention_days)
    assert snapshot_days == 90
    assert daily_days == 365


# ---- openai_billing.py getattr fallback still works ----


def test_openai_billing_getattr_fallback_returns_default(monkeypatch):
    """``services/cost/openai_billing.py`` uses
    ``getattr(settings.provider_cost, "openai_admin_api_key", "")``
    because that field is not in atlas's ProviderCostConfig today.
    The substrate copy mirrors atlas, so the getattr fallback still
    returns the empty default."""
    settings = _reload_settings_in_standalone_mode(monkeypatch)
    fallback = getattr(settings.provider_cost, "openai_admin_api_key", "") or ""
    assert fallback == ""
