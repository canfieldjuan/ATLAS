import importlib.util
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "test_adapter_live.py"
_SPEC = importlib.util.spec_from_file_location("test_adapter_live", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_auth_headers_omits_authorization_when_token_missing():
    assert _MODULE._auth_headers(None) == {}
    assert _MODULE._auth_headers("") == {}


def test_auth_headers_includes_bearer_token():
    assert _MODULE._auth_headers("token-123") == {"Authorization": "Bearer token-123"}


def test_discover_vendor_returns_first_tracked_vendor(monkeypatch):
    calls = []

    def _fake_get(url, *, token=None):
        calls.append((url, token))
        return {"signals": [{"vendor_name": "  Salesforce  "}]}

    monkeypatch.setattr(_MODULE, "_get", _fake_get)

    assert _MODULE._discover_vendor("http://atlas", "tok") == "Salesforce"
    assert calls == [("http://atlas/api/v1/b2b/tenant/signals?limit=1", "tok")]


def test_discover_vendor_returns_none_for_blank_signal_name(monkeypatch):
    monkeypatch.setattr(
        _MODULE,
        "_get",
        lambda url, *, token=None: {"signals": [{"vendor_name": "   "}]},
    )

    assert _MODULE._discover_vendor("http://atlas", "tok") is None


def test_main_requires_token(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["test_adapter_live.py"])

    assert _MODULE.main() == 2
    assert "ATLAS_TOKEN or --token is required" in capsys.readouterr().out


def test_main_requires_vendor_when_discovery_fails(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["test_adapter_live.py", "--token", "tok"])
    monkeypatch.setattr(_MODULE, "_discover_vendor", lambda base, token: None)

    assert _MODULE.main() == 2
    assert "Unable to discover a tracked vendor" in capsys.readouterr().out


def test_main_returns_error_when_health_check_fails(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_adapter_live.py", "--base-url", "http://atlas", "--token", "tok", "--vendor", "HubSpot"],
    )
    monkeypatch.setattr(_MODULE, "_get", lambda url, *, token=None: {"_error": "connection refused"})

    assert _MODULE.main() == 1
    output = capsys.readouterr().out
    assert "Server not reachable at http://atlas" in output
    assert "connection refused" in output


def test_main_runs_all_checks_with_discovered_vendor(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_adapter_live.py", "--base-url", "http://atlas", "--token", "tok"],
    )
    monkeypatch.setattr(_MODULE, "_discover_vendor", lambda base, token: "HubSpot")

    health_calls = []

    def _fake_get(url, *, token=None):
        health_calls.append((url, token))
        return {"ok": True}

    call_log = []

    monkeypatch.setattr(_MODULE, "_get", _fake_get)
    monkeypatch.setattr(
        _MODULE,
        "test_high_intent",
        lambda base, token, vendor: call_log.append(("high_intent", base, token, vendor)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_high_intent_vendor_scoped",
        lambda base, token, vendor: call_log.append(("high_intent_vendor", base, token, vendor)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_vendor_profile",
        lambda base, token, vendor: call_log.append(("vendor_profile", base, token, vendor)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_search_reviews",
        lambda base, token: call_log.append(("search_reviews", base, token)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_search_reviews_filtered",
        lambda base, token: call_log.append(("search_reviews_filtered", base, token)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_search_reviews_churn_intent",
        lambda base, token: call_log.append(("search_reviews_churn_intent", base, token)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_null_semantics",
        lambda base, token: call_log.append(("null_semantics", base, token)) or [],
    )
    monkeypatch.setattr(
        _MODULE,
        "test_suppression_impact",
        lambda base, token: call_log.append(("suppression_impact", base, token)) or [],
    )

    assert _MODULE.main() == 0
    assert health_calls == [("http://atlas/api/v1/ping", None)]
    assert call_log == [
        ("high_intent", "http://atlas", "tok", "HubSpot"),
        ("high_intent_vendor", "http://atlas", "tok", "HubSpot"),
        ("vendor_profile", "http://atlas", "tok", "HubSpot"),
        ("search_reviews", "http://atlas", "tok"),
        ("search_reviews_filtered", "http://atlas", "tok"),
        ("search_reviews_churn_intent", "http://atlas", "tok"),
        ("null_semantics", "http://atlas", "tok"),
        ("suppression_impact", "http://atlas", "tok"),
    ]
