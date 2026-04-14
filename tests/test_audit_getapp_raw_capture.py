"""Tests for GetApp raw capture audit script helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_getapp_raw_capture.py"
    spec = spec_from_file_location("audit_getapp_raw_capture", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_target_metadata_accepts_json_string():
    module = _load_script_module()

    metadata = module._normalize_target_metadata('{"region":"us","tags":["enterprise"]}')

    assert metadata == {"region": "us", "tags": ["enterprise"]}


def test_normalize_target_metadata_rejects_non_mapping_json():
    module = _load_script_module()

    metadata = module._normalize_target_metadata('["not","a","mapping"]')

    assert metadata == {}


def test_error_report_is_structured():
    module = _load_script_module()

    class _Args:
        vendor_name = "ClickUp"
        method = "browser"
        page = 1

    report = module._error_report(_Args(), RuntimeError("Page.goto failed"))

    assert report["classification"] == "fetch_error"
    assert report["vendor_name"] == "ClickUp"
    assert "Page.goto failed" in report["error"]
