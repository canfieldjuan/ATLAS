"""Tests for Trustpilot target hygiene audit helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_trustpilot_target_hygiene.py"
    spec = spec_from_file_location("audit_trustpilot_target_hygiene", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_capture_result_marks_404_as_broken():
    module = _load_script_module()

    result = module._classify_capture_result(
        {
            "status_code": 404,
            "analysis": {
                "employer_fields_present": False,
                "title_fields_present": False,
            },
        }
    )

    assert result == "broken_target"


def test_classify_capture_result_marks_field_empty_named_account_pages():
    module = _load_script_module()

    result = module._classify_capture_result(
        {
            "status_code": 200,
            "analysis": {
                "employer_fields_present": False,
                "title_fields_present": False,
            },
        }
    )

    assert result == "field_empty_for_named_account"


def test_normalize_target_metadata_accepts_json_string():
    module = _load_script_module()

    metadata = module._normalize_target_metadata('{"region":"us"}')

    assert metadata == {"region": "us"}
