"""Tests for GetApp target hygiene audit helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "audit_getapp_target_hygiene.py"
    spec = spec_from_file_location("audit_getapp_target_hygiene", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_capture_result_marks_context_only_without_employer():
    module = _load_script_module()

    result = module._classify_capture_result(
        {
            "status_code": 200,
            "analysis": {
                "employer_fields_present": False,
                "title_fields_present": True,
                "company_size_fields_present": True,
            },
        }
    )

    assert result == "context_only_no_employer"


def test_classify_capture_result_marks_employer_identity_present():
    module = _load_script_module()

    result = module._classify_capture_result(
        {
            "status_code": 200,
            "analysis": {
                "employer_fields_present": True,
                "title_fields_present": True,
                "company_size_fields_present": True,
            },
        }
    )

    assert result == "employer_identity_present"
