import importlib.util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_reasoning_rollout_readiness.py"
)
_SPEC = importlib.util.spec_from_file_location("check_reasoning_rollout_readiness", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)

_as_dict = _MODULE._as_dict
_exit_code = _MODULE._exit_code
_REQUIRED_MIGRATIONS = _MODULE._REQUIRED_MIGRATIONS
_REQUIRED_TABLES = _MODULE._REQUIRED_TABLES
_REQUIRED_TASKS = _MODULE._REQUIRED_TASKS
_reasoning_v2_schema_predicate = _MODULE._reasoning_v2_schema_predicate
_status = _MODULE._status


def test_as_dict_parses_json_strings():
    result = _as_dict('{"scheduled_scope_strategy":"competitive_sets"}')

    assert result == {"scheduled_scope_strategy": "competitive_sets"}


def test_status_marks_required_failures_as_fail():
    result = _status("required_migrations", False, detail={"missing": ["261_b2b_competitive_sets"]})

    assert result["name"] == "required_migrations"
    assert result["status"] == "fail"
    assert result["required"] is True


def test_status_marks_optional_failures_as_warn():
    result = _status("competitive_sets_seeded", False, required=False, detail={"count": 0})

    assert result["status"] == "warn"
    assert result["required"] is False


def test_reasoning_v2_schema_predicate_accepts_canonical_versions():
    predicate = _reasoning_v2_schema_predicate("schema_version")

    assert "IN ('v2', '2')" in predicate
    assert "LIKE 'v2.%'" in predicate
    assert "LIKE '2.%'" in predicate


def test_required_rollout_migrations_include_report_subscription_tables():
    assert "265_b2b_report_subscriptions" in _REQUIRED_MIGRATIONS
    assert "266_b2b_report_subscription_delivery_log" in _REQUIRED_MIGRATIONS
    assert "268_b2b_report_subscription_delivery_dry_run_status" in _REQUIRED_MIGRATIONS


def test_required_rollout_tables_include_report_subscription_tables():
    assert "b2b_report_subscriptions" in _REQUIRED_TABLES
    assert "b2b_report_subscription_delivery_log" in _REQUIRED_TABLES


def test_required_rollout_tasks_include_report_delivery():
    assert "b2b_reasoning_synthesis" in _REQUIRED_TASKS
    assert "b2b_report_subscription_delivery" in _REQUIRED_TASKS


def test_exit_code_is_nonzero_when_required_check_fails():
    checks = [
        _status("required_tables", True),
        _status("required_migrations", False),
        _status("competitive_sets_seeded", False, required=False),
    ]

    assert _exit_code(checks) == 1


def test_exit_code_is_zero_when_only_warnings_exist():
    checks = [
        _status("required_tables", True),
        _status("competitive_sets_seeded", False, required=False),
    ]

    assert _exit_code(checks) == 0
