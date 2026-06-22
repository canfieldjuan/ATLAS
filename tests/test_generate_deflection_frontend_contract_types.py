import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_deflection_frontend_contract_types.py"


MOD = runpy.run_path(str(SCRIPT))


def test_deflection_frontend_contract_types_include_backend_projection_fields() -> None:
    rendered = MOD["render_types"]()

    assert "export type DeflectionSnapshotSummary" in rendered
    assert "non_repeat_ticket_count: number;" in rendered
    assert (
        'DEFLECTION_SNAPSHOT_SUMMARY_OPTIONAL_FIELDS = ["source_date_start", '
        '"source_date_end", "source_window_days"]'
    ) in rendered
    assert "source_date_start?: string | null;" in rendered
    assert "source_date_end?: string | null;" in rendered
    assert "source_window_days?: number | null;" in rendered
    assert "ticket_count: number;" in rendered
    assert (
        'export type DeflectionResultPageSnapshot = Pick<DeflectionSnapshot, "summary" | '
        '"top_questions" | "top_blind_spots">;'
    ) in rendered
    assert "unknown" not in rendered
    assert "any" not in rendered


def test_deflection_frontend_contract_types_check_rejects_stale_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    output.write_text("// stale\n", encoding="utf-8")

    assert MOD["main"](["--output", str(output), "--check"]) == 1


def test_deflection_frontend_contract_types_check_accepts_generated_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"

    assert MOD["main"](["--output", str(output)]) == 0
    assert MOD["main"](["--output", str(output), "--check"]) == 0


def test_deflection_frontend_contract_types_fail_closed_on_unmapped_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    contract["snapshot_projection"]["fields"][0]["projected_fields"] = [
        *contract["snapshot_projection"]["fields"][0]["projected_fields"],
        "new_backend_only_metric",
    ]

    try:
        MOD["render_types"](contract)
    except ValueError as exc:
        assert "new_backend_only_metric" in str(exc)
    else:
        raise AssertionError("unknown backend projection field should fail closed")


def test_deflection_frontend_contract_types_reject_unknown_optional_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    contract["snapshot_projection"]["fields"][0]["optional_projected_fields"] = [
        *contract["snapshot_projection"]["fields"][0]["optional_projected_fields"],
        "not_actually_projected",
    ]

    try:
        MOD["render_types"](contract)
    except ValueError as exc:
        assert "not_actually_projected" in str(exc)
    else:
        raise AssertionError("optional backend fields must also be projected")
