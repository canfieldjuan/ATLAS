import copy
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_deflection_frontend_contract_types.py"
DEFLECTION_PRODUCT_SURFACE_MANIFEST = (
    ROOT / "tests" / "maturity_sweep" / "deflection_product_surface_manifest.json"
)


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


def test_deflection_api_contract_metadata_includes_result_page_projection_fields() -> None:
    rendered = MOD["render_api_contract"]()

    assert 'DEFLECTION_SNAPSHOT_SCHEMA_VERSION = "deflection.v1"' in rendered
    assert (
        'DEFLECTION_RESULT_PAGE_SNAPSHOT_FIELDS = Object.freeze(["summary", '
        '"top_questions", "top_blind_spots"])'
    ) in rendered
    assert (
        'DEFLECTION_SNAPSHOT_SUMMARY_FIELDS = Object.freeze(["generated", '
        '"drafted_answer_count", "no_proven_answer_count"'
    ) in rendered
    assert '"non_repeat_ticket_count"' in rendered
    assert (
        'DEFLECTION_SNAPSHOT_TOP_QUESTION_FIELDS = Object.freeze(["rank", '
        '"question", "ticket_count", "weighted_frequency", "customer_wording"])'
    ) in rendered
    assert (
        'DEFLECTION_SNAPSHOT_TOP_BLIND_SPOT_FIELDS = Object.freeze(["rank", '
        '"question", "ticket_count"])'
    ) in rendered
    assert "source_ids" not in rendered


def test_deflection_report_model_types_include_backend_projection_fields() -> None:
    rendered = MOD["render_report_model_types"]()

    assert 'export const DEFLECTION_REPORT_MODEL_SCHEMA_VERSION = "deflection.v1" as const;' in rendered
    assert (
        'DEFLECTION_REPORT_SECTION_IDS = ["support_tax", "source_file", '
        '"seo_targets", "ranked_questions", "priority_fix_queue"'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_CONDITIONAL_SECTION_IDS = ["source_file", '
        '"outcome_diagnostics"]'
    ) in rendered
    assert "export type DeflectionReportPriorityFixQueueItem" in rendered
    assert "status_counts: Record<string, number>;" in rendered
    assert "support_cost_basis: DeflectionReportPriorityFixQueueSupportCostBasis;" in rendered
    assert "csat_signal: DeflectionReportPriorityFixQueueCsatSignal;" in rendered
    assert "product_gap_summary?: string;" in rendered
    assert "customer_vocabulary?: string[];" in rendered
    assert "cost_period?: string;" in rendered
    assert "cost_confidence?: string;" in rendered
    assert "jira_template?: DeflectionReportPriorityFixQueueJiraTemplate;" in rendered
    assert "export type DeflectionReportPriorityFixQueueJiraTemplate" in rendered
    assert "  product_gap_summary: string;" in rendered
    assert "  customer_vocabulary: string[];" in rendered
    assert "top_evidence: DeflectionReportPriorityFixQueueTopEvidence[];" in rendered
    assert "review_key: string;" in rendered
    assert "suppression_reason: string;" in rendered
    assert "suppression_reason_label: string;" in rendered
    assert "source_date_window: DeflectionReportSupportTaxSourceDateWindow | null;" in rendered
    assert "term_mappings: DeflectionReportQuestionDetailsTermMappings[];" in rendered
    assert "outcome_diagnostics: DeflectionReportQuestionOutcomeDiagnostics | null;" in rendered
    assert "evidence_tier?: string;" in rendered
    assert "routing_signals?: DeflectionReportPriorityFixQueueRoutingSignals;" in rendered
    assert "source_file?: DeflectionReportSourceFileData;" in rendered
    assert "outcome_diagnostics?: DeflectionReportOutcomeDiagnosticsData;" in rendered
    assert "export type DeflectionStructuredReport" in rendered
    assert "any" not in rendered


def test_deflection_report_model_types_publish_hosted_safe_allowlists() -> None:
    rendered = MOD["render_report_model_types"]()

    assert (
        'DEFLECTION_REPORT_TOP_UNRESOLVED_REPEATS_HOSTED_CONSUMER_SAFE_FIELDS = '
        '["items", "top_item_count", "support_cost_basis"]'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_HOSTED_CONSUMER_SAFE_FIELDS = '
        '["rank", "question", "status", "owner_lane", "evidence_tier", '
        '"routing_signals", "product_gap_summary", "customer_vocabulary", '
        '"cost_period", "cost_confidence", "jira_template", "confidence", '
        '"recommended_action", "ticket_count", "estimated_support_cost", '
        '"priority_score", "priority_drivers", "csat_signal"]'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_JIRA_TEMPLATE_HOSTED_CONSUMER_SAFE_FIELDS = '
        '["recommended_title", "question", "owner_lane", "product_gap_summary", '
        '"ticket_count", "estimated_support_cost", "cost_period", "cost_confidence", '
        '"evidence_tier", "customer_vocabulary", "recommended_action"]'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_CSAT_SIGNAL_HOSTED_CONSUMER_SAFE_FIELDS = '
        '["status", "csat_present_count", "negative_csat_ticket_count", "numeric_average"]'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_ROUTING_SIGNALS_HOSTED_CONSUMER_SAFE_FIELDS = '
        '["tags", "product_area", "custom_product_area"]'
    ) in rendered
    assert (
        "DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_TOP_EVIDENCE_HOSTED_CONSUMER_SAFE_FIELDS = "
        "[]"
    ) in rendered
    assert '"review_key", "suppression_reason", "suppression_reason_label"' in rendered
    assert "hosted_consumer_safe_fields: string[];" not in rendered


def test_deflection_report_model_types_publish_hosted_field_shapes() -> None:
    rendered = MOD["render_report_model_types"]()

    assert "export const DEFLECTION_REPORT_HOSTED_FIELD_SHAPES = {" in rendered
    assert '"support_tax": {' in rendered
    assert '"source_date_window": "object",' in rendered
    assert '"support_tax.source_date_window": {' in rendered
    assert '"source_window_days": "scalar",' in rendered
    assert '"question_details.rows": {' in rendered
    assert '"term_mappings": "object_array",' in rendered
    assert '"question_details.rows.term_mappings": {' in rendered
    assert '"source_id_count": "scalar",' in rendered
    assert '"outcome_diagnostics.rows": {' in rendered
    assert '"status_mix": "scalar",' in rendered
    assert '"suppressed_repeat_review_queue": {' in rendered
    assert '"reason_counts": "record",' in rendered
    assert '"priority_fix_queue.items": {' in rendered
    assert '"customer_vocabulary": "scalar_array",' in rendered
    assert '"jira_template": "object",' in rendered
    assert '"priority_fix_queue.items.jira_template": {' in rendered
    assert '"cost_confidence": "scalar",' in rendered


def test_deflection_report_model_api_contract_includes_backend_projection_fields() -> None:
    rendered = MOD["render_report_model_api_contract"]()

    assert 'DEFLECTION_REPORT_MODEL_SCHEMA_VERSION = "deflection.v1"' in rendered
    assert (
        'DEFLECTION_REPORT_SECTION_IDS = Object.freeze(["support_tax", "source_file", '
        '"seo_targets", "ranked_questions", "priority_fix_queue"'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_CONDITIONAL_SECTION_IDS = Object.freeze(["source_file", '
        '"outcome_diagnostics"])'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_FIELDS = Object.freeze(["rank", '
        '"repeat_key", "cluster_id"'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_SUPPRESSED_REPEAT_REVIEW_QUEUE_ITEMS_FIELDS = '
        'Object.freeze(["rank", "repeat_key", "cluster_id"'
    ) in rendered
    assert '"review_key"' in rendered
    assert '"suppression_reason"' in rendered
    assert '"suppression_reason_label"' in rendered
    assert '"top_evidence"' in rendered
    assert (
        'DEFLECTION_REPORT_QUESTION_DETAILS_ROWS_FIELDS = Object.freeze(["rank", '
        '"question", "customer_wording"'
    ) in rendered
    assert '"source_ids"' in rendered
    assert '"evidence_quotes"' in rendered
    assert "as const" not in rendered


def test_deflection_report_model_api_contract_publishes_hosted_safe_allowlists() -> None:
    rendered = MOD["render_report_model_api_contract"]()

    assert (
        'DEFLECTION_REPORT_TOP_UNRESOLVED_REPEATS_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["items", "top_item_count", "support_cost_basis"])'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["rank", "question", "status", "owner_lane", "evidence_tier", '
        '"routing_signals", "product_gap_summary", "customer_vocabulary", '
        '"cost_period", "cost_confidence", "jira_template", "confidence", '
        '"recommended_action", "ticket_count", "estimated_support_cost", '
        '"priority_score", "priority_drivers", "csat_signal"])'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_JIRA_TEMPLATE_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["recommended_title", "question", "owner_lane", "product_gap_summary", '
        '"ticket_count", "estimated_support_cost", "cost_period", "cost_confidence", '
        '"evidence_tier", "customer_vocabulary", "recommended_action"])'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_CSAT_SIGNAL_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["status", "csat_present_count", "negative_csat_ticket_count", '
        '"numeric_average"])'
    ) in rendered
    assert (
        'DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_ROUTING_SIGNALS_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["tags", "product_area", "custom_product_area"])'
    ) in rendered
    assert (
        "DEFLECTION_REPORT_PRIORITY_FIX_QUEUE_ITEMS_TOP_EVIDENCE_HOSTED_CONSUMER_SAFE_FIELDS = "
        "Object.freeze([])"
    ) in rendered
    assert (
        'DEFLECTION_REPORT_SUPPRESSED_REPEAT_REVIEW_QUEUE_ITEMS_HOSTED_CONSUMER_SAFE_FIELDS = '
        'Object.freeze(["rank", "question", "status", "owner_lane", "evidence_tier", '
        '"routing_signals", "product_gap_summary", "customer_vocabulary", '
        '"cost_period", "cost_confidence", "jira_template", "confidence", '
        '"recommended_action", "ticket_count", "estimated_support_cost", '
        '"priority_score", "priority_drivers", "csat_signal", "review_key", '
        '"suppression_reason", "suppression_reason_label"])'
    ) in rendered


def test_deflection_report_model_api_contract_publishes_hosted_field_shapes() -> None:
    rendered = MOD["render_report_model_api_contract"]()

    assert "export const DEFLECTION_REPORT_HOSTED_FIELD_SHAPES = Object.freeze({" in rendered
    assert '"support_tax": Object.freeze({' in rendered
    assert '"source_date_window": "object",' in rendered
    assert '"support_tax.source_date_window": Object.freeze({' in rendered
    assert '"question_details.rows": Object.freeze({' in rendered
    assert '"term_mappings": "object_array",' in rendered
    assert '"outcome_diagnostics.rows": Object.freeze({' in rendered
    assert '"status_mix": "scalar",' in rendered
    assert '"suppressed_repeat_review_queue": Object.freeze({' in rendered
    assert '"reason_counts": "record",' in rendered


def test_deflection_report_hosted_shape_map_rejects_non_scalar_without_metadata() -> None:
    base = MOD["deflection_report_model_contract_shape"]()
    cases = [
        ("support_tax", None, "nested_object_fields", "source_date_window"),
        ("question_details", "rows", "nested_collection_fields", "term_mappings"),
        ("priority_fix_queue", None, "record_fields", "status_counts"),
    ]

    for section_id, collection_field, metadata_key, field in cases:
        contract = copy.deepcopy(base)
        sections = {
            section["id"]: section
            for section in contract["report_projection"]["sections"]
        }
        owner = sections[section_id]
        if collection_field:
            owner = owner["collection"]
            assert owner["field"] == collection_field

        if metadata_key == "record_fields":
            owner[metadata_key] = [
                record_field for record_field in owner[metadata_key] if record_field != field
            ]
        else:
            owner[metadata_key] = [
                entry for entry in owner[metadata_key] if entry["field"] != field
            ]

        try:
            MOD["render_report_model_api_contract"](contract)
        except ValueError as exc:
            assert field in str(exc)
            assert "non-scalar type" in str(exc)
        else:
            raise AssertionError(f"{field} should require hosted shape metadata")


def test_generated_deflection_api_contracts_are_enrolled_in_product_surface_manifest() -> None:
    manifest = json.loads(DEFLECTION_PRODUCT_SURFACE_MANIFEST.read_text(encoding="utf-8"))
    manifest_files = set(manifest["files"])

    assert MOD["DEFAULT_API_OUTPUT"].relative_to(ROOT).as_posix() in manifest_files
    assert (
        MOD["DEFAULT_REPORT_MODEL_API_OUTPUT"].relative_to(ROOT).as_posix()
        in manifest_files
    )


def test_deflection_frontend_contract_types_check_rejects_stale_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    api_output = tmp_path / "snapshot-contract.js"
    report_model_output = tmp_path / "deflectionReportModel.ts"
    report_model_api_output = tmp_path / "report-model-contract.js"
    output.write_text("// stale\n", encoding="utf-8")
    api_output.write_text(MOD["render_api_contract"](), encoding="utf-8")
    report_model_output.write_text(MOD["render_report_model_types"](), encoding="utf-8")
    report_model_api_output.write_text(
        MOD["render_report_model_api_contract"](),
        encoding="utf-8",
    )

    assert (
        MOD["main"](
            [
                "--output",
                str(output),
                "--api-output",
                str(api_output),
                "--report-model-output",
                str(report_model_output),
                "--report-model-api-output",
                str(report_model_api_output),
                "--check",
            ]
        )
        == 1
    )


def test_deflection_api_contract_metadata_check_rejects_stale_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    api_output = tmp_path / "snapshot-contract.js"
    report_model_output = tmp_path / "deflectionReportModel.ts"
    report_model_api_output = tmp_path / "report-model-contract.js"
    output.write_text(MOD["render_types"](), encoding="utf-8")
    api_output.write_text("// stale\n", encoding="utf-8")
    report_model_output.write_text(MOD["render_report_model_types"](), encoding="utf-8")
    report_model_api_output.write_text(
        MOD["render_report_model_api_contract"](),
        encoding="utf-8",
    )

    assert (
        MOD["main"](
            [
                "--output",
                str(output),
                "--api-output",
                str(api_output),
                "--report-model-output",
                str(report_model_output),
                "--report-model-api-output",
                str(report_model_api_output),
                "--check",
            ]
        )
        == 1
    )


def test_deflection_report_model_types_check_rejects_stale_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    api_output = tmp_path / "snapshot-contract.js"
    report_model_output = tmp_path / "deflectionReportModel.ts"
    report_model_api_output = tmp_path / "report-model-contract.js"
    output.write_text(MOD["render_types"](), encoding="utf-8")
    api_output.write_text(MOD["render_api_contract"](), encoding="utf-8")
    report_model_output.write_text("// stale\n", encoding="utf-8")
    report_model_api_output.write_text(
        MOD["render_report_model_api_contract"](),
        encoding="utf-8",
    )

    assert (
        MOD["main"](
            [
                "--output",
                str(output),
                "--api-output",
                str(api_output),
                "--report-model-output",
                str(report_model_output),
                "--report-model-api-output",
                str(report_model_api_output),
                "--check",
            ]
        )
        == 1
    )


def test_deflection_report_model_api_contract_check_rejects_stale_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    api_output = tmp_path / "snapshot-contract.js"
    report_model_output = tmp_path / "deflectionReportModel.ts"
    report_model_api_output = tmp_path / "report-model-contract.js"
    output.write_text(MOD["render_types"](), encoding="utf-8")
    api_output.write_text(MOD["render_api_contract"](), encoding="utf-8")
    report_model_output.write_text(MOD["render_report_model_types"](), encoding="utf-8")
    report_model_api_output.write_text("// stale\n", encoding="utf-8")

    assert (
        MOD["main"](
            [
                "--output",
                str(output),
                "--api-output",
                str(api_output),
                "--report-model-output",
                str(report_model_output),
                "--report-model-api-output",
                str(report_model_api_output),
                "--check",
            ]
        )
        == 1
    )


def test_deflection_frontend_contract_types_check_accepts_generated_output(tmp_path) -> None:
    output = tmp_path / "deflectionSnapshot.ts"
    api_output = tmp_path / "snapshot-contract.js"
    report_model_output = tmp_path / "deflectionReportModel.ts"
    report_model_api_output = tmp_path / "report-model-contract.js"

    common_args = [
        "--output",
        str(output),
        "--api-output",
        str(api_output),
        "--report-model-output",
        str(report_model_output),
        "--report-model-api-output",
        str(report_model_api_output),
    ]

    assert MOD["main"](common_args) == 0
    assert MOD["main"]([*common_args, "--check"]) == 0


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
        MOD["render_api_contract"](contract)
    except ValueError as exc:
        assert "not_actually_projected" in str(exc)
    else:
        raise AssertionError("optional backend fields must also be projected")


def test_deflection_report_model_types_fail_closed_on_unmapped_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    contract["report_projection"]["sections"][0]["projected_fields"] = [
        *contract["report_projection"]["sections"][0]["projected_fields"],
        "new_backend_only_metric",
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        assert "new_backend_only_metric" in str(exc)
    else:
        raise AssertionError("unknown backend report field should fail closed")


def test_deflection_report_model_types_reject_unknown_optional_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    contract["report_projection"]["sections"][0]["optional_projected_fields"] = [
        *contract["report_projection"]["sections"][0]["optional_projected_fields"],
        "not_actually_projected",
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        assert "not_actually_projected" in str(exc)
    else:
        raise AssertionError("optional report fields must also be projected")


def test_deflection_report_model_types_reject_unknown_hosted_safe_section_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    section = contract["report_projection"]["sections"][0]
    section["hosted_consumer_safe_fields"] = [
        *section["hosted_consumer_safe_fields"],
        "raw_unprojected_hosted_field",
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        assert "raw_unprojected_hosted_field" in str(exc)
    else:
        raise AssertionError("hosted-safe section fields must also be projected")


def test_deflection_report_model_types_reject_unknown_hosted_safe_collection_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    sections = {
        section["id"]: section
        for section in contract["report_projection"]["sections"]
    }
    collection = sections["priority_fix_queue"]["collection"]
    collection["hosted_consumer_safe_fields"] = [
        *collection["hosted_consumer_safe_fields"],
        "raw_unprojected_hosted_field",
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        assert "raw_unprojected_hosted_field" in str(exc)
    else:
        raise AssertionError("hosted-safe collection fields must also be projected")


def test_deflection_report_model_types_reject_unknown_hosted_safe_nested_field() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    sections = {
        section["id"]: section
        for section in contract["report_projection"]["sections"]
    }
    collection = sections["priority_fix_queue"]["collection"]
    nested = {
        entry["field"]: entry
        for entry in collection["nested_object_fields"]
    }["csat_signal"]
    nested["hosted_consumer_safe_fields"] = [
        *nested["hosted_consumer_safe_fields"],
        "raw_unprojected_hosted_field",
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        assert "raw_unprojected_hosted_field" in str(exc)
    else:
        raise AssertionError("hosted-safe nested fields must also be projected")


def test_deflection_report_model_types_reject_hosted_non_scalar_without_shape_metadata() -> None:
    contract = MOD["deflection_report_model_contract_shape"]()
    section = {
        entry["id"]: entry
        for entry in contract["report_projection"]["sections"]
    }["support_tax"]
    section["nested_object_fields"] = [
        entry
        for entry in section["nested_object_fields"]
        if entry["field"] != "source_date_window"
    ]

    try:
        MOD["render_report_model_types"](contract)
    except ValueError as exc:
        message = str(exc)
        assert "source_date_window" in message
        assert "no record/nested shape metadata" in message
    else:
        raise AssertionError("hosted-safe non-scalar report fields need shape metadata")
