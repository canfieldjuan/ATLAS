from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "maturity_sweep_advisory.yml"

EXPECTED_RATCHET_GROUPS = {
    "extracted-content-pipeline": [
        "baseline_extracted_content_pipeline.json",
        "--sensitive-glob '**/billing/**'",
        "--sensitive-glob '**/*deletion*'",
    ],
    "phase-c1-extracted-core": [
        "for lane in extracted_reasoning_core extracted_quality_gate extracted_evidence_to_story; do",
        'baseline_${lane}.json',
        '--sensitive-glob "${lane}/**"',
    ],
    "phase-c2-competitive-intelligence": [
        "extracted_competitive_intelligence",
        "baseline_extracted_competitive_intelligence.json",
    ],
    "phase-c3-llm-infrastructure": [
        "extracted_llm_infrastructure",
        "baseline_extracted_llm_infrastructure.json",
    ],
    "phase-c4-scripts": [
        "python scripts/maturity_sweep.py scripts",
        "baseline_scripts.json",
    ],
    "atlas-brain-api": [
        "python scripts/maturity_sweep.py atlas_brain/api",
        "baseline_atlas_brain_api.json",
        "--sensitive-glob '**/invoicing/**'",
    ],
    "atlas-brain-auth": [
        "python scripts/maturity_sweep.py atlas_brain/auth",
        "baseline_atlas_brain_auth.json",
        "--sensitive-glob '**/*'",
    ],
    "atlas-brain-autonomous": [
        "python scripts/maturity_sweep.py atlas_brain/autonomous",
        "baseline_atlas_brain_autonomous.json",
    ],
    "atlas-brain-mcp": [
        "python scripts/maturity_sweep.py atlas_brain/mcp",
        "baseline_atlas_brain_mcp.json",
    ],
    "atlas-brain-services-b2b": [
        "python scripts/maturity_sweep.py atlas_brain/services/b2b",
        "baseline_atlas_brain_services_b2b.json",
    ],
    "atlas-brain-b2a-support": [
        "for lane in alerts brand discovery escalation events jobs memory modes orchestration pipelines presence schemas templates utils; do",
        'baseline_atlas_brain_${lane}.json',
    ],
    "atlas-brain-b2b-service-comms": [
        "services/email_webhooks",
        "services/speaker_id",
        'baseline_atlas_brain_${slug}.json',
    ],
    "atlas-brain-b2c-core-risk": [
        "for lane in reasoning security storage; do",
        "atlas_brain/security/**",
        "atlas_brain/storage/**",
    ],
    "atlas-brain-b2d-runtime-control": [
        "for lane in agents capabilities tools; do",
        "--sensitive-glob '**/security.py'",
    ],
    "atlas-brain-b2e-scraping": [
        "python scripts/maturity_sweep.py atlas_brain/services/scraping",
        "baseline_atlas_brain_services_scraping.json",
        "atlas_brain/services/scraping/**",
    ],
}


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _section(text: str, start_marker: str, end_marker: str | None = None) -> str:
    start = text.index(start_marker)
    if end_marker is None:
        return text[start:]
    return text[start : text.index(end_marker, start + len(start_marker))]


def test_unit_and_advisory_job_keeps_existing_sweep_coverage() -> None:
    section = _section(
        _workflow_text(),
        "  maturity-sweep-unit-and-advisory:\n",
        "  maturity-sweep-ratchets:\n",
    )

    assert "tests/test_maturity_sweep.py" in section
    assert "tests/test_detect_retired_failure_modes.py" in section
    assert "tests/test_maturity_sweep_advisory_workflow.py" in section
    assert "tests/test_retired_failure_detector_workflow.py" in section
    assert "continue-on-error: true" in section
    assert "python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --top 25" in section


def test_ratchet_matrix_preserves_all_current_command_groups() -> None:
    section = _section(
        _workflow_text(),
        "  maturity-sweep-ratchets:\n",
        "  maturity-sweep:\n",
    )

    assert "fail-fast: false" in section
    assert section.count("          - group: ") == len(EXPECTED_RATCHET_GROUPS)
    assert section.count("--baseline") == 15
    assert section.count("--min-score 8") == 15
    for group, markers in EXPECTED_RATCHET_GROUPS.items():
        assert f"- group: {group}" in section
        for marker in markers:
            assert marker in section


def test_aggregate_job_preserves_stable_maturity_sweep_context() -> None:
    section = _section(_workflow_text(), "  maturity-sweep:\n")

    assert "name: maturity-sweep" in section
    assert "needs:" in section
    assert "- maturity-sweep-unit-and-advisory" in section
    assert "- maturity-sweep-ratchets" in section
    assert "if: always()" in section
    assert "needs.maturity-sweep-unit-and-advisory.result" in section
    assert "needs.maturity-sweep-ratchets.result" in section
    assert 'exit "$failed"' in section
