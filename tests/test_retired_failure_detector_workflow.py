from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "retired_failure_detector.yml"
MATURITY_WORKFLOW = ROOT / ".github" / "workflows" / "maturity_sweep_advisory.yml"
UPLOAD_ARTIFACT_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def load_workflow(path: Path = WORKFLOW) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def workflow_on(payload: dict[str, object]) -> dict[str, object]:
    on_block = payload.get(True, payload.get("on"))
    assert isinstance(on_block, dict)
    return on_block


def job_steps(payload: dict[str, object]) -> list[dict[str, object]]:
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    job = jobs.get("retired-failure-detector")
    assert isinstance(job, dict)
    steps = job.get("steps")
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def step_by_name(steps: list[dict[str, object]], name: str) -> dict[str, object]:
    for step in steps:
        if step.get("name") == name:
            return step
    raise AssertionError(f"missing workflow step: {name}")


def test_workflow_runs_on_pull_request_without_target_token_posture() -> None:
    payload = load_workflow()
    on_block = workflow_on(payload)

    assert "pull_request" in on_block
    assert "pull_request_target" not in on_block
    assert payload["permissions"] == {"contents": "read"}


def test_workflow_fetches_base_ref_and_writes_detector_json_artifact() -> None:
    steps = job_steps(load_workflow())
    resolve_step = step_by_name(steps, "Resolve base ref")
    run_step = step_by_name(steps, "Run retired failure detector")

    resolve_script = str(resolve_step.get("run", ""))
    run_script = str(run_step.get("run", ""))

    assert "refs/remotes/origin/${base_ref}" in resolve_script
    assert "scripts/detect_retired_failure_modes.py" in run_script
    assert '--base "origin/${base_ref}"' in run_script
    assert "--json-out artifacts/retired-failure-detector/retired-failure-signals.json" in run_script


def test_workflow_uploads_pinned_detector_artifact_without_signal_failure_step() -> None:
    steps = job_steps(load_workflow())
    upload_step = step_by_name(steps, "Upload retired failure detector artifact")
    all_run_text = "\n".join(str(step.get("run", "")) for step in steps)

    assert upload_step["uses"] == f"actions/upload-artifact@{UPLOAD_ARTIFACT_SHA}"
    assert upload_step["with"] == {
        "name": "retired-failure-detector-signals-${{ github.run_id }}",
        "path": "artifacts/retired-failure-detector",
        "if-no-files-found": "error",
        "retention-days": 14,
    }
    assert "jq" not in all_run_text
    assert "signals | length" not in all_run_text


def test_workflow_contract_test_is_enrolled_in_maturity_sweep_pr_ci() -> None:
    workflow = MATURITY_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/test_retired_failure_detector_workflow.py" in workflow
