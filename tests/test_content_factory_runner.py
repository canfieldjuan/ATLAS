"""Tests for the Content Factory stage runner.

Open WebUI (the HTTP chat call) is the external boundary and is mocked; the store
and contracts are real, so run_stage's extract + validate + persist path is
exercised for real against a temp filesystem.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

import atlas_brain.services.content_factory_runner as runner
from atlas_brain.services.content_factory_store import ArtifactStoreError
from atlas_brain.services.content_factory_runner import (
    WorkerError,
    call_worker,
    extract_json,
)
from atlas_brain.services.content_factory_store import job_dir

BRIEF_JSON = '{"schema": "content_brief.v1", "project_id": "resolution-audit", "request_raw": "x"}'


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- extract_json ---


@pytest.mark.parametrize(
    "text",
    [
        BRIEF_JSON,
        "```json\n" + BRIEF_JSON + "\n```",
        "```\n" + BRIEF_JSON + "\n```",
        "Here is the brief:\n" + BRIEF_JSON + "\nHope that helps.",
    ],
)
def test_extract_json_variants(text):
    data = extract_json(text)
    assert data is not None and data["schema"] == "content_brief.v1"


@pytest.mark.parametrize(
    "text", ["no json here", "", "[1, 2, 3]", "{not valid}", "```json\n[]\n```"]
)
def test_extract_json_returns_none(text):
    assert extract_json(text) is None


# --- call_worker (OWUI HTTP boundary mocked) ---


def test_call_worker_returns_content(monkeypatch):
    resp = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(resp))
    assert call_worker("m", "u", api_key="k") == "hi"


def test_call_worker_http_error_raises_worker_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("url", 500, "err", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WorkerError):
        call_worker("m", "u", api_key="k")


def test_call_worker_missing_message_raises(monkeypatch):
    resp = json.dumps({"choices": []}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(resp))
    with pytest.raises(WorkerError):
        call_worker("m", "u", api_key="k")


# --- run_stage (worker mocked, real store) ---


def test_run_stage_persists_valid_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: BRIEF_JSON)
    rec = runner.run_stage("job1", "brief", "cf-brief-architect", "req", api_key="k", root=tmp_path)
    assert Path(rec["path"]).exists() and rec["schema"] == "content_brief.v1"


def test_run_stage_extracts_from_fenced_prose(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "call_worker", lambda *a, **k: "Sure:\n```json\n" + BRIEF_JSON + "\n```"
    )
    rec = runner.run_stage("job1", "brief", "m", "req", api_key="k", root=tmp_path)
    assert Path(rec["path"]).exists()


def test_run_stage_no_json_raises_and_persists_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: "I cannot help with that.")
    with pytest.raises(WorkerError):
        runner.run_stage("job1", "brief", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job1", root=tmp_path).exists()


def test_run_stage_invalid_artifact_not_persisted(tmp_path, monkeypatch):
    bad = (
        '{"schema": "evidence_packet.v1", "project_id": "p", '
        '"evidence": [{"id": "e", "quote": "q", "source_id": ""}]}'
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: bad)
    with pytest.raises(ValidationError):
        runner.run_stage("job1", "evidence", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job1", root=tmp_path).exists()


# --- editor stage: deterministic copy-verification enforcement (Phase 4.2) ---

_CLEAN_BODY = "Support leaders: the Resolution Audit ranks your repeated tickets."
_BAD_BODY = "Guaranteed savings for every team."


def _editor_json(body, recommendation="revise", **extra):
    audit = {
        "schema": "editorial_audit.v2",
        "project_id": "resolution-audit",
        "edited_body_markdown": body,
        "recommendation": recommendation,
    }
    audit.update(extra)
    return json.dumps(audit)


def _stored(rec):
    return json.loads(Path(rec["path"]).read_text())


def _stage_path(root, job_id, stage):
    return job_dir(job_id, root=root) / f"{stage}.json"


def test_editor_stage_injects_deterministic_pass(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json(_CLEAN_BODY, "promote"))
    rec = runner.run_stage("job1", "audit", "cf-editor", None, api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "pass"
    assert stored["recommendation"] == "promote"  # clean copy may promote


def test_editor_worker_cannot_self_promote_overclaim(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    # Worker claims a passing verdict + promote on copy that actually overclaims.
    reply = _editor_json(_BAD_BODY, "promote", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):  # injected fail vs promote -> #2116 guard rejects
        runner.run_stage("job1", "audit", "m", None, api_key="k", root=tmp_path)
    assert not _stage_path(tmp_path, "job1", "audit").exists()


def test_editor_overclaim_revise_persists_with_fail(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json(_BAD_BODY, "revise"))
    rec = runner.run_stage("job1", "audit", "m", None, api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("guaranteed-savings" in h for h in stored["copy_verification"]["hits"])


def test_editor_worker_claimed_verdict_is_overridden(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    # Worker asserts pass on bad copy but only recommends revise; the deterministic
    # verdict must still overwrite the worker's claim.
    reply = _editor_json(_BAD_BODY, "revise", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job1", "audit", "m", None, api_key="k", root=tmp_path)
    assert _stored(rec)["copy_verification"]["verdict"] == "fail"


def test_non_editor_stage_gets_no_copy_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: BRIEF_JSON)
    rec = runner.run_stage("job1", "brief", "m", "req", api_key="k", root=tmp_path)
    assert "copy_verification" not in _stored(rec)


def test_empty_edited_copy_cannot_promote(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    # Fail closed: an empty edited body means nothing was verified, so a worker cannot
    # self-promote by omitting the edited copy.
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json("", "promote"))
    with pytest.raises(ValidationError):
        runner.run_stage("job1", "audit", "m", None, api_key="k", root=tmp_path)
    assert not _stage_path(tmp_path, "job1", "audit").exists()


def test_empty_edited_copy_revise_persists_with_fail(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json("   ", "revise"))
    rec = runner.run_stage("job1", "audit", "m", None, api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("unverified-copy" in h for h in stored["copy_verification"]["hits"])


def test_custom_stage_audit_is_also_gated(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job1", ["e1"], approved=False)
    # A custom (non-"audit") stage emitting editorial_audit.v1 must not bypass the gate:
    # gating is by schema, so its self-promotion of overclaiming copy is still rejected.
    reply = _editor_json(_BAD_BODY, "promote", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):
        runner.run_stage("job1", "audit-v2", "m", None, api_key="k", root=tmp_path)
    assert not _stage_path(tmp_path, "job1", "audit-v2").exists()


def test_enforce_overwrites_worker_supplied_advisory_warnings():
    """The checklist is computed from the edited copy, never taken from the
    worker: a fabricated empty list must not blind the reviewer."""
    from atlas_brain.services.content_factory_runner import _enforce_copy_verification

    artifact = {
        "schema": "editorial_audit.v2",
        "project_id": "p",
        "edited_body_markdown": "We draft the answer for every repeated ticket.",
        "recommendation": "revise",
        "advisory_warnings": [],
    }
    _enforce_copy_verification(artifact)
    assert any(
        w.startswith("unqualified-answer-claim:")
        for w in artifact["advisory_warnings"]
    )
    assert artifact["advisory_warnings"][-1].startswith("reminder:")


def test_enforce_clears_warnings_with_empty_body():
    from atlas_brain.services.content_factory_runner import _enforce_copy_verification

    artifact = {
        "schema": "editorial_audit.v2",
        "project_id": "p",
        "edited_body_markdown": "  ",
        "recommendation": "revise",
        "advisory_warnings": ["fabricated: looks reviewed"],
    }
    _enforce_copy_verification(artifact)
    assert artifact["advisory_warnings"] == []
    assert artifact["copy_verification"]["verdict"] == "fail"


def test_run_stage_persists_deterministic_warnings_and_normalizes_v1(tmp_path, monkeypatch):
    """Reachability proof at the real entrypoint (#2181 round 2): a worker
    reply tagged v1 with a fabricated empty checklist is normalized to v2 and
    persisted with the DETERMINISTIC advisory warnings."""
    _seed_draft(tmp_path, "job-adv", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "editorial_audit.v1",
            "project_id": "p",
            "edited_body_markdown": "We draft the answer for every repeated ticket.",
            "recommendation": "revise",
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-adv", "audit", "cf-editor-verifier", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert rec["schema"] == "editorial_audit.v3"
    assert stored["schema"] == "editorial_audit.v3"
    assert any(
        w.startswith("unqualified-answer-claim:")
        for w in stored["advisory_warnings"]
    )
    assert stored["advisory_warnings"][-1].startswith("reminder:")
    assert stored["copy_verification"]["verdict"] == "pass"


def test_run_stage_rejects_contradictory_v2_version(tmp_path, monkeypatch):
    """Round-18 regression: a worker reply already tagged v2 keeps its own
    schema_version, so contradictory metadata fails validation instead of
    being silently rewritten to 2."""
    _seed_draft(tmp_path, "job-v2v", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "editorial_audit.v2",
            "schema_version": 999,
            "project_id": "p",
            "edited_body_markdown": "Clean copy.",
            "recommendation": "revise",
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):
        runner.run_stage("job-v2v", "audit", "m", None, api_key="k", root=tmp_path)
    assert not _stage_path(tmp_path, "job-v2v", "audit").exists()


# --- Phase 6: repurposing + image-prompt gates at the real entrypoint ---


def _seed_draft(root, job_id, source_ids, revision=1, project="p", approved=True):
    """Establish the approved source state Phase 6 readiness requires: a
    draft with claim lineage plus an audit that promoted it."""
    from atlas_brain.services.content_factory_store import write_artifact

    write_artifact(job_id, "draft", {
        "schema": "draft.v1",
        "project_id": project,
        "revision": revision,
        "body_markdown": "seed",
        "claims": [{"text": f"claim {sid}", "source_id": sid} for sid in source_ids],
    }, root=root)
    if approved:
        # Mirror what run_stage does: bind the approval to the draft bytes.
        fingerprint = runner._draft_fingerprint(job_id, root)
        write_artifact(job_id, "audit", {
            "source_draft_fingerprint": fingerprint,
            "schema": "editorial_audit.v3",
            "schema_version": 3,
            "project_id": project,
            # Must approve THIS revision -- a stale audit authorizes nothing.
            "draft_revision": revision,
            "edited_body_markdown": "seed",
            "copy_verification": {"verdict": "pass", "hits": []},
            "recommendation": "promote",
        }, root=root)


def test_run_stage_overwrites_variant_verdicts_and_blocks_bad_ship(tmp_path, monkeypatch):
    """A worker cannot ship an overclaiming variant: the runner recomputes
    each variant's verdict from its own body, so a self-asserted
    ready_to_publish becomes invalid and nothing persists."""
    reply = json.dumps(
        {
            "schema": "repurposing.v1",
            "project_id": "p",
            "variants": [
                {
                    "channel": "linkedin",
                    "body_markdown": "We guarantee savings for every team.",
                    "derived_from_claims": ["e1"],
                    "copy_verification": {"verdict": "pass", "hits": []},
                }
            ],
            "ready_to_publish": True,
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    # Seed a draft whose lineage covers the variant, so this test isolates
    # the CONTRACT behaviour (self-promotion) rather than tripping the
    # separate lineage check first.
    _seed_draft(tmp_path, "job-rp", ["e1"])
    with pytest.raises(ValidationError):
        runner.run_stage("job-rp", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_run_stage_persists_clean_variants_with_computed_verdicts(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-rp2", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "repurposing.v1",
            "project_id": "p",
            "variants": [
                {
                    "channel": "linkedin",
                    "body_markdown": "Repeat tickets quietly consume agent hours.",
                    "derived_from_claims": ["e1"],
                }
            ],
            "ready_to_publish": False,
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-rp2", "repurposing", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["variants"][0]["copy_verification"]["verdict"] == "pass"
    assert stored["variants"][0]["advisory_warnings"][-1].startswith("reminder:")


def test_run_stage_blank_variant_body_fails_closed(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-rp3", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "repurposing.v1",
            "project_id": "p",
            "variants": [
                {
                    "channel": "x",
                    "body_markdown": "   ",
                    "derived_from_claims": ["e1"],
                    "copy_verification": {"verdict": "pass", "hits": []},
                }
            ],
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    # blank body is rejected by the contract (NonEmptyStr) -- nothing persists
    with pytest.raises(ValidationError):
        runner.run_stage("job-rp3", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_run_stage_gates_image_prompt_text(tmp_path, monkeypatch):
    """Banned copy inside a PROMPT would be rendered into the artwork, where
    no text check would see it -- the gate runs on the prompt itself."""
    _seed_draft(tmp_path, "job-img", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "image_prompt.v1",
            "project_id": "p",
            "prompts": [
                {
                    "purpose": "hero",
                    "prompt_text": "poster reading guaranteed savings for every team",
                }
            ],
            "copy_verification": {"verdict": "pass", "hits": []},
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-img", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("guaranteed-savings" in hit for hit in stored["copy_verification"]["hits"])


def test_run_stage_image_prompt_pii_is_caught(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-img2", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "image_prompt.v1",
            "project_id": "p",
            "prompts": [
                {"purpose": "card", "prompt_text": "business card reading bob@example.com"}
            ],
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-img2", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    # The VERDICT never carries the raw value (the persisted-evidence
    # theorem); the prompt text itself is the artifact's payload and stays
    # so a human can see what to fix -- same as a draft body.
    assert "bob@example.com" not in json.dumps(stored["copy_verification"])
    assert stored["copy_verification"]["hits"] == ["prompt 1: email: <redacted>"]


def test_run_stage_clean_image_prompt_passes(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-img3", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "image_prompt.v1",
            "project_id": "p",
            "prompts": [
                {"purpose": "hero", "prompt_text": "a tidy office desk in soft morning light"}
            ],
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-img3", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass"


def test_stage_schema_mismatch_still_enforced_for_phase6(tmp_path, monkeypatch):
    """A repurposing artifact cannot land under the image_prompt stage."""
    _seed_draft(tmp_path, "job-mix", ["e1"], approved=False)
    reply = json.dumps(
        {
            "schema": "repurposing.v1",
            "project_id": "p",
            "variants": [
                {"channel": "x", "body_markdown": "clean", "derived_from_claims": ["e1"]}
            ],
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError):
        runner.run_stage("job-mix", "image_prompt", "m", None, api_key="k", root=tmp_path)


def test_negative_prompt_naming_banned_terms_still_passes(tmp_path, monkeypatch):
    """Guard's second side: a negative prompt is an EXCLUSION list, so naming
    a banned phrase there is the correct designer response to the threat --
    it must not trip the gate."""
    _seed_draft(tmp_path, "job-neg", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{
            "purpose": "hero",
            "prompt_text": "a tidy office desk in soft morning light",
            "negative_prompt": "blurry, watermark, text, guaranteed savings, phone number",
        }],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-neg", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass"
    assert stored["prompts"][0]["negative_prompt"].startswith("blurry")


def test_positive_prompt_with_banned_claim_still_fails(tmp_path, monkeypatch):
    """The other side of the same guard stays closed."""
    _seed_draft(tmp_path, "job-pos", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{
            "purpose": "hero",
            "prompt_text": "poster reading guaranteed savings",
            "negative_prompt": "blurry",
        }],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-pos", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"


def test_worker_cannot_self_declare_ready_to_generate(tmp_path, monkeypatch):
    """The runner recomputes the verdict, so a failing prompt set cannot be
    persisted as renderable no matter what the worker claims."""
    _seed_draft(tmp_path, "job-selfgen", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "poster reading guaranteed savings"}],
        "copy_verification": {"verdict": "pass", "hits": []},
        "ready_to_generate": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):
        runner.run_stage("job-selfgen", "image_prompt", "m", None, api_key="k", root=tmp_path)
    assert not _stage_path(tmp_path, "job-selfgen", "image_prompt").exists()


# --- review round 2 on #2192 ---


def test_fabricated_lineage_blocks_shipping(tmp_path, monkeypatch):
    """Non-blank lineage is not REAL lineage: an id the draft never
    established is still an orphan."""
    _seed_draft(tmp_path, "job-lin", ["e1"])
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["completely-fabricated-id"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="absent from the draft"):
        runner.run_stage("job-lin", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_real_lineage_ships(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-lin2", ["e1", "e2"])
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e2"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-lin2", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is True


def test_unready_package_skips_lineage_check(tmp_path, monkeypatch):
    """An unready package is a legitimate intermediate state."""
    _seed_draft(tmp_path, "job-lin3", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["not-yet-real"]}],
        "ready_to_publish": False,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-lin3", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert Path(rec["path"]).exists()


def test_missing_draft_fails_closed_on_ship(tmp_path, monkeypatch):
    called = False
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    def unexpected_worker(*args, **kwargs):
        nonlocal called
        called = True
        return reply

    monkeypatch.setattr(runner, "call_worker", unexpected_worker)
    with pytest.raises(ArtifactStoreError, match="requires a committed draft"):
        runner.run_stage("job-nodraft", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert called is False


def test_international_phone_in_prompt_fails(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-intl", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "Call us at +442079460958"}],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-intl", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    assert "442079460958" not in json.dumps(stored["copy_verification"])


def test_prompts_verified_independently_no_cross_synthesis(tmp_path, monkeypatch):
    """Joining items must not synthesize a claim no single prompt makes."""
    _seed_draft(tmp_path, "job-split", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [
            {"purpose": "a", "prompt_text": "a warm kitchen, results guaranteed"},
            {"purpose": "b", "prompt_text": "savings account paperwork on a desk"},
        ],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-split", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass", stored["copy_verification"]


def test_hits_identify_the_offending_prompt(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-which", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [
            {"purpose": "a", "prompt_text": "a clean desk"},
            {"purpose": "b", "prompt_text": "poster reading guaranteed savings"},
        ],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-which", "image_prompt", "m", None, api_key="k", root=tmp_path)
    hits = json.loads(Path(rec["path"]).read_text())["copy_verification"]["hits"]
    assert all(h.startswith("prompt 2:") for h in hits), hits


# --- review round 3 on #2192 ---


def test_stale_source_revision_blocks_shipping(tmp_path, monkeypatch):
    """Overlapping claim ids must not let a package built on superseded
    copy ship."""
    _seed_draft(tmp_path, "job-rev", ["e1"], revision=2)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "source_draft_revision": 1,
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="superseded copy"):
        runner.run_stage("job-rev", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_matching_source_revision_ships(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-rev2", ["e1"], revision=2)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "source_draft_revision": 2,
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-rev2", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is True


def test_image_prompt_readiness_also_checks_revision(tmp_path, monkeypatch):
    """ImagePromptSet gets the same tie to the approved draft."""
    _seed_draft(tmp_path, "job-imgrev", ["e1"], revision=3)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "source_draft_revision": 1,
        "prompts": [{"purpose": "hero", "prompt_text": "a tidy desk"}],
        "ready_to_generate": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="superseded copy"):
        runner.run_stage("job-imgrev", "image_prompt", "m", None, api_key="k", root=tmp_path)


def test_string_false_readiness_persists_as_intermediate(tmp_path, monkeypatch):
    """A weak worker's "false" normalizes to False, so lineage approval is not
    required and the intermediate package persists (runner and schema agree)."""
    _seed_draft(tmp_path, "job-strfalse", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["not-yet-real"]}],
        "ready_to_publish": "false",
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-strfalse", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is False


@pytest.mark.parametrize("text", [
    "poster reading do not guarantee savings",
    "a sign that says we never guarantee savings",
])
def test_negated_banned_phrase_in_prompt_still_fails(text, tmp_path, monkeypatch):
    """Prose negation does not un-draw words a renderer is told to paint."""
    _seed_draft(tmp_path, "job-neg2", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": text}],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-neg2", "image_prompt", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail", stored["copy_verification"]


def test_body_copy_keeps_prose_negation_semantics():
    """The literal matcher is prompt-only; body copy still reads denials as
    denials (the #2181 contract is untouched)."""
    from atlas_brain.services.content_factory_copy_verification import verify_copy

    assert verify_copy("We do not promise guaranteed savings.").verdict == "pass"


# --- review round 4 on #2192 ---


def test_unaudited_draft_cannot_ship(tmp_path, monkeypatch):
    """Existence of a draft proves it ran, not that anything approved it."""
    _seed_draft(tmp_path, "job-noaudit", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="recommending 'promote'"):
        runner.run_stage("job-noaudit", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_revise_audit_cannot_ship(tmp_path, monkeypatch):
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-revise", ["e1"], approved=False)
    write_artifact("job-revise", "audit", {
        "schema": "editorial_audit.v2", "project_id": "p",
        "edited_body_markdown": "seed",
        "copy_verification": {"verdict": "fail", "hits": ["guaranteed-savings: x"]},
        "recommendation": "revise",
    }, root=tmp_path)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="recommending 'promote'"):
        runner.run_stage("job-revise", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_cross_project_derivation_blocked(tmp_path, monkeypatch):
    """Matching revision + overlapping ids are not evidence across projects."""
    _seed_draft(tmp_path, "job-xproj", ["e1"], project="source")
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "other",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="project mismatch"):
        runner.run_stage("job-xproj", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_stale_audit_cannot_approve_newer_draft(tmp_path, monkeypatch):
    """A revision-1 approval does not authorize revision-2 copy."""
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-staleaudit", ["e1"], revision=2, approved=False)
    write_artifact("job-staleaudit", "audit", {
        "schema": "editorial_audit.v2", "project_id": "p",
        "draft_revision": 1,
        "edited_body_markdown": "seed",
        "copy_verification": {"verdict": "pass", "hits": []},
        "recommendation": "promote",
    }, root=tmp_path)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p", "source_draft_revision": 2,
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="audit approved draft revision"):
        runner.run_stage("job-staleaudit", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_foreign_project_audit_cannot_approve(tmp_path, monkeypatch):
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-xaudit", ["e1"], approved=False)
    write_artifact("job-xaudit", "audit", {
        "schema": "editorial_audit.v2", "project_id": "elsewhere",
        "edited_body_markdown": "seed",
        "copy_verification": {"verdict": "pass", "hits": []},
        "recommendation": "promote",
    }, root=tmp_path)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="audit project"):
        runner.run_stage("job-xaudit", "repurposing", "m", None, api_key="k", root=tmp_path)




# --- review round 6: generative oracle for the contact classifier ---------
#
# Built from grammars rather than a fixture list, so the decision is
# exercised across the space instead of the examples that happened to be
# reported. Both error directions are asserted.

_INTENTS = ["Call", "Text", "Dial", "Reach us at", "Phone", "Ring"]
_DIALABLE = [
    "1-800-GOT-JUNK", "1-800-FLOWERS",          # vanity, letters
    "+442079460958", "+44 20 7946 0958",        # E.164
    "0044 20 7946 0958",                        # 00 prefix
    "(555) 123-4567", "555-123-4567",           # NANP
    "555.123.4567", "5551234567",               # separators / unbroken
    "07700 900123", "07700900123",              # local, long
]
_DESCRIPTIVE = [
    "RGB palette 255 255 255", "canvas 1920 1080 pixels",
    "calendar showing 2026-07-25", "invoice dated 07/25/2026",
    "a wall clock showing 9:45", "a room with 3 windows and 2 chairs",
    "address plaque reading 12 Elm", "serial 12345678 engraved on a plate",
    "ISO 4217 currency chart", "recipe calls for 2 cups and 3 eggs",
    "a tidy desk in soft morning light",
]
_SCENES = ["a poster of {}", "signage reading {}", "{} on a storefront window"]


def _prompt_verdict(text, tmp_path, monkeypatch, job):
    _seed_draft(tmp_path, job, ["e1"], approved=False)
    reply = json.dumps({"schema": "image_prompt.v1", "project_id": "p",
                        "prompts": [{"purpose": "hero", "prompt_text": text}]})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage(job, "image_prompt", "m", None, api_key="k", root=tmp_path)
    return json.loads(Path(rec["path"]).read_text())["copy_verification"]


@pytest.mark.parametrize("number", _DIALABLE)
@pytest.mark.parametrize("intent", _INTENTS)
def test_oracle_dial_intent_plus_number_always_fails(number, intent, tmp_path, monkeypatch):
    cv = _prompt_verdict(f"{intent} {number} today", tmp_path, monkeypatch, "job-or1")
    assert cv["verdict"] == "fail", (intent, number, cv)


@pytest.mark.parametrize("text", _DESCRIPTIVE)
@pytest.mark.parametrize("scene", _SCENES)
def test_oracle_descriptive_numbers_never_fail(text, scene, tmp_path, monkeypatch):
    """No dial intent and no dialable structure -> digits are description."""
    cv = _prompt_verdict(scene.format(text), tmp_path, monkeypatch, "job-or2")
    assert cv["verdict"] == "pass", (scene, text, cv)


@pytest.mark.parametrize("number", ["+442079460958", "(555) 123-4567", "555-123-4567"])
def test_oracle_structural_forms_fail_without_intent(number, tmp_path, monkeypatch):
    """E.164 and NANP are unambiguous: no intent word needed."""
    cv = _prompt_verdict(f"a poster showing {number}", tmp_path, monkeypatch, "job-or3")
    assert cv["verdict"] == "fail", (number, cv)


@pytest.mark.parametrize("addr", [
    "bob@example.com", "josé@example.com", "user@例え.テスト", "a.b+tag@sub.domain.co.uk",
])
def test_oracle_email_any_script_fails(addr, tmp_path, monkeypatch):
    cv = _prompt_verdict(f"a card reading {addr}", tmp_path, monkeypatch, "job-or4")
    assert cv["verdict"] == "fail", (addr, cv)


@pytest.mark.parametrize("separator", [".", "\u3002", "\uff0e", "\uff61"])
def test_oracle_idna_equivalent_domain_separators_fail(
    separator, tmp_path, monkeypatch
):
    """Every IDNA domain-label separator has the ASCII-dot decision."""
    address = f"user@例え{separator}テスト"
    cv = _prompt_verdict(
        f"a card reading {address}", tmp_path, monkeypatch, "job-idna"
    )
    assert cv["verdict"] == "fail", (address, cv)


# --- review round 7 on #2192 ---


_CASINGS = ["1-800-flowers", "1-800-FLOWERS", "1-800-Flowers", "1-800-gOt-JuNk"]


@pytest.mark.parametrize("number", _CASINGS)
@pytest.mark.parametrize("intent", ["Call", "Dial", "Text"])
def test_vanity_recognition_is_case_independent(number, intent, tmp_path, monkeypatch):
    """Vanity spelling is case-insensitive; attachment (hyphen vs space) is
    what separates the number from the next word."""
    cv = _prompt_verdict(f"{intent} {number} today", tmp_path, monkeypatch, "job-case")
    assert cv["verdict"] == "fail", (intent, number, cv)


@pytest.mark.parametrize("trailing", ["today", "now", "for details", "to book"])
def test_trailing_lowercase_word_not_absorbed(trailing, tmp_path, monkeypatch):
    """The other side: a space-joined word must not extend the token past
    the E.164 bound and thereby hide a real number."""
    cv = _prompt_verdict(f"Ring 07700 900123 {trailing}", tmp_path, monkeypatch, "job-trail")
    assert cv["verdict"] == "fail", (trailing, cv)


def test_same_revision_draft_replacement_invalidates_approval(tmp_path, monkeypatch):
    """A rerun of the draft stage keeps revision 1 but changes the body, so
    the old approval must not carry to text nobody reviewed."""
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-swap", ["e1"])          # draft + matching audit
    write_artifact("job-swap", "draft", {              # rerun, same revision
        "schema": "draft.v1", "project_id": "p", "revision": 1,
        "body_markdown": "completely different body",
        "claims": [{"text": "claim e2", "source_id": "e2"}],
    }, root=tmp_path)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e2"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="does not match the draft currently"):
        runner.run_stage("job-swap", "repurposing", "m", None, api_key="k", root=tmp_path)


def test_image_prompt_readiness_also_content_bound(tmp_path, monkeypatch):
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-swap2", ["e1"])
    write_artifact("job-swap2", "draft", {
        "schema": "draft.v1", "project_id": "p", "revision": 1,
        "body_markdown": "replaced body", "claims": [{"text": "c", "source_id": "e1"}],
    }, root=tmp_path)
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "a tidy desk"}],
        "ready_to_generate": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="does not match the draft currently"):
        runner.run_stage("job-swap2", "image_prompt", "m", None, api_key="k", root=tmp_path)


def test_audit_stage_stamps_the_fingerprint(tmp_path, monkeypatch):
    """The binding is runner-set, not worker-supplied."""
    _seed_draft(tmp_path, "job-stamp", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "editorial_audit.v2", "project_id": "p",
        "edited_body_markdown": "Clean edited copy.",
        "recommendation": "revise",
        "source_draft_fingerprint": "worker-supplied-lie",
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-stamp", "audit", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["source_draft_fingerprint"] == runner._draft_fingerprint("job-stamp", tmp_path)
    assert stored["source_draft_fingerprint"] != "worker-supplied-lie"


def test_source_bound_stage_rejects_prebuilt_prompt_before_dispatch(
    tmp_path, monkeypatch
):
    """Already-built text cannot claim the independently snapshotted draft."""
    _seed_draft(tmp_path, "job-prebuilt", ["e1"], approved=False)
    called = False

    def unexpected_worker(*args, **kwargs):
        nonlocal called
        called = True
        return _editor_json("unreachable", "revise")

    monkeypatch.setattr(runner, "call_worker", unexpected_worker)
    with pytest.raises(ArtifactStoreError, match="runner-owned prompt"):
        runner.run_stage(
            "job-prebuilt",
            "audit",
            "m",
            "request already built from source A",
            api_key="k",
            root=tmp_path,
        )
    assert called is False


def test_runner_prompt_uses_draft_current_at_run_stage_entry(tmp_path, monkeypatch):
    """A pre-entry A->B replacement makes the worker receive B, never stale A."""
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-pre-entry", ["e1"], approved=False)
    write_artifact(
        "job-pre-entry",
        "draft",
        {
            "schema": "draft.v1",
            "project_id": "p",
            "revision": 1,
            "body_markdown": "source B committed before run_stage",
            "claims": [{"text": "claim e1 from B", "source_id": "e1"}],
        },
        root=tmp_path,
    )
    observed = []

    def worker(_model, user_content, **_kwargs):
        observed.append(user_content)
        return _editor_json("Worker response about source B.", "revise")

    monkeypatch.setattr(runner, "call_worker", worker)
    rec = runner.run_stage(
        "job-pre-entry",
        "audit",
        "m",
        None,
        api_key="k",
        root=tmp_path,
    )

    stored = json.loads(Path(rec["path"]).read_text())
    assert len(observed) == 1
    assert "source B committed before run_stage" in observed[0]
    assert "claim e1 from B" in observed[0]
    assert stored["source_draft_fingerprint"] == runner._draft_fingerprint(
        "job-pre-entry", tmp_path
    )


@pytest.mark.parametrize("stage", ["audit", "audit-v2", "repurposing", "image_prompt"])
def test_source_bound_stage_rejects_callback_that_ignores_draft(
    stage, tmp_path, monkeypatch
):
    """No source stage lets a stale lambda inherit the current fingerprint."""
    job_id = f"job-ignored-draft-{stage}"
    _seed_draft(tmp_path, job_id, ["e1"], approved=False)
    called = False

    def unexpected_worker(*args, **kwargs):
        nonlocal called
        called = True
        return _editor_json("unreachable", "revise")

    monkeypatch.setattr(runner, "call_worker", unexpected_worker)
    with pytest.raises(ArtifactStoreError, match="runner-owned prompt"):
        runner.run_stage(
            job_id,
            stage,
            "m",
            lambda _draft: "stale prompt from source A",
            api_key="k",
            root=tmp_path,
        )
    assert called is False


@pytest.mark.parametrize("stage", ["audit", "audit-v2", "repurposing", "image_prompt"])
def test_source_bound_stage_requires_committed_draft_before_dispatch(
    stage, tmp_path, monkeypatch
):
    """A missing committed draft is not a source snapshot and must not become
    `Committed draft JSON:null`."""
    called = False

    def unexpected_worker(*args, **kwargs):
        nonlocal called
        called = True
        return _editor_json("unreachable", "revise")

    monkeypatch.setattr(runner, "call_worker", unexpected_worker)
    with pytest.raises(ArtifactStoreError, match="requires a committed draft"):
        runner.run_stage(
            f"job-no-draft-{stage}",
            stage,
            "m",
            None,
            api_key="k",
            root=tmp_path,
        )
    assert called is False


def test_audit_rejects_draft_replaced_while_worker_runs(tmp_path, monkeypatch):
    """The audit fingerprint binds the pre-dispatch source, not the draft that
    happens to be current after the worker returns."""
    from atlas_brain.services.content_factory_store import write_artifact

    _seed_draft(tmp_path, "job-dispatch-audit", ["e1"], approved=False)

    def replace_draft_during_worker(*args, **kwargs):
        write_artifact("job-dispatch-audit", "draft", {
            "schema": "draft.v1",
            "project_id": "p",
            "revision": 1,
            "body_markdown": "same revision, different source B",
            "claims": [{"text": "claim e1 changed", "source_id": "e1"}],
        }, root=tmp_path)
        return _editor_json("Worker response about source A.", "revise")

    monkeypatch.setattr(runner, "call_worker", replace_draft_during_worker)
    with pytest.raises(ArtifactStoreError, match="changed while the worker was running"):
        runner.run_stage(
            "job-dispatch-audit",
            "audit",
            "m",
            None,
            api_key="k",
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("stage", "artifact"),
    [
        (
            "repurposing",
            {
                "schema": "repurposing.v1",
                "project_id": "p",
                "variants": [{
                    "channel": "email",
                    "body_markdown": "Intermediate copy about source A.",
                    "derived_from_claims": ["e1"],
                }],
                "ready_to_publish": False,
            },
        ),
        (
            "image_prompt",
            {
                "schema": "image_prompt.v1",
                "project_id": "p",
                "prompts": [{
                    "purpose": "hero",
                    "prompt_text": "an image derived from source A",
                }],
                "ready_to_generate": False,
            },
        ),
    ],
)
def test_phase6_rejects_draft_replaced_while_worker_runs(
    stage, artifact, tmp_path, monkeypatch
):
    """Equivalent pre-dispatch binding applies to both Phase 6 workers, even
    before their intermediate artifacts claim readiness."""
    from atlas_brain.services.content_factory_store import write_artifact

    job_id = f"job-dispatch-{stage}"
    _seed_draft(tmp_path, job_id, ["e1"])

    def replace_draft_during_worker(*args, **kwargs):
        write_artifact(job_id, "draft", {
            "schema": "draft.v1",
            "project_id": "p",
            "revision": 1,
            "body_markdown": "same revision, different source B",
            "claims": [{"text": "claim e1 changed", "source_id": "e1"}],
        }, root=tmp_path)
        return json.dumps(artifact)

    monkeypatch.setattr(runner, "call_worker", replace_draft_during_worker)
    with pytest.raises(ArtifactStoreError, match="changed while the worker was running"):
        runner.run_stage(
            job_id,
            stage,
            "m",
            None,
            api_key="k",
            root=tmp_path,
        )


def test_unchanged_draft_keeps_approval_valid(tmp_path, monkeypatch):
    """The other side: an untouched draft still ships."""
    _seed_draft(tmp_path, "job-stable", ["e1"])
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-stable", "repurposing", "m", None, api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is True


# --- round 8: shape-first PII classification, v3 contract, job lock -------

_DIAL_WORDS = ["call", "calling", "phone", "telephone", "text", "contact", "reach", "dial"]
_DESCRIPTIVE_NUMERALS = ["255 255 255", "1920 1080", "2026-07-25", "100 200 300", "1920x1080"]


@pytest.mark.parametrize("word", _DIAL_WORDS)
@pytest.mark.parametrize("numeral", _DESCRIPTIVE_NUMERALS)
def test_oracle_descriptive_numbers_survive_unrelated_dial_words(word, numeral):
    """Class-closure (#2192 round 8): a descriptive number must still render
    when an UNRELATED dial word appears in the same prompt. The old rule
    searched for intent globally, so "a person calling across a room, RGB
    palette 255 255 255" was rejected as contact PII."""
    prompt = f"a {word} booth in the background, {numeral} colour grade, wide shot"
    assert runner._prompt_contact_hits(prompt) == []


@pytest.mark.parametrize(
    "prompt",
    [
        "a call center scene, 1920 1080 resolution poster",
        "a person calling across a room, RGB palette 255 255 255",
        "phone booth photographed at 100 200 300 RGB",
        "wide shot. call sheet on the table. 1920 1080 export",
        "a telephone on a desk, shot at 24 70 mm",
    ],
)
def test_compound_noun_scenes_render(prompt):
    """The intent words here are noun modifiers, not dial verbs. Separating
    them is why grouped shapes need same-segment government within 2 tokens
    rather than a wider window."""
    assert runner._prompt_contact_hits(prompt) == []


@pytest.mark.parametrize(
    "numeral",
    ["+44 20 7946 0958", "555 234 5678", "555.1234", "1-800-FLOWERS",
     "1-800-flowers", "07700 900123", "5552345678"],
)
@pytest.mark.parametrize(
    "scene", ["a poster of {}", "signage reading {}", "{} on a storefront window"]
)
def test_dialable_shapes_fail_with_no_intent_word(numeral, scene):
    """Unambiguous dial shapes carry their own evidence: no dial verb needed,
    in any scene and any casing."""
    assert runner._prompt_contact_hits(scene.format(numeral)) != []


def test_unbroken_run_is_a_serial_until_intent_appears():
    """Documented residual: an unbroken digit run that is NOT a valid NANP
    number is shape-identical to a serial, so it renders on its own and only
    fails once a dial verb governs it."""
    assert runner._prompt_contact_hits("a plate engraved serial 12345678") == []
    assert runner._prompt_contact_hits("a phone on a desk. serial 12345678 engraved") == []
    assert runner._prompt_contact_hits("text me 12345678 today") != []


def test_v2_audit_stays_frozen_and_readable():
    """#2192 round 8: v2 shipped in #2181, so it must keep validating
    byte-for-byte and must NOT learn the fingerprint field -- extra='forbid'
    means a field added there makes new audits unreadable to a v2 consumer."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2, EditorialAuditV3

    v2_payload = {
        "schema": "editorial_audit.v2",
        "project_id": "p",
        "edited_body_markdown": "Clean copy.",
        "recommendation": "revise",
    }
    EditorialAuditV2.model_validate(v2_payload)  # still readable

    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        EditorialAuditV2.model_validate({**v2_payload, "source_draft_fingerprint": "abc"})

    v3 = EditorialAuditV3.model_validate(
        {**v2_payload, "schema": "editorial_audit.v3", "schema_version": 3,
         "source_draft_fingerprint": "abc"}
    )
    assert v3.source_draft_fingerprint == "abc"


def test_run_stage_normalizes_v2_audit_to_v3(tmp_path, monkeypatch):
    """A v2-tagged worker reply upgrades cleanly, because its metadata is
    self-consistent. The anti-laundering rule still holds for contradictory
    metadata (see test_run_stage_rejects_contradictory_v2_version)."""
    _seed_draft(tmp_path, "job-v23", ["e1"], approved=False)
    reply = json.dumps({
        "schema": "editorial_audit.v2",
        "schema_version": 2,
        "project_id": "p",
        "edited_body_markdown": "Clean copy.",
        "recommendation": "revise",
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-v23", "audit", "m", None, api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["schema"] == "editorial_audit.v3"
    assert stored["schema_version"] == 3


def _job_lock_is_free(job_id, root):
    """Probe the lock from a SECOND file description: flock conflicts between
    separate open() calls even inside one process, so a non-blocking attempt
    reports truthfully whether the runner holds it."""
    import fcntl

    path = Path(root) / ".locks" / f"{job_id}.lock"
    if not path.exists():
        return True
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True


def test_run_stage_holds_job_lock_across_validation_and_write(tmp_path, monkeypatch):
    """#2192 round 8 (TOCTOU): validating the draft fingerprint proves nothing
    if the draft can be replaced before the artifact commits. The lock must be
    HELD while readiness is being decided, and released afterwards."""
    _seed_draft(tmp_path, "job-lock", ["e1"])
    observed = []
    real_enforce = runner._enforce_lineage

    def probing_enforce(artifact, job_id, root):
        observed.append(_job_lock_is_free(job_id, root))
        return real_enforce(artifact, job_id, root)

    monkeypatch.setattr(runner, "_enforce_lineage", probing_enforce)
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p", "source_draft_revision": 1,
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    runner.run_stage("job-lock", "repurposing", "m", None, api_key="k", root=tmp_path)

    assert observed == [False], "readiness was decided outside the job lock"
    assert _job_lock_is_free("job-lock", tmp_path), "lock leaked after run_stage"


def test_job_lock_is_reentrant_within_a_thread(tmp_path):
    """run_stage holds the lock and then calls write_artifact, which takes it
    again -- that must not deadlock."""
    from atlas_brain.services.content_factory_store import job_lock

    with job_lock("job-re", root=tmp_path):
        with job_lock("job-re", root=tmp_path):
            assert not _job_lock_is_free("job-re", tmp_path)
    assert _job_lock_is_free("job-re", tmp_path)


def test_job_lock_excludes_a_second_process(tmp_path):
    """The exclusion is real, not just an in-process flag."""
    from atlas_brain.services.content_factory_store import job_lock

    assert _job_lock_is_free("job-x", tmp_path)
    with job_lock("job-x", root=tmp_path):
        assert not _job_lock_is_free("job-x", tmp_path)
    assert _job_lock_is_free("job-x", tmp_path)


def test_job_lock_identity_includes_the_root(tmp_path):
    """The same job id under two roots is two stores, not a re-entrant lock."""
    from atlas_brain.services.content_factory_store import job_lock

    first = tmp_path / "first"
    second = tmp_path / "second"
    with job_lock("job-root", root=first):
        with job_lock("job-root", root=second):
            assert not _job_lock_is_free("job-root", first)
            assert not _job_lock_is_free("job-root", second)

        # Unwind. Asserting only "both held while nested" cannot distinguish a
        # correct release from one that never happened -- a leaked inner lock
        # looks identical inside the block. Leaving the inner scope must free
        # the inner root and leave the outer one held.
        assert _job_lock_is_free("job-root", second)
        assert not _job_lock_is_free("job-root", first)

    assert _job_lock_is_free("job-root", first)


# --- round 9: vanity grammar both directions and committed-state atomicity ---

_SPEC_DIGITS = ["8", "16", "24", "32", "64", "1920", "1080", "4096"]
_SPEC_WORDS = ["bit", "color", "pixel", "style", "float", "depth", "res"]


@pytest.mark.parametrize("lead", _SPEC_DIGITS)
@pytest.mark.parametrize("word", _SPEC_WORDS)
def test_oracle_hyphenated_renderer_specs_render(lead, word):
    """Grammar-derived (#2192 round 9): letters ATTACHED to digits do not make
    a vanity number. A renderer spec's leading group is not an area code, and
    its keypad mapping is not a valid NANP number, so the whole class renders
    -- "16-bit-color" and "1920-1080-pixel" were rejected before."""
    assert runner._prompt_contact_hits(f"a {lead}-{word}-{word} render, studio light") == []


def test_oracle_every_separator_partition_of_vanity_suffix_fails():
    """Grammar-derived other side: attachment choices cannot make a NANP
    vanity number disappear. This generates every join/space/hyphen/dot
    partition of a held-out suffix rather than listing review spellings."""
    word = "CLEANUP"
    separators = ("", " ", "-", ".")
    for prefix in ("800", "1 800"):
        for choices in product(separators, repeat=len(word) - 1):
            suffix = "".join(
                char + (choices[index] if index < len(choices) else "")
                for index, char in enumerate(word)
            )
            prompt = f"Call {prefix} {suffix}"
            assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize("prefix", ["+44 800", "0044 800", "+81 3"])
@pytest.mark.parametrize("separator", ["", " ", "-", "."])
@pytest.mark.parametrize("word", ["FLOWERS", "CLEANUP", "PLUMBER"])
def test_oracle_international_vanity_with_dial_evidence_fails(
    prefix, separator, word
):
    """Explicit international prefix + intent + keypad spelling is contact PII."""
    suffix = separator.join(word)
    prompt = f"Call {prefix} {suffix} today"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize("prefix", ["+81", "+44", "0044"])
@pytest.mark.parametrize("whitespace", [" ", "\t", "\u00a0"])
@pytest.mark.parametrize("width", range(1, 9))
def test_oracle_international_vanity_accepts_numeric_whitespace_runs(
    prefix, whitespace, width
):
    """Whitespace formatting cannot erase explicit international dial evidence."""
    gap = whitespace * width
    prompt = f"Call {prefix}{gap}3 FLOWERS today"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize("lead", ["212", "305", "415", "617", "800"])
@pytest.mark.parametrize(
    "art_direction",
    ["art deco sign", "blue mural", "soft focus", "red carpet", "new typography"],
)
def test_oracle_three_digit_art_direction_is_not_detached_vanity(
    lead, art_direction
):
    """A keypad coincidence in ordinary prose is not contact evidence."""
    prompt = f"room {lead} {art_direction}, editorial photograph"
    assert runner._prompt_contact_hits(prompt) == [], prompt


@pytest.mark.parametrize("whitespace", ["  ", "\t\t", "\u00a0\u00a0"])
def test_numeric_whitespace_runs_do_not_promote_detached_prose(whitespace):
    """A wider separator is formatting, not dial intent."""
    prompt = f"room 212{whitespace}art deco sign, editorial photograph"
    assert runner._prompt_contact_hits(prompt) == [], prompt


@pytest.mark.parametrize("prefix", ["+44", "\uff0b44", "0044"])
@pytest.mark.parametrize("separator", [" ", "/", "\uff0f", "\uff0e", "\uff0d", "\u00a0"])
@pytest.mark.parametrize("word", ["FLOWERS", "\uff26\uff2c\uff2f\uff37\uff25\uff32\uff33", "\ufb02OWERS"])
def test_oracle_compatibility_equivalent_phonewords_share_one_verdict(
    prefix, separator, word
):
    """NFKC spellings and every admitted separator preserve phone evidence."""
    prompt = f"Call {prefix}{separator}800{separator}{word} today"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize(
    "renderer_spec",
    [
        "\uff11\uff16\uff0f\uff42\uff49\uff54\uff0f\uff43\uff4f\uff4c\uff4f\uff52",
        "\uff11\uff19\uff12\uff10\uff0f\uff11\uff10\uff18\uff10\uff0f\uff50\uff49\uff58\uff45\uff4c",
        "\uff18\uff0d\uff42\uff49\uff54\uff0d\uff53\uff54\uff59\uff4c\uff45",
    ],
)
def test_compatibility_normalization_does_not_promote_renderer_specs(renderer_spec):
    """The normalized parser still requires a dialable prefix/mapping."""
    assert runner._prompt_contact_hits(f"studio render {renderer_spec}") == []


@pytest.mark.parametrize(
    "renderer_value",
    [
        "+1920/1080",
        "+3840/2160",
        "+2026/07/25",
        "+255/255/255",
    ],
)
def test_slash_separated_renderer_numbers_require_dial_evidence(renderer_value):
    """An explicit plus does not turn dimensions, dates, or RGB into a phone."""
    prompt = f"display {renderer_value} landscape artwork"
    assert runner._prompt_contact_hits(prompt) == [], prompt


@pytest.mark.parametrize(
    "dial_candidate",
    [
        "+44/800/FLOWERS",
        "+44/800/3569377",
        "0044/800/FLOWERS",
    ],
)
def test_slash_separated_candidates_with_structural_dial_intent_fail(
    dial_candidate,
):
    """Slash parity remains when direct dial syntax supplies the evidence."""
    prompt = f"Call {dial_candidate} today"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize(
    "dial_candidate",
    [
        "+44 (800) FLOWERS",
        "+44 (800) F-L-O-W-E-R-S",
        "0044 (800) FLOWERS",
    ],
)
def test_parenthesized_phonewords_share_numeric_group_verdict(dial_candidate):
    """Parentheses are formatting inside the same bounded dial grammar."""
    prompt = f"Call {dial_candidate} today"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize("line_break", ["\n", "\r", "\r\n"])
def test_single_line_break_preserves_structural_dial_intent(line_break):
    """One logical CTA line break is formatting, not a dial-evidence boundary."""
    prompt = f"Call{line_break}+44 800 FLOWERS"
    assert runner._prompt_contact_hits(prompt) != [], prompt


@pytest.mark.parametrize("line_break", ["\n\n", "\r\r", "\r\n\r\n"])
def test_paragraph_break_does_not_bridge_dial_intent(line_break):
    """The finite bridge does not govern a candidate in another paragraph."""
    prompt = f"Call{line_break}+44 800 FLOWERS"
    assert runner._prompt_contact_hits(prompt) == [], prompt


@pytest.mark.parametrize("intent", ["Call", "Text", "Contact", "Reach"])
@pytest.mark.parametrize("candidate", ["212 ART DECO", "+44 800 FLOWERS"])
@pytest.mark.parametrize(
    ("bridge", "has_dial_evidence"),
    [
        (" ", True),
        (": ", True),
        (" me ", True),
        (" us at ", True),
        (" for room ", False),
        (" sheet for room ", False),
        (" about suite ", False),
        (". room ", False),
    ],
)
def test_oracle_detached_phonewords_require_structural_bridge(
    intent, candidate, bridge, has_dial_evidence
):
    """Evidence-keyed oracle: syntax, not intent-category proximity, decides."""
    prompt = f"{intent}{bridge}{candidate} sign"
    assert bool(runner._prompt_contact_hits(prompt)) is has_dial_evidence, prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "room 212 art deco contact sheet",
        "room 212 art deco text treatment",
        "poster for room 212 art deco, contact sheet layout",
        "room 800 flowers contact sheet",
    ],
)
def test_trailing_renderer_nouns_do_not_bridge_detached_phonewords(prompt):
    """Renderer terms after an ambiguous candidate are not reverse dial syntax."""
    assert runner._prompt_contact_hits(prompt) == [], prompt


@pytest.mark.parametrize(
    "prompt",
    ["255 255 255 blue tint", "1920 1080 pixel export", "100 200 300 RGB values",
     "a poster of serial 12345678 engraved on a plate"],
)
def test_trailing_word_extension_does_not_over_reach(prompt):
    """Extending a candidate by trailing words must not manufacture a hit:
    the vanity test still demands an area-code prefix and a valid mapping."""
    assert runner._prompt_contact_hits(prompt) == []


def test_failed_commit_restores_previous_artifact_and_index(tmp_path, monkeypatch):
    """An ordinary raised commit failure restores worktree/index hygiene."""
    from atlas_brain.services import content_factory_store as store

    _seed_draft(tmp_path, "job-resid", ["e1"])
    draft_path = Path(tmp_path) / "jobs" / "job-resid" / "draft.json"
    before = draft_path.read_bytes()

    # Exercise the real Git process: add succeeds, while commit rejects the
    # malformed author date after the target has been written and staged.
    monkeypatch.setenv("GIT_AUTHOR_DATE", "not-a-git-date")
    with pytest.raises(store.ArtifactStoreError):
        store.write_artifact("job-resid", "draft", {
            "schema": "draft.v1", "project_id": "p", "revision": 2,
            "body_markdown": "REPLACED body nobody approved",
            "claims": [{"text": "claim e1", "source_id": "e1"}],
        }, root=tmp_path)

    assert draft_path.read_bytes() == before, "failed write left residue on disk"
    staged = subprocess.run(
        ["git", "-C", str(draft_path.parent), "diff", "--cached", "--name-only"],
        capture_output=True, text=True,
    ).stdout
    assert "draft.json" not in staged, "failed write left draft.json staged"


def test_failed_first_write_removes_the_file(tmp_path, monkeypatch):
    """A stage that had no previous artifact must not leave a partial one."""
    from atlas_brain.services import content_factory_store as store

    monkeypatch.setenv("GIT_AUTHOR_DATE", "not-a-git-date")
    with pytest.raises(store.ArtifactStoreError):
        store.write_artifact("job-first", "draft", {
            "schema": "draft.v1", "project_id": "p", "revision": 1,
            "body_markdown": "body", "claims": [{"text": "c", "source_id": "e1"}],
        }, root=tmp_path)
    job = Path(tmp_path) / "jobs" / "job-first"
    assert not (job / "draft.json").exists()
    assert subprocess.run(
        ["git", "-C", str(job), "ls-files", "--error-unmatch", "draft.json"],
        capture_output=True,
    ).returncode != 0


def test_uncommitted_crash_residue_cannot_authorize_ready_artifact(
    tmp_path, monkeypatch
):
    """Canonical readers use Git HEAD, so even residue left when cleanup never
    runs cannot replace the draft/audit pair that authorizes readiness."""
    _seed_draft(tmp_path, "job-crash", ["e1"])
    job = Path(tmp_path) / "jobs" / "job-crash"
    draft_path = job / "draft.json"
    committed = subprocess.run(
        ["git", "-C", str(job), "show", "HEAD:draft.json"],
        check=True,
        capture_output=True,
    ).stdout
    draft_path.write_text(json.dumps({
        "schema": "draft.v1",
        "project_id": "p",
        "revision": 1,
        "body_markdown": "UNCOMMITTED replacement nobody approved",
        "claims": [{"text": "other", "source_id": "unapproved"}],
    }))
    subprocess.run(
        ["git", "-C", str(job), "add", "--", "draft.json"],
        check=True,
    )

    reply = json.dumps({
        "schema": "repurposing.v1",
        "project_id": "p",
        "source_draft_revision": 1,
        "variants": [{
            "channel": "email",
            "body_markdown": "Clean copy.",
            "derived_from_claims": ["e1"],
        }],
        "ready_to_publish": True,
    })
    response = json.dumps({
        "choices": [{"message": {"content": reply}}],
    }).encode()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(response),
    )
    record = runner.run_stage(
        "job-crash",
        "repurposing",
        "m",
        None,
        api_key="k",
        root=tmp_path,
    )

    stored = json.loads(Path(record["path"]).read_text())
    assert stored["ready_to_publish"] is True
    assert subprocess.run(
        ["git", "-C", str(job), "show", "HEAD:draft.json"],
        check=True,
        capture_output=True,
    ).stdout == committed
    assert b"UNCOMMITTED replacement" in draft_path.read_bytes()


# --- #2201: default-ignorable PII-gate bypass -----------------------------

# One representative per default-ignorable family a producer can reach for.
_ZERO_WIDTH = [
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "﻿",  # zero width no-break space / BOM
    "­",  # soft hyphen
    "⁠",  # word joiner
    "️",  # variation selector-16
    "᠎",  # Mongolian vowel separator
]
# Every seam in a phone string: before the prefix, inside it, between numeric
# groups, inside a group, and inside the vanity suffix.
_SEAM_TEMPLATES = [
    "Call +44{z}800 FLOWERS",
    "Call +{z}44 800 FLOWERS",
    "call 1-800-FLOW{z}ERS now",
    "reach 555-123{z}-4567",
    "reach 555{z}-123-4567",
    "call me at 555{z}1234567",
    "text 5552345{z}678",
]


@pytest.mark.parametrize("zw", _ZERO_WIDTH)
@pytest.mark.parametrize("template", _SEAM_TEMPLATES)
def test_zero_width_insertion_cannot_bypass_prompt_pii_gate(zw, template):
    """#2201: a default-ignorable character renders as nothing, so inserting
    one must not change the verdict. Before the fix it defeated the prefix,
    the structural NANP pattern and the vanity suffix alike."""
    assert runner._prompt_contact_hits(template.format(z=zw)) != []


@pytest.mark.parametrize("zw", _ZERO_WIDTH)
def test_zero_width_insertion_cannot_bypass_email_gates(zw):
    """Same class on the address side, in both the prompt gate and the
    merged body-copy gate."""
    from atlas_brain.services.content_factory_copy_verification import verify_copy

    assert runner._prompt_contact_hits("write to a" + zw + "lice@example.com") != []
    assert runner._prompt_contact_hits("write to alice@example" + zw + ".com") != []
    assert verify_copy("write to alice@example" + zw + ".com").verdict == "fail"


@pytest.mark.parametrize("zw", _ZERO_WIDTH)
def test_zero_width_insertion_cannot_bypass_body_copy_phone_gate(zw):
    """The bypass also defeated `verify_copy`, which gates the editorial
    audit's promote decision -- a wider surface than the prompt gate."""
    from atlas_brain.services.content_factory_copy_verification import verify_copy

    assert verify_copy("Call 555-123" + zw + "-4567 today").verdict == "fail"


@pytest.mark.parametrize("zw", _ZERO_WIDTH)
def test_redacted_evidence_never_carries_a_hidden_address(zw):
    """The digit theorem covered phone digits, but an address's LETTERS are
    not digits: a zero-width character made it evade the email pattern and
    persist intact in claim evidence."""
    from atlas_brain.services.content_factory_copy_verification import _redact_pii

    redacted = _redact_pii("reach alice@example" + zw + ".com now")
    assert "alice@example" not in redacted
    assert "<redacted-email>" in redacted


@pytest.mark.parametrize(
    "prompt",
    [
        "a person calling across a room, RGB palette 255 255 255",
        "a 16-bit-color render, studio light",
        "a call center scene, 1920 1080 resolution poster",
        "a poster of serial 12345678 engraved on a plate",
    ],
)
def test_scan_view_does_not_create_false_positives(prompt):
    """Stripping must not join tokens into a hit that was not there."""
    assert runner._prompt_contact_hits(prompt) == []


def test_scan_view_keeps_whitespace_controls():
    """Newlines and tabs are real separators: dropping them would glue a word
    onto a following number and manufacture candidates."""
    from atlas_brain.services.content_factory_copy_verification import scan_view

    assert scan_view("room 255\nx255 blue\tgrade") == "room 255\nx255 blue\tgrade"
    assert scan_view("a​b c") == "ab c"


# --- #2201 round 12: recipient-marking bridge ----------------------------

_MESSAGE_INTENTS = ["Call", "Text", "SMS", "WhatsApp", "dial", "phone", "contact"]
_BRIDGES = ["", "to ", "me at ", "us on ", "to me at "]


@pytest.mark.parametrize("intent", _MESSAGE_INTENTS)
@pytest.mark.parametrize("bridge", _BRIDGES)
def test_recipient_bridge_cross_product_fails(intent, bridge):
    """#2201: the bridge vocabulary omitted the dative `to`, so `Text to +44
    800 FLOWERS` passed while `Text +44 800 FLOWERS` failed -- a one-word
    connector defeated the gate. Evidence-keyed across every admitted intent
    and bridge form."""
    assert runner._prompt_contact_hits(f"{intent} {bridge}+44 800 FLOWERS") != []


@pytest.mark.parametrize(
    "prompt",
    [
        # `to` must bridge only a RECIPIENT, never descriptive prose.
        "text to room 212 art deco",
        "poster for room 212 art deco, contact sheet layout",
        "text treatment for room 212 art deco",
        # Possessive determiners are deliberately NOT bridge words: they occur
        # in ordinary renderer prose, so admitting them would read a
        # typography instruction as a dial instruction.
        "text your 1920 1080 export",
        "call our 1920 1080 render sheet",
        "a call center scene, 1920 1080 resolution poster",
    ],
)
def test_recipient_bridge_does_not_admit_renderer_prose(prompt):
    assert runner._prompt_contact_hits(prompt) == []


# --- #2201 round 13: one shared default-ignorable predicate ---------------


def _all_default_ignorable_samples():
    """Sampled from the SHARED range table, not a hand-written list.

    Deriving the corpus from the predicate is what makes this class-closed:
    a codepoint added to the table is covered automatically, and a partial
    second definition (the U+034F miss) cannot pass unnoticed.
    """
    from atlas_brain.schemas.content_factory import _DEFAULT_IGNORABLE_RANGES

    samples = []
    for low, high in _DEFAULT_IGNORABLE_RANGES:
        for codepoint in {low, (low + high) // 2, high}:
            samples.append(chr(codepoint))
    return samples


_DEFAULT_IGNORABLE_SAMPLES = _all_default_ignorable_samples()
_CONTACT_SEAMS = [
    "Call +44{z}800 FLOWERS",
    "Call +{z}44 800 FLOWERS",
    "call 1-800-FLOW{z}ERS now",
    "reach 555-123{z}-4567",
    "call me at 555{z}1234567",
]


@pytest.mark.parametrize("ignorable", _DEFAULT_IGNORABLE_SAMPLES)
@pytest.mark.parametrize("seam", _CONTACT_SEAMS)
def test_every_default_ignorable_fails_the_prompt_gate(ignorable, seam):
    """#2201 round 13: `scan_view` tested Cf/Cc plus a partial hand-built set,
    so U+034F (category Mn) still defeated the gate after the zero-width class
    was closed. Any rule phrased as "Cf plus some ranges" leaves the bypass
    open by construction -- the shared predicate is the only correct test."""
    assert runner._prompt_contact_hits(seam.format(z=ignorable)) != []


@pytest.mark.parametrize("ignorable", _DEFAULT_IGNORABLE_SAMPLES)
def test_every_default_ignorable_fails_the_body_copy_gate(ignorable):
    from atlas_brain.services.content_factory_copy_verification import verify_copy

    assert verify_copy("Call 555-123" + ignorable + "-4567 today").verdict == "fail"


@pytest.mark.parametrize("ignorable", _DEFAULT_IGNORABLE_SAMPLES)
def test_no_default_ignorable_leaks_an_address_through_redaction(ignorable):
    """U+034F left the address FULLY intact in persisted claim evidence --
    the digit theorem does not help, because an address has no digits."""
    from atlas_brain.services.content_factory_copy_verification import _redact_pii

    redacted = _redact_pii("reach alice@example" + ignorable + ".com now")
    assert "alice@example" not in redacted


def test_scan_view_and_routing_key_share_one_predicate():
    """Two definitions is the defect. Assert the scan actually uses the
    shared predicate, so a future local copy fails here first."""
    from atlas_brain.schemas.content_factory import is_default_ignorable
    from atlas_brain.services.content_factory_copy_verification import scan_view

    for ignorable in _DEFAULT_IGNORABLE_SAMPLES:
        assert is_default_ignorable(ignorable)
        assert scan_view("a" + ignorable + "b") == "ab"


# --- #2201 round 14: dial-verb evidence + stage admission ----------------

_RENDERER_VERBS = ["center", "place", "align", "overlay", "position", "set"]
_RENDERER_BRIDGES = ["on", "at"]
_RENDERER_DIMENSIONS = ["1920 1080", "255 255 255", "100 200 300", "12 34 56 78"]


@pytest.mark.parametrize("verb", _RENDERER_VERBS)
@pytest.mark.parametrize("bridge", _RENDERER_BRIDGES)
@pytest.mark.parametrize("dimension", _RENDERER_DIMENSIONS)
def test_overloaded_marker_as_object_is_renderer_prose(verb, bridge, dimension):
    """#2201 round 14: `text` is a dial verb in "text me at <number>" and a
    typography noun in "center text on 1920 1080 canvas" -- both put an
    admitted bridge word between marker and number, so the finite bridge
    vocabulary alone cannot separate them. Position does: an imperative CTA
    heads its clause; here the marker is the object of an earlier verb."""
    assert runner._prompt_contact_hits(f"{verb} text {bridge} {dimension} canvas") == []


@pytest.mark.parametrize(
    "prompt",
    [
        "Text to +44 800 FLOWERS",
        "Text me at +44 800 FLOWERS",
        "Contact us on +44 800 FLOWERS",
        "text me 12 34 56 78",
        # Unambiguous markers are NOT position-restricted, so an ordinary CTA
        # with a discourse lead-in still fails.
        "please call me at 12 34 56 78",
        "now dial me at 12 34 56 78",
    ],
)
def test_clause_heading_marker_still_governs(prompt):
    assert runner._prompt_contact_hits(prompt) != []


def test_audit_v2_stage_rejects_a_mislabeled_artifact(tmp_path):
    """#2201 round 14: the runner names `audit-v2` as a source-bound stage but
    the store had no entry, so it was a CUSTOM stage admitting ANY artifact --
    a content_brief committed successfully as audit-v2.json, and that schema
    is outside _SOURCE_BOUND_SCHEMAS so the source comparison was skipped."""
    from atlas_brain.services.content_factory_store import (
        STAGE_SCHEMAS,
        ArtifactStoreError,
        write_artifact,
    )

    assert "audit-v2" in STAGE_SCHEMAS
    with pytest.raises(ArtifactStoreError, match="stage/schema mismatch"):
        write_artifact(
            "job-av2",
            "audit-v2",
            {"schema": "content_brief.v1", "project_id": "p", "request_raw": "x"},
            root=tmp_path,
        )


def test_audit_v2_stage_accepts_an_editorial_audit(tmp_path):
    """The other side: the stage must still admit what it is for."""
    from atlas_brain.services.content_factory_store import write_artifact

    record = write_artifact(
        "job-av2ok",
        "audit-v2",
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "edited_body_markdown": "Clean copy.",
            "recommendation": "revise",
        },
        root=tmp_path,
    )
    assert record["schema"] == "editorial_audit.v2"
