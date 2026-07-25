"""Tests for the Content Factory stage runner.

Open WebUI (the HTTP chat call) is the external boundary and is mocked; the store
and contracts are real, so run_stage's extract + validate + persist path is
exercised for real against a temp filesystem.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from pydantic import ValidationError

import atlas_brain.services.content_factory_runner as runner
from atlas_brain.services.content_factory_store import ArtifactStoreError
from atlas_brain.services.content_factory_runner import (
    WorkerError,
    call_worker,
    extract_json,
    run_stage,
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


def test_editor_stage_injects_deterministic_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json(_CLEAN_BODY, "promote"))
    rec = runner.run_stage("job1", "audit", "cf-editor", "req", api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "pass"
    assert stored["recommendation"] == "promote"  # clean copy may promote


def test_editor_worker_cannot_self_promote_overclaim(tmp_path, monkeypatch):
    # Worker claims a passing verdict + promote on copy that actually overclaims.
    reply = _editor_json(_BAD_BODY, "promote", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):  # injected fail vs promote -> #2116 guard rejects
        runner.run_stage("job1", "audit", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job1", root=tmp_path).exists()


def test_editor_overclaim_revise_persists_with_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json(_BAD_BODY, "revise"))
    rec = runner.run_stage("job1", "audit", "m", "req", api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("guaranteed-savings" in h for h in stored["copy_verification"]["hits"])


def test_editor_worker_claimed_verdict_is_overridden(tmp_path, monkeypatch):
    # Worker asserts pass on bad copy but only recommends revise; the deterministic
    # verdict must still overwrite the worker's claim.
    reply = _editor_json(_BAD_BODY, "revise", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job1", "audit", "m", "req", api_key="k", root=tmp_path)
    assert _stored(rec)["copy_verification"]["verdict"] == "fail"


def test_non_editor_stage_gets_no_copy_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: BRIEF_JSON)
    rec = runner.run_stage("job1", "brief", "m", "req", api_key="k", root=tmp_path)
    assert "copy_verification" not in _stored(rec)


def test_empty_edited_copy_cannot_promote(tmp_path, monkeypatch):
    # Fail closed: an empty edited body means nothing was verified, so a worker cannot
    # self-promote by omitting the edited copy.
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json("", "promote"))
    with pytest.raises(ValidationError):
        runner.run_stage("job1", "audit", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job1", root=tmp_path).exists()


def test_empty_edited_copy_revise_persists_with_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: _editor_json("   ", "revise"))
    rec = runner.run_stage("job1", "audit", "m", "req", api_key="k", root=tmp_path)
    stored = _stored(rec)
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("unverified-copy" in h for h in stored["copy_verification"]["hits"])


def test_custom_stage_audit_is_also_gated(tmp_path, monkeypatch):
    # A custom (non-"audit") stage emitting editorial_audit.v1 must not bypass the gate:
    # gating is by schema, so its self-promotion of overclaiming copy is still rejected.
    reply = _editor_json(_BAD_BODY, "promote", copy_verification={"verdict": "pass", "hits": []})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):
        runner.run_stage("job1", "audit-v2", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job1", root=tmp_path).exists()


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
    reply = json.dumps(
        {
            "schema": "editorial_audit.v1",
            "project_id": "p",
            "edited_body_markdown": "We draft the answer for every repeated ticket.",
            "recommendation": "revise",
        }
    )
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-adv", "audit", "cf-editor-verifier", "req", api_key="k", root=tmp_path)
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
        runner.run_stage("job-v2v", "audit", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job-v2v", root=tmp_path).exists()


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
        runner.run_stage("job-rp", "repurposing", "m", "req", api_key="k", root=tmp_path)


def test_run_stage_persists_clean_variants_with_computed_verdicts(tmp_path, monkeypatch):
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
    rec = runner.run_stage("job-rp2", "repurposing", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["variants"][0]["copy_verification"]["verdict"] == "pass"
    assert stored["variants"][0]["advisory_warnings"][-1].startswith("reminder:")


def test_run_stage_blank_variant_body_fails_closed(tmp_path, monkeypatch):
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
        runner.run_stage("job-rp3", "repurposing", "m", "req", api_key="k", root=tmp_path)


def test_run_stage_gates_image_prompt_text(tmp_path, monkeypatch):
    """Banned copy inside a PROMPT would be rendered into the artwork, where
    no text check would see it -- the gate runs on the prompt itself."""
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
    rec = runner.run_stage("job-img", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    assert any("guaranteed-savings" in hit for hit in stored["copy_verification"]["hits"])


def test_run_stage_image_prompt_pii_is_caught(tmp_path, monkeypatch):
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
    rec = runner.run_stage("job-img2", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    # The VERDICT never carries the raw value (the persisted-evidence
    # theorem); the prompt text itself is the artifact's payload and stays
    # so a human can see what to fix -- same as a draft body.
    assert "bob@example.com" not in json.dumps(stored["copy_verification"])
    assert stored["copy_verification"]["hits"] == ["prompt 1: email: <redacted>"]


def test_run_stage_clean_image_prompt_passes(tmp_path, monkeypatch):
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
    rec = runner.run_stage("job-img3", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass"


def test_stage_schema_mismatch_still_enforced_for_phase6(tmp_path, monkeypatch):
    """A repurposing artifact cannot land under the image_prompt stage."""
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
        runner.run_stage("job-mix", "image_prompt", "m", "req", api_key="k", root=tmp_path)


def test_negative_prompt_naming_banned_terms_still_passes(tmp_path, monkeypatch):
    """Guard's second side: a negative prompt is an EXCLUSION list, so naming
    a banned phrase there is the correct designer response to the threat --
    it must not trip the gate."""
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{
            "purpose": "hero",
            "prompt_text": "a tidy office desk in soft morning light",
            "negative_prompt": "blurry, watermark, text, guaranteed savings, phone number",
        }],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-neg", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass"
    assert stored["prompts"][0]["negative_prompt"].startswith("blurry")


def test_positive_prompt_with_banned_claim_still_fails(tmp_path, monkeypatch):
    """The other side of the same guard stays closed."""
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{
            "purpose": "hero",
            "prompt_text": "poster reading guaranteed savings",
            "negative_prompt": "blurry",
        }],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-pos", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"


def test_worker_cannot_self_declare_ready_to_generate(tmp_path, monkeypatch):
    """The runner recomputes the verdict, so a failing prompt set cannot be
    persisted as renderable no matter what the worker claims."""
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "poster reading guaranteed savings"}],
        "copy_verification": {"verdict": "pass", "hits": []},
        "ready_to_generate": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ValidationError):
        runner.run_stage("job-selfgen", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    assert not job_dir("job-selfgen", root=tmp_path).exists()


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
        runner.run_stage("job-lin", "repurposing", "m", "req", api_key="k", root=tmp_path)


def test_real_lineage_ships(tmp_path, monkeypatch):
    _seed_draft(tmp_path, "job-lin2", ["e1", "e2"])
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e2"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-lin2", "repurposing", "m", "req", api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is True


def test_unready_package_skips_lineage_check(tmp_path, monkeypatch):
    """An unready package is a legitimate intermediate state."""
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["not-yet-real"]}],
        "ready_to_publish": False,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-lin3", "repurposing", "m", "req", api_key="k", root=tmp_path)
    assert Path(rec["path"]).exists()


def test_missing_draft_fails_closed_on_ship(tmp_path, monkeypatch):
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["e1"]}],
        "ready_to_publish": True,
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    with pytest.raises(ArtifactStoreError, match="readable draft artifact"):
        runner.run_stage("job-nodraft", "repurposing", "m", "req", api_key="k", root=tmp_path)


def test_international_phone_in_prompt_fails(tmp_path, monkeypatch):
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": "Call us at +442079460958"}],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-intl", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "fail"
    assert "442079460958" not in json.dumps(stored["copy_verification"])


def test_prompts_verified_independently_no_cross_synthesis(tmp_path, monkeypatch):
    """Joining items must not synthesize a claim no single prompt makes."""
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [
            {"purpose": "a", "prompt_text": "a warm kitchen, results guaranteed"},
            {"purpose": "b", "prompt_text": "savings account paperwork on a desk"},
        ],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-split", "image_prompt", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["copy_verification"]["verdict"] == "pass", stored["copy_verification"]


def test_hits_identify_the_offending_prompt(tmp_path, monkeypatch):
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [
            {"purpose": "a", "prompt_text": "a clean desk"},
            {"purpose": "b", "prompt_text": "poster reading guaranteed savings"},
        ],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-which", "image_prompt", "m", "req", api_key="k", root=tmp_path)
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
        runner.run_stage("job-rev", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
    rec = runner.run_stage("job-rev2", "repurposing", "m", "req", api_key="k", root=tmp_path)
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
        runner.run_stage("job-imgrev", "image_prompt", "m", "req", api_key="k", root=tmp_path)


def test_string_false_readiness_persists_as_intermediate(tmp_path, monkeypatch):
    """A weak worker's "false" normalizes to False, so no draft is required
    and the intermediate package persists (runner and schema agree)."""
    reply = json.dumps({
        "schema": "repurposing.v1", "project_id": "p",
        "variants": [{"channel": "x", "body_markdown": "Clean copy.",
                      "derived_from_claims": ["not-yet-real"]}],
        "ready_to_publish": "false",
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-strfalse", "repurposing", "m", "req", api_key="k", root=tmp_path)
    assert json.loads(Path(rec["path"]).read_text())["ready_to_publish"] is False


@pytest.mark.parametrize("text", [
    "poster reading do not guarantee savings",
    "a sign that says we never guarantee savings",
])
def test_negated_banned_phrase_in_prompt_still_fails(text, tmp_path, monkeypatch):
    """Prose negation does not un-draw words a renderer is told to paint."""
    reply = json.dumps({
        "schema": "image_prompt.v1", "project_id": "p",
        "prompts": [{"purpose": "hero", "prompt_text": text}],
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-neg2", "image_prompt", "m", "req", api_key="k", root=tmp_path)
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
        runner.run_stage("job-noaudit", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
        runner.run_stage("job-revise", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
        runner.run_stage("job-xproj", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
        runner.run_stage("job-staleaudit", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
        runner.run_stage("job-xaudit", "repurposing", "m", "req", api_key="k", root=tmp_path)




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
    reply = json.dumps({"schema": "image_prompt.v1", "project_id": "p",
                        "prompts": [{"purpose": "hero", "prompt_text": text}]})
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage(job, "image_prompt", "m", "req", api_key="k", root=tmp_path)
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
        runner.run_stage("job-swap", "repurposing", "m", "req", api_key="k", root=tmp_path)


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
        runner.run_stage("job-swap2", "image_prompt", "m", "req", api_key="k", root=tmp_path)


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
    rec = runner.run_stage("job-stamp", "audit", "m", "req", api_key="k", root=tmp_path)
    stored = json.loads(Path(rec["path"]).read_text())
    assert stored["source_draft_fingerprint"] == runner._draft_fingerprint("job-stamp", tmp_path)
    assert stored["source_draft_fingerprint"] != "worker-supplied-lie"


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
    rec = runner.run_stage("job-stable", "repurposing", "m", "req", api_key="k", root=tmp_path)
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
    reply = json.dumps({
        "schema": "editorial_audit.v2",
        "schema_version": 2,
        "project_id": "p",
        "edited_body_markdown": "Clean copy.",
        "recommendation": "revise",
    })
    monkeypatch.setattr(runner, "call_worker", lambda *a, **k: reply)
    rec = runner.run_stage("job-v23", "audit", "m", "req", api_key="k", root=tmp_path)
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
    runner.run_stage("job-lock", "repurposing", "m", "req", api_key="k", root=tmp_path)

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
