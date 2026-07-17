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
        "schema": "editorial_audit.v1",
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
