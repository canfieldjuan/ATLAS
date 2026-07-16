"""Tests for the Content Factory artifact store.

Uses a real temp filesystem and real git (not mocks) per the real-adapters rule:
the store's whole job is to persist + version files on disk, so mocking git/fs
would test nothing real.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas_brain.services.content_factory_store import (
    ArtifactStoreError,
    job_dir,
    write_artifact,
)

BRIEF = {
    "schema": "content_brief.v1",
    "project_id": "resolution-audit",
    "request_raw": "x",
}
DRAFT = {"schema": "draft.v1", "project_id": "p", "body_markdown": "hello"}
# Fails a contract invariant: blank source_id on an evidence row.
EVIDENCE_BAD = {
    "schema": "evidence_packet.v1",
    "project_id": "p",
    "evidence": [{"id": "e1", "quote": "q", "source_id": ""}],
}


def _commits(job: Path) -> int:
    r = subprocess.run(
        ["git", "-C", str(job), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    )
    return int(r.stdout.strip()) if r.returncode == 0 else 0


def test_write_valid_artifact_persists_and_commits(tmp_path):
    rec = write_artifact("job1", "brief", BRIEF, root=tmp_path)
    p = Path(rec["path"])
    assert p.exists() and p.name == "brief.json"
    assert json.loads(p.read_text())["schema"] == "content_brief.v1"
    assert rec["sha"] and rec["schema"] == "content_brief.v1"
    assert _commits(job_dir("job1", root=tmp_path)) == 1


def test_canonical_schema_key_persisted(tmp_path):
    # The model attribute is artifact_schema, but the file must use "schema".
    rec = write_artifact("job1", "brief", BRIEF, root=tmp_path)
    data = json.loads(Path(rec["path"]).read_text())
    assert "schema" in data and "artifact_schema" not in data


def test_invalid_artifact_is_not_written(tmp_path):
    with pytest.raises(ValidationError):
        write_artifact("job1", "evidence", EVIDENCE_BAD, root=tmp_path)
    assert not (job_dir("job1", root=tmp_path) / "evidence.json").exists()
    # validation happens before mkdir, so no job folder is created either
    assert not job_dir("job1", root=tmp_path).exists()


def test_missing_schema_tag_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_artifact("job1", "brief", {"project_id": "p", "request_raw": "x"}, root=tmp_path)


@pytest.mark.parametrize("bad", ["../etc", "a/b", "..", ".", "a..b", "", ".hidden", "x/../y"])
def test_unsafe_job_id_rejected(tmp_path, bad):
    with pytest.raises(ArtifactStoreError):
        write_artifact(bad, "brief", BRIEF, root=tmp_path)


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "x/../y", ""])
def test_unsafe_stage_rejected(tmp_path, bad):
    with pytest.raises(ArtifactStoreError):
        write_artifact("job1", bad, BRIEF, root=tmp_path)


def test_valid_segments_with_dots_and_dashes_accepted(tmp_path):
    # boundary: single dots and dashes are fine; only ".." is rejected
    rec = write_artifact("2026-07-16-job.1", "brief", BRIEF, root=tmp_path)
    assert Path(rec["path"]).exists()


def test_second_stage_same_job_adds_commit(tmp_path):
    write_artifact("job1", "brief", BRIEF, root=tmp_path)
    write_artifact("job1", "draft", DRAFT, root=tmp_path)
    assert _commits(job_dir("job1", root=tmp_path)) == 2


def test_rewrite_identical_content_makes_no_empty_commit(tmp_path):
    write_artifact("job1", "brief", BRIEF, root=tmp_path)
    rec = write_artifact("job1", "brief", BRIEF, root=tmp_path)
    assert _commits(job_dir("job1", root=tmp_path)) == 1
    assert rec["sha"]  # still returns the current HEAD


def test_rewrite_changed_content_makes_new_commit(tmp_path):
    write_artifact("job1", "brief", BRIEF, root=tmp_path)
    changed = {**BRIEF, "channel": "linkedin"}
    write_artifact("job1", "brief", changed, root=tmp_path)
    assert _commits(job_dir("job1", root=tmp_path)) == 2
