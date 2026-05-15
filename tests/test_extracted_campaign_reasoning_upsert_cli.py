"""Tests for the campaign reasoning context upsert CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "upsert_extracted_campaign_reasoning_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "upsert_extracted_campaign_reasoning_contexts",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
upsert_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upsert_cli)


class _Repository:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def save_context(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"ctx-{len(self.calls)}"


class _FailingRepository(_Repository):
    def __init__(self, *, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    async def save_context(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("database write failed")
        return f"ctx-{len(self.calls)}"


class _Pool:
    def __init__(self, matches: list[Any] | None = None) -> None:
        self.matches = list(matches or [])
        self.fetchval_calls: list[dict[str, Any]] = []
        self.closed = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append({"query": query, "args": args})
        return self.matches.pop(0) if self.matches else None

    async def close(self) -> None:
        self.closed = True


def _read_audit_entries(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_context_rows_accepts_array_wrapper_and_mapping_index() -> None:
    """Host files can be arrays, wrapper objects, or keyed context maps."""

    assert upsert_cli._context_rows([{"target_id": "a"}]) == [{"target_id": "a"}]
    assert upsert_cli._context_rows({"contexts": [{"target_id": "b"}]}) == [
        {"target_id": "b"}
    ]
    assert upsert_cli._context_rows({
        "contexts": {
            "opp-1": {"top_theses": [{"summary": "Pricing pressure"}]},
        }
    }) == [
        {
            "target_id": "opp-1",
            "context": {"top_theses": [{"summary": "Pricing pressure"}]},
        }
    ]


def test_row_selectors_use_cli_row_and_matching_fields() -> None:
    """Selectors come from explicit CLI values plus row metadata."""

    selectors = upsert_cli._row_selectors(
        {
            "selectors": ["row-selector", "row-selector-2"],
            "target_id": "opp-1",
            "company_name": "Acme",
            "contact_email": "buyer@example.com",
        },
        ["cli-selector"],
    )

    assert selectors == (
        "cli-selector",
        "row-selector",
        "row-selector-2",
        "opp-1",
        "Acme",
        "buyer@example.com",
    )


def test_row_context_prefers_nested_context() -> None:
    """A nested context field is saved as the reasoning payload."""

    row = {
        "target_id": "opp-1",
        "selectors": ["opp-1"],
        "context": {"top_theses": [{"summary": "Renewal risk"}]},
        "ignored": "metadata outside context",
    }

    assert upsert_cli._row_context(row) == {
        "top_theses": [{"summary": "Renewal risk"}],
    }


def test_dry_run_result_reports_prepared_write_count() -> None:
    """Dry-run output is derived from already validated prepared rows."""

    prepared = upsert_cli._prepare_contexts(
        payload={
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "Renewal risk"}]},
                }
            ]
        },
        default_account_id="acct-default",
        default_target_mode="vendor_retention",
        extra_selectors=[],
    )

    assert upsert_cli._dry_run_result(prepared) == {
        "status": "dry_run",
        "would_upsert": 1,
    }


def test_dry_run_result_reports_validated_opportunities() -> None:
    """Dry-run output can report DB opportunity validation without writes."""

    assert upsert_cli._dry_run_result(
        [{"selectors": ("opp-1",)}],
        validated_opportunities=True,
    ) == {
        "status": "dry_run",
        "would_upsert": 1,
        "validated_opportunities": 1,
    }


@pytest.mark.asyncio
async def test_upsert_contexts_saves_each_row_with_defaults() -> None:
    """Rows without account/mode values inherit CLI defaults."""

    repository = _Repository()

    result = await upsert_cli._upsert_contexts(
        repository,  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "Renewal risk"}]},
                },
                {
                    "selectors": ["opp-2", "Acme"],
                    "account_id": "acct-row",
                    "target_mode": "challenger_intel",
                    "reasoning_context": {"proof_points": [{"label": "source"}]},
                },
            ]
        },
        default_account_id="acct-default",
        default_target_mode="vendor_retention",
        extra_selectors=[],
    )

    assert result == {"status": "ok", "upserted": 2, "ids": ["ctx-1", "ctx-2"]}
    first = repository.calls[0]
    assert first["scope"].account_id == "acct-default"
    assert first["target_mode"] == "vendor_retention"
    assert first["selectors"] == ("opp-1",)
    assert first["context"]["top_theses"][0]["summary"] == "Renewal risk"
    second = repository.calls[1]
    assert second["scope"].account_id == "acct-row"
    assert second["target_mode"] == "challenger_intel"
    assert second["selectors"] == ("opp-2", "Acme")
    assert "reasoning_context" in second["context"]


@pytest.mark.asyncio
async def test_upsert_contexts_preserves_commas_inside_selector_values() -> None:
    """Exact selector values such as company names are not split on commas."""

    repository = _Repository()

    await upsert_cli._upsert_contexts(
        repository,  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "company_name": "Acme, Inc.",
                    "context": {"top_theses": [{"summary": "Renewal risk"}]},
                }
            ]
        },
        default_account_id="acct-default",
        default_target_mode="vendor_retention",
        extra_selectors=["Global, LLC"],
    )

    assert repository.calls[0]["selectors"] == ("Global, LLC", "Acme, Inc.")


@pytest.mark.asyncio
async def test_upsert_contexts_appends_metadata_audit_log(tmp_path: Path) -> None:
    """Audit entries record row metadata without serializing full context payloads."""

    repository = _Repository()
    audit_log = tmp_path / "audit" / "reasoning-upserts.jsonl"

    result = await upsert_cli._upsert_contexts(
        repository,  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "target_id": "opp-1",
                    "account_id": "acct-row",
                    "target_mode": "vendor_retention",
                    "context": {
                        "top_theses": [{"summary": "Renewal risk"}],
                        "proof_points": [{"label": "source"}],
                    },
                }
            ]
        },
        default_account_id="acct-default",
        default_target_mode="",
        extra_selectors=["Acme"],
        audit_log=audit_log,
    )

    entries = _read_audit_entries(audit_log)
    assert result["ids"] == ["ctx-1"]
    assert entries[0]["action"] == "upsert_campaign_reasoning_context"
    assert entries[0]["context_id"] == "ctx-1"
    assert entries[0]["account_id"] == "acct-row"
    assert entries[0]["target_mode"] == "vendor_retention"
    assert entries[0]["selectors"] == ["Acme", "opp-1"]
    assert entries[0]["context_keys"] == ["proof_points", "top_theses"]
    assert "top_theses" not in entries[0]
    assert entries[0]["recorded_at"].endswith("Z")


@pytest.mark.asyncio
async def test_upsert_contexts_audits_each_successful_row_before_later_failure(
    tmp_path: Path,
) -> None:
    """Committed rows keep audit records even when a later write fails."""

    repository = _FailingRepository(fail_on_call=2)
    audit_log = tmp_path / "reasoning-upserts.jsonl"

    with pytest.raises(RuntimeError, match="database write failed"):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={
                "contexts": [
                    {
                        "target_id": "opp-1",
                        "context": {"top_theses": [{"summary": "x"}]},
                    },
                    {
                        "target_id": "opp-2",
                        "context": {"top_theses": [{"summary": "y"}]},
                    },
                ]
            },
            default_account_id="acct-default",
            default_target_mode="vendor_retention",
            extra_selectors=[],
            audit_log=audit_log,
        )

    entries = _read_audit_entries(audit_log)
    assert [entry["row_index"] for entry in entries] == [1]
    assert entries[0]["context_id"] == "ctx-1"


@pytest.mark.asyncio
async def test_upsert_contexts_appends_audit_log_across_runs_and_rows(
    tmp_path: Path,
) -> None:
    """Audit logs append JSONL entries and preserve one-based row indexes."""

    audit_log = tmp_path / "reasoning-upserts.jsonl"

    await upsert_cli._upsert_contexts(
        _Repository(),  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "x"}]},
                },
                {
                    "target_id": "opp-2",
                    "context": {"top_theses": [{"summary": "y"}]},
                },
            ]
        },
        default_account_id="acct-default",
        default_target_mode="vendor_retention",
        extra_selectors=[],
        audit_log=audit_log,
    )
    await upsert_cli._upsert_contexts(
        _Repository(),  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "target_id": "opp-3",
                    "context": {"top_theses": [{"summary": "z"}]},
                },
            ]
        },
        default_account_id="acct-default",
        default_target_mode="vendor_retention",
        extra_selectors=[],
        audit_log=audit_log,
    )

    entries = _read_audit_entries(audit_log)
    assert [entry["row_index"] for entry in entries] == [1, 2, 1]
    assert [entry["context_id"] for entry in entries] == ["ctx-1", "ctx-2", "ctx-1"]


@pytest.mark.asyncio
async def test_upsert_contexts_validates_opportunity_matches_before_writing() -> None:
    """Optional validation requires a live opportunity match for every row."""

    repository = _Repository()
    pool = _Pool(matches=[1])

    result = await upsert_cli._upsert_contexts(
        repository,  # type: ignore[arg-type]
        payload={
            "contexts": [
                {
                    "target_id": "opp-1",
                    "account_id": "acct-1",
                    "target_mode": "Vendor_Retention",
                    "context": {"top_theses": [{"summary": "x"}]},
                },
            ]
        },
        default_account_id="",
        default_target_mode="",
        extra_selectors=["Acme"],
        validate_opportunities=True,
        opportunity_pool=pool,
    )

    assert result["upserted"] == 1
    assert len(repository.calls) == 1
    call = pool.fetchval_calls[0]
    assert 'FROM "campaign_opportunities"' in call["query"]
    assert "status = 'active'" in call["query"]
    assert "target_id = ANY($3::text[])" in call["query"]
    assert call["args"] == (
        "acct-1",
        "vendor_retention",
        ["Acme", "opp-1"],
        ["acme", "opp-1"],
    )


@pytest.mark.asyncio
async def test_upsert_contexts_rejects_unmatched_opportunity_before_writing() -> None:
    """Validation failure happens before the reasoning row is saved."""

    repository = _Repository()
    pool = _Pool(matches=[None])

    with pytest.raises(ValueError, match="opportunity validation failed for rows: 1"):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={
                "contexts": [
                    {
                        "target_id": "missing-opp",
                        "context": {"top_theses": [{"summary": "x"}]},
                    },
                ]
            },
            default_account_id="acct-1",
            default_target_mode="vendor_retention",
            extra_selectors=[],
            validate_opportunities=True,
            opportunity_pool=pool,
        )

    assert repository.calls == []


@pytest.mark.asyncio
async def test_upsert_contexts_reports_all_unmatched_opportunity_rows_before_writing() -> None:
    """Validation collects every missing row before rejecting the batch."""

    repository = _Repository()
    pool = _Pool(matches=[1, None, None])

    with pytest.raises(
        ValueError,
        match="opportunity validation failed for rows: 2, 3",
    ):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={
                "contexts": [
                    {
                        "target_id": "opp-1",
                        "context": {"top_theses": [{"summary": "x"}]},
                    },
                    {
                        "target_id": "missing-2",
                        "context": {"top_theses": [{"summary": "y"}]},
                    },
                    {
                        "target_id": "missing-3",
                        "context": {"top_theses": [{"summary": "z"}]},
                    },
                ]
            },
            default_account_id="acct-1",
            default_target_mode="vendor_retention",
            extra_selectors=[],
            validate_opportunities=True,
            opportunity_pool=pool,
        )

    assert len(pool.fetchval_calls) == 3
    assert repository.calls == []


@pytest.mark.asyncio
async def test_main_dry_run_with_opportunity_validation_opens_db_but_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    """Validated dry-runs check opportunity matches but skip context writes."""

    payload_path = tmp_path / "contexts.json"
    payload_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "x"}]},
                }
            ]
        }),
        encoding="utf-8",
    )
    pool = _Pool(matches=[1])

    async def create_pool(database_url: str) -> Any:
        assert database_url == "postgres://example"
        return pool

    monkeypatch.setattr(upsert_cli, "_create_pool", create_pool)
    exit_code = await upsert_cli._main_from_args([
        str(payload_path),
        "--database-url",
        "postgres://example",
        "--dry-run",
        "--validate-opportunities",
        "--json",
    ])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "dry_run",
        "would_upsert": 1,
        "validated_opportunities": 1,
    }
    assert pool.closed is True


@pytest.mark.asyncio
async def test_upsert_contexts_rejects_rows_without_selectors() -> None:
    """An unselectable row would never be read back, so fail loudly."""

    repository = _Repository()

    with pytest.raises(ValueError, match="row 1 has no selectors"):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={"context": {"top_theses": [{"summary": "x"}]}},
            default_account_id="acct-1",
            default_target_mode="vendor_retention",
            extra_selectors=[],
        )
    assert repository.calls == []


@pytest.mark.asyncio
async def test_main_dry_run_skips_database_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    """Dry-run should validate the file without needing DB credentials."""

    payload_path = tmp_path / "contexts.json"
    payload_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "x"}]},
                }
            ]
        }),
        encoding="utf-8",
    )

    async def create_pool(database_url: str) -> Any:
        raise AssertionError("dry-run must not open a database pool")

    monkeypatch.setattr(upsert_cli, "_create_pool", create_pool)
    exit_code = await upsert_cli._main_from_args([
        str(payload_path),
        "--dry-run",
        "--json",
    ])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "dry_run",
        "would_upsert": 1,
    }


@pytest.mark.asyncio
async def test_main_dry_run_does_not_write_audit_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run remains read-only even when --audit-log is supplied."""

    payload_path = tmp_path / "contexts.json"
    audit_log = tmp_path / "audit.jsonl"
    payload_path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "context": {"top_theses": [{"summary": "x"}]},
                }
            ]
        }),
        encoding="utf-8",
    )

    async def create_pool(database_url: str) -> Any:
        raise AssertionError("dry-run must not open a database pool")

    monkeypatch.setattr(upsert_cli, "_create_pool", create_pool)
    await upsert_cli._main_from_args([
        str(payload_path),
        "--dry-run",
        "--audit-log",
        str(audit_log),
    ])

    assert not audit_log.exists()


@pytest.mark.asyncio
async def test_upsert_contexts_validates_all_rows_before_writing() -> None:
    """A malformed later row should not leave a partially applied file."""

    repository = _Repository()

    with pytest.raises(ValueError, match="row 2 has no selectors"):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={
                "contexts": [
                    {
                        "target_id": "opp-1",
                        "context": {"top_theses": [{"summary": "x"}]},
                    },
                    {"context": {"top_theses": [{"summary": "y"}]}},
                ]
            },
            default_account_id="acct-1",
            default_target_mode="vendor_retention",
            extra_selectors=[],
        )
    assert repository.calls == []


@pytest.mark.asyncio
async def test_upsert_contexts_rejects_rows_without_context() -> None:
    """Do not overwrite an existing useful row with an empty payload."""

    repository = _Repository()

    with pytest.raises(ValueError, match="row 1 has no context"):
        await upsert_cli._upsert_contexts(
            repository,  # type: ignore[arg-type]
            payload={"target_id": "opp-1"},
            default_account_id="acct-1",
            default_target_mode="vendor_retention",
            extra_selectors=[],
        )
    assert repository.calls == []
