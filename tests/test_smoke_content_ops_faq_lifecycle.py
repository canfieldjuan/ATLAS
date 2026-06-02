from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/smoke_content_ops_faq_lifecycle.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_lifecycle",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"


class _Pool:
    def __init__(
        self,
        *,
        existing_relations=None,
        update_hits: bool = True,
    ) -> None:
        self.existing_relations = (
            set(existing_relations)
            if existing_relations is not None
            else {"ticket_faq_markdown"}
        )
        self.update_hits = update_hits
        self.rows: list[dict] = []
        self.fetchval_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.execute_calls: list[dict] = []
        self.closed = False

    async def fetchval(self, query, *args):
        self.fetchval_calls.append({"query": str(query), "args": args})
        if "to_regclass" in str(query):
            return args[0] if args and args[0] in self.existing_relations else None
        faq_id = f"faq-uuid-{len(self.rows) + 1}"
        self.rows.append({
            "id": faq_id,
            "account_id": args[0],
            "target_id": args[1],
            "target_mode": args[2],
            "title": args[3],
            "markdown": args[4],
            "items": args[5],
            "source_count": args[6],
            "ticket_source_count": args[7],
            "output_checks": args[8],
            "warnings": args[9],
            "metadata": args[10],
            "status": "draft",
        })
        return faq_id

    async def fetch(self, query, *args):
        self.fetch_calls.append({"query": str(query), "args": args})
        if "UPDATE ticket_faq_markdown" in str(query):
            if not self.update_hits:
                return []
            updated_rows = []
            for row in self.rows:
                if row["id"] == args[0] and row["account_id"] == args[2]:
                    row["status"] = args[1]
                    updated_rows.append(dict(row))
            return updated_rows
        account_id = args[0] if args else ""
        status = args[1] if "status = $2" in str(query) and len(args) > 1 else None
        target_mode_index = 2 if "target_mode = $3" in str(query) else None
        target_mode = args[target_mode_index] if target_mode_index is not None and len(args) > target_mode_index else None
        rows = [
            row
            for row in self.rows
            if row["account_id"] == account_id
            and (status is None or row["status"] == status)
            and (target_mode is None or row["target_mode"] == target_mode)
        ]
        return rows

    async def execute(self, query, *args):
        self.execute_calls.append({"query": str(query), "args": args})
        if not self.update_hits:
            return "UPDATE 0"
        updated = 0
        for row in self.rows:
            if row["id"] == args[0] and row["account_id"] == args[2]:
                row["status"] = args[1]
                updated += 1
        return f"UPDATE {updated}"

    async def close(self):
        self.closed = True


async def _return_pool(pool):
    return pool


def _args(**overrides):
    values = {
        "path": SUPPORT_TICKET_CSV,
        "source_format": "csv",
        "target_mode": "vendor_retention",
        "title": "Customer Ticket FAQ",
        "account_id": "acct-smoke",
        "user_id": None,
        "min_source_rows": 2,
        "min_saved_faqs": 1,
        "review_status": "published",
        "export_limit": 20,
        "max_text_chars": 1200,
        "allow_ingestion_warnings": False,
        "default_field": [],
        "output_result": None,
        "json": False,
        "database_url": "postgres://example",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_generates_exports_reviews_and_reexports(monkeypatch):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 4
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["output_checks"] == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert payload["normalization_warnings"] == {
        "warning_count": 0,
        "warnings_by_code": {},
        "warning_sample": [],
        "warnings_truncated": False,
    }
    assert payload["input_profile"]["status"] == "ok"
    assert payload["input_profile"]["raw_row_count"] == 4
    assert payload["input_profile"]["raw_row_count_source"] == "csv_rows"
    assert payload["input_profile"]["usable_source_count"] == 4
    assert payload["input_profile"]["warning_count"] == 0
    assert payload["input_profile"]["usable_source_ratio"] == 1.0
    assert payload["draft_export"]["rows"][0]["id"] == "faq-uuid-1"
    assert payload["draft_export"]["rows"][0]["status"] == "draft"
    assert payload["reviewed_export"]["rows"][0]["id"] == "faq-uuid-1"
    assert payload["reviewed_export"]["rows"][0]["status"] == "published"
    assert "# Customer Ticket FAQ" in payload["reviewed_export"]["rows"][0]["markdown"]
    assert pool.closed is True
    assert pool.execute_calls


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_persists_1000_row_json_bundle(monkeypatch, tmp_path):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    source = tmp_path / "support_ticket_bundle.json"
    source.write_text(
        json.dumps({
            "support_tickets": [
                {
                    "ticket_id": f"ticket-lifecycle-{index}",
                    "source_type": "support_ticket",
                    "subject": "Billing renewal question",
                    "message": "How do I confirm my renewal invoice before payment?",
                    "pain_category": "billing",
                }
                for index in range(1000)
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    code, payload = await smoke.run_faq_lifecycle_smoke(
        _args(
            path=source,
            source_format="json",
            title="Customer Ticket FAQ Lifecycle Scale Smoke",
            min_source_rows=1000,
            export_limit=5,
            default_field=[
                "company_name=Acme Billing",
                "contact_email=billing@example.com",
                "vendor_name=Atlas Billing",
            ],
        )
    )

    assert code == 0
    assert payload["ok"] is True
    assert payload["source_rows"] == 1000
    assert payload["input_profile"]["status"] == "ok"
    assert payload["input_profile"]["raw_row_count"] == 1000
    assert payload["input_profile"]["raw_row_count_source"] == (
        "json_bundle.support_tickets"
    )
    assert payload["input_profile"]["usable_source_count"] == 1000
    assert payload["input_profile"]["warning_count"] == 0
    assert payload["input_profile"]["usable_source_ratio"] == 1.0
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert payload["generation"]["source_count"] == 1000
    assert payload["generation"]["ticket_source_count"] == 1000
    assert payload["generation"]["items"][0]["ticket_count"] == 1000
    assert len(payload["generation"]["items"][0]["source_ids"]) == 1000
    assert payload["generation"]["items"][0]["source_ids"][0] == "ticket-lifecycle-0"
    assert payload["generation"]["items"][0]["source_ids"][-1] == "ticket-lifecycle-999"
    assert payload["lifecycle_summary"] == {
        "status": "ok",
        "source": str(source),
        "source_format": "json",
        "source_rows": 1000,
        "input_profile": payload["input_profile"],
        "source_count": 1000,
        "ticket_source_count": 1000,
        "generated_item_count": 1,
        "output_checks": {
            "uses_user_vocabulary": True,
            "condensed": True,
            "has_action_items": True,
            "resolution_evidence_scoped": True,
        },
        "saved_faq_count": 1,
        "draft_export_count": 1,
        "reviewed_export_count": 1,
        "review_status": "published",
        "error_count": 0,
        "errors": [],
    }

    draft = payload["draft_export"]["rows"][0]
    reviewed = payload["reviewed_export"]["rows"][0]
    assert draft["source_count"] == 1000
    assert draft["ticket_source_count"] == 1000
    assert draft["status"] == "draft"
    assert draft["items"][0]["ticket_count"] == 1000
    assert len(draft["items"][0]["source_ids"]) == 1000
    assert reviewed["source_count"] == 1000
    assert reviewed["ticket_source_count"] == 1000
    assert reviewed["status"] == "published"
    assert "# Customer Ticket FAQ Lifecycle Scale Smoke" in reviewed["markdown"]
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_reports_normalization_warning_codes(monkeypatch, tmp_path):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))
    source = tmp_path / "support_ticket_bundle.json"
    source.write_text(
        json.dumps({
            "support_tickets": [
                {
                    "ticket_id": f"ticket-warning-{index}",
                    "source_type": "support_ticket",
                    "subject": "Billing renewal question",
                    "message": "How do I confirm my renewal invoice before payment?",
                    "pain_category": "billing",
                    "company_name": "Acme Billing",
                    "contact_email": "billing@example.com",
                }
                for index in range(3)
            ],
        })
        + "\n",
        encoding="utf-8",
    )

    code, payload = await smoke.run_faq_lifecycle_smoke(
        _args(path=source, source_format="json", min_source_rows=3)
    )

    assert code == 1
    assert payload["generation"] is None
    assert payload["input_profile"]["status"] == "ok"
    assert payload["input_profile"]["raw_row_count"] == 3
    assert payload["input_profile"]["raw_row_count_source"] == (
        "json_bundle.support_tickets"
    )
    assert payload["input_profile"]["usable_source_count"] == 3
    assert payload["input_profile"]["warning_count"] == 3
    assert payload["input_profile"]["warnings_by_code"] == {
        "missing_vendor_name": 3
    }
    assert payload["input_profile"]["usable_source_ratio"] == 1.0
    assert payload["normalization_warnings"]["warning_count"] == 3
    assert payload["normalization_warnings"]["warnings_by_code"] == {
        "missing_vendor_name": 3
    }
    assert payload["normalization_warnings"]["warning_sample"][0]["code"] == (
        "missing_vendor_name"
    )
    assert payload["normalization_warnings"]["warnings_truncated"] is False
    assert any("missing_vendor_name=3" in error for error in payload["errors"])
    assert pool.execute_calls == []
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_fails_closed_when_table_missing(monkeypatch):
    pool = _Pool(existing_relations=())
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 1
    assert any("ticket_faq_markdown" in error for error in payload["errors"])
    assert payload["generation"] is None
    assert payload["lifecycle_summary"]["status"] == "failed"
    assert payload["lifecycle_summary"]["source_rows"] == 4
    assert payload["lifecycle_summary"]["input_profile"] == payload["input_profile"]
    assert payload["lifecycle_summary"]["source_count"] is None
    assert payload["lifecycle_summary"]["ticket_source_count"] is None
    assert payload["lifecycle_summary"]["generated_item_count"] is None
    assert payload["lifecycle_summary"]["output_checks"] is None
    assert payload["lifecycle_summary"]["saved_faq_count"] == 0
    assert payload["lifecycle_summary"]["draft_export_count"] is None
    assert payload["lifecycle_summary"]["reviewed_export_count"] is None
    assert payload["lifecycle_summary"]["review_status"] == "published"
    assert payload["lifecycle_summary"]["error_count"] == 1
    assert payload["lifecycle_summary"]["errors"] == payload["errors"]
    assert len(pool.fetchval_calls) == 1
    assert pool.execute_calls == []
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_writes_result_when_pool_creation_fails(
    monkeypatch, tmp_path
):
    async def fail_create_pool(*_args, **_kwargs):
        raise RuntimeError("database connection slots exhausted")

    result_path = tmp_path / "lifecycle_failure.json"
    monkeypatch.setattr(smoke, "_create_pool", fail_create_pool)

    code, payload = await smoke.run_faq_lifecycle_smoke(
        _args(output_result=result_path)
    )

    assert code == 1
    assert payload["ok"] is False
    assert payload["source"] == str(SUPPORT_TICKET_CSV)
    assert payload["source_rows"] == 0
    assert payload["input_profile"]["status"] == "not_started"
    assert payload["generation"] is None
    assert payload["draft_export"] is None
    assert payload["reviewed_export"] is None
    assert payload["saved_ids"] == []
    assert payload["errors"] == ["RuntimeError: database connection slots exhausted"]
    assert payload["lifecycle_summary"]["status"] == "failed"
    assert payload["lifecycle_summary"]["source_rows"] == 0
    assert payload["lifecycle_summary"]["error_count"] == 1
    assert payload["lifecycle_summary"]["errors"] == payload["errors"]
    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == payload


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_reports_review_status_miss(monkeypatch):
    pool = _Pool(update_hits=False)
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(_args())

    assert code == 1
    assert payload["saved_ids"] == ["faq-uuid-1"]
    assert any("review status update missed saved FAQ id" in error for error in payload["errors"])
    assert payload["reviewed_export"] is None
    assert pool.closed is True


@pytest.mark.asyncio
async def test_faq_lifecycle_smoke_marks_input_profile_error_on_load_failure(monkeypatch, tmp_path):
    pool = _Pool()
    monkeypatch.setattr(smoke, "_create_pool", lambda *_args, **_kwargs: _return_pool(pool))

    code, payload = await smoke.run_faq_lifecycle_smoke(
        _args(path=tmp_path / "missing.json", source_format="json")
    )

    assert code == 1
    assert payload["generation"] is None
    assert payload["input_profile"]["status"] == "error"
    assert "No such file or directory" in payload["input_profile"]["error"]
    assert "No such file or directory" in payload["input_profile"]["raw_row_count_error"]
    assert any("No such file or directory" in error for error in payload["errors"])
    assert pool.execute_calls == []
    assert pool.closed is True


def test_faq_lifecycle_print_payload_includes_input_profile(capsys) -> None:
    smoke._print_payload(
        {
            "ok": True,
            "input_profile": {
                "status": "ok",
                "usable_source_count": 1000,
                "raw_row_count": 1000,
                "warning_count": 0,
            },
            "saved_ids": ["faq-uuid-1"],
            "review_status": "published",
        },
        as_json=False,
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Content Ops FAQ lifecycle smoke passed:" in captured.out
    assert "input_status=ok" in captured.out
    assert "source_rows=1000/1000" in captured.out
    assert "saved_faqs=1" in captured.out
    assert "review_status=published" in captured.out


def test_faq_lifecycle_print_payload_includes_input_profile_on_failure(capsys) -> None:
    smoke._print_payload(
        {
            "ok": False,
            "input_profile": {
                "status": "ok",
                "usable_source_count": 46,
                "raw_row_count": 1000,
                "skipped_row_count": 954,
                "missing_source_text_count": 954,
                "warning_count": 954,
            },
            "errors": ["expected at least 500 source row(s), got 46"],
        },
        as_json=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Content Ops FAQ lifecycle smoke failed:" in captured.err
    assert "input_status=ok" in captured.err
    assert "source_rows=46/1000" in captured.err
    assert "skipped_rows=954" in captured.err
    assert "missing_source_text=954" in captured.err
    assert "warnings=954" in captured.err
    assert "- expected at least 500 source row(s), got 46" in captured.err


def test_faq_lifecycle_print_payload_summary_json(capsys) -> None:
    smoke._print_payload(
        {
            "ok": True,
            "lifecycle_summary": {
                "status": "ok",
                "source_rows": 1000,
                "saved_faq_count": 1,
                "draft_export_count": 1,
                "reviewed_export_count": 1,
            },
            "reviewed_export": {
                "rows": [
                    {
                        "markdown": "# full markdown body should not print",
                    }
                ]
            },
        },
        as_json=False,
        summary_json=True,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == ""
    assert output == {
        "status": "ok",
        "source_rows": 1000,
        "saved_faq_count": 1,
        "draft_export_count": 1,
        "reviewed_export_count": 1,
    }
    assert "full markdown body" not in captured.out


def test_faq_lifecycle_print_payload_summary_json_on_failure(capsys) -> None:
    smoke._print_payload(
        {
            "ok": False,
            "lifecycle_summary": {
                "status": "failed",
                "source_rows": 46,
                "error_count": 1,
                "errors": ["expected at least 500 source row(s), got 46"],
            },
            "errors": ["expected at least 500 source row(s), got 46"],
        },
        as_json=False,
        summary_json=True,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == ""
    assert output["status"] == "failed"
    assert output["source_rows"] == 46
    assert output["error_count"] == 1
    assert output["errors"] == ["expected at least 500 source row(s), got 46"]


def test_faq_lifecycle_print_payload_full_json_wins_over_summary_json(capsys) -> None:
    smoke._print_payload(
        {
            "ok": True,
            "lifecycle_summary": {"status": "ok"},
            "reviewed_export": {"rows": [{"markdown": "# full markdown"}]},
        },
        as_json=True,
        summary_json=True,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == ""
    assert output["lifecycle_summary"] == {"status": "ok"}
    assert output["reviewed_export"]["rows"][0]["markdown"] == "# full markdown"


def test_faq_lifecycle_smoke_rejects_invalid_args() -> None:
    args = _args(min_saved_faqs=0)

    with pytest.raises(SystemExit, match="--min-saved-faqs must be positive"):
        smoke._validate_args(args)
