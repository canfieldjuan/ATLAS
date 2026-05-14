from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.blog_blueprint_ingest import (
    BlogBlueprintDataFormat,
    load_blog_blueprints_from_file,
    normalize_blog_blueprint_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/load_extracted_blog_blueprints.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "load_extracted_blog_blueprints",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self) -> None:
        self.fetchval_results: list[str] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((str(query), args))
        assert self.fetchval_results, "unexpected extra fetchval call"
        return self.fetchval_results.pop(0)

    async def close(self):
        self.closed = True


def test_load_blog_blueprints_from_json_payload_normalizes_rows(tmp_path: Path) -> None:
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps({
            "blueprints": [
                {
                    "target_mode": "vendor_retention",
                    "topic_type": "pricing_pressure",
                    "title": "Acme pricing pressure",
                    "sections": [{"id": "intro"}],
                    "tags": ["pricing"],
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_blog_blueprints_from_file(path)

    assert loaded.source == str(path)
    assert loaded.warnings == ()
    blueprint = loaded.blueprints[0]
    assert blueprint.target_mode == "vendor_retention"
    assert blueprint.topic_type == "pricing_pressure"
    assert blueprint.slug == "acme-pricing-pressure"
    assert blueprint.suggested_title == "Acme pricing pressure"
    assert blueprint.payload["sections"] == [{"id": "intro"}]
    assert "title" not in blueprint.payload


def test_load_blog_blueprints_from_bare_json_object(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({
            "target_mode": "vendor_retention",
            "title": "Single blueprint",
        }),
        encoding="utf-8",
    )

    loaded = load_blog_blueprints_from_file(path)

    assert len(loaded.blueprints) == 1
    assert loaded.blueprints[0].slug == "single-blueprint"


def test_load_blog_blueprints_preserves_payload_data_on_bare_object(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({
            "target_mode": "vendor_retention",
            "title": "Single blueprint",
            "data": [{"section": "intro"}],
            "rows": [{"section": "proof"}],
        }),
        encoding="utf-8",
    )

    loaded = load_blog_blueprints_from_file(path)

    assert len(loaded.blueprints) == 1
    assert loaded.blueprints[0].slug == "single-blueprint"
    assert loaded.blueprints[0].payload["data"] == [{"section": "intro"}]
    assert loaded.blueprints[0].payload["rows"] == [{"section": "proof"}]
    assert loaded.warnings == ()


def test_load_blog_blueprints_prefers_explicit_blueprints_wrapper(tmp_path: Path) -> None:
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps({
            "topic_type": "pricing",
            "blueprints": [{"title": "Wrapped blueprint"}],
        }),
        encoding="utf-8",
    )

    loaded = load_blog_blueprints_from_file(path, target_mode="vendor_retention")

    assert len(loaded.blueprints) == 1
    assert loaded.blueprints[0].slug == "wrapped-blueprint"
    assert loaded.blueprints[0].target_mode == "vendor_retention"
    assert loaded.warnings == ()


def test_load_blog_blueprints_rejects_ambiguous_extension(tmp_path: Path) -> None:
    path = tmp_path / "blueprints.txt"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot infer blog blueprint format"):
        load_blog_blueprints_from_file(path)


def test_blog_blueprint_data_format_is_exported_for_type_hints() -> None:
    assert BlogBlueprintDataFormat is not None


def test_normalize_blog_blueprint_rows_applies_defaults_and_skips_bad_rows() -> None:
    loaded = normalize_blog_blueprint_rows(
        [
            {"title": "Renewal story", "topic_type": "renewal"},
            {"slug": "missing-mode"},
            "bad-row",
        ],
        target_mode="vendor_retention",
    )

    assert len(loaded.blueprints) == 2
    assert loaded.blueprints[0].target_mode == "vendor_retention"
    assert loaded.blueprints[0].topic_type == "renewal"
    assert loaded.blueprints[1].slug == "missing-mode"
    assert loaded.warnings[-1].code == "row_not_object"
    assert loaded.warnings[-1].row_index == 3


def test_normalize_blog_blueprint_rows_skips_missing_target_mode() -> None:
    loaded = normalize_blog_blueprint_rows([{"title": "No mode"}])

    assert loaded.blueprints == ()
    assert loaded.warnings[0].code == "missing_target_mode"
    assert loaded.warnings[0].field == "target_mode"


def test_normalize_blog_blueprint_rows_uses_topic_without_storing_it_as_payload() -> None:
    loaded = normalize_blog_blueprint_rows([
        {"topic": "Pricing story", "target_mode": "vendor_retention"}
    ])

    assert loaded.blueprints[0].slug == "pricing-story"
    assert "topic" not in loaded.blueprints[0].payload


@pytest.mark.asyncio
async def test_blog_blueprint_import_cli_dry_run_outputs_json(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    cli = _load_cli_module()
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps([{"title": "Pricing story"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "load",
            str(path),
            "--account-id",
            "acct_1",
            "--target-mode",
            "vendor_retention",
            "--dry-run",
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["dry_run"] is True
    assert output["loaded"] == 1
    assert output["saved"] == 0
    assert output["account_id"] == "acct_1"


@pytest.mark.asyncio
async def test_blog_blueprint_import_cli_requires_database_url(
    monkeypatch,
    tmp_path,
) -> None:
    cli = _load_cli_module()
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps([{"title": "Pricing story", "target_mode": "vendor_retention"}]),
        encoding="utf-8",
    )
    monkeypatch.delenv("EXTRACTED_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(cli.sys, "argv", ["load", str(path), "--account-id", "acct_1"])

    with pytest.raises(SystemExit, match="Missing --database-url"):
        await cli._main()


@pytest.mark.asyncio
async def test_blog_blueprint_import_cli_wires_pool_and_closes(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    cli = _load_cli_module()
    path = tmp_path / "blueprints.json"
    path.write_text(
        json.dumps([{"title": "Pricing story", "target_mode": "vendor_retention"}]),
        encoding="utf-8",
    )
    pool = _Pool()
    pool.fetchval_results = ["bp_1"]
    created_urls: list[str] = []

    async def create_pool(database_url):
        created_urls.append(database_url)
        return pool

    monkeypatch.setattr(cli, "_create_pool", create_pool)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "load",
            str(path),
            "--account-id",
            "acct_1",
            "--database-url",
            "postgres://example",
            "--json",
        ],
    )

    exit_code = await cli._main()

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert created_urls == ["postgres://example"]
    assert pool.closed is True
    assert output["saved"] == 1
    assert output["saved_ids"] == ["bp_1"]
    query, args = pool.fetchval_calls[0]
    assert "INSERT INTO blog_blueprints" in query
    assert args[0] == "acct_1"
    assert args[1] == "vendor_retention"
