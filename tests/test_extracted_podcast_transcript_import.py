from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.podcast_postgres_import import (
    import_podcast_transcripts,
)
from extracted_content_pipeline.podcast_transcript_data import (
    PodcastTranscriptWarning,
    load_podcast_transcripts_from_file,
    normalize_podcast_transcript_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_extracted_podcast_transcript_import.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_extracted_podcast_transcript_import",
        CLI,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pool:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query, *args):
        self.executed.append((str(query), args))
        return "EXECUTE"

    async def close(self):
        return None


def test_load_json_array(tmp_path: Path) -> None:
    payload = [
        {"episode_id": "ep-1", "title": "First", "transcript_text": "x" * 300},
        {"episode_id": "ep-2", "title": "Second", "transcript_text": "y" * 300},
    ]
    file = tmp_path / "episodes.json"
    file.write_text(json.dumps(payload), encoding="utf-8")

    result = load_podcast_transcripts_from_file(file)

    assert {row["episode_id"] for row in result.transcripts} == {"ep-1", "ep-2"}
    assert all(row["title"] for row in result.transcripts)


def test_load_json_wrapped_object(tmp_path: Path) -> None:
    payload = {"transcripts": [{"episode_id": "ep-1", "transcript_text": "x" * 300}]}
    file = tmp_path / "wrap.json"
    file.write_text(json.dumps(payload), encoding="utf-8")

    result = load_podcast_transcripts_from_file(file)

    assert len(result.transcripts) == 1
    assert result.transcripts[0]["episode_id"] == "ep-1"


def test_load_jsonl_parses_line_delimited(tmp_path: Path) -> None:
    file = tmp_path / "ep.jsonl"
    rows = [
        {"episode_id": "ep-1", "title": "One", "transcript_text": "x" * 300},
        {"episode_id": "ep-2", "title": "Two", "transcript_text": "y" * 300},
    ]
    file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = load_podcast_transcripts_from_file(file)

    assert {row["episode_id"] for row in result.transcripts} == {"ep-1", "ep-2"}


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    file = tmp_path / "ep.jsonl"
    file.write_text(
        json.dumps({"episode_id": "ep-1", "transcript_text": "x" * 300}) + "\n\n\n"
        + json.dumps({"episode_id": "ep-2", "transcript_text": "y" * 300}) + "\n",
        encoding="utf-8",
    )

    result = load_podcast_transcripts_from_file(file)

    assert len(result.transcripts) == 2


def test_load_jsonl_invalid_line_raises(tmp_path: Path) -> None:
    file = tmp_path / "ep.jsonl"
    file.write_text(
        json.dumps({"episode_id": "ep-1", "transcript_text": "x" * 300}) + "\n"
        + "{not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_podcast_transcripts_from_file(file)


def test_load_csv_with_extra_columns(tmp_path: Path) -> None:
    file = tmp_path / "ep.csv"
    file.write_text(
        "episode_id,title,transcript_text,custom_meta\n"
        "ep-9,Hello,abc def ghi jkl,custom-value\n",
        encoding="utf-8",
    )

    result = load_podcast_transcripts_from_file(file)

    assert len(result.transcripts) == 1
    row = result.transcripts[0]
    assert row["episode_id"] == "ep-9"
    assert row["title"] == "Hello"
    assert row["raw_payload"]["custom_meta"] == "custom-value"


def test_load_txt_uses_filename_as_episode_id(tmp_path: Path) -> None:
    file = tmp_path / "ep-42.txt"
    file.write_text("Episode 42 title line\n\nBody starts here.\n", encoding="utf-8")

    result = load_podcast_transcripts_from_file(file)

    assert len(result.transcripts) == 1
    row = result.transcripts[0]
    assert row["episode_id"] == "ep-42"
    assert row["title"] == "Episode 42 title line"
    assert "Body starts here." in row["transcript_text"]


def test_load_srt_strips_timestamps_and_indices(tmp_path: Path) -> None:
    srt = (
        "1\n"
        "00:00:01,500 --> 00:00:04,200\n"
        "Hello there\n"
        "\n"
        "2\n"
        "00:00:04,500 --> 00:00:07,800\n"
        "How are you\n"
    )
    file = tmp_path / "talk.srt"
    file.write_text(srt, encoding="utf-8")

    result = load_podcast_transcripts_from_file(file)

    assert len(result.transcripts) == 1
    body = result.transcripts[0]["transcript_text"]
    assert "00:00:01" not in body
    assert "Hello there" in body
    assert "How are you" in body


def test_short_transcript_emits_warning() -> None:
    result = normalize_podcast_transcript_rows([
        {"episode_id": "ep-x", "title": "T", "transcript_text": "short"},
    ])
    codes = {w.code for w in result.warnings}
    assert "transcript_too_short" in codes


def test_missing_episode_id_skipped_with_warning() -> None:
    result = normalize_podcast_transcript_rows([
        {"title": "no-id", "transcript_text": "x" * 300},
    ])
    assert result.transcripts == ()
    assert any(w.code == "missing_episode_id" for w in result.warnings)


@pytest.mark.asyncio
async def test_import_inserts_rows() -> None:
    pool = _Pool()
    rows = [{"episode_id": "ep-1", "title": "T", "transcript_text": "x" * 300}]

    result = await import_podcast_transcripts(
        pool,
        rows,
        scope=TenantScope(account_id="acct_1"),
    )

    assert result.inserted == 1
    assert result.skipped == 0
    assert "ep-1" in result.episode_ids
    insert_queries = [q for q, _ in pool.executed if "INSERT INTO" in q]
    assert len(insert_queries) == 1


@pytest.mark.asyncio
async def test_import_dry_run_does_not_touch_db() -> None:
    pool = _Pool()
    rows = [{"episode_id": "ep-1", "title": "T", "transcript_text": "x" * 300}]

    result = await import_podcast_transcripts(
        pool,
        rows,
        scope=TenantScope(account_id="acct_1"),
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.inserted == 1
    assert pool.executed == []


@pytest.mark.asyncio
async def test_import_replace_existing_deletes_first() -> None:
    pool = _Pool()
    rows = [{"episode_id": "ep-1", "title": "T", "transcript_text": "x" * 300}]

    await import_podcast_transcripts(
        pool,
        rows,
        scope=TenantScope(account_id="acct_1"),
        replace_existing=True,
    )

    queries = [q for q, _ in pool.executed]
    assert any("DELETE FROM" in q for q in queries)
    assert any("INSERT INTO" in q for q in queries)


@pytest.mark.asyncio
async def test_import_skips_row_without_episode_id_when_pre_normalized() -> None:
    """When the caller already normalized rows (the CLI codepath), the
    importer's own loop catches missing episode_ids and counts them as
    skipped + adds a missing_episode_id warning."""

    pool = _Pool()
    rows = [
        {"title": "no id", "transcript_text": "x" * 300},
        {
            "episode_id": "ep-2",
            "title": "ok",
            "transcript_text": "y" * 300,
            "raw_payload": {},
        },
    ]

    result = await import_podcast_transcripts(
        pool,
        rows,
        scope=TenantScope(account_id="acct_1"),
        normalize=False,
    )

    assert result.inserted == 1
    assert result.skipped == 1
    assert "ep-2" in result.episode_ids
    assert any(w.code == "missing_episode_id" for w in result.warnings)


@pytest.mark.asyncio
async def test_import_normalizes_and_drops_no_id_row_with_warning() -> None:
    """When normalize=True (default), the no-id row is dropped during
    normalization with a missing_episode_id warning, before the importer
    loop sees it."""

    pool = _Pool()
    rows = [
        {"title": "no id", "transcript_text": "x" * 300},
        {"episode_id": "ep-2", "title": "ok", "transcript_text": "y" * 300},
    ]

    result = await import_podcast_transcripts(
        pool,
        rows,
        scope=TenantScope(account_id="acct_1"),
    )

    assert result.inserted == 1
    assert "ep-2" in result.episode_ids
    assert any(w.code == "missing_episode_id" for w in result.warnings)


def test_cli_module_loads_and_exposes_main() -> None:
    module = _load_cli_module()
    assert hasattr(module, "_main")
    assert hasattr(module, "_parse_args")
