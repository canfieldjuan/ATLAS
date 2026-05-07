from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extracted_evidence_to_story.sources import (
    EvidenceStoryLoadError,
    UnsupportedSourceType,
    load_evidence_story_sources,
    write_evidence_story_sources,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/build_evidence_to_story_sources.py"
GOLDEN_MANIFEST = (
    ROOT
    / "extracted_evidence_to_story/fixtures/evidence_to_story_v0_golden/inputs/manifest.json"
)


def _write_manifest(tmp_path: Path, *, sources: list[dict]) -> Path:
    (tmp_path / "yt.txt").write_text("Transcript text", encoding="utf-8")
    (tmp_path / "article.txt").write_text("Article text", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"story_id": "case_001", "sources": sources}),
        encoding="utf-8",
    )
    return path


def _default_sources() -> list[dict]:
    return [
        {
            "type": "youtube_transcript",
            "title": "Video treatment",
            "url": "https://example.com/video",
            "text_path": "yt.txt",
            "metadata": {"channel": "Example"},
        },
        {
            "type": "news_article",
            "title": "Reported article",
            "url": "https://example.com/article",
            "text_path": "article.txt",
            "metadata": {"publication": "Example News"},
        },
    ]


def test_load_evidence_story_sources_normalizes_stage_one_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, sources=_default_sources())

    loaded = load_evidence_story_sources(manifest)

    assert loaded.story_id == "case_001"
    assert [source.source_id for source in loaded.sources] == [
        "src_yt_01",
        "src_news_01",
    ]
    assert loaded.sources[0].text == "Transcript text"
    assert loaded.sources[1].metadata == {"publication": "Example News"}
    assert loaded.as_dict()["sources"][0]["type"] == "youtube_transcript"


def test_load_evidence_story_sources_rejects_duplicate_source_types(tmp_path: Path) -> None:
    sources = _default_sources()
    sources.append({
        "type": "news_article",
        "title": "Second article",
        "url": "https://example.com/article-2",
        "text_path": "article.txt",
    })
    manifest = _write_manifest(tmp_path, sources=sources)

    with pytest.raises(EvidenceStoryLoadError, match=r"youtube_transcript=1, news_article=2"):
        load_evidence_story_sources(manifest)


def test_load_evidence_story_sources_validates_type_mix_before_reading_text(tmp_path: Path) -> None:
    """Two-pass loader: validation runs before any text-file I/O.

    Manifest has 2 youtube_transcripts (invalid mix) but the second one
    points at a non-existent text file. If validation ran after reads,
    we'd see "text file not found"; with the two-pass design we see
    the type-mix error first.
    """

    sources = [
        {
            "type": "youtube_transcript",
            "title": "first",
            "url": "https://example.com/a",
            "text_path": "yt.txt",
        },
        {
            "type": "youtube_transcript",
            "title": "second",
            "url": "https://example.com/b",
            "text_path": "does_not_exist.txt",
        },
    ]
    manifest = _write_manifest(tmp_path, sources=sources)

    with pytest.raises(EvidenceStoryLoadError, match=r"news_article=0"):
        load_evidence_story_sources(manifest)


def test_load_evidence_story_sources_rejects_text_path_escaping_manifest_dir(tmp_path: Path) -> None:
    """A relative text_path that escapes the manifest's parent directory is rejected."""

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("escapee", encoding="utf-8")
    try:
        sources = _default_sources()
        sources[0]["text_path"] = "../outside.txt"
        manifest = _write_manifest(tmp_path, sources=sources)

        with pytest.raises(EvidenceStoryLoadError, match="escapes manifest directory"):
            load_evidence_story_sources(manifest)
    finally:
        outside.unlink(missing_ok=True)


def test_load_evidence_story_sources_rejects_unsupported_source_type(tmp_path: Path) -> None:
    sources = _default_sources()
    sources[0]["type"] = "court_record"
    manifest = _write_manifest(tmp_path, sources=sources)

    with pytest.raises(UnsupportedSourceType, match="court_record"):
        load_evidence_story_sources(manifest)


def test_load_evidence_story_sources_rejects_missing_type_with_unsupported_error(tmp_path: Path) -> None:
    """Missing or empty `type` is also categorised as UnsupportedSourceType."""

    sources = _default_sources()
    del sources[0]["type"]
    manifest = _write_manifest(tmp_path, sources=sources)

    with pytest.raises(UnsupportedSourceType):
        load_evidence_story_sources(manifest)


def test_load_evidence_story_sources_rejects_missing_text_file(tmp_path: Path) -> None:
    sources = _default_sources()
    sources[0]["text_path"] = "missing.txt"
    manifest = _write_manifest(tmp_path, sources=sources)

    with pytest.raises(EvidenceStoryLoadError, match="text file not found"):
        load_evidence_story_sources(manifest)


def test_load_evidence_story_sources_rejects_empty_text_file(tmp_path: Path) -> None:
    (tmp_path / "yt.txt").write_text("", encoding="utf-8")
    (tmp_path / "article.txt").write_text("Article text", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"story_id": "case_001", "sources": _default_sources()}),
        encoding="utf-8",
    )

    with pytest.raises(EvidenceStoryLoadError, match="text file is empty"):
        load_evidence_story_sources(path)


def test_write_evidence_story_sources_writes_sources_json(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, sources=_default_sources())
    output_dir = tmp_path / "story_package"

    output_path = write_evidence_story_sources(manifest, output_dir)

    assert output_path == output_dir / "sources.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["story_id"] == "case_001"
    assert payload["sources"][0]["source_id"] == "src_yt_01"


def test_golden_fixture_manifest_loads_stage_one_sources() -> None:
    loaded = load_evidence_story_sources(GOLDEN_MANIFEST)

    assert loaded.story_id == "real_noid_dominos_hostage_standoff"
    assert [source.source_id for source in loaded.sources] == [
        "src_yt_01",
        "src_news_01",
    ]
    assert "Stealing a Slice" in loaded.sources[0].text
    assert "UPI" in loaded.sources[1].metadata["publication"]


def test_golden_fixture_round_trips_to_expected_sources_json() -> None:
    """Loader output is byte-for-byte identical to the shipped expected fixture.

    The expected file IS the Stage-1 contract; this test pins the contract
    so loader changes can't drift from it silently.
    """

    expected = json.loads(
        (GOLDEN_MANIFEST.parents[1] / "expected/sources.json").read_text(encoding="utf-8")
    )
    loaded = load_evidence_story_sources(GOLDEN_MANIFEST)
    assert loaded.as_dict() == expected


def test_stage_one_cli_prints_sources_json(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, sources=_default_sources())

    completed = subprocess.run(
        [sys.executable, str(CLI), str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["story_id"] == "case_001"
    assert payload["sources"][1]["source_id"] == "src_news_01"


def test_stage_one_cli_writes_output_dir(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, sources=_default_sources())
    output_dir = tmp_path / "story_package"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "sources.json" in completed.stdout
    assert (output_dir / "sources.json").exists()
