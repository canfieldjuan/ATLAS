"""Stage-1 source loading for the Evidence-to-Story product."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, cast


SourceType = Literal["youtube_transcript", "news_article"]

SUPPORTED_SOURCE_TYPES: tuple[str, ...] = ("youtube_transcript", "news_article")
SOURCE_ID_PREFIXES: Mapping[str, str] = {
    "youtube_transcript": "yt",
    "news_article": "news",
}
SOURCES_FILENAME = "sources.json"


class EvidenceStoryLoadError(ValueError):
    """Raised when a Stage-1 source manifest cannot be loaded."""


class UnsupportedSourceType(EvidenceStoryLoadError):
    """Raised when a manifest source type is outside the v0 contract."""


@dataclass(frozen=True)
class SourceRecord:
    """Normalized Stage-1 source record."""

    source_id: str
    type: SourceType
    title: str
    url: str
    text: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "type": self.type,
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceStorySources:
    """Stage-1 output payload for the evidence-to-story package."""

    story_id: str
    sources: tuple[SourceRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "sources": [source.as_dict() for source in self.sources],
        }


def load_evidence_story_sources(
    manifest_path: str | Path,
) -> EvidenceStorySources:
    """Load the v0 Stage-1 manifest into normalized source records."""

    path = Path(manifest_path)
    manifest = _load_manifest(path)
    story_id = _required_text(manifest, "story_id", context="manifest")
    source_specs = manifest.get("sources")
    if not isinstance(source_specs, Sequence) or isinstance(source_specs, (str, bytes)):
        raise EvidenceStoryLoadError("manifest.sources must be an array")

    type_counts: dict[str, int] = {}
    sources: list[SourceRecord] = []
    for index, spec in enumerate(source_specs, start=1):
        if not isinstance(spec, Mapping):
            raise EvidenceStoryLoadError(f"source {index} must be an object")
        source_type = _source_type(spec, index=index)
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
        source_id = _source_id(source_type, type_counts[source_type])
        text_path = _required_text(spec, "text_path", context=f"source {index}")
        text = _read_source_text(path.parent, text_path, index=index)
        metadata = spec.get("metadata") if isinstance(spec.get("metadata"), Mapping) else {}
        sources.append(
            SourceRecord(
                source_id=source_id,
                type=source_type,
                title=_required_text(spec, "title", context=f"source {index}"),
                url=_optional_text(spec.get("url")),
                text=text,
                metadata=dict(metadata),
            )
        )

    _validate_v0_source_mix(type_counts)
    return EvidenceStorySources(story_id=story_id, sources=tuple(sources))


def write_evidence_story_sources(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Write Stage-1 `sources.json` into an evidence-to-story package."""

    loaded = load_evidence_story_sources(manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / SOURCES_FILENAME
    output_path.write_text(
        f"{json.dumps(loaded.as_dict(), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return output_path


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceStoryLoadError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceStoryLoadError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(data, Mapping):
        raise EvidenceStoryLoadError("manifest must be a JSON object")
    return data


def _source_type(spec: Mapping[str, Any], *, index: int) -> SourceType:
    value = _required_text(spec, "type", context=f"source {index}")
    if value not in SUPPORTED_SOURCE_TYPES:
        raise UnsupportedSourceType(
            f"source {index} type {value!r} is not supported; "
            f"expected one of {', '.join(SUPPORTED_SOURCE_TYPES)}"
        )
    return cast(SourceType, value)


def _validate_v0_source_mix(type_counts: Mapping[str, int]) -> None:
    if any(type_counts.get(source_type, 0) != 1 for source_type in SUPPORTED_SOURCE_TYPES):
        raise EvidenceStoryLoadError(
            "manifest must include exactly one youtube_transcript source "
            "and one news_article source"
        )


def _source_id(source_type: str, count: int) -> str:
    prefix = SOURCE_ID_PREFIXES[source_type]
    return f"src_{prefix}_{count:02d}"


def _read_source_text(base_dir: Path, text_path: str, *, index: int) -> str:
    path = Path(text_path)
    if not path.is_absolute():
        path = base_dir / path
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise EvidenceStoryLoadError(f"source {index} text file not found: {path}") from exc
    if not text:
        raise EvidenceStoryLoadError(f"source {index} text file is empty: {path}")
    return text


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    value = mapping.get(key)
    text = _optional_text(value)
    if not text:
        raise EvidenceStoryLoadError(f"{context}.{key} must be a non-empty string")
    return text


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
