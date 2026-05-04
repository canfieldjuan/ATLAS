"""Transcript data adapters for podcast repurposing.

Mirrors campaign_customer_data.py with two extra format branches:
- ``.txt``  one-episode plain text. Filename stem becomes the episode_id;
            first non-empty line becomes the title; remainder is the
            transcript body.
- ``.srt``  SRT subtitle file. Index lines and timestamp lines are
            stripped; caption text is concatenated with single spaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .campaign_ports import TenantScope
from .podcast_ports import PodcastTranscript


TranscriptDataFormat = Literal["auto", "json", "csv", "txt", "srt"]

# Lower bound below which the loader emits a transcript_too_short warning.
_MIN_USEFUL_TRANSCRIPT_CHARS = 200

# SRT timestamp lines look like ``00:00:01,500 --> 00:00:04,200``.
_SRT_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*$"
)
_SRT_INDEX_RE = re.compile(r"^\s*\d+\s*$")


@dataclass(frozen=True)
class PodcastTranscriptWarning:
    """Non-fatal validation warning for one loaded transcript row."""

    code: str
    message: str
    row_index: int | None = None
    field: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.row_index is not None:
            data["row_index"] = self.row_index
        if self.field:
            data["field"] = self.field
        return data


@dataclass(frozen=True)
class PodcastTranscriptLoadResult:
    """Normalized transcripts plus validation warnings from a transcript file."""

    transcripts: tuple[dict[str, Any], ...]
    warnings: tuple[PodcastTranscriptWarning, ...] = ()
    source: str | None = None

    def warning_dicts(self) -> list[dict[str, Any]]:
        return [warning.as_dict() for warning in self.warnings]


def load_podcast_transcripts_from_file(
    path: str | Path,
    *,
    file_format: TranscriptDataFormat = "auto",
) -> PodcastTranscriptLoadResult:
    """Load podcast transcripts from JSON, CSV, plain text, or SRT.

    For ``.txt`` and ``.srt`` files, the filename stem is used as the
    episode_id and the loader returns a single-row result. For ``.json``
    and ``.csv``, an array of rows is expected (or a wrapped object with
    one of the keys ``transcripts``, ``episodes``, ``rows``, ``data``).
    """

    source = Path(path)
    resolved_format = _resolve_format(source, file_format)
    if resolved_format == "csv":
        rows = _load_csv_rows(source)
    elif resolved_format == "json":
        rows = _load_json_rows(source)
    elif resolved_format == "txt":
        rows = _load_txt_rows(source)
    else:
        rows = _load_srt_rows(source)
    result = normalize_podcast_transcript_rows(rows)
    return PodcastTranscriptLoadResult(
        transcripts=result.transcripts,
        warnings=result.warnings,
        source=str(source),
    )


def normalize_podcast_transcript_rows(
    rows: Sequence[Any],
) -> PodcastTranscriptLoadResult:
    """Normalize loose transcript rows and collect non-fatal warnings."""

    transcripts: list[dict[str, Any]] = []
    warnings: list[PodcastTranscriptWarning] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            warnings.append(
                PodcastTranscriptWarning(
                    code="row_not_object",
                    row_index=index,
                    message="Skipped row because it is not an object.",
                )
            )
            continue
        normalized = normalize_podcast_transcript(row)
        if normalized is None:
            warnings.append(
                PodcastTranscriptWarning(
                    code="missing_episode_id",
                    field="episode_id",
                    row_index=index,
                    message="Skipped row because it has no episode_id.",
                )
            )
            continue
        transcripts.append(normalized)
        warnings.extend(_validation_warnings(normalized, row_index=index))
    return PodcastTranscriptLoadResult(
        transcripts=tuple(transcripts),
        warnings=tuple(warnings),
    )


def normalize_podcast_transcript(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize one row into the canonical transcript dict shape.

    Returns None if the row has no usable episode_id.
    """

    episode_id = _clean_text(row.get("episode_id"))
    if not episode_id:
        return None

    transcript_text = _clean_text(row.get("transcript_text") or row.get("text"))
    title = _clean_text(row.get("title")) or ""
    duration = _coerce_int(row.get("duration_seconds") or row.get("duration"))
    publish_date = _clean_text(row.get("publish_date") or row.get("published_at")) or None
    host_name = _clean_text(row.get("host_name") or row.get("host")) or None
    guest_name = _clean_text(row.get("guest_name") or row.get("guest")) or None
    source_url = _clean_text(row.get("source_url") or row.get("url")) or None

    known_fields = {
        "episode_id",
        "title",
        "transcript_text",
        "text",
        "duration_seconds",
        "duration",
        "publish_date",
        "published_at",
        "host_name",
        "host",
        "guest_name",
        "guest",
        "source_url",
        "url",
    }
    raw_payload = {
        str(key): value
        for key, value in row.items()
        if key not in known_fields and value not in (None, "", [], {})
    }

    return {
        "episode_id": episode_id,
        "title": title,
        "transcript_text": transcript_text,
        "duration_seconds": duration,
        "publish_date": publish_date,
        "host_name": host_name,
        "guest_name": guest_name,
        "source_url": source_url,
        "raw_payload": raw_payload,
    }


@dataclass(frozen=True)
class FilePodcastTranscriptRepository:
    """TranscriptRepository backed by loaded transcript rows.

    Lets the example/offline runner read transcripts from disk without a
    Postgres pool.
    """

    transcripts: Sequence[Mapping[str, Any]]
    warnings: Sequence[PodcastTranscriptWarning] = field(default_factory=tuple)
    source: str | None = None

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        file_format: TranscriptDataFormat = "auto",
    ) -> "FilePodcastTranscriptRepository":
        loaded = load_podcast_transcripts_from_file(path, file_format=file_format)
        return cls(
            transcripts=loaded.transcripts,
            warnings=loaded.warnings,
            source=loaded.source,
        )

    async def read_transcripts(
        self,
        *,
        scope: TenantScope,
        episode_id: str | None = None,
        limit: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[PodcastTranscript]:
        del scope
        del filters
        rows = list(self.transcripts)
        if episode_id is not None:
            target = str(episode_id).strip()
            rows = [row for row in rows if str(row.get("episode_id") or "").strip() == target]
        rows = rows[: max(0, limit)]
        return [_row_to_transcript(row) for row in rows]


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _resolve_format(path: Path, file_format: TranscriptDataFormat) -> Literal["json", "csv", "txt", "srt"]:
    if file_format != "auto":
        return file_format
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".txt":
        return "txt"
    if suffix == ".srt":
        return "srt"
    raise ValueError(f"Cannot infer transcript file format from suffix: {path}")


def _load_json_rows(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, Mapping):
        raise ValueError("JSON transcript data must be an object or array")
    for key in ("transcripts", "episodes", "rows", "data"):
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [dict(data)]


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        for row in reader:
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                cleaned_value = _coerce_csv_value(value)
                if cleaned_value not in (None, ""):
                    cleaned[str(key)] = cleaned_value
            rows.append(cleaned)
    return rows


def _load_txt_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = ""
    body_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            title = stripped
            body_start = index + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return [{
        "episode_id": path.stem,
        "title": title,
        "transcript_text": body,
    }]


def _load_srt_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    captions: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _SRT_INDEX_RE.match(line):
            continue
        if _SRT_TIMESTAMP_RE.match(line):
            continue
        captions.append(line)
    return [{
        "episode_id": path.stem,
        "title": path.stem,
        "transcript_text": " ".join(captions).strip(),
    }]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_csv_value(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    if text[0] not in "[{":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _coerce_int(value: Any) -> int | None:
    if value in (None, "", []):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validation_warnings(
    transcript: Mapping[str, Any],
    *,
    row_index: int,
) -> list[PodcastTranscriptWarning]:
    warnings: list[PodcastTranscriptWarning] = []
    body = str(transcript.get("transcript_text") or "")
    if not body.strip():
        warnings.append(
            PodcastTranscriptWarning(
                code="empty_transcript",
                field="transcript_text",
                row_index=row_index,
                message="Transcript text is empty.",
            )
        )
    elif len(body) < _MIN_USEFUL_TRANSCRIPT_CHARS:
        warnings.append(
            PodcastTranscriptWarning(
                code="transcript_too_short",
                field="transcript_text",
                row_index=row_index,
                message=(
                    f"Transcript text is shorter than {_MIN_USEFUL_TRANSCRIPT_CHARS} "
                    "characters; extraction quality may suffer."
                ),
            )
        )
    if not str(transcript.get("title") or "").strip():
        warnings.append(
            PodcastTranscriptWarning(
                code="missing_title",
                field="title",
                row_index=row_index,
                message="Transcript has no title; episode_id will be used as the title.",
            )
        )
    return warnings


def _row_to_transcript(row: Mapping[str, Any]) -> PodcastTranscript:
    raw_payload = row.get("raw_payload") or {}
    if not isinstance(raw_payload, Mapping):
        raw_payload = {}
    episode_id = str(row.get("episode_id") or "")
    # Fall back to episode_id when no explicit title was provided so
    # downstream prompts always see a non-empty title field. The
    # missing_title warning still fires earlier in normalization.
    title = str(row.get("title") or "").strip() or episode_id
    return PodcastTranscript(
        episode_id=episode_id,
        title=title,
        transcript_text=str(row.get("transcript_text") or ""),
        duration_seconds=_coerce_int(row.get("duration_seconds")),
        publish_date=row.get("publish_date") or None,
        host_name=row.get("host_name") or None,
        guest_name=row.get("guest_name") or None,
        source_url=row.get("source_url") or None,
        raw_payload=dict(raw_payload),
    )


__all__ = [
    "TranscriptDataFormat",
    "PodcastTranscriptLoadResult",
    "PodcastTranscriptWarning",
    "FilePodcastTranscriptRepository",
    "load_podcast_transcripts_from_file",
    "normalize_podcast_transcript_rows",
    "normalize_podcast_transcript",
]
