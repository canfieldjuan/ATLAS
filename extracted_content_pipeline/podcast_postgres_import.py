"""Postgres importer for podcast transcripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from .campaign_ports import TenantScope
from .podcast_transcript_data import (
    PodcastTranscriptLoadResult,
    PodcastTranscriptWarning,
    normalize_podcast_transcript_rows,
)


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class PodcastTranscriptImportResult:
    inserted: int
    skipped: int
    dry_run: bool
    replace_existing: bool
    episode_ids: tuple[str, ...]
    warnings: tuple[PodcastTranscriptWarning, ...] = ()
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "inserted": self.inserted,
            "skipped": self.skipped,
            "dry_run": self.dry_run,
            "replace_existing": self.replace_existing,
            "episode_ids": list(self.episode_ids),
            "warnings": [warning.as_dict() for warning in self.warnings],
        }
        if self.source:
            data["source"] = self.source
        return data


async def import_podcast_transcripts(
    db: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: TenantScope | None = None,
    transcripts_table: str = "podcast_transcripts",
    replace_existing: bool = False,
    dry_run: bool = False,
    normalize: bool = True,
    warnings: Sequence[PodcastTranscriptWarning] = (),
    source: str | None = None,
) -> PodcastTranscriptImportResult:
    """Insert podcast transcript rows into the product-owned Postgres table."""

    table = _identifier(transcripts_table)
    loaded = _loaded_rows(rows, normalize=normalize)
    all_warnings = list(warnings) + list(loaded.warnings)
    prepared: list[JsonDict] = []
    skipped = 0
    for index, transcript in enumerate(loaded.transcripts, start=1):
        episode_id = str(transcript.get("episode_id") or "").strip()
        if not episode_id:
            skipped += 1
            all_warnings.append(
                PodcastTranscriptWarning(
                    code="missing_episode_id",
                    field="episode_id",
                    row_index=index,
                    message="Skipped row because it does not contain an episode_id.",
                )
            )
            continue
        prepared.append(dict(transcript))

    episode_ids = tuple(str(row["episode_id"]) for row in prepared)
    if dry_run:
        return PodcastTranscriptImportResult(
            inserted=len(prepared),
            skipped=skipped,
            dry_run=True,
            replace_existing=replace_existing,
            episode_ids=episode_ids,
            warnings=tuple(all_warnings),
            source=source or loaded.source,
        )

    # Empty string is the sentinel for "no tenant scope". The migration sets
    # account_id NOT NULL DEFAULT '' so the UNIQUE (account_id, episode_id)
    # constraint actually prevents duplicates in single-tenant deployments.
    account_id = (scope or TenantScope()).account_id or ""
    if replace_existing and episode_ids:
        await db.execute(
            f"""
            DELETE FROM {table}
             WHERE account_id = $1
               AND episode_id = ANY($2::text[])
            """,
            account_id,
            list(episode_ids),
        )
    for row in prepared:
        await db.execute(
            f"""
            INSERT INTO {table} (
                account_id, episode_id, title, transcript_text,
                duration_seconds, publish_date, host_name, guest_name,
                source_url, raw_payload, status
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10::jsonb,
                'active'
            )
            """,
            account_id,
            row.get("episode_id"),
            _clean(row.get("title")),
            row.get("transcript_text") or "",
            row.get("duration_seconds"),
            row.get("publish_date"),
            _clean(row.get("host_name")),
            _clean(row.get("guest_name")),
            _clean(row.get("source_url")),
            _jsonb(row.get("raw_payload") or {}),
        )
    return PodcastTranscriptImportResult(
        inserted=len(prepared),
        skipped=skipped,
        dry_run=False,
        replace_existing=replace_existing,
        episode_ids=episode_ids,
        warnings=tuple(all_warnings),
        source=source or loaded.source,
    )


def _loaded_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    normalize: bool,
) -> PodcastTranscriptLoadResult:
    if normalize:
        return normalize_podcast_transcript_rows(rows)
    return PodcastTranscriptLoadResult(
        transcripts=tuple(dict(row) for row in rows if isinstance(row, Mapping)),
    )


def _jsonb(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str, separators=(",", ":"))


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _identifier(value: str) -> str:
    parts = str(value or "").strip().split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid SQL identifier: {value!r}")
    for part in parts:
        if not all(char.isalnum() or char == "_" for char in part):
            raise ValueError(f"invalid SQL identifier: {value!r}")
    return ".".join(f'"{part}"' for part in parts)


__all__ = [
    "PodcastTranscriptImportResult",
    "import_podcast_transcripts",
]
