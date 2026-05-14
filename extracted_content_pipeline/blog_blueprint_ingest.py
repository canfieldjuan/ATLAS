"""File adapter for standalone blog-blueprint ingestion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from .blog_blueprint_postgres import BlogBlueprint


BlogBlueprintDataFormat = Literal["auto", "json"]


@dataclass(frozen=True)
class BlogBlueprintWarning:
    """Non-fatal warning for one loaded blueprint row."""

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
class BlogBlueprintLoadResult:
    """Normalized blog blueprints plus non-fatal validation warnings."""

    blueprints: tuple[BlogBlueprint, ...]
    warnings: tuple[BlogBlueprintWarning, ...] = ()
    source: str | None = None

    def warning_dicts(self) -> list[dict[str, Any]]:
        return [warning.as_dict() for warning in self.warnings]

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded": len(self.blueprints),
            "skipped": len(self.warnings),
            "source": self.source,
            "warnings": self.warning_dicts(),
        }


def load_blog_blueprints_from_file(
    path: str | Path,
    *,
    file_format: BlogBlueprintDataFormat = "auto",
    target_mode: str | None = None,
    topic_type: str | None = None,
) -> BlogBlueprintLoadResult:
    """Load host-supplied blog blueprints from a JSON file."""

    source = Path(path)
    resolved_format = _resolve_format(source, file_format)
    if resolved_format != "json":
        raise ValueError(f"Unsupported blog blueprint file format: {resolved_format}")
    result = normalize_blog_blueprint_rows(
        _load_json_rows(source),
        target_mode=target_mode,
        topic_type=topic_type,
    )
    return BlogBlueprintLoadResult(
        blueprints=result.blueprints,
        warnings=result.warnings,
        source=str(source),
    )


def normalize_blog_blueprint_rows(
    rows: Sequence[Any],
    *,
    target_mode: str | None = None,
    topic_type: str | None = None,
) -> BlogBlueprintLoadResult:
    """Normalize loose JSON rows into ``BlogBlueprint`` objects."""

    blueprints: list[BlogBlueprint] = []
    warnings: list[BlogBlueprintWarning] = []
    default_target_mode = _clean(target_mode)
    default_topic_type = _clean(topic_type)
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            warnings.append(
                BlogBlueprintWarning(
                    code="row_not_object",
                    row_index=index,
                    message="Skipped row because it is not an object.",
                )
            )
            continue
        normalized, row_warnings = _normalize_row(
            row,
            row_index=index,
            default_target_mode=default_target_mode,
            default_topic_type=default_topic_type,
        )
        warnings.extend(row_warnings)
        if normalized is not None:
            blueprints.append(normalized)
    return BlogBlueprintLoadResult(
        blueprints=tuple(blueprints),
        warnings=tuple(warnings),
    )


def _normalize_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    default_target_mode: str | None,
    default_topic_type: str | None,
) -> tuple[BlogBlueprint | None, list[BlogBlueprintWarning]]:
    warnings: list[BlogBlueprintWarning] = []
    target_mode = _clean(row.get("target_mode")) or default_target_mode
    if not target_mode:
        warnings.append(
            BlogBlueprintWarning(
                code="missing_target_mode",
                row_index=row_index,
                field="target_mode",
                message="Skipped row because target_mode is missing.",
            )
        )
        return None, warnings

    title = _clean(row.get("suggested_title")) or _clean(row.get("title"))
    slug = _clean(row.get("slug")) or _slugify(title or row.get("topic") or "")
    if not slug:
        warnings.append(
            BlogBlueprintWarning(
                code="missing_slug",
                row_index=row_index,
                field="slug",
                message="Skipped row because slug is missing and no title can derive it.",
            )
        )
        return None, warnings

    resolved_topic_type = (
        _clean(row.get("topic_type"))
        or default_topic_type
        or "blog_post"
    )
    typed_keys = {"target_mode", "topic_type", "slug", "suggested_title", "title"}
    payload = {str(key): value for key, value in row.items() if str(key) not in typed_keys}
    return (
        BlogBlueprint(
            target_mode=target_mode,
            topic_type=resolved_topic_type,
            slug=slug,
            suggested_title=title or slug.replace("-", " ").title(),
            payload=payload,
        ),
        warnings,
    )


def _resolve_format(
    path: Path,
    file_format: BlogBlueprintDataFormat,
) -> Literal["json"]:
    if file_format == "json":
        return "json"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    raise ValueError(f"Cannot infer blog blueprint format from file suffix: {path}")


def _load_json_rows(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return list(data)
    if not isinstance(data, Mapping):
        raise ValueError("JSON blog blueprint data must be an object or array")
    for key in ("blueprints", "rows", "data"):
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [dict(data)]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


__all__ = [
    "BlogBlueprintLoadResult",
    "BlogBlueprintWarning",
    "load_blog_blueprints_from_file",
    "normalize_blog_blueprint_rows",
]
