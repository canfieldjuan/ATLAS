#!/usr/bin/env python3
"""Build small messy support-ticket CSV fixtures from source-row JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


DEFAULT_LIMIT = 8
FIELDNAMES = ("ticket_id", "subject", "message", "resolution_text", "pain_category")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = load_source_rows(args.source_rows, limit=args.limit)
    if not rows:
        raise SystemExit("No source rows found.")
    tickets = support_ticket_rows(rows)
    manifest = write_messy_csv_fixtures(tickets, args.output_dir)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Wrote {len(manifest['cases'])} messy CSV fixture(s) to {args.output_dir}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_rows", type=Path, help="Content Ops source-row JSON/JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 3:
        raise SystemExit("--limit must be at least 3")
    return args


def load_source_rows(path: Path, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    if limit < 1:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            rows = _rows_from_mapping(data)
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            rows = list(data)
        else:
            raise ValueError("Source rows must be a JSON array, object bundle, or JSONL file.")
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    return out[:limit]


def _rows_from_mapping(data: Mapping[str, Any]) -> list[Any]:
    for key in ("sources", "rows", "data", "tickets", "support_tickets"):
        value = data.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    return [dict(data)]


def support_ticket_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    tickets: list[dict[str, str]] = []
    for index, row in enumerate(source_rows, start=1):
        ticket_id = _first_text(row, ("ticket_id", "source_id", "id", "complaint_id")) or f"ticket-{index}"
        subject = (
            _first_text(row, ("subject", "source_title", "title", "issue", "pain_category"))
            or f"Imported support row {index}"
        )
        message = _first_text(
            row,
            (
                "message",
                "body",
                "text",
                "description",
                "complaint",
                "consumer_complaint_narrative",
                "narrative",
            ),
        )
        resolution = _first_text(
            row,
            (
                "resolution_text",
                "answer",
                "company_response",
                "company_response_to_consumer",
            ),
        )
        category = _first_text(row, ("pain_category", "issue", "category", "product", "topic"))
        tickets.append({
            "ticket_id": ticket_id,
            "subject": subject,
            "message": message,
            "resolution_text": resolution,
            "pain_category": category,
        })
    return tickets


def write_messy_csv_fixtures(
    ticket_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    rows = [dict(row) for row in ticket_rows if _first_text(row, ("message", "subject"))]
    if len(rows) < 3:
        raise ValueError("At least three text-bearing rows are required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _write_case(
            output_dir,
            "bom_utf8.csv",
            rows[:4],
            delimiter=",",
            encoding="utf-8-sig",
            expected="parsed",
            notes="UTF-8 BOM header with browser-export punctuation.",
        ),
        _write_case(
            output_dir,
            "cp1252_semicolon.csv",
            _with_suffix(rows[:4], " Buyer\u2019s wording kept intact."),
            delimiter=";",
            encoding="cp1252",
            expected="parsed",
            notes="cp1252 export and semicolon delimiter.",
        ),
        _write_case(
            output_dir,
            "tab_delimited.csv",
            rows[:4],
            delimiter="\t",
            encoding="utf-8",
            expected="parsed",
            notes="tab-delimited help-desk export.",
        ),
        _write_case(
            output_dir,
            "html_bodies.csv",
            _with_html(rows[:4]),
            delimiter=",",
            encoding="utf-8",
            expected="parsed",
            notes="HTML body text and entities in the message column.",
        ),
        _write_leading_metadata_case(output_dir, rows[:3]),
        _write_extra_cells_case(output_dir, rows[:3]),
        _write_short_rows_case(output_dir, rows[:3]),
        _write_multiline_case(output_dir, rows[:3]),
    ]
    manifest = {
        "source": "deflection_messy_csv_fixtures",
        "row_count": len(rows),
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_case(
    output_dir: Path,
    filename: str,
    rows: Sequence[Mapping[str, str]],
    *,
    delimiter: str,
    encoding: str,
    expected: str,
    notes: str,
) -> dict[str, str]:
    path = output_dir / filename
    text = _render_csv(rows, delimiter=delimiter)
    if encoding == "cp1252":
        path.write_bytes(text.encode("cp1252", errors="replace"))
    else:
        path.write_text(text, encoding=encoding)
    return {"name": filename, "path": filename, "expected": expected, "notes": notes}


def _write_leading_metadata_case(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    path = output_dir / "leading_metadata_row.csv"
    path.write_text(
        "Zendesk ticket export generated 2026-06-09\n" + _render_csv(rows, delimiter=","),
        encoding="utf-8",
    )
    return {
        "name": path.name,
        "path": path.name,
        "expected": "parsed",
        "notes": "Provider title row before a plausible header should be skipped.",
    }


def _write_extra_cells_case(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    path = output_dir / "ragged_extra_cells.csv"
    first = rows[0]
    lines = [
        ",".join(FIELDNAMES),
        ",".join([
            first.get("ticket_id", ""),
            first.get("subject", ""),
            "unquoted, comma splits this row",
            first.get("resolution_text", ""),
            first.get("pain_category", ""),
        ]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "name": path.name,
        "path": path.name,
        "expected": "fail_loud",
        "notes": "Extra unquoted delimiter must raise instead of shifting columns.",
    }


def _write_short_rows_case(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    path = output_dir / "ragged_short_rows.csv"
    first, second = rows[0], rows[1]
    lines = [
        ",".join(FIELDNAMES),
        _csv_line([first.get(key, "") for key in FIELDNAMES]),
        _csv_line([second.get("ticket_id", ""), second.get("subject", "")]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "name": path.name,
        "path": path.name,
        "expected": "parsed_partial",
        "notes": "Short rows parse, but missing body text must remain visible in counts.",
    }


def _write_multiline_case(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    multiline = [dict(row) for row in rows]
    multiline[0]["message"] = (
        f"{multiline[0].get('message', '')}\n"
        "Customer pasted a second line with commas, tabs, and semicolons."
    )
    return _write_case(
        output_dir,
        "quoted_multiline.csv",
        multiline,
        delimiter=",",
        encoding="utf-8",
        expected="parsed",
        notes="Quoted multiline body with embedded delimiters.",
    )


def _render_csv(rows: Sequence[Mapping[str, str]], *, delimiter: str) -> str:
    lines = [delimiter.join(FIELDNAMES)]
    for row in rows:
        lines.append(_csv_line([str(row.get(key, "")) for key in FIELDNAMES], delimiter=delimiter))
    return "\n".join(lines) + "\n"


def _csv_line(values: Sequence[str], *, delimiter: str = ",") -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="")
    writer.writerow(list(values))
    return buffer.getvalue()


def _with_suffix(rows: Sequence[Mapping[str, str]], suffix: str) -> list[dict[str, str]]:
    out = [dict(row) for row in rows]
    out[0]["message"] = f"{out[0].get('message', '')} {suffix}"
    return out


def _with_html(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    out = [dict(row) for row in rows]
    for row in out:
        row["message"] = f"<p>{row.get('message', '')}</p><p>Needs review &amp; policy check.</p>"
    return out


def _first_text(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
