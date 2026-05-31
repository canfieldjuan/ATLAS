from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "docs"
    / "extraction"
    / "validation"
    / "fixtures"
    / "faq_deflection_live_upload_sample.csv"
)
EXPECTED_HEADER = [
    "ticket_id",
    "created_at",
    "subject",
    "message",
    "status",
    "tags",
]


def _rows() -> list[dict[str, str]]:
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_HEADER
        return list(reader)


def test_live_upload_fixture_has_representative_ticket_shape() -> None:
    rows = _rows()

    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"closed"}
    assert len({row["ticket_id"] for row in rows}) == len(rows)
    assert all(row["message"].strip().endswith("?") for row in rows)
    assert all(row["tags"].strip() for row in rows)


def test_live_upload_fixture_covers_snapshot_worthy_themes() -> None:
    text = " ".join(
        f"{row['subject']} {row['message']} {row['tags']}".lower()
        for row in _rows()
    )

    for theme in ("export", "billing", "security", "team"):
        assert theme in text
