from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_synthetic_support_tickets.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_support_tickets", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tmp_path, name, extra_args=()):
    gen = load_generator()
    out = tmp_path / name
    assert gen.main(["--output-dir", str(out), *extra_args]) == 0
    return out


def read_rows(out_dir, encoding="utf-8", delimiter=","):
    text = (out_dir / "tickets.csv").read_bytes().decode(encoding)
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


# --- determinism -----------------------------------------------------------

def test_same_seed_is_byte_identical(tmp_path):
    a = run(tmp_path, "a", ["--seed", "7"])
    b = run(tmp_path, "b", ["--seed", "7"])
    assert (a / "tickets.csv").read_bytes() == (b / "tickets.csv").read_bytes()
    assert (a / "ground_truth.json").read_bytes() == (
        b / "ground_truth.json"
    ).read_bytes()


def test_different_seed_differs(tmp_path):
    a = run(tmp_path, "a", ["--seed", "7"])
    b = run(tmp_path, "b", ["--seed", "8"])
    assert (a / "tickets.csv").read_bytes() != (b / "tickets.csv").read_bytes()


# --- ground truth ------------------------------------------------------------

def test_ground_truth_matches_rows(tmp_path):
    out = run(tmp_path, "labeled", ["--seed", "11"])
    rows = read_rows(out)
    truth = json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))

    assert truth["total_tickets"] == len(rows)
    by_id = {row["ticket_id"]: row for row in rows}
    assert set(truth["ticket_to_intent"]) == set(by_id)

    for cluster in truth["expected_clusters"]:
        assert cluster["size"] == len(cluster["ticket_ids"])
        for ticket_id in cluster["ticket_ids"]:
            row = by_id[ticket_id]
            assert truth["ticket_to_intent"][ticket_id] == cluster["intent"]
            assert row["pain_category"] == cluster["pain_category"]
            # resolution coverage is per-intent: always or never
            assert bool(row["resolution_text"].strip()) == cluster["has_resolution"]


# --- injectors ---------------------------------------------------------------

def test_no_labels_empties_pain_category(tmp_path):
    out = run(tmp_path, "raw", ["--seed", "7", "--no-labels"])
    rows = read_rows(out)
    assert rows and all(row["pain_category"] == "" for row in rows)


def test_unmapped_body_column_renames_message(tmp_path):
    out = run(tmp_path, "unmapped", ["--seed", "7", "--unmapped-body-column"])
    header = (
        (out / "tickets.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    )
    assert "customer_msg" in header
    assert "message" not in header


def test_utf16_encoding_not_utf8_readable(tmp_path):
    out = run(tmp_path, "utf16", ["--seed", "7", "--encoding", "utf-16"])
    raw = (out / "tickets.csv").read_bytes()
    assert raw[:2] in (b"\xff\xfe", b"\xfe\xff")  # BOM present
    assert "ticket_id" in raw.decode("utf-16")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:  # pragma: no cover - would mean the injector silently did nothing
        raise AssertionError("utf-16 output unexpectedly decoded as utf-8")


def test_junk_rows_add_banner_blank_and_short(tmp_path):
    clean = run(tmp_path, "clean", ["--seed", "7"])
    junk = run(tmp_path, "junk", ["--seed", "7", "--junk-rows"])
    clean_lines = (clean / "tickets.csv").read_text(encoding="utf-8").splitlines()
    junk_lines = (junk / "tickets.csv").read_text(encoding="utf-8").splitlines()
    assert len(junk_lines) == len(clean_lines) + 3
    assert junk_lines[0].startswith("Exported by HelpDesk Pro")
    assert junk_lines[1].startswith("ticket_id")  # header comes second


def test_semicolon_delimiter(tmp_path):
    out = run(tmp_path, "semi", ["--seed", "7", "--delimiter", ";"])
    rows = read_rows(out, delimiter=";")
    assert rows and rows[0]["ticket_id"]


# --- round trip through the real ingestion package ---------------------------

def test_clean_output_round_trips_with_no_warnings(tmp_path):
    from extracted_content_pipeline.support_ticket_input_package import (
        build_support_ticket_input_package,
    )

    out = run(tmp_path, "clean", ["--seed", "7"])
    rows = read_rows(out)
    package = build_support_ticket_input_package([dict(row) for row in rows])

    assert package.warnings == ()
    assert package.metadata["included_row_count"] == len(rows)
    assert package.metadata["skipped_row_count"] == 0
    # intents with resolution templates guarantee evidence is present
    assert package.metadata["support_ticket_resolution_evidence_present"] is True


def test_invalid_base_date_exits_2(tmp_path):
    gen = load_generator()
    code = gen.main(
        ["--output-dir", str(tmp_path / "bad"), "--base-date", "not-a-date"]
    )
    assert code == 2
