from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_review_misses.py"

SEED_ONLY = """
## Ledger

| Date | Escaped issue | Missed by (human / AI / CI) | Root cause | New gate added | Owner |
|---|---|---|---|---|---|
| _seed_ | _First real entry goes here. Until then this row documents the format._ | - | - | - | - |

## Lifecycle
queue not archive
"""

WITH_ROWS = """
## Ledger

| Date | Escaped issue | Missed by (human / AI / CI) | Root cause | New gate added | Owner |
|---|---|---|---|---|---|
| _seed_ | _placeholder_ | - | - | - | - |
| 2026-06-06 | Missing auth check on admin endpoint | Human review | No auth rule fired | Added R3 trigger | Dev A |
| 2026-06-07 | Fail-open regex a Copilot bot finding caught late | Human review | weak fixture | (none yet) | Dev B |
| 2026-06-08 | Flaky retry double-send | CI | missing idempotency test | Added retry test | Dev C |

## Lifecycle
queue not archive
"""


def load():
    spec = importlib.util.spec_from_file_location("summarize_review_misses", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_only_parses_to_no_real_rows():
    mod = load()
    rows = mod.parse_ledger(SEED_ONLY)
    assert rows == []
    assert mod.summarize(rows)["total"] == 0


def test_parses_real_rows_and_skips_placeholder():
    mod = load()
    rows = mod.parse_ledger(WITH_ROWS)
    assert len(rows) == 3  # seed/placeholder row excluded
    assert rows[0]["issue"].startswith("Missing auth check")


def test_summary_buckets():
    mod = load()
    summary = mod.summarize(mod.parse_ledger(WITH_ROWS))
    assert summary["total"] == 3
    assert summary["by_human"] == 2
    assert summary["by_ci"] == 1
    # Two human misses, one of which names a Copilot/bot finding.
    assert summary["ai_missed_by_human"] == 1


def test_gated_vs_ungated():
    mod = load()
    summary = mod.summarize(mod.parse_ledger(WITH_ROWS))
    assert summary["gated"] == 2  # auth + retry rows name a gate
    assert summary["ungated"] == 1  # the "(none yet)" row


def test_empty_cell_detection():
    mod = load()
    assert mod._is_empty("-")
    assert mod._is_empty("")
    assert mod._is_empty("_seed_")
    assert not mod._is_empty("Added R3 trigger")


def test_fail_on_ungated_exit_codes(tmp_path: Path):
    mod = load()
    seed = tmp_path / "seed.md"
    seed.write_text(SEED_ONLY, encoding="utf-8")
    # Seed-only: nothing ungated -> exit 0 even with the flag.
    assert mod.main(["--ledger", str(seed), "--fail-on-ungated"]) == 0

    rows = tmp_path / "rows.md"
    rows.write_text(WITH_ROWS, encoding="utf-8")
    # Has an ungated row -> 0 without the flag, 1 with it.
    assert mod.main(["--ledger", str(rows)]) == 0
    assert mod.main(["--ledger", str(rows), "--fail-on-ungated"]) == 1


def test_missing_ledger_is_usage_error(tmp_path: Path):
    mod = load()
    assert mod.main(["--ledger", str(tmp_path / "nope.md")]) == 2


def test_missing_ledger_table_fails_closed(tmp_path: Path):
    # A real file with no "## Ledger" table must fail, not report seed-only
    # success -- even (especially) under --fail-on-ungated.
    mod = load()
    assert mod._ledger_table_present("## Something else\nno table here\n") is False
    assert mod._ledger_table_present(SEED_ONLY) is True
    wrong = tmp_path / "wrong.md"
    wrong.write_text("# Some other doc\n\nno ledger table at all\n", encoding="utf-8")
    assert mod.main(["--ledger", str(wrong)]) == 2
    assert mod.main(["--ledger", str(wrong), "--fail-on-ungated"]) == 2
