from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "drain_hardening.py"

PREAMBLE = """\
# HARDENING.md

Park non-blocking hardening discoveries here. Newest entries go first.

## Entry Format

```md
## YYYY-MM-DD

### <short title>
- File/location:
```

## Parked Items"""

FOOTER = """\
> **Atlas blog** parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md);
> scan that file too when working those lanes."""


def load_tool():
    name = "drain_hardening"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _entry(date_str: str, title: str) -> str:
    return f"## {date_str}\n\n### {title}\n- File/location: `x.py`\n- Effort: S"


def _hardening(*entries: str, footer: bool = True) -> str:
    parts = [PREAMBLE, *entries]
    if footer:
        parts.append(FOOTER)
    return "\n\n".join(parts) + "\n"


def test_parse_ignores_fenced_example_in_preamble():
    tool = load_tool()
    text = _hardening(_entry("2026-05-29", "npm audit"))
    _, region, footer = tool.split_sections(text)
    entries = tool.parse_entries(region)
    # only the real entry, not the ## YYYY-MM-DD example inside the fenced block
    assert [e.date_str for e in entries] == ["2026-05-29"]
    assert "ATLAS-HARDENING.md" in footer


def test_drain_moves_stale_keeps_fresh_and_preserves_footer(tmp_path):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    archive = tmp_path / "archive.md"
    hardening.write_text(
        _hardening(_entry("2026-06-01", "fresh"), _entry("2026-01-10", "stale")),
        encoding="utf-8",
    )

    drained = tool.drain(hardening, archive, today=date(2026, 6, 6), max_age_days=90)

    assert [e.date_str for e in drained] == ["2026-01-10"]
    out = hardening.read_text(encoding="utf-8")
    assert "2026-06-01" in out and "fresh" in out      # fresh kept
    assert "2026-01-10" not in out                      # stale removed
    assert "# HARDENING.md" in out                       # preamble preserved
    assert "ATLAS-HARDENING.md" in out                   # footer preserved
    archived = archive.read_text(encoding="utf-8")
    assert "2026-01-10" in archived and "stale" in archived


def test_drain_preserves_undated_leading_text(tmp_path):
    # An undated note / typo'd heading under "## Parked Items" must survive a drain
    # triggered by an unrelated stale entry -- it is carried through, not dropped.
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    archive = tmp_path / "archive.md"
    note = "## 2026-6-01\n\n### typo-dated note that should not be lost"
    body = "\n\n".join([PREAMBLE, note, _entry("2026-01-10", "stale"), FOOTER]) + "\n"
    hardening.write_text(body, encoding="utf-8")

    drained = tool.drain(hardening, archive, today=date(2026, 6, 6), max_age_days=90)

    assert [e.date_str for e in drained] == ["2026-01-10"]
    out = hardening.read_text(encoding="utf-8")
    assert "typo-dated note that should not be lost" in out  # carried through, not dropped
    assert "ATLAS-HARDENING.md" in out                        # footer still preserved
    assert "2026-01-10" not in out                            # stale entry removed


def test_archive_append_accumulates_without_rewrite(tmp_path):
    tool = load_tool()
    archive = tmp_path / "archive.md"
    first = [tool.Entry(date_str="2026-01-10", body="## 2026-01-10\n\n### one")]
    second = [tool.Entry(date_str="2026-02-11", body="## 2026-02-11\n\n### two")]

    tool.append_to_archive(archive, first)
    tool.append_to_archive(archive, second)

    text = archive.read_text(encoding="utf-8")
    assert text.count("# HARDENING archive") == 1   # header written once
    assert "### one" in text and "### two" in text   # both blocks accumulated


def test_archive_append_writes_header_into_preexisting_empty_file(tmp_path):
    # A zero-length archive (manually created or truncated) must still get the header.
    tool = load_tool()
    archive = tmp_path / "archive.md"
    archive.write_text("", encoding="utf-8")

    tool.append_to_archive(archive, [tool.Entry(date_str="2026-01-10", body="## 2026-01-10\n\n### one")])

    text = archive.read_text(encoding="utf-8")
    assert text.count("# HARDENING archive") == 1   # header present, not malformed
    assert "### one" in text


def test_drain_noop_leaves_file_byte_identical(tmp_path):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    archive = tmp_path / "archive.md"
    original = _hardening(_entry("2026-06-01", "fresh"))
    hardening.write_text(original, encoding="utf-8")

    drained = tool.drain(hardening, archive, today=date(2026, 6, 6), max_age_days=90)

    assert drained == []
    assert hardening.read_text(encoding="utf-8") == original  # untouched, no rewrite
    assert not archive.exists()


def test_drain_is_idempotent(tmp_path):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    archive = tmp_path / "archive.md"
    hardening.write_text(
        _hardening(_entry("2026-06-01", "fresh"), _entry("2026-01-10", "stale")),
        encoding="utf-8",
    )

    tool.drain(hardening, archive, today=date(2026, 6, 6), max_age_days=90)
    second = tool.drain(hardening, archive, today=date(2026, 6, 6), max_age_days=90)

    assert second == []


def test_unparseable_date_is_kept():
    tool = load_tool()
    entries = [tool.Entry(date_str="2026-13-99", body="## 2026-13-99\n\n### weird")]
    kept, drained = tool.partition_by_age(entries, today=date(2026, 6, 6), max_age_days=1)
    assert kept and not drained


def test_check_warns_on_stale_entry_but_exits_zero(tmp_path, capsys):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    hardening.write_text(_hardening(_entry("2026-01-10", "stale")), encoding="utf-8")

    rc = tool.main(
        ["check", "--hardening-file", str(hardening), "--today", "2026-06-06"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARNING" in out and "days old" in out
    assert "drain_hardening.py drain" in out


def test_check_ok_when_within_thresholds(tmp_path, capsys):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    hardening.write_text(_hardening(_entry("2026-06-01", "fresh")), encoding="utf-8")

    rc = tool.main(
        ["check", "--hardening-file", str(hardening), "--today", "2026-06-06"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK:" in out


def test_check_warns_on_size(tmp_path, capsys):
    tool = load_tool()
    hardening = tmp_path / "HARDENING.md"
    hardening.write_text(_hardening(_entry("2026-06-01", "fresh")), encoding="utf-8")

    rc = tool.main(
        [
            "check",
            "--hardening-file",
            str(hardening),
            "--today",
            "2026-06-06",
            "--max-lines",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "exceeds threshold 3" in out
