#!/usr/bin/env python3
"""Verify MCP tool counts claimed in CLAUDE.md match @mcp.tool decorators."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
MCP_DIR = REPO_ROOT / "atlas_brain" / "mcp"

HEADER_TO_FILE = {
    "CRM": "crm_server.py",
    "Email": "email_server.py",
    "Twilio": "twilio_server.py",
    "Calendar": "calendar_server.py",
    "Invoicing": "invoicing_server.py",
    "Invoicing Readonly": "invoicing_readonly_server.py",
    "Content Ops Deflection Readonly": "content_ops_deflection_readonly_server.py",
    "Content Ops Marketer Verify": "content_ops_marketer_verify_server.py",
    "Intelligence": "intelligence_server.py",
    "Universal Scraper": "scraper_server.py",
    "Memory": "memory_server.py",
    "B2B Churn Intelligence": "_b2b_sum",
}

HEADER_PATTERN = re.compile(
    r"^###\s+(?P<name>.+?)\s+MCP Server"
    r"\s*\(\s*(?P<count>\d+\+?)\s+tools\s*(?:,[^)]*)?\)",
    re.MULTILINE,
)
MCP_HEADING_PATTERN = re.compile(
    r"^###\s+(?P<name>.+?)\s+MCP Server(?P<suffix>.*)$",
    re.MULTILINE,
)
TOOL_DECORATOR_PATTERN = re.compile(r"^\s*@mcp\.tool(?:\s*\(|\s*$)")

MISSING_FILE = "MISSING_FILE"
MISSING_DIR = "MISSING_DIR"


def count_decorators(path: Path) -> int | str:
    if not path.exists():
        return MISSING_FILE
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if TOOL_DECORATOR_PATTERN.match(line)
    )


def actual_count_for(file_key: str) -> int | str:
    if file_key == "_b2b_sum":
        b2b = MCP_DIR / "b2b"
        if not b2b.is_dir():
            return MISSING_DIR
        total = 0
        for path in sorted(b2b.glob("*.py")):
            count = count_decorators(path)
            if isinstance(count, str):
                return count
            total += count
        return total
    return count_decorators(MCP_DIR / file_key)


def audit_claims(text: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    matched_spans: set[tuple[int, int]] = set()
    seen_known: set[str] = set()
    for match in HEADER_PATTERN.finditer(text):
        matched_spans.add(match.span())
        name = match.group("name").strip()
        claimed = match.group("count")
        file_key = HEADER_TO_FILE.get(name)
        if file_key is None:
            rows.append((name, claimed, "?", "UNKNOWN"))
            continue
        seen_known.add(name)

        actual = actual_count_for(file_key)
        if isinstance(actual, str):
            rows.append((name, claimed, "N/A", actual))
        elif claimed.endswith("+"):
            rows.append((name, claimed, str(actual), "DRIFT (soft count)"))
        else:
            status = "OK" if int(claimed) == actual else "DRIFT"
            rows.append((name, claimed, str(actual), status))

    for heading in MCP_HEADING_PATTERN.finditer(text):
        if heading.span() in matched_spans:
            continue
        name = heading.group("name").strip()
        suffix = heading.group("suffix").strip() or "<missing count>"
        rows.append((name, suffix, "N/A", "MALFORMED"))

    for name in HEADER_TO_FILE:
        if name not in seen_known:
            rows.append((name, "MISSING", "N/A", "MISSING"))
    return rows


def main() -> int:
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return 2

    rows = audit_claims(CLAUDE_MD.read_text(encoding="utf-8"))
    name_w = max(len(row[0]) for row in rows)
    print(f"{'server'.ljust(name_w)}  claimed  actual  status")
    print("-" * (name_w + 28))
    drift = False
    for name, claimed, actual, status in rows:
        if status != "OK":
            drift = True
        print(f"{name.ljust(name_w)}  {claimed:>7}  {actual:>6}  {status}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
