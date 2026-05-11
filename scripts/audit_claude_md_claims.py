#!/usr/bin/env python3
"""Verify MCP tool counts claimed in CLAUDE.md match @mcp.tool decorators.

Parses the per-server section headers in CLAUDE.md
(e.g. "### Email MCP Server (9 tools)") and compares each claim
against the actual count of @mcp.tool() decorators in the matching
atlas_brain/mcp/*_server.py file. For b2b_churn_server, sums across
atlas_brain/mcp/b2b/*.py because the server is split into modules.

Exits 0 if all claims match the code. Exits 1 if any drift.

Usage:
    python scripts/audit_claude_md_claims.py
"""
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
    "Intelligence": "intelligence_server.py",
    "Universal Scraper": "scraper_server.py",
    "Memory": "memory_server.py",
    "B2B Churn Intelligence": "_b2b_sum",
}

HEADER_PATTERN = re.compile(
    r"^###\s+(?P<name>.+?)\s+MCP Server\s*\(\s*(?P<count>\d+\+?)(?:\s+tools)?",
    re.MULTILINE,
)


def count_decorators(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("@mcp.tool")
    )


def actual_count_for(file_key: str) -> int:
    if file_key == "_b2b_sum":
        b2b = MCP_DIR / "b2b"
        if not b2b.is_dir():
            return -1
        return sum(count_decorators(p) for p in sorted(b2b.glob("*.py")))
    return count_decorators(MCP_DIR / file_key)


def main() -> int:
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return 2

    text = CLAUDE_MD.read_text(encoding="utf-8")
    rows = []
    for m in HEADER_PATTERN.finditer(text):
        name = m.group("name").strip()
        claimed = m.group("count")
        file_key = HEADER_TO_FILE.get(name)
        if file_key is None:
            # Unknown server name in a header -- flag it rather than ignore.
            rows.append((name, claimed, "?", "UNKNOWN"))
            continue
        actual = actual_count_for(file_key)
        if claimed.endswith("+"):
            # Soft / loose claim -- always drift since we have an exact count.
            status = "DRIFT (soft count)"
        else:
            status = "OK" if int(claimed) == actual else "DRIFT"
        rows.append((name, claimed, str(actual), status))

    if not rows:
        print("No '### ... MCP Server (N tools)' headers found in CLAUDE.md.")
        return 1

    name_w = max(len(r[0]) for r in rows)
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
