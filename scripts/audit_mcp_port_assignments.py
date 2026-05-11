#!/usr/bin/env python3
"""Verify MCP port numbers in atlas_brain/config.py match CLAUDE.md.

`MCPConfig` in atlas_brain/config.py declares the canonical port for
each MCP server (e.g., `crm_port: int = Field(default=8056, ...)`).
CLAUDE.md references these same numbers in two shapes:

  A. Env-var docs:        `ATLAS_MCP_CRM_PORT=8056`
  B. Markdown table rows: `| CRM | 8056 | 10 | ... |` (PR #457+)

Either or both may be present; both should match the config source.

Exits 0 if every doc claim matches config. Exits 1 on any drift.

Usage:
    python scripts/audit_mcp_port_assignments.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = REPO_ROOT / "atlas_brain" / "config.py"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

ENV_VAR_LINE = re.compile(r"^ATLAS_MCP_([A-Z][A-Z_]+)_PORT\s*=\s*(\d{4,5})", re.MULTILINE)
TABLE_ROW = re.compile(
    r"^\|\s*([A-Za-z][A-Za-z0-9 +/()-]+?)\s*\|\s*(\d{4,5})\s*\|",
    re.MULTILINE,
)

# Normalize doc names to the lowercase keys MCPConfig uses (e.g. "CRM" -> "crm",
# "B2B Churn Intelligence" -> "b2b_churn", "Universal Scraper" -> "scraper").
NAME_NORMALIZER = {
    "crm": "crm",
    "email": "email",
    "twilio": "twilio",
    "calendar": "calendar",
    "invoicing": "invoicing",
    "intelligence": "intelligence",
    "b2b_churn": "b2b_churn",
    "scraper": "scraper",
    "universal scraper": "scraper",
    "memory": "memory",
    "memory (graphiti+postgres)": "memory",
    "b2b churn intelligence": "b2b_churn",
}


def config_ports() -> dict[str, int]:
    """Walk atlas_brain/config.py, find MCPConfig, extract <name>_port defaults."""
    tree = ast.parse(CONFIG_PY.read_text())
    ports: dict[str, int] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == "MCPConfig"):
            continue
        for sub in node.body:
            if not isinstance(sub, ast.AnnAssign) or not isinstance(sub.target, ast.Name):
                continue
            if not sub.target.id.endswith("_port"):
                continue
            key = sub.target.id[: -len("_port")]
            value_node = sub.value
            # Default is positional or keyword in Field(default=N, ...).
            if isinstance(value_node, ast.Call):
                for kw in value_node.keywords:
                    if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                        ports[key] = kw.value.value
                        break
                else:
                    # Try positional first argument.
                    if value_node.args and isinstance(value_node.args[0], ast.Constant):
                        ports[key] = value_node.args[0].value
            elif isinstance(value_node, ast.Constant):
                ports[key] = value_node.value
    return ports


def doc_claims(text: str) -> list[tuple[int, str, int, str]]:
    """Return list of (line_no, normalized_name, port, source_kind)."""
    claims: list[tuple[int, str, int, str]] = []
    # Env-var style.
    for m in ENV_VAR_LINE.finditer(text):
        env_name = m.group(1).lower()
        port = int(m.group(2))
        line_no = text[: m.start()].count("\n") + 1
        norm = NAME_NORMALIZER.get(env_name)
        if norm is not None:
            claims.append((line_no, norm, port, "env"))
    # Markdown-table style: only for rows containing a 4-5-digit port and a
    # known server name in the first cell.
    for m in TABLE_ROW.finditer(text):
        raw = m.group(1).strip().lower()
        port = int(m.group(2))
        norm = NAME_NORMALIZER.get(raw)
        if norm is None:
            continue
        line_no = text[: m.start()].count("\n") + 1
        claims.append((line_no, norm, port, "table"))
    return claims


def main() -> int:
    if not CONFIG_PY.exists():
        print(f"config.py not found at {CONFIG_PY}", file=sys.stderr)
        return 2
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return 2

    truth = config_ports()
    if not truth:
        print("Could not locate MCPConfig port assignments.", file=sys.stderr)
        return 2

    print("config.py MCPConfig ports:")
    for k in sorted(truth):
        print(f"  {k:<14} {truth[k]}")
    print()

    text = CLAUDE_MD.read_text()
    claims = doc_claims(text)
    if not claims:
        print("No port claims found in CLAUDE.md (env-var or table).")
        return 1

    print(f"CLAUDE.md port claims (env-var or table): {len(claims)}")
    print("-" * 60)

    drift = False
    for line_no, name, port, kind in claims:
        expected = truth.get(name)
        if expected is None:
            print(f"line {line_no:>4} [{kind:<5}] {name:<14} port={port}  UNKNOWN (not in config.py)")
            drift = True
        elif expected != port:
            print(f"line {line_no:>4} [{kind:<5}] {name:<14} port={port}  DRIFT (config has {expected})")
            drift = True
        else:
            print(f"line {line_no:>4} [{kind:<5}] {name:<14} port={port}  OK")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
