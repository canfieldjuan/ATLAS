#!/usr/bin/env python3
"""Verify MCP tool-name inventories in CLAUDE.md match @mcp.tool functions."""
from __future__ import annotations

import ast
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
    "Intelligence": "intelligence_server.py",
    "Universal Scraper": "scraper_server.py",
    "Memory": "memory_server.py",
    "B2B Churn Intelligence": "_b2b_sum",
}

HEADER_PATTERN = re.compile(
    r"^###\s+(?P<name>.+?)\s+MCP Server", re.MULTILINE
)
BACKTICK_IDENT = re.compile(r"`([a-z][a-z0-9_]{3,})`")


def section_slice(text: str, start: int) -> str:
    """Return the substring from `start` up to the next server heading."""
    rest = text[start:]
    match = re.search(r"\n(?:## |### )", rest)
    return rest if match is None else rest[: match.start()]


def doc_claims(text: str) -> tuple[dict[str, set[str]], list[str]]:
    """Return (known_claims, unknown_headers)."""
    claims: dict[str, set[str]] = {}
    unknown: list[str] = []
    for match in HEADER_PATTERN.finditer(text):
        name = match.group("name").strip()
        if name not in HEADER_TO_FILE:
            unknown.append(name)
            continue
        section = section_slice(text, match.end())
        claims[name] = set(BACKTICK_IDENT.findall(section))
    return claims, unknown


def tool_names_in_file(path: Path) -> set[str]:
    if not path.exists():
        return set()

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "tool"
                and isinstance(target.value, ast.Name)
                and target.value.id == "mcp"
            ):
                names.add(node.name)
                break
    return names


def actual_for(file_key: str) -> set[str]:
    if file_key == "_b2b_sum":
        out: set[str] = set()
        b2b = MCP_DIR / "b2b"
        if b2b.is_dir():
            for path in sorted(b2b.glob("*.py")):
                out |= tool_names_in_file(path)
        return out
    return tool_names_in_file(MCP_DIR / file_key)


def main() -> int:
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}", file=sys.stderr)
        return 2

    claims, unknown_headers = doc_claims(CLAUDE_MD.read_text(encoding="utf-8"))
    if not claims and not unknown_headers:
        print("No '### ... MCP Server' headers found in CLAUDE.md.")
        return 1

    drift = False
    if unknown_headers:
        drift = True
        print("DRIFT: unknown MCP Server header(s) in CLAUDE.md:")
        for name in unknown_headers:
            print(f"  - {name!r} not in HEADER_TO_FILE (rename or new server?)")

    for name in sorted(claims):
        file_key = HEADER_TO_FILE[name]
        actual = actual_for(file_key)
        claimed = claims[name]
        missing_in_doc = actual - claimed
        extra_in_doc = claimed - actual
        status = "OK" if not (missing_in_doc or extra_in_doc) else "DRIFT"
        print(
            f"\n{name} ({file_key})  "
            f"claimed={len(claimed)}  actual={len(actual)}  {status}"
        )
        if missing_in_doc:
            drift = True
            print("  missing in doc (in code, not in CLAUDE.md):")
            for missing in sorted(missing_in_doc):
                print(f"    - {missing}")
        if extra_in_doc:
            drift = True
            print("  extra in doc (in CLAUDE.md, not in code):")
            for extra in sorted(extra_in_doc):
                print(f"    - {extra}")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
