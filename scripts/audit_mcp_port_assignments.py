#!/usr/bin/env python3
"""Verify MCP port claims in CLAUDE.md match MCPConfig defaults."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ENV_PORT_PATTERN = re.compile(r"\bATLAS_MCP_([A-Z0-9_]+)_PORT\s*=\s*(\d+)\b")
SSE_PORT_PATTERN = re.compile(r"SSE HTTP mode \(port\s+(\d+)\)")
MODULE_PATTERN = re.compile(r"python -m atlas_brain\.mcp\.([a-z0-9_]+)_server\b[^\n#]*\s--sse\b")


@dataclass(frozen=True)
class PortClaim:
    name: str
    port: int
    source: str
    line_no: int


@dataclass(frozen=True)
class PortAuditRow:
    name: str
    status: str
    expected: int | None
    documented: tuple[PortClaim, ...]


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").strip("_")


def _field_default_int(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field":
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, int):
                    return keyword.value.value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def mcp_config_ports(config_text: str) -> dict[str, int]:
    tree = ast.parse(config_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MCPConfig":
            ports: dict[str, int] = {}
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign):
                    continue
                if not isinstance(statement.target, ast.Name):
                    continue
                field_name = statement.target.id
                if not field_name.endswith("_port"):
                    continue
                default = _field_default_int(statement.value)
                if default is not None:
                    ports[field_name.removesuffix("_port")] = default
            return ports
    raise ValueError("MCPConfig class not found")


def documented_port_claims(doc_text: str) -> dict[str, list[PortClaim]]:
    claims: dict[str, list[PortClaim]] = {}
    pending_sse_port: tuple[int, int] | None = None

    for line_no, line in enumerate(doc_text.splitlines(), start=1):
        for match in ENV_PORT_PATTERN.finditer(line):
            name = _normalize_name(match.group(1))
            claim = PortClaim(name=name, port=int(match.group(2)), source="env", line_no=line_no)
            claims.setdefault(name, []).append(claim)

        sse_match = SSE_PORT_PATTERN.search(line)
        if sse_match:
            pending_sse_port = (int(sse_match.group(1)), line_no)
            continue

        module_match = MODULE_PATTERN.search(line)
        if module_match and pending_sse_port:
            port, port_line_no = pending_sse_port
            name = _normalize_name(module_match.group(1))
            claim = PortClaim(name=name, port=port, source="sse-example", line_no=port_line_no)
            claims.setdefault(name, []).append(claim)
            pending_sse_port = None
    return claims


def audit_ports(config_ports: dict[str, int], claims: dict[str, list[PortClaim]]) -> list[PortAuditRow]:
    rows: list[PortAuditRow] = []
    all_names = sorted(set(config_ports) | set(claims))
    for name in all_names:
        expected = config_ports.get(name)
        documented = tuple(claims.get(name, ()))
        documented_ports = {claim.port for claim in documented}

        if expected is None:
            status = "EXTRA"
        elif not documented:
            status = "MISSING"
        elif documented_ports == {expected}:
            status = "OK"
        elif len(documented_ports) > 1:
            status = "CONFLICT"
        else:
            status = "DRIFT"
        rows.append(PortAuditRow(name=name, status=status, expected=expected, documented=documented))
    return rows


def _format_claims(claims: tuple[PortClaim, ...]) -> str:
    if not claims:
        return "-"
    return ", ".join(f"{claim.port} ({claim.source}:line {claim.line_no})" for claim in claims)


def main() -> int:
    if len(sys.argv) not in (1, 3):
        print("usage: audit_mcp_port_assignments.py [CLAUDE_MD CONFIG_PY]", file=sys.stderr)
        return 2

    doc_path = Path(sys.argv[1]) if len(sys.argv) == 3 else Path("CLAUDE.md")
    config_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("atlas_brain/config.py")

    if not doc_path.exists():
        print(f"doc file not found: {doc_path}", file=sys.stderr)
        return 2
    if not config_path.exists():
        print(f"config file not found: {config_path}", file=sys.stderr)
        return 2

    try:
        config_ports = mcp_config_ports(config_path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as exc:
        print(f"failed to parse MCPConfig: {exc}", file=sys.stderr)
        return 2

    rows = audit_ports(
        config_ports=config_ports,
        claims=documented_port_claims(doc_path.read_text(encoding="utf-8")),
    )

    drift = False
    print(f"doc: {doc_path}")
    print(f"config: {config_path}")
    print("-" * 72)
    for row in rows:
        if row.status != "OK":
            drift = True
        expected = "-" if row.expected is None else str(row.expected)
        print(f"{row.status:<10} {row.name:<16} config={expected:<5} docs={_format_claims(row.documented)}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
