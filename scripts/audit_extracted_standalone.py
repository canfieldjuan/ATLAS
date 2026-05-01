#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "extracted_content_pipeline"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    module: str
    statement: str
    kind: str


def _python_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _shim_kind(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return "empty"
    if all(
        line.startswith("from atlas_brain.")
        or line.startswith("from atlas_brain import")
        or line.startswith("import atlas_brain")
        for line in lines
    ):
        return "bridge_shim"
    return "hard_import"


def _statement(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment:
        return " ".join(segment.strip().split())
    return type(node).__name__


def audit() -> list[Finding]:
    findings: list[Finding] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                Finding(
                    path=str(path.relative_to(ROOT)),
                    line=exc.lineno or 0,
                    module="<syntax-error>",
                    statement=str(exc),
                    kind="syntax_error",
                )
            )
            continue
        kind = _shim_kind(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "atlas_brain" or alias.name.startswith("atlas_brain."):
                        findings.append(
                            Finding(
                                path=str(path.relative_to(ROOT)),
                                line=node.lineno,
                                module=alias.name,
                                statement=_statement(source, node),
                                kind=kind,
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "atlas_brain" or module.startswith("atlas_brain."):
                    findings.append(
                        Finding(
                            path=str(path.relative_to(ROOT)),
                            line=node.lineno,
                            module=module,
                            statement=_statement(source, node),
                            kind=kind,
                        )
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit extracted_content_pipeline for Atlas runtime coupling."
    )
    parser.add_argument(
        "--fail-on-debt",
        action="store_true",
        help="Return non-zero when any atlas_brain runtime import remains.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args()

    findings = audit()
    counts: dict[str, int] = {}
    for item in findings:
        counts[item.kind] = counts.get(item.kind, 0) + 1

    if args.json:
        print(
            json.dumps(
                {
                    "total": len(findings),
                    "counts": counts,
                    "findings": [asdict(item) for item in findings],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Atlas runtime import findings: {len(findings)}")
        for kind, count in sorted(counts.items()):
            print(f"  {kind}: {count}")
        if findings:
            print()
            for item in findings:
                print(f"{item.path}:{item.line}: [{item.kind}] {item.statement}")

    if args.fail_on_debt and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
