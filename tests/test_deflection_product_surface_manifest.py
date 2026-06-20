from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_deflection_product_surface_manifest.py"

SPEC = importlib.util.spec_from_file_location(
    "check_deflection_product_surface_manifest",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _write_manifest(root: Path, patterns: list[str], files: list[str]) -> Path:
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"patterns": patterns, "files": files}),
        encoding="utf-8",
    )
    return manifest


def test_manifest_checker_passes_when_discovered_matches_expected(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    product_file = tmp_path / "portfolio-ui" / "src" / "App.tsx"
    product_file.parent.mkdir(parents=True)
    product_file.write_text("export default function App() { return null; }\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        ["portfolio-ui/src/App.tsx"],
        ["portfolio-ui/src/App.tsx"],
    )

    monkeypatch.chdir(tmp_path)

    assert checker.main(["--manifest", str(manifest)]) == 0
    assert "deflection product surface manifest ok: 1 file(s)" in capsys.readouterr().out


def test_manifest_checker_fails_for_missing_expected_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest = _write_manifest(
        tmp_path,
        ["portfolio-ui/src/App.tsx"],
        ["portfolio-ui/src/App.tsx"],
    )

    monkeypatch.chdir(tmp_path)

    assert checker.main(["--manifest", str(manifest)]) == 1
    output = capsys.readouterr().out
    assert "manifest lists missing file(s):" in output
    assert "- portfolio-ui/src/App.tsx" in output


def test_manifest_checker_fails_for_discovered_untracked_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    product_file = tmp_path / "portfolio-ui" / "api" / "content-ops" / "deflection" / "nested" / "route.ts"
    product_file.parent.mkdir(parents=True)
    product_file.write_text("export default async function handler() {}\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        ["portfolio-ui/api/content-ops/deflection/**/*"],
        [],
    )

    monkeypatch.chdir(tmp_path)

    assert checker.main(["--manifest", str(manifest)]) == 1
    output = capsys.readouterr().out
    assert "manifest is missing discovered file(s):" in output
    assert "- portfolio-ui/api/content-ops/deflection/nested/route.ts" in output
