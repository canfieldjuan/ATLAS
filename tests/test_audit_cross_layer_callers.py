from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.audit_helpers import REPO_ROOT, load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_cross_layer_callers")


def test_cli_warns_for_non_diff_reference_to_changed_symbol(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(repo, "pkg/lib.py", "def normalize(value):\n    return value.strip().lower()\n")
    _commit(repo, "change shared helper")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN: changed symbols have references outside this PR diff" in result.stdout
    assert "normalize (pkg/lib.py:1)" in result.stdout
    assert "scripts/cli.py" in result.stdout
    assert "normalize(value)" in result.stdout


def test_cli_ignores_references_inside_changed_files(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(repo, "pkg/lib.py", "def normalize(value):\n    return value.strip().lower()\n")
    _write(
        repo,
        "scripts/cli.py",
        "from pkg.lib import normalize\n\n\ndef run(value):\n    return normalize(value).lower()\n",
    )
    _commit(repo, "change helper and caller")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed python symbols: 1" in result.stdout
    assert "OK: no non-diff references found" in result.stdout


def test_cli_passes_for_non_python_diff(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(repo, "README.md", "changed docs\n")
    _commit(repo, "change docs")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed python symbols: 0" in result.stdout
    assert "OK: no changed Python symbols found" in result.stdout


def test_cli_ignores_added_python_files(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(repo, "pkg/new_helper.py", "def normalize(value):\n    return value\n")
    _commit(repo, "add helper")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "changed python symbols: 0" in result.stdout
    assert "OK: no changed Python symbols found" in result.stdout


def test_cli_uses_word_boundary_for_symbol_references(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(
        repo,
        "scripts/cli.py",
        "\n".join((
            "from pkg.lib import normalize",
            "",
            "def run(value):",
            "    normalize_path = value",
            "    _normalize = value",
            "    normalizer = value",
            "    return normalize(value)",
            "",
        )),
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add boundary fixture")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _write(repo, "pkg/lib.py", "def normalize(value):\n    return value.strip().lower()\n")
    _commit(repo, "change shared helper")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "scripts/cli.py:7: return normalize(value)" in result.stdout
    assert "normalize_path" not in result.stdout
    assert "_normalize" not in result.stdout
    assert "normalizer" not in result.stdout


def test_cli_reports_method_attribute_calls_without_bare_name_noise(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(
        repo,
        "pkg/service.py",
        "\n".join((
            "class FAQService:",
            "    def generate(self, value):",
            "        return value.strip().lower() + '!'",
            "",
        )),
    )
    _write(
        repo,
        "scripts/method_cli.py",
        "\n".join((
            "from pkg.service import FAQService",
            "",
            "def generate(value):",
            "    return value",
            "",
            "def run(service, value):",
            "    generated = value",
            "    return service.generate(value)",
            "",
        )),
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add method fixture")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _write(
        repo,
        "pkg/service.py",
        "\n".join((
            "class FAQService:",
            "    def generate(self, value):",
            "        return value.strip().lower()",
            "",
        )),
    )
    _commit(repo, "change method")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAQService.generate (pkg/service.py:2)" in result.stdout
    assert "scripts/method_cli.py:8: return service.generate(value)" in result.stdout
    assert "def generate(value)" not in result.stdout
    assert "generated = value" not in result.stdout


def test_cli_excludes_docs_and_json_references(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)
    _write(repo, "README.md", "normalize(value) should be called here\n")
    _write(repo, "config.json", '{"example": "normalize(value)"}\n')
    _write(repo, "pkg/lib.py", "def normalize(value):\n    return value.strip().lower()\n")
    _commit(repo, "change shared helper")

    result = _run(repo)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "README.md" not in result.stdout
    assert "config.json" not in result.stdout
    assert "scripts/cli.py" in result.stdout


def test_cli_returns_two_for_missing_base_ref(tmp_path: Path) -> None:
    repo = _write_fixture_repo(tmp_path)

    result = _run(repo, "does-not-exist-ref")

    assert result.returncode == 2
    assert "cross-layer caller audit error: base ref not found" in result.stderr


def test_validate_repo_path_rejects_path_traversal(auditor) -> None:
    with pytest.raises(auditor.AuditError, match="unsafe repository path"):
        auditor.validate_repo_path("../outside.py")


def test_validate_repo_path_rejects_absolute_path(auditor) -> None:
    with pytest.raises(auditor.AuditError, match="unsafe repository path"):
        auditor.validate_repo_path("/tmp/outside.py")


def _write_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _write(repo, "pkg/lib.py", "def normalize(value):\n    return value.strip()\n")
    _write(
        repo,
        "scripts/cli.py",
        "from pkg.lib import normalize\n\n\ndef run(value):\n    return normalize(value)\n",
    )
    _write(repo, "README.md", "fixture\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    return repo


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_cross_layer_callers.py"),
            *args,
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout
