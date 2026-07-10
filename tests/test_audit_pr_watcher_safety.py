from __future__ import annotations

import pytest
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_pr_watcher_safety.py"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def _repo(tmp_path: Path, doc: str = "Watcher reports readiness only.\n") -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "AGENTS.md").write_text(doc, encoding="utf-8")
    return repo


def test_safe_repo_and_local_watcher_pass(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    watcher = tmp_path / "atlas-pr-watch"
    watcher.write_text("print('ready_for_human_merge')\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "session.env").write_text('AUTO_MERGE="0"\n', encoding="utf-8")

    result = _run(tmp_path, "--repo-root", str(repo), "--watcher-bin", str(watcher), "--config-dir", str(config_dir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: watcher docs/config/source grant no merge authority" in result.stdout


def test_fails_on_truthy_auto_merge_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "session.env").write_text('AUTO_MERGE="1"\n', encoding="utf-8")

    result = _run(tmp_path, "--repo-root", str(repo), "--config-dir", str(config_dir))

    assert result.returncode == 1
    assert "watcher config must use AUTO_MERGE=0" in result.stdout


def test_fails_on_exported_truthy_auto_merge_config(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "session.env").write_text("export AUTO_MERGE=1\n", encoding="utf-8")

    result = _run(tmp_path, "--repo-root", str(repo), "--config-dir", str(config_dir))

    assert result.returncode == 1
    assert "watcher config must use AUTO_MERGE=0" in result.stdout


def test_fails_on_truthy_auto_merge_config_with_shell_comment(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "session.env").write_text(
        "AUTO_MERGE=1 # temporary\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--config-dir", str(config_dir))

    assert result.returncode == 1
    assert "watcher config must use AUTO_MERGE=0" in result.stdout


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "enabled"])
def test_fails_on_truthy_auto_merge_doc_examples(tmp_path: Path, truthy: str) -> None:
    repo = _repo(tmp_path, f'AUTO_MERGE="{truthy}"\n')

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 1
    assert "repo docs/templates must not grant merge authority" in result.stdout


def test_fails_on_merge_command_in_watcher_source(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    watcher = tmp_path / "atlas-pr-watch"
    watcher.write_text('subprocess.run(["gh", "pr", "merge", "123", "--delete-branch"])\n', encoding="utf-8")

    result = _run(tmp_path, "--repo-root", str(repo), "--watcher-bin", str(watcher))

    assert result.returncode == 1
    assert "must not contain PR merge/delete-branch commands" in result.stdout


def test_fails_on_merge_command_in_watcher_wrapper(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wrapper = tmp_path / "atlas-pr-watch-and-wake"
    wrapper.write_text('gh pr merge "$PR" --delete-branch\n', encoding="utf-8")

    result = _run(tmp_path, "--repo-root", str(repo), "--watcher-wrapper", str(wrapper))

    assert result.returncode == 1
    assert "atlas-pr-watch-and-wake" in result.stdout
    assert "must not contain PR merge/delete-branch commands" in result.stdout


def test_fails_on_merge_command_in_systemd_unit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    (systemd_dir / "atlas-pr-watch@.service").write_text(
        "ExecStart=/usr/bin/gh pr merge 123 --squash\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--systemd-dir", str(systemd_dir))

    assert result.returncode == 1
    assert "atlas-pr-watch@.service" in result.stdout
    assert "must not contain PR merge/delete-branch commands" in result.stdout


def test_fails_on_merge_command_in_systemd_drop_in(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dropin_dir = tmp_path / "systemd" / "atlas-pr-watch@.service.d"
    dropin_dir.mkdir(parents=True)
    (dropin_dir / "wake-bridge.conf").write_text(
        "ExecStart=%h/.local/bin/merge-wrapper --delete-branch\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--systemd-dir", str(tmp_path / "systemd"))

    assert result.returncode == 1
    assert "wake-bridge.conf" in result.stdout
    assert "must not contain PR merge/delete-branch commands" in result.stdout


def test_fails_on_docs_that_grant_watcher_merge_authority(tmp_path: Path) -> None:
    repo = _repo(
        tmp_path,
        textwrap.dedent(
            """\
            | Signal | Merge allowed? |
            |---|---|
            | Scheduled watcher | Yes |

            The watcher may merge after standing authorization.
            """
        ),
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 1
    assert "repo docs/templates must not grant merge authority" in result.stdout


@pytest.mark.parametrize("surface", ["timer", "notification", "bridge", "wake bridge"])
def test_fails_on_docs_that_grant_wake_surface_merge_authority(
    tmp_path: Path,
    surface: str,
) -> None:
    repo = _repo(tmp_path, f"The {surface} may merge once checks are green.\n")

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 1
    assert "repo docs/templates must not grant merge authority" in result.stdout


def test_repo_only_passes_when_local_watcher_is_absent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 0, result.stdout + result.stderr


def test_fails_on_repo_watcher_source_with_merge_command(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "watch_owned_pr.sh").write_text(
        "#!/usr/bin/env bash\ngh pr merge \"$PR\" --squash\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 1
    assert "watcher executable must not contain PR merge/delete-branch commands" in result.stdout


def test_repo_watcher_source_status_only_passes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "watch_owned_pr.sh").write_text(
        "#!/usr/bin/env bash\necho MERGE-READY\n",
        encoding="utf-8",
    )

    result = _run(tmp_path, "--repo-root", str(repo), "--repo-only")

    assert result.returncode == 0, result.stdout + result.stderr
