from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pre_push_audit_runs_plan_auditors_for_touched_plan(tmp_path):
    repo = tmp_path / "repo"
    _write_fixture_repo(repo)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "-M", "main")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")

    plan = repo / "plans" / "PR-Wrapper-Smoke.md"
    plan.parent.mkdir()
    (repo / "scripts" / "wrapper_smoke.py").write_text("print('ok')\n", encoding="utf-8")
    plan.write_text(_plan_text(total_loc=39), encoding="utf-8")
    _git(repo, "add", "plans/PR-Wrapper-Smoke.md", "scripts/wrapper_smoke.py")
    _git(repo, "commit", "-m", "add wrapper smoke")

    result = subprocess.run(
        ["bash", str(repo / "scripts" / "pre_push_audit.sh")],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Plan shape: plans/PR-Wrapper-Smoke.md" in result.stdout
    assert "Plan files touched: plans/PR-Wrapper-Smoke.md" in result.stdout
    assert "Plan diff size: plans/PR-Wrapper-Smoke.md" in result.stdout
    assert "all checks passed" in result.stdout


def _write_fixture_repo(repo: Path) -> None:
    (repo / "scripts").mkdir(parents=True)
    for name in (
        "audit_claude_md_claims.py",
        "audit_mcp_port_assignments.py",
        "audit_plan_doc.py",
        "audit_plan_doc_files_touched.py",
        "audit_plan_doc_diff_size.py",
        "pre_push_audit.sh",
    ):
        (repo / "scripts" / name).write_text(
            (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    _write_claude_md(repo / "CLAUDE.md")
    _write_config(repo / "atlas_brain" / "config.py")
    _write_mcp_servers(repo / "atlas_brain" / "mcp")


def _write_claude_md(path: Path) -> None:
    path.write_text(
        """# Fixture

ATLAS_MCP_CRM_PORT=8056
ATLAS_MCP_EMAIL_PORT=8057
ATLAS_MCP_TWILIO_PORT=8058
ATLAS_MCP_CALENDAR_PORT=8059
ATLAS_MCP_INVOICING_PORT=8060
ATLAS_MCP_INTELLIGENCE_PORT=8061
ATLAS_MCP_B2B_CHURN_PORT=8062

### CRM MCP Server (10 tools)
# SSE HTTP mode (port 8056)
python -m atlas_brain.mcp.crm_server --sse
### Email MCP Server (9 tools)
# SSE HTTP mode (port 8057)
python -m atlas_brain.mcp.email_server --sse
### Twilio MCP Server (10 tools)
# SSE HTTP mode (port 8058)
python -m atlas_brain.mcp.twilio_server --sse
### Calendar MCP Server (8 tools)
# SSE HTTP mode (port 8059)
python -m atlas_brain.mcp.calendar_server --sse
### Invoicing MCP Server (18 tools)
# SSE HTTP mode (port 8060)
python -m atlas_brain.mcp.invoicing_server --sse
### Intelligence MCP Server (33 tools)
# SSE HTTP mode (port 8061)
python -m atlas_brain.mcp.intelligence_server --sse
### B2B Churn Intelligence MCP Server (83 tools)
# SSE HTTP mode (port 8062)
python -m atlas_brain.mcp.b2b_churn_server --sse
### Universal Scraper MCP Server (5 tools)
# SSE HTTP mode (port 8063)
python -m atlas_brain.mcp.scraper_server --sse
### Memory MCP Server (15 tools)
# SSE HTTP mode (port 8064)
python -m atlas_brain.mcp.memory_server --sse
""",
        encoding="utf-8",
    )


def _write_config(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        """from pydantic import Field

class MCPConfig:
    crm_port: int = Field(default=8056)
    email_port: int = Field(default=8057)
    twilio_port: int = Field(default=8058)
    calendar_port: int = Field(default=8059)
    invoicing_port: int = Field(default=8060)
    intelligence_port: int = Field(default=8061)
    b2b_churn_port: int = Field(default=8062)
    scraper_port: int = Field(default=8063)
    memory_port: int = Field(default=8064)
""",
        encoding="utf-8",
    )


def _write_mcp_servers(mcp_dir: Path) -> None:
    mcp_dir.mkdir(parents=True)
    counts = {
        "crm_server.py": 10,
        "email_server.py": 9,
        "twilio_server.py": 10,
        "calendar_server.py": 8,
        "invoicing_server.py": 18,
        "intelligence_server.py": 33,
        "scraper_server.py": 5,
        "memory_server.py": 15,
    }
    for filename, count in counts.items():
        (mcp_dir / filename).write_text(_decorators(count), encoding="utf-8")
    (mcp_dir / "b2b").mkdir()
    (mcp_dir / "b2b" / "fixture.py").write_text(_decorators(83), encoding="utf-8")


def _decorators(count: int) -> str:
    return "\n".join("@mcp.tool\ndef tool_%s():\n    pass\n" % idx for idx in range(count))


def _plan_text(*, total_loc: int) -> str:
    return """# PR-Wrapper-Smoke

## Why this slice exists

Test fixture.

## Scope (this PR)

### Files touched

- `plans/PR-Wrapper-Smoke.md`
- `scripts/wrapper_smoke.py`

## Mechanism

Test fixture.

## Intentional

Test fixture.

## Deferred

None.

## Verification

```bash
bash scripts/pre_push_audit.sh
```

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Wrapper-Smoke.md` | 38 |
| `scripts/wrapper_smoke.py` | 1 |
| **Total** | **~%d** |
""" % total_loc


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
