#!/usr/bin/env python3
"""Generate a technical-debt register and audit summary."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


REGISTER_HEADER = [
    "id",
    "rank",
    "date_identified",
    "category",
    "priority_score",
    "status",
    "owner",
    "component",
    "file_ref",
    "symptom",
    "impact_summary",
    "risk_level",
    "frequency",
    "effort_days",
    "target_window",
    "acceptance_criteria",
    "evidence",
    "last_verified",
]


@dataclass
class PytestCollectInfo:
    """Test collection summary."""

    collected: int
    errors: int
    raw_output: str
    return_code: int
    timed_out: bool = False


@dataclass
class DebtItem:
    """Debt finding."""

    debt_id: str
    category: str
    component: str
    symptom: str
    impact_summary: str
    impact: int
    frequency: int
    risk: int
    effort: int
    effort_days: int
    target_window: str
    acceptance_criteria: str
    evidence_note: str
    refs: list[str] = field(default_factory=list)

    def score(self) -> float:
        """Compute priority score."""
        return round((self.impact * self.frequency * self.risk) / self.effort, 1)

    def risk_level(self) -> str:
        """Map numeric risk to label."""
        if self.risk >= 5:
            return "critical"
        if self.risk >= 4:
            return "high"
        if self.risk >= 3:
            return "medium"
        return "low"


def run_cmd(args: list[str], cwd: Path, timeout: int = 300) -> tuple[int, str, bool]:
    """Run a shell command and return code/output/timed_out."""
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return 124, (stdout + "\n" + stderr).strip(), True
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode, output, False


def repo_root() -> Path:
    """Resolve repository root from script path."""
    return Path(__file__).resolve().parent.parent


def rel_ref(root: Path, path: Path, line: int | None = None) -> str:
    """Build a relative file reference with optional line number."""
    rel = path.relative_to(root).as_posix()
    if line is None:
        return rel
    return f"{rel}:{line}"


def read_text(path: Path) -> str:
    """Read file text safely."""
    return path.read_text(encoding="utf-8", errors="replace")


def find_line(path: Path, needle: str) -> int | None:
    """Find first line containing literal text."""
    if not path.exists():
        return None
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        if needle in line:
            return line_no
    return None


def find_line_after(path: Path, needle: str, start_line: int) -> int | None:
    """Find first line after start line containing literal text."""
    if not path.exists():
        return None
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        if line_no <= start_line:
            continue
        if needle in line:
            return line_no
    return None


def find_regex_lines(path: Path, pattern: str) -> list[int]:
    """Find line numbers that match a regex."""
    if not path.exists():
        return []
    regex = re.compile(pattern)
    matches: list[int] = []
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            matches.append(line_no)
    return matches


def tracked_files(root: Path) -> list[str]:
    """Get tracked files from git."""
    code, output, _ = run_cmd(["git", "ls-files"], cwd=root, timeout=60)
    if code != 0:
        raise RuntimeError("git ls-files failed")
    return [line.strip() for line in output.splitlines() if line.strip()]


def untracked_files(root: Path) -> list[str]:
    """Get untracked files that are not ignored."""
    code, output, _ = run_cmd(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        timeout=60,
    )
    if code != 0:
        raise RuntimeError("git ls-files --others failed")
    return [line.strip() for line in output.splitlines() if line.strip()]


def repository_files(root: Path) -> list[str]:
    """Get tracked plus untracked repository files."""
    combined = set(tracked_files(root))
    combined.update(untracked_files(root))
    return sorted(combined)


def parse_env_keys(env_path: Path) -> list[str]:
    """Parse env var keys from a .env-like file."""
    if not env_path.exists():
        return []
    keys: list[str] = []
    for raw_line in read_text(env_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            keys.append(key)
    return keys


def collect_baseline(root: Path) -> dict[str, object]:
    """Collect baseline repo metrics."""
    files = repository_files(root)
    py_count = sum(1 for f in files if f.endswith(".py"))
    ts_count = sum(1 for f in files if f.endswith(".ts") or f.endswith(".tsx"))
    test_count = sum(1 for f in files if f.startswith("tests/") or "/__tests__/" in f)
    env_local = root / ".env.local"
    return {
        "python_files": py_count,
        "ts_files": ts_count,
        "test_files": test_count,
        "env_local_path": env_local,
        "env_local_keys": parse_env_keys(env_local),
    }


def parse_collected_tests(output: str) -> int:
    """Parse collected test count from pytest output."""
    match = re.search(r"(\d+)\s+tests?\s+collected", output)
    if match:
        return int(match.group(1))
    return 0


def parse_error_count(output: str) -> int:
    """Parse error count from pytest output."""
    match = re.search(r"(\d+)\s+errors?\s+in", output)
    if match:
        return int(match.group(1))
    collecting = re.findall(r"^ERROR collecting\s+(\S+)", output, flags=re.MULTILINE)
    return len(collecting)


def run_pytest_collect(root: Path, enabled: bool) -> PytestCollectInfo:
    """Run pytest collect-only and parse summary."""
    if not enabled:
        return PytestCollectInfo(0, 0, "pytest collect skipped", 0, False)
    code, output, timed_out = run_cmd(
        ["pytest", "--collect-only", "-q"],
        cwd=root,
        timeout=300,
    )
    return PytestCollectInfo(
        collected=parse_collected_tests(output),
        errors=parse_error_count(output),
        raw_output=output,
        return_code=code,
        timed_out=timed_out,
    )


def extract_pytest_error_refs(root: Path, output: str) -> list[str]:
    """Extract file references from pytest error output."""
    refs: list[str] = []
    seen: set[str] = set()
    for rel_path in re.findall(r"^ERROR collecting\s+(\S+)", output, flags=re.MULTILINE):
        path = root / rel_path
        if path.exists():
            candidate = rel_ref(root, path, 1)
            if candidate not in seen:
                refs.append(candidate)
                seen.add(candidate)
    pattern = r"^([A-Za-z0-9_./-]+\.py):(\d+): in"
    for rel_path, line in re.findall(pattern, output, flags=re.MULTILINE):
        path = root / rel_path
        if path.exists() and path.resolve().is_relative_to(root.resolve()):
            candidate = rel_ref(root, path, int(line))
            if candidate not in seen:
                refs.append(candidate)
                seen.add(candidate)
    return refs[:8]


def duplicate_migration_versions(root: Path) -> dict[str, list[Path]]:
    """Find duplicated migration numeric prefixes."""
    migrations_dir = root / "atlas_brain" / "storage" / "migrations"
    versions: dict[str, list[Path]] = {}
    for sql_path in sorted(migrations_dir.glob("*.sql")):
        match = re.match(r"(\d+)_", sql_path.name)
        if not match:
            continue
        versions.setdefault(match.group(1), []).append(sql_path)
    return {version: paths for version, paths in versions.items() if len(paths) > 1}


def detect_td001(root: Path) -> DebtItem | None:
    """Detect duplicate migration versions."""
    duplicates = duplicate_migration_versions(root)
    if not duplicates:
        return None
    refs: list[str] = []
    for version in sorted(duplicates):
        for sql_path in sorted(duplicates[version]):
            refs.append(rel_ref(root, sql_path, 1))
    init_path = root / "atlas_brain" / "storage" / "migrations" / "__init__.py"
    schema_line = find_line(init_path, "version INTEGER PRIMARY KEY")
    insert_line = find_line(init_path, "ON CONFLICT (version) DO NOTHING")
    if schema_line:
        refs.append(rel_ref(root, init_path, schema_line))
    if insert_line:
        refs.append(rel_ref(root, init_path, insert_line))
    evidence = "Duplicate migration versions: " + ", ".join(sorted(duplicates.keys()))
    return DebtItem(
        debt_id="TD-001",
        category="Data/Migrations",
        component="atlas_brain_storage",
        symptom="Duplicate migration version prefixes share schema_migrations keys.",
        impact_summary="Schema drift or skipped migrations during deploy.",
        impact=5,
        frequency=5,
        risk=5,
        effort=2,
        effort_days=2,
        target_window="2026-Q1",
        acceptance_criteria="Every migration filename has a unique numeric prefix.",
        evidence_note=evidence,
        refs=refs,
    )


def detect_td002(root: Path) -> DebtItem | None:
    """Detect hardcoded compose credentials."""
    refs: list[str] = []
    compose_files = [root / "docker-compose.yml", root / "atlas_video-processing" / "docker-compose.yml"]
    pattern = r"POSTGRES_(PASSWORD|USER)\s*[:=]\s*[^$\s]"
    for compose_path in compose_files:
        for line in find_regex_lines(compose_path, pattern):
            refs.append(rel_ref(root, compose_path, line))
    if not refs:
        return None
    return DebtItem(
        debt_id="TD-002",
        category="Security/Ops",
        component="compose",
        symptom="Compose files contain inline database credentials.",
        impact_summary="Credential leakage and environment drift risk.",
        impact=5,
        frequency=4,
        risk=4,
        effort=2,
        effort_days=2,
        target_window="2026-Q1",
        acceptance_criteria="Compose uses env substitution for credentials.",
        evidence_note="Hardcoded POSTGRES_USER/POSTGRES_PASSWORD entries found.",
        refs=refs,
    )


def detect_td003(root: Path) -> DebtItem | None:
    """Detect unconditional mock camera registration."""
    path = root / "atlas_video-processing" / "atlas_vision" / "api" / "main.py"
    import_line = find_line(path, "from ..devices.cameras.mock import create_mock_cameras")
    loop_line = find_line(path, "for camera in create_mock_cameras():")
    guard_line = find_line(path, "settings.camera.register_mock_cameras")
    if not import_line or not loop_line:
        return None
    if guard_line and guard_line < loop_line:
        return None
    refs = [rel_ref(root, path, import_line), rel_ref(root, path, loop_line)]
    return DebtItem(
        debt_id="TD-003",
        category="RuntimeCorrectness",
        component="atlas_vision_startup",
        symptom="Atlas Vision startup always registers mock cameras.",
        impact_summary="Synthetic devices can appear in production runtime.",
        impact=5,
        frequency=4,
        risk=4,
        effort=2,
        effort_days=2,
        target_window="2026-Q1",
        acceptance_criteria="Mock cameras are disabled unless an explicit dev flag is enabled.",
        evidence_note="Mock camera registration loop exists without a dev-flag guard.",
        refs=refs,
    )


def detect_td004(root: Path) -> DebtItem | None:
    """Detect migration coupling to discovery initialization."""
    discovery = root / "atlas_brain" / "discovery" / "service.py"
    main = root / "atlas_brain" / "main.py"
    run_line = find_line(discovery, "await run_migrations(pool)")
    init_line = find_line(main, "await init_discovery()")
    if not run_line or not init_line:
        return None
    refs = [rel_ref(root, discovery, run_line), rel_ref(root, main, init_line)]
    return DebtItem(
        debt_id="TD-004",
        category="Architecture",
        component="atlas_brain_startup",
        symptom="Database migrations run from discovery initialization path.",
        impact_summary="Schema bootstrap depends on unrelated feature toggles.",
        impact=5,
        frequency=3,
        risk=5,
        effort=2,
        effort_days=2,
        target_window="2026-Q1",
        acceptance_criteria="Migrations run from a dedicated startup stage.",
        evidence_note="run_migrations is called inside discovery service init.",
        refs=refs,
    )


def detect_td005(root: Path) -> DebtItem | None:
    """Detect hardcoded tenant coupling in comms context registration."""
    core = root / "atlas_comms" / "core" / "config.py"
    service = root / "atlas_comms" / "service.py"
    tools = root / "atlas_brain" / "tools" / "scheduling.py"
    core_line = find_line(core, "EFFINGHAM_MAIDS_CONTEXT = BusinessContext(")
    service_line = find_line(service, "router.register_context(EFFINGHAM_MAIDS_CONTEXT)")
    tools_line = find_line(tools, "router.register_context(EFFINGHAM_MAIDS_CONTEXT)")
    if not core_line or (not service_line and not tools_line):
        return None
    refs = [rel_ref(root, core, core_line)]
    if service_line:
        refs.append(rel_ref(root, service, service_line))
    if tools_line:
        refs.append(rel_ref(root, tools, tools_line))
    return DebtItem(
        debt_id="TD-005",
        category="ProductCoupling",
        component="atlas_comms_contexts",
        symptom="Effingham context is hardcoded and auto-injected.",
        impact_summary="Multi-tenant operation is constrained by hidden defaults.",
        impact=5,
        frequency=4,
        risk=4,
        effort=3,
        effort_days=3,
        target_window="2026-Q2",
        acceptance_criteria="Default contexts are data-driven and tenant-agnostic.",
        evidence_note="EFFINGHAM_MAIDS_CONTEXT appears in config and startup paths.",
        refs=refs,
    )


def detect_td006(root: Path) -> DebtItem | None:
    """Detect silent fallbacks to stub integrations."""
    real = root / "atlas_brain" / "comms" / "real_services.py"
    base = root / "atlas_comms" / "services" / "base.py"
    lines = [
        find_line(real, "return StubEmailService()"),
        find_line(real, "return StubSMSService()"),
        find_line(real, "return StubCalendarService()"),
    ]
    lines = [line for line in lines if line]
    base_line = find_line(base, "class StubCalendarService")
    if not lines or not base_line:
        return None
    refs = [rel_ref(root, real, line) for line in lines]
    refs.append(rel_ref(root, base, base_line))
    return DebtItem(
        debt_id="TD-006",
        category="Reliability",
        component="comms_services",
        symptom="Factory methods silently return stub service implementations.",
        impact_summary="Integration outages can be masked as successful calls.",
        impact=5,
        frequency=4,
        risk=4,
        effort=3,
        effort_days=3,
        target_window="2026-Q2",
        acceptance_criteria="Disabled integrations return explicit operator-visible errors.",
        evidence_note="Stub* services are returned by runtime factories.",
        refs=refs,
    )


def detect_td007(root: Path, info: PytestCollectInfo) -> DebtItem | None:
    """Detect test collection blockers."""
    if info.errors <= 0:
        return None
    refs = extract_pytest_error_refs(root, info.raw_output)
    note = f"pytest --collect-only: collected={info.collected}, errors={info.errors}, rc={info.return_code}"
    return DebtItem(
        debt_id="TD-007",
        category="Testing",
        component="test_harness",
        symptom="Test collection fails in baseline environments.",
        impact_summary="Feedback loop reliability is reduced for CI and local checks.",
        impact=4,
        frequency=5,
        risk=4,
        effort=3,
        effort_days=3,
        target_window="2026-Q2",
        acceptance_criteria="Unit test collection succeeds without privileged sockets or heavy optional deps.",
        evidence_note=note,
        refs=refs,
    )


def detect_td008(root: Path) -> DebtItem | None:
    """Detect unimplemented SignalWire audio streaming path."""
    path = root / "atlas_comms" / "providers" / "signalwire.py"
    todo_line = find_line(path, "TODO: Implement via SignalWire's Stream noun")
    if not todo_line:
        return None
    pass_line = find_line_after(path, "pass", todo_line)
    refs = [rel_ref(root, path, todo_line)]
    if pass_line:
        refs.append(rel_ref(root, path, pass_line))
    return DebtItem(
        debt_id="TD-008",
        category="FeatureCompletion",
        component="signalwire_provider",
        symptom="SignalWire stream_audio_to_call path is not implemented.",
        impact_summary="Audio streaming appears available but no media is sent.",
        impact=4,
        frequency=4,
        risk=4,
        effort=3,
        effort_days=3,
        target_window="2026-Q2",
        acceptance_criteria="Audio stream path is implemented or hard-fails as unsupported.",
        evidence_note="stream_audio_to_call still contains TODO marker.",
        refs=refs,
    )


def detect_td009(root: Path) -> DebtItem | None:
    """Detect env loading inconsistency across services."""
    brain = root / "atlas_brain" / "main.py"
    comms = root / "atlas_comms" / "core" / "config.py"
    vision = root / "atlas_video-processing" / "atlas_vision" / "core" / "config.py"
    brain_line = find_line(brain, 'load_dotenv(_env_root / ".env.local", override=True)')
    comms_line = find_line(comms, 'env_file=".env"')
    vision_line = find_line(vision, 'model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_")')
    if not brain_line or not comms_line or not vision_line:
        return None
    refs = [
        rel_ref(root, brain, brain_line),
        rel_ref(root, comms, comms_line),
        rel_ref(root, vision, vision_line),
    ]
    return DebtItem(
        debt_id="TD-009",
        category="Configuration",
        component="env_loading",
        symptom="Services do not share one env precedence contract.",
        impact_summary="Settings may diverge across brain/comms/vision processes.",
        impact=4,
        frequency=4,
        risk=3,
        effort=3,
        effort_days=3,
        target_window="2026-Q2",
        acceptance_criteria="All services follow one documented .env/.env.local precedence model.",
        evidence_note="Brain explicitly loads .env.local while other services rely on different patterns.",
        refs=refs,
    )


def detect_td010(root: Path) -> DebtItem | None:
    """Detect simulation-only stream processor logic."""
    path = root / "atlas_video-processing" / "processing" / "video_stream_processor" / "video_stream_processor.py"
    broker_line = find_line(path, "KAFKA_BROKER =")
    placeholder_line = find_line(path, "Placeholder function to simulate OpenCV processing")
    result_line = find_line(path, "Processed_Metadata_Example")
    if not broker_line or not placeholder_line or not result_line:
        return None
    refs = [
        rel_ref(root, path, broker_line),
        rel_ref(root, path, placeholder_line),
        rel_ref(root, path, result_line),
    ]
    return DebtItem(
        debt_id="TD-010",
        category="RuntimeCorrectness",
        component="video_processor",
        symptom="Video processor still uses simulation placeholder logic.",
        impact_summary="Pipeline health can be misread as production ready.",
        impact=3,
        frequency=3,
        risk=3,
        effort=2,
        effort_days=2,
        target_window="2026-Q2",
        acceptance_criteria="Processor executes real decode/detection path or is isolated as demo-only.",
        evidence_note="Current path returns fixed simulated metadata.",
        refs=refs,
    )


def collect_debt_items(root: Path, pytest_info: PytestCollectInfo) -> list[DebtItem]:
    """Build all active debt findings."""
    items = [
        detect_td001(root),
        detect_td002(root),
        detect_td003(root),
        detect_td004(root),
        detect_td005(root),
        detect_td006(root),
        detect_td007(root, pytest_info),
        detect_td008(root),
        detect_td009(root),
        detect_td010(root),
    ]
    return [item for item in items if item is not None]


def rank_items(items: list[DebtItem]) -> list[DebtItem]:
    """Sort items by descending priority score then debt id."""
    return sorted(items, key=lambda item: (-item.score(), item.debt_id))


def write_register(path: Path, audit_date: str, ranked: list[DebtItem]) -> None:
    """Write debt register CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REGISTER_HEADER)
        for rank, item in enumerate(ranked, start=1):
            writer.writerow(
                [
                    item.debt_id,
                    rank,
                    audit_date,
                    item.category,
                    f"{item.score():.1f}",
                    "open",
                    "unassigned",
                    item.component,
                    "; ".join(item.refs),
                    item.symptom,
                    item.impact_summary,
                    item.risk_level(),
                    item.frequency,
                    item.effort_days,
                    item.target_window,
                    item.acceptance_criteria,
                    item.evidence_note,
                    audit_date,
                ]
            )


def audit_lines(audit_date: str, baseline: dict[str, object], info: PytestCollectInfo, ranked: list[DebtItem]) -> list[str]:
    """Render audit markdown content."""
    env_path = baseline["env_local_path"]
    env_keys = baseline["env_local_keys"]
    lines = [
        "# Atlas Technical Debt Baseline Audit",
        "",
        f"Date: {audit_date}",
        f"Scope: `{repo_root().as_posix()}`",
        "Audit type: Automated repository scan",
        "",
        "## Method",
        "",
        "1. Count tracked source files by language and test area.",
        "2. Read `.env.local` key inventory.",
        "3. Run deterministic debt rule checks against known hotspots.",
        "4. Run `pytest --collect-only -q` for collectability signal.",
        "5. Rank with `priority_score = (impact * frequency * risk) / effort`.",
        "",
        "## Verified Baseline Facts",
        "",
        f"- Python files: {baseline['python_files']}",
        f"- TS/TSX files: {baseline['ts_files']}",
        f"- Test files: {baseline['test_files']}",
        f"- `.env.local` path verified: `{env_path}`",
        "- `.env.local` keys currently present:",
    ]
    for key in env_keys:
        lines.append(f"  - `{key}`")
    lines.extend(
        [
            "- Test collection signal:",
            f"  - `{info.collected} tests collected`",
            f"  - `{info.errors} collection errors`",
            f"  - `pytest return code: {info.return_code}`",
            "",
            "## Prioritized Top Debt Items",
            "",
            "| Rank | ID | Score | Category | Finding | Verified Evidence |",
            "|---|---|---:|---|---|---|",
        ]
    )
    if not ranked:
        lines.append("| - | - | - | - | No active debt findings detected by current rules. | - |")
        return lines
    for rank, item in enumerate(ranked, start=1):
        refs = "; ".join(item.refs)
        lines.append(
            f"| {rank} | {item.debt_id} | {item.score():.1f} | {item.category} | {item.symptom} | `{refs}` |"
        )
    return lines


def write_audit(path: Path, lines: list[str]) -> None:
    """Write audit markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Generate technical debt register files.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Audit date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/technical-debt",
        help="Output directory relative to repo root.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest collect-only scan.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    root = repo_root()
    output_dir = root / args.output_dir
    baseline = collect_baseline(root)
    pytest_info = run_pytest_collect(root, enabled=not args.skip_pytest)
    ranked = rank_items(collect_debt_items(root, pytest_info))
    dated_register = output_dir / f"debt-register-{args.date}.csv"
    latest_register = output_dir / "debt-register-latest.csv"
    write_register(dated_register, args.date, ranked)
    write_register(latest_register, args.date, ranked)
    audit = audit_lines(args.date, baseline, pytest_info, ranked)
    dated_audit = output_dir / f"initial-audit-{args.date}.md"
    latest_audit = output_dir / "initial-audit-latest.md"
    write_audit(dated_audit, audit)
    write_audit(latest_audit, audit)
    if not ranked:
        print("No debt items detected by current rules.")
    print(f"Wrote {dated_register.relative_to(root)}")
    print(f"Wrote {latest_register.relative_to(root)}")
    print(f"Wrote {dated_audit.relative_to(root)}")
    print(f"Wrote {latest_audit.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
