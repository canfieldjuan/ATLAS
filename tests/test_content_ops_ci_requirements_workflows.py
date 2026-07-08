from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT_OPS_REQUIREMENTS = ROOT / "requirements.content_ops_ci.txt"

WORKFLOWS = {
    "stripe_paid": ROOT
    / ".github/workflows/atlas_content_ops_deflection_stripe_paid_checks.yml",
    "macro_writeback": ROOT
    / ".github/workflows/atlas_content_ops_macro_writeback_checks.yml",
    "input_provider": ROOT
    / ".github/workflows/atlas_content_ops_input_provider_checks.yml",
    "deflection_report": ROOT
    / ".github/workflows/atlas_content_ops_deflection_report_checks.yml",
}

WORKFLOW_CANARIES = {
    "stripe_paid": (
        "tests/test_atlas_billing_stripe_hardening.py",
        "tests/test_atlas_billing_content_ops_deflection_stripe_paid.py",
        "tests/test_atlas_billing_content_ops_deflection_paid_flow.py",
    ),
    "macro_writeback": (
        "tests/test_autonomous_faq_macro_writeback_scheduled_publish.py",
        "tests/test_extracted_ticket_faq_macro_writeback_zendesk.py",
        "tests/test_scheduler.py::TestDefaults::test_content_ops_faq_macro_writeback_default_seed_respects_opt_in",
    ),
    "input_provider": (
        "tests/test_atlas_content_ops_input_provider.py",
        "tests/test_atlas_content_ops_infrastructure.py",
        "tests/test_support_ticket_provider_landing_blog_execute.py",
    ),
    "deflection_report": (
        "tests/test_content_ops_deflection_report.py::test_postgres_deflection_report_store_live_round_trips_paid_gate",
        "tests/test_deflection_delta_automation_task.py",
        "tests/test_run_deflection_full_report_qa_live_runner.py",
    ),
}

# Required by NAME (specifier-agnostic): the guard's intent is that these
# packages stay present in the subset file, not that they carry a particular
# specifier shape (the file is ==pinned as of #2035 G1.2). The excluded side
# below was already name-based; this makes both sides symmetric.
REQUIRED_PACKAGES = (
    "torch",
    "numpy",
    "asyncpg",
    "mcp",
    "stripe",
    "fpdf2",
    "markdown",
    "feedparser",
    "curl_cffi",
    "beautifulsoup4",
    "pytest",
    "pytest-asyncio",
)

EXCLUDED_HEAVY_PACKAGES = (
    "torchaudio",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "nemo_toolkit",
    "opencv-python",
    "sounddevice",
    "soundfile",
    "webrtcvad",
    "piper-tts",
    "kokoro",
    "playwright",
    "playwright-stealth",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def requirement_lines(path: Path) -> set[str]:
    lines: set[str] = set()
    for raw_line in read(path).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.add(line)
    return lines


def requirement_name(requirement: str) -> str:
    name = re.split(r"\\s|[<>=!~;[]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def test_content_ops_ci_requirements_keep_needed_packages_without_heavy_stack() -> None:
    requirements = requirement_lines(CONTENT_OPS_REQUIREMENTS)
    requirement_names = {requirement_name(line) for line in requirements}

    for package in REQUIRED_PACKAGES:
        assert requirement_name(package) in requirement_names
    for package in EXCLUDED_HEAVY_PACKAGES:
        assert requirement_name(package) not in requirement_names


def test_content_ops_workflows_install_and_cache_the_subset_requirements() -> None:
    for workflow in WORKFLOWS.values():
        text = read(workflow)

        assert '"requirements.content_ops_ci.txt"' in text
        assert '- "requirements.txt"' not in text
        assert "cache-dependency-path: requirements.content_ops_ci.txt" in text
        assert "pip install -r requirements.content_ops_ci.txt" in text
        assert "pip install -r requirements.txt" not in text


def test_content_ops_workflows_keep_existing_test_targets_enrolled() -> None:
    for name, workflow in WORKFLOWS.items():
        text = read(workflow)

        assert "tests/test_content_ops_ci_requirements_workflows.py" in text
        for target in WORKFLOW_CANARIES[name]:
            assert target in text
