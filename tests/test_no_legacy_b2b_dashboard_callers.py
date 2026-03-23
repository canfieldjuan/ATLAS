from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_TOKENS = ("/api/v1/b2b/dashboard", "/b2b/dashboard")

# These files are allowed to reference legacy paths for migration support.
ALLOWED_FILES = {
    "atlas_brain/api/b2b_dashboard.py",
    "atlas_brain/api/b2b_tenant_dashboard.py",
    "atlas_brain/main.py",
    "tests/test_b2b_tenant_data_freshness.py",
    "tests/test_no_legacy_b2b_dashboard_callers.py",
}

SCAN_ROOTS = (
    "atlas_brain/api",
    "atlas_brain/autonomous",
    "atlas-churn-ui/src",
    "atlas-intel-ui/src",
    "scripts",
)

TEXT_FILE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".sh", ".md", ".yaml", ".yml", ".json"}


def _iter_source_files():
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_FILE_SUFFIXES:
                continue
            yield path


def test_no_legacy_b2b_dashboard_path_callers_outside_migration_files():
    offenders: list[str] = []
    for path in _iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(token in text for token in LEGACY_TOKENS):
            offenders.append(rel)

    assert not offenders, f"Legacy b2b/dashboard callers found: {offenders}"
