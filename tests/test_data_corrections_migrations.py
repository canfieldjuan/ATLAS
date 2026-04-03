from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "atlas_brain" / "storage" / "migrations"


def _read(name: str) -> str:
    return (MIGRATIONS / name).read_text()


def test_consumer_corrections_keeps_suppress_source() -> None:
    sql = _read("122_consumer_corrections.sql")
    assert "'suppress_source'" in sql


def test_forward_fix_restores_suppress_source_constraint() -> None:
    sql = _read("254_restore_suppress_source_correction_type.sql")
    assert "chk_correction_type" in sql
    assert "'suppress_source'" in sql
