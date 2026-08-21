"""Failure-branch tests for scripts/check_contact_write_boundary.py.

The point of these fixtures is to prove the detector FIRES, not just that it
runs. A guard whose tests only show clean trees passing is indistinguishable
from a guard that matches nothing at all, which is the exact failure mode
tests/test_maturity_sweep.py was written to pin for the maturity sweep.

Every planted violation below is a shape a real bypass could take: a fresh
INSERT in a new service, a raw write in a script, an f-string built query, a
schema-qualified table name.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_contact_write_boundary.py"

SPEC = importlib.util.spec_from_file_location("contact_write_boundary", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _write(tmp_path: Path, rel: str, source: str) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _classify(tmp_path: Path, baseline: dict | None = None):
    findings = MOD.scan(tmp_path)
    return MOD.classify(findings, baseline or {})


# ---------------------------------------------------------------------------
# The detector must fire
# ---------------------------------------------------------------------------

def test_planted_insert_outside_provider_is_blocking(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/services/rogue_writer.py",
        'async def create(conn, name):\n'
        '    await conn.execute("""\n'
        '        INSERT INTO contacts (id, full_name)\n'
        '        VALUES ($1, $2)\n'
        '    """, 1, name)\n',
    )
    blocking, _ = _classify(tmp_path)
    assert len(blocking) == 1, "detector failed to fire on a planted INSERT"
    assert blocking[0].path == "atlas_brain/services/rogue_writer.py"
    assert blocking[0].operation == "INSERT"


def _seed_inventory(tmp_path: Path) -> Path:
    """Write an inventory baseline matching the current tree.

    Without one, `main()` fails closed on the missing file, and an exit-code
    test would pass for that reason instead of the one it names.
    """
    import json as _json

    baseline_dir = tmp_path / "tests" / "contact_write_boundary"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "baseline.json"
    findings, unanalyzable = MOD.scan_tree(tmp_path)
    path.write_text(
        _json.dumps(MOD.build_baseline(findings, unanalyzable)), encoding="utf-8"
    )
    return path


def test_planted_insert_makes_main_exit_nonzero(tmp_path: Path) -> None:
    """The CI contract is the exit code, not the finding list."""
    _write(
        tmp_path,
        "scripts/sneaky_import.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    _seed_inventory(tmp_path)
    exit_code = MOD.main(["--root", str(tmp_path)])
    assert exit_code == 1, "planted INSERT did not fail the build"


def test_fstring_built_update_is_detected(tmp_path: Path) -> None:
    """The real provider update is an f-string; a bypass could be too."""
    _write(
        tmp_path,
        "scripts/patcher.py",
        'def patch(cols):\n'
        '    return f"UPDATE contacts SET {cols} WHERE id = $1"\n',
    )
    _, new_mutations = _classify(tmp_path)
    assert [f.operation for f in new_mutations] == ["UPDATE"]


def test_concatenated_sql_is_detected(tmp_path: Path) -> None:
    """`"INSERT INTO " + "contacts (...)"` is one statement at runtime.

    Scanning each ast.Constant in isolation missed this; the recognizer now
    constant-folds `+` chains and f-strings before matching.
    """
    _write(
        tmp_path,
        "scripts/split_sql.py",
        'SQL = "INSERT INTO " + "contacts (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_quoted_table_identifier_is_detected(tmp_path: Path) -> None:
    """`INSERT INTO "contacts" (...)` is valid SQL and must not evade."""
    _write(
        tmp_path,
        "scripts/quoted.py",
        'SQL = \'INSERT INTO "contacts" (full_name) VALUES ($1)\'\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_runtime_built_table_name_is_blocking_in_scope(tmp_path: Path) -> None:
    """`"INSERT INTO " + table` cannot be cleared by reading the literal."""
    _write(
        tmp_path,
        "atlas_brain/services/dynamic.py",
        'def q(table):\n    return "INSERT INTO " + table + " (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert any(f.operation == "DYNAMIC" for f in blocking)


def test_runtime_built_table_name_outside_scope_is_not_blocking(tmp_path: Path) -> None:
    """Unrelated subsystems parameterize their own tables by design.

    The repo has 18 such sites (podcast/campaign/reddit importers, a generic
    migration runner). Failing the build on those is how a gate gets a
    reputation for noise and gets switched off.
    """
    _write(
        tmp_path,
        "extracted_content_pipeline/importer.py",
        'def q(table):\n    return "INSERT INTO " + table + " (a) VALUES ($1)"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert [f.operation for f in new_mutations] == ["DYNAMIC"]


def test_lowercase_prose_ending_in_update_is_not_dynamic(tmp_path: Path) -> None:
    """Regression pin for a case-insensitive rule that flagged nine files.

    "DRY RUN: Would update" is English. SQL keywords in this codebase are
    uppercase, which is what separates the two without inventing a parser.
    """
    _write(
        tmp_path,
        "scripts/chatty.py",
        'A = "DRY RUN: Would update"\nB = "would update"\nC = ", update"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_every_row_creation_form_is_blocking(tmp_path: Path) -> None:
    """INSERT is not the only way a row reaches the table.

    MERGE can insert, COPY bulk-loads, and SELECT ... INTO creates and fills.
    All three write rows while skipping the provider.
    """
    _write(
        tmp_path,
        "atlas_brain/services/other_writes.py",
        # The MERGE carries an insert branch: an update-only MERGE creates no
        # rows and is deliberately non-blocking (see
        # test_merge_without_insert_branch_is_not_a_create).
        'A = "MERGE INTO contacts USING staging ON contacts.id = staging.id '
        'WHEN NOT MATCHED THEN INSERT (full_name) VALUES (staging.name)"\n'
        'B = "COPY contacts (full_name) FROM STDIN"\n'
        'C = "SELECT full_name INTO contacts FROM staging"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert {f.operation for f in blocking} == {"MERGE", "COPY", "SELECT_INTO"}


def test_sql_migration_files_are_scanned(tmp_path: Path) -> None:
    """Python is not the only way a statement reaches the database."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/999_backfill.sql",
        "INSERT INTO contacts (full_name) SELECT name FROM legacy;\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.path for f in blocking] == [
        "atlas_brain/storage/migrations/999_backfill.sql"
    ]


def test_sql_comments_are_not_executable(tmp_path: Path) -> None:
    """Migration 358 documents a rollback recipe in a comment; it is not a write."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/998_notes.sql",
        "-- UPDATE contacts SET lead_stage = 'estimate_booked';\n"
        "/* INSERT INTO contacts (full_name) VALUES ('x'); */\n"
        "CREATE INDEX idx_demo ON contacts (email);\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_dashes_inside_a_string_literal_cannot_hide_a_write(tmp_path: Path) -> None:
    """`--` inside a literal is data, not a comment.

    The first draft stripped comments with `--[^\\n]*`, so
    `SELECT 'harmless -- '; INSERT INTO contacts (...)` on one line hid a real
    INSERT from the gate entirely. A regex cannot tokenize SQL.
    """
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/900_evade.sql",
        "SELECT 'harmless -- '; INSERT INTO contacts (full_name) VALUES ('hidden');\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking if f.operation == "INSERT"] == ["INSERT"]


def test_escaped_quote_inside_literal_is_handled(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/901_escape.sql",
        "INSERT INTO contacts (full_name) VALUES ('O''Brien -- x');\n",
    )
    blocking, _ = _classify(tmp_path)
    assert any(f.operation == "INSERT" for f in blocking)


def test_dollar_quoted_body_does_not_swallow_following_statements(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/902_dollar.sql",
        "DO $$ BEGIN PERFORM 1; END $$;\n"
        "INSERT INTO contacts (full_name) VALUES ('y');\n",
    )
    blocking, _ = _classify(tmp_path)
    assert any(f.operation == "INSERT" for f in blocking)


def test_nested_block_comment_is_fully_stripped(tmp_path: Path) -> None:
    """PostgreSQL block comments nest; a non-nesting stripper ends early."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/903_nested.sql",
        "/* outer /* inner */ INSERT INTO contacts (full_name) VALUES ('x'); */\n"
        "SELECT 1;\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_production_module_named_test_is_not_exempt(tmp_path: Path) -> None:
    """Exemption is by location, never by basename.

    13 real production modules start with `test_` outside any tests directory
    (`scripts/test_adapter_live.py`, `atlas_brain/test_token_tracking.py`, ...).
    A basename rule let every one of them carry a write past the gate.
    """
    _write(
        tmp_path,
        "scripts/test_adapter_live.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.path for f in blocking] == ["scripts/test_adapter_live.py"]


def test_docstrings_are_not_executable_sql(tmp_path: Path) -> None:
    """Prose about code is not a statement the database runs."""
    _write(
        tmp_path,
        "atlas_brain/services/documented.py",
        '"""Module that will INSERT INTO contacts (full_name) VALUES ($1)."""\n'
        'def f():\n'
        '    """Helper that runs UPDATE contacts SET full_name = $1."""\n'
        '    return 1\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_two_statements_with_identical_snippets_are_both_counted(tmp_path: Path) -> None:
    """Per-file dedup keys on (operation, line), not on the snippet.

    Keying on the baseline key (which omits the line) silently collapsed one of
    the provider's nine UPDATE statements, under-reporting the inventory this
    gate exists to pin.
    """
    _write(
        tmp_path,
        "atlas_brain/services/twice.py",
        'A = "UPDATE contacts SET updated_at = NOW() WHERE id = $1"\n'
        'B = "UPDATE contacts SET updated_at = NOW() WHERE id = $1"\n',
    )
    findings, _ = MOD.scan_tree(tmp_path)
    updates = [f for f in findings if f.operation == "UPDATE"]
    assert len(updates) == 2, "identical statements on different lines both count"


def test_schema_qualified_and_delete_are_detected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/purge.py",
        'A = "INSERT INTO public.contacts (full_name) VALUES ($1)"\n'
        'B = "DELETE FROM contacts WHERE id = $1"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]
    assert [f.operation for f in new_mutations] == ["DELETE"]


# ---------------------------------------------------------------------------
# The detector must NOT fire (false positives are how a gate gets disabled)
# ---------------------------------------------------------------------------

def test_approved_provider_module_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/services/crm_provider.py",
        'SQL = "INSERT INTO contacts (id, full_name) VALUES ($1, $2)"\n'
        'UPD = "UPDATE contacts SET full_name = $2 WHERE id = $1"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_prose_docstring_is_not_a_write(tmp_path: Path) -> None:
    """Regression pin: 'Create/update contacts in the Atlas CRM.' is English.

    The first draft of this detector matched bare `update contacts` and flagged
    scripts/import_calendar_contacts.py's module docstring.
    """
    _write(
        tmp_path,
        "scripts/importer.py",
        '"""Create/update contacts in the Atlas CRM."""\n'
        'HELP = "This tool will insert into contacts only via the provider."\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_commented_out_sql_is_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/legacy.py",
        "# INSERT INTO contacts (full_name) VALUES ($1)\n"
        "VALUE = 1\n",
    )
    blocking, _ = _classify(tmp_path)
    assert blocking == []


def test_neighbouring_tables_do_not_match(tmp_path: Path) -> None:
    """`contact_interactions` and `contacts_archive` are different tables."""
    _write(
        tmp_path,
        "scripts/other_tables.py",
        'A = "INSERT INTO contact_interactions (contact_id) VALUES ($1)"\n'
        'B = "INSERT INTO contacts_archive (id) VALUES ($1)"\n'
        'C = "UPDATE contact_interactions SET summary = $1"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_tests_directory_is_exempt(tmp_path: Path) -> None:
    """Test fixtures legitimately contain SQL."""
    _write(
        tmp_path,
        "tests/test_something.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_unparsable_file_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    """A file the gate cannot parse must not be reported as clean.

    The first draft returned `[]` for unreadable/unparsable files, which meant a
    bypass hiding behind a syntax error would leave the gate saying OK. Silence
    is not evidence of absence. The repo's maturity sweep flagged both swallowed
    excepts, and it was right to.
    """
    _write(tmp_path, "scripts/broken.py", "def (:\n")
    _write(tmp_path, "scripts/fine.py", "VALUE = 1\n")

    findings, unanalyzable = MOD.scan_tree(tmp_path)
    assert findings == []
    assert len(unanalyzable) == 1
    assert "scripts/broken.py" in unanalyzable[0]
    assert "unparsable" in unanalyzable[0]


def test_unparsable_file_fails_the_build(tmp_path: Path) -> None:
    _write(tmp_path, "scripts/broken.py", "def (:\n")
    _seed_inventory(tmp_path)
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_known_unanalyzable_file_can_be_baselined(tmp_path: Path) -> None:
    """A deliberate fixture may be recorded, but only explicitly."""
    _write(tmp_path, "scripts/broken.py", "def (:\n")
    findings, unanalyzable = MOD.scan_tree(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(MOD.build_baseline(findings, unanalyzable)), encoding="utf-8"
    )
    # The inventory baseline is passed explicitly: it defaults to
    # <root>/tests/contact_write_boundary/baseline.json, which this fixture tree
    # does not have, and a missing inventory now fails closed by design.
    assert MOD.main(
        [
            "--root", str(tmp_path),
            "--baseline", str(baseline_path),
            "--inventory-baseline", str(baseline_path),
        ]
    ) == 0


# ---------------------------------------------------------------------------
# Baseline behavior
# ---------------------------------------------------------------------------

def test_baselined_update_is_not_reported_but_a_new_one_is(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/known_writer.py",
        'SQL = "UPDATE contacts SET business_context_id = $1 WHERE id = $2"\n',
    )
    baseline = MOD.build_baseline(MOD.scan(tmp_path))
    _, new_mutations = _classify(tmp_path, baseline)
    assert new_mutations == [], "a baselined write should be silent"

    _write(
        tmp_path,
        "scripts/new_writer.py",
        'SQL = "UPDATE contacts SET full_name = $1 WHERE id = $2"\n',
    )
    _, new_mutations = _classify(tmp_path, baseline)
    assert [f.path for f in new_mutations] == ["scripts/new_writer.py"]


def test_insert_is_never_silenced_by_the_baseline(tmp_path: Path) -> None:
    """An INSERT must fail the build even if someone baselines it.

    Otherwise `--update-baseline` becomes a one-command bypass of the guard.
    """
    _write(
        tmp_path,
        "scripts/rogue.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    baseline = MOD.build_baseline(MOD.scan(tmp_path))
    blocking, _ = _classify(tmp_path, baseline)
    assert len(blocking) == 1, "baselining an INSERT must not disable the gate"


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------

def test_repository_is_currently_clean() -> None:
    """The tree must pass its own gate, with the committed baseline.

    The baseline is part of the contract, not an optional flag: it records the
    18 pre-existing dynamic-target sites in unrelated subsystems.
    """
    baseline = ROOT / "tests" / "contact_write_boundary" / "baseline.json"
    assert MOD.main(["--root", str(ROOT), "--baseline", str(baseline)]) == 0


def test_repository_has_exactly_the_known_insert_sites() -> None:
    """Pins the fact that makes this gate enforceable: the approved surface is
    two INSERT sites in one module. If a third appears legitimately, this test
    is the place to make that decision consciously."""
    findings = MOD.scan(ROOT)
    inserts = [
        f for f in findings
        if f.operation == "INSERT" and not MOD._is_test_path(f.path)
    ]
    assert {f.path for f in inserts} == {"atlas_brain/services/crm_provider.py"}
    assert len(inserts) == 2, f"expected 2 provider INSERT sites, found {len(inserts)}"

    # Ground truth: `grep -c 'UPDATE contacts' crm_provider.py` == 11. Pinning
    # the count catches a recognizer regression that silently drops one.
    provider_updates = [
        f for f in findings
        if f.operation == "UPDATE" and f.path == "atlas_brain/services/crm_provider.py"
    ]
    assert len(provider_updates) == 11, (
        f"expected 11 provider UPDATE statements, found {len(provider_updates)}"
    )


def test_baseline_file_matches_the_tree() -> None:
    """The committed baseline must not drift from reality.

    Asserts on `writer_inventory` rather than `known_writes`: the latter is
    empty today (every writer is allow-listed), so comparing it would be an
    empty-in/empty-out assertion that passes no matter what the tree does.
    """
    baseline_path = ROOT / "tests" / "contact_write_boundary" / "baseline.json"
    assert baseline_path.exists(), "committed baseline is missing"
    committed = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = MOD.build_baseline(MOD.scan(ROOT))

    assert committed["writer_inventory"], "inventory is empty; the scan found nothing"
    assert committed["writer_inventory"] == current["writer_inventory"], (
        "contact writer inventory drifted; refresh with --update-baseline "
        "and review the diff as part of the change"
    )
    assert committed["known_writes"] == current["known_writes"]


def test_baseline_inventory_is_not_vacuous() -> None:
    """A baseline that records nothing cannot detect drift in anything."""
    baseline_path = ROOT / "tests" / "contact_write_boundary" / "baseline.json"
    committed = json.loads(baseline_path.read_text(encoding="utf-8"))
    inventory = committed["writer_inventory"]
    inserts = [entry for entry in inventory if "::INSERT::" in entry]
    updates = [entry for entry in inventory if "::UPDATE::" in entry]
    assert len(inserts) == 2, "expected both provider INSERT sites in the inventory"
    assert len(updates) >= 11, "expected the known UPDATE surface in the inventory"


# ---------------------------------------------------------------------------
# Round 4: literal-vs-executable lexing, TRUNCATE, MERGE branches
# ---------------------------------------------------------------------------

def test_escape_string_does_not_erase_following_statements(tmp_path: Path) -> None:
    r"""`E'it\'s'` closes at the backslash-escaped quote in a naive lexer.

    The previous lexer then treated the rest of the line as a comment and
    erased a real INSERT, exiting 0.
    """
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/904_escape.sql",
        "SELECT E'it\\'s -- still literal'; "
        "INSERT INTO contacts (full_name) VALUES ('x');\n",
    )
    blocking, _ = _classify(tmp_path)
    assert any(f.operation == "INSERT" for f in blocking)


def test_sql_quoted_as_data_is_not_executable(tmp_path: Path) -> None:
    """Literal contents are data. Reading them invents statements.

    `SELECT 'Example only: INSERT INTO contacts (...)'` was reported as a
    blocking write, which would red the build over a doc string in a migration.
    """
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/905_inert.sql",
        "SELECT 'Example only: INSERT INTO contacts (full_name) VALUES (''x'')';\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_truncate_surfaces_in_the_inventory(tmp_path: Path) -> None:
    """TRUNCATE deletes every row without the provider or the audit ledger."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/906_truncate.sql",
        "TRUNCATE TABLE contacts;\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert [f.operation for f in new_mutations] == ["TRUNCATE"]
    assert blocking == [], "classified with the deferred UPDATE/DELETE policy"


def test_merge_without_insert_branch_is_not_a_create(tmp_path: Path) -> None:
    """An update-only MERGE creates no rows, so it must not red the build."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/907_merge_update.sql",
        "MERGE INTO contacts USING staging s ON s.id = contacts.id "
        "WHEN MATCHED THEN UPDATE SET full_name = s.name;\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert [f.operation for f in new_mutations] == ["MERGE_UPDATE"]


def test_merge_with_insert_branch_is_blocking(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/908_merge_insert.sql",
        "MERGE INTO contacts USING staging s ON s.id = contacts.id "
        "WHEN NOT MATCHED THEN INSERT (full_name) VALUES (s.name);\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["MERGE"]


def test_select_before_insert_does_not_report_phantom_select_into(tmp_path: Path) -> None:
    """`SELECT 1; INSERT INTO contacts (...)` is one write, not two."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/909_two_statements.sql",
        "SELECT 1;\nINSERT INTO contacts (full_name) VALUES ('x');\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


# ---------------------------------------------------------------------------
# Round 5: executable dollar bodies, inventory enforcement
# ---------------------------------------------------------------------------

def test_do_block_body_is_executable_sql(tmp_path: Path) -> None:
    """`DO $$ ... $$` runs its body; blanking it hid a real writer.

    The migration runner submits each file's complete SQL text, so an INSERT
    inside a DO block reaches the database exactly like a top-level one.
    """
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/910_do.sql",
        "DO $$ BEGIN INSERT INTO contacts (full_name) VALUES ('x'); END $$;\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_function_body_is_executable_sql(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/911_func.sql",
        "CREATE FUNCTION f() RETURNS void AS $body$ BEGIN "
        "INSERT INTO contacts (full_name) VALUES ('y'); END $body$ LANGUAGE plpgsql;\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_literal_inside_a_dollar_body_stays_inert(tmp_path: Path) -> None:
    """Recursing into the body must still blank literals nested in it."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/912_notice.sql",
        "DO $$ BEGIN RAISE NOTICE 'INSERT INTO contacts is documented here'; END $$;\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_new_write_in_the_allowed_module_fails_without_a_baseline_update(
    tmp_path: Path,
) -> None:
    """classify() alone cannot see this: the module is allow-listed.

    Without an inventory comparison a PR could add a third INSERT inside
    crm_provider.py, or delete an inventoried writer, and still exit 0 -- the
    drift record would never actually be checked.
    """
    import json as _json

    provider = tmp_path / "atlas_brain" / "services" / "crm_provider.py"
    provider.parent.mkdir(parents=True, exist_ok=True)
    provider.write_text(
        'A = "INSERT INTO contacts (a) VALUES ($1)"\n'
        'B = "INSERT INTO contacts (b) VALUES ($1)"\n',
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "tests" / "contact_write_boundary"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    findings, unanalyzable = MOD.scan_tree(tmp_path)
    (baseline_dir / "baseline.json").write_text(
        _json.dumps(MOD.build_baseline(findings, unanalyzable)), encoding="utf-8"
    )
    assert MOD.main(["--root", str(tmp_path)]) == 0

    provider.write_text(
        provider.read_text() + 'C = "INSERT INTO contacts (c) VALUES ($1)"\n',
        encoding="utf-8",
    )
    assert MOD.main(["--root", str(tmp_path)]) == 1, (
        "a new write inside the allow-listed module must not pass silently"
    )


def test_removing_an_inventoried_writer_also_fails(tmp_path: Path) -> None:
    import json as _json

    provider = tmp_path / "atlas_brain" / "services" / "crm_provider.py"
    provider.parent.mkdir(parents=True, exist_ok=True)
    provider.write_text(
        'A = "INSERT INTO contacts (a) VALUES ($1)"\n'
        'B = "INSERT INTO contacts (b) VALUES ($1)"\n',
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "tests" / "contact_write_boundary"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    findings, unanalyzable = MOD.scan_tree(tmp_path)
    (baseline_dir / "baseline.json").write_text(
        _json.dumps(MOD.build_baseline(findings, unanalyzable)), encoding="utf-8"
    )

    provider.write_text('A = "INSERT INTO contacts (a) VALUES ($1)"\n', encoding="utf-8")
    assert MOD.main(["--root", str(tmp_path)]) == 1


# ---------------------------------------------------------------------------
# Round 6: EXECUTE, missing baseline, same-line writes
# ---------------------------------------------------------------------------

def test_execute_of_a_literal_is_executable_sql(tmp_path: Path) -> None:
    """`EXECUTE 'INSERT INTO contacts ...'` runs the literal.

    Round 4 taught the lexer that literals are inert data, which is right
    everywhere except EXECUTE position, where the literal *is* the statement.
    """
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/913_execute.sql",
        "DO $$ BEGIN EXECUTE 'INSERT INTO contacts (full_name) VALUES (''x'')'; "
        "END $$;\n",
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_literal_not_in_execute_position_stays_inert(tmp_path: Path) -> None:
    """The EXECUTE rule must not re-break the round-4 fix."""
    _write(
        tmp_path,
        "atlas_brain/storage/migrations/914_inert.sql",
        "SELECT 'Example only: INSERT INTO contacts (full_name) VALUES (''x'')';\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_missing_inventory_baseline_fails_closed(tmp_path: Path) -> None:
    """Deleting the baseline must not switch inventory enforcement off.

    Skipping the comparison when the file is absent made removal a one-line
    way to disable the check, in the same diff as the writer it would surface.
    """
    _write(
        tmp_path,
        "atlas_brain/services/crm_provider.py",
        'A = "INSERT INTO contacts (a) VALUES ($1)"\n',
    )
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_two_distinct_writes_on_one_line_are_both_kept(tmp_path: Path) -> None:
    """Dedup keys on (operation, line, snippet).

    (operation, line) alone collapsed two different statements sharing a line,
    which is the mirror of the earlier bug where the snippet-only key collapsed
    two statements sharing a snippet.
    """
    _write(
        tmp_path,
        "atlas_brain/services/same_line.py",
        'A, B = "INSERT INTO contacts (a) VALUES ($1)", '
        '"INSERT INTO contacts (b) VALUES ($1)"\n',
    )
    findings, _ = MOD.scan_tree(tmp_path)
    assert len([f for f in findings if f.operation == "INSERT"]) == 2


# ---------------------------------------------------------------------------
# Round 7: Python-embedded SQL lexing, driver-level writes
# ---------------------------------------------------------------------------

def test_comment_inside_python_embedded_sql_does_not_hide_a_write(
    tmp_path: Path,
) -> None:
    """PostgreSQL treats `/* note */` as whitespace, so this is a write.

    The .sql path was lexed from round 4; the Python path was still matching
    raw literal values, so the same defect existed on the other side.
    """
    _write(
        tmp_path,
        "atlas_brain/services/commented.py",
        'SQL = "INSERT /* audit note */ INTO contacts (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_sql_quoted_as_data_inside_python_is_not_executable(tmp_path: Path) -> None:
    """The mirror direction: literal data in a Python string is not a write."""
    _write(
        tmp_path,
        "atlas_brain/services/documented.py",
        "SQL = \"SELECT 'Example only: INSERT INTO contacts (full_name)'\"\n",
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_driver_level_table_write_is_detected(tmp_path: Path) -> None:
    """asyncpg can insert rows with no SQL statement to scan.

    `copy_records_to_table("contacts", ...)` writes directly, bypassing the
    provider and leaving nothing for a text matcher to find.
    """
    _write(
        tmp_path,
        "atlas_brain/services/bulk.py",
        "async def load(conn, rows):\n"
        '    await conn.copy_records_to_table("contacts", records=rows)\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["DRIVER_WRITE"]


def test_driver_write_via_keyword_target_is_detected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/services/bulk_kw.py",
        "async def load(conn, rows):\n"
        '    await conn.copy_records_to_table(table_name="contacts", records=rows)\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["DRIVER_WRITE"]


def test_driver_write_to_another_table_is_not_flagged(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "atlas_brain/services/bulk_other.py",
        "async def load(conn, rows):\n"
        '    await conn.copy_records_to_table("invoices", records=rows)\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


# ---------------------------------------------------------------------------
# Round 8: inventory multiplicity
# ---------------------------------------------------------------------------

def _seed_provider(tmp_path: Path, source: str) -> Path:
    provider = tmp_path / "atlas_brain" / "services" / "crm_provider.py"
    provider.parent.mkdir(parents=True, exist_ok=True)
    provider.write_text(source, encoding="utf-8")
    _seed_inventory(tmp_path)
    return provider


_DUPLICATE_WRITES = (
    'A = "INSERT INTO contacts (a) VALUES ($1)"\n'
    'B = "INSERT INTO contacts (a) VALUES ($1)"\n'
)


def test_adding_a_byte_for_byte_duplicate_writer_is_drift(tmp_path: Path) -> None:
    """Two identical writes on different lines share a baseline key.

    Set arithmetic reported no drift for either direction, so an approved
    module could grow an extra INSERT silently. The comparison is a multiset.
    """
    provider = _seed_provider(tmp_path, _DUPLICATE_WRITES)
    assert MOD.main(["--root", str(tmp_path)]) == 0

    provider.write_text(
        _DUPLICATE_WRITES + 'C = "INSERT INTO contacts (a) VALUES ($1)"\n',
        encoding="utf-8",
    )
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_removing_one_of_two_identical_writers_is_drift(tmp_path: Path) -> None:
    provider = _seed_provider(tmp_path, _DUPLICATE_WRITES)
    provider.write_text(
        'A = "INSERT INTO contacts (a) VALUES ($1)"\n', encoding="utf-8"
    )
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_unchanged_duplicate_writers_are_not_drift(tmp_path: Path) -> None:
    """The multiset must not report drift against an unchanged tree."""
    _seed_provider(tmp_path, _DUPLICATE_WRITES)
    assert MOD.main(["--root", str(tmp_path)]) == 0


# ---------------------------------------------------------------------------
# The workflow's own contract
# ---------------------------------------------------------------------------

def test_trusted_base_invocation_ignores_a_widened_pr_allowlist(
    tmp_path: Path,
) -> None:
    """Run the shape the trusted job runs: base checker, PR tree as data.

    Until this PR merges, `pull_request_target` has no base definition of the
    workflow, so the enforcement job cannot execute in CI on the PR that
    introduces it. This test is the standing proof of the wiring that the CI
    run cannot yet provide -- and it stays useful afterwards, because it fails
    if the trusted-base property is ever lost.
    """
    import json as _json
    import subprocess
    import sys

    base = tmp_path / "base"
    pr = tmp_path / "pr"
    for tree in (base, pr):
        (tree / "scripts").mkdir(parents=True, exist_ok=True)
        (tree / "tests" / "contact_write_boundary").mkdir(parents=True, exist_ok=True)

    # Base carries the real checker and an empty inventory.
    (base / "scripts" / "check_contact_write_boundary.py").write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    empty = _json.dumps(MOD.build_baseline([], []))
    (base / "tests" / "contact_write_boundary" / "baseline.json").write_text(
        empty, encoding="utf-8"
    )

    # The PR plants a forbidden writer AND widens its own allow-list to permit
    # it, then records that state in its own inventory.
    (pr / "atlas_brain" / "services").mkdir(parents=True, exist_ok=True)
    (pr / "atlas_brain" / "services" / "evil.py").write_text(
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n', encoding="utf-8"
    )
    widened = SCRIPT.read_text(encoding="utf-8").replace(
        'INSERT_ALLOWED = (PROVIDER_MODULE,)',
        'INSERT_ALLOWED = (PROVIDER_MODULE, "atlas_brain/services/evil.py")',
    )
    assert "evil.py" in widened, "fixture failed to widen the allow-list"
    (pr / "scripts" / "check_contact_write_boundary.py").write_text(
        widened, encoding="utf-8"
    )
    pr_findings, pr_unanalyzable = MOD.scan_tree(pr)
    (pr / "tests" / "contact_write_boundary" / "baseline.json").write_text(
        _json.dumps(MOD.build_baseline(pr_findings, pr_unanalyzable)), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(base / "scripts" / "check_contact_write_boundary.py"),
            "--root", str(pr),
            "--baseline", str(base / "tests" / "contact_write_boundary" / "baseline.json"),
            "--inventory-baseline",
            str(pr / "tests" / "contact_write_boundary" / "baseline.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        "the base checker must reject a writer the PR allow-listed for itself\n"
        f"{result.stdout}"
    )
    assert "evil.py" in result.stdout

    # And the PR's own checker would have passed it -- the property being kept.
    self_result = subprocess.run(
        [
            sys.executable,
            str(pr / "scripts" / "check_contact_write_boundary.py"),
            "--root", str(pr),
            "--baseline", str(pr / "tests" / "contact_write_boundary" / "baseline.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert self_result.returncode == 0, (
        "fixture is not demonstrating the difference between base and PR checkers"
    )


def test_blocking_context_has_exactly_one_producer() -> None:
    """`contact-write-boundary` must be emitted only by the trusted event.

    Registering `pull_request` alongside `pull_request_target` on one workflow
    made GitHub emit two check runs of that name per update -- the
    ordinary-event copy skipped by the job guard. Atlas's readiness consumers
    pick the latest run by name and count `skipped` as green
    (`scripts/pr_watcher.py`, `scripts/watch_owned_pr.sh`), so the skipped copy
    could mask a failed enforcement run and report a forbidden writer as
    merge-ready.
    """
    import yaml

    workflows = ROOT / ".github" / "workflows"
    producers: dict[str, list[str]] = {}
    for path in workflows.glob("*.yml"):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            continue
        # PyYAML parses the bare `on:` key as the boolean True.
        triggers = spec.get("on", spec.get(True)) or {}
        trigger_names = set(triggers) if isinstance(triggers, dict) else {triggers}
        for job_name in (spec.get("jobs") or {}):
            producers.setdefault(job_name, []).extend(sorted(trigger_names))

    assert producers.get("contact-write-boundary") == ["pull_request_target"], (
        "the blocking context must have exactly one producer, the trusted "
        f"event; found {producers.get('contact-write-boundary')}"
    )
    # The advisory job must never share the blocking name.
    assert "contact-write-boundary" not in producers.get(
        "contact-write-boundary-selfcheck", []
    )


# ---------------------------------------------------------------------------
# Round 13: alias form, statement position
# ---------------------------------------------------------------------------

def test_insert_with_table_alias_is_detected(tmp_path: Path) -> None:
    """`INSERT INTO contacts AS c (...)` is valid PostgreSQL and was a bypass."""
    _write(
        tmp_path,
        "atlas_brain/services/aliased.py",
        'SQL = "INSERT INTO contacts AS c (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.operation for f in blocking] == ["INSERT"]


def test_statement_opening_a_cte_body_is_detected(tmp_path: Path) -> None:
    """CTE-bodied writes are real writes.

    Pinned because an attempt to suppress prose false positives by requiring a
    statement-start position silently dropped exactly these: the provider opens
    CTE bodies with `WITH v AS (\\n  UPDATE contacts ...`. That attempt was
    reverted; see ATLAS #2304 for the prose false positive it was chasing.
    """
    _write(
        tmp_path,
        "atlas_brain/services/cte.py",
        'SQL = """\n'
        '    WITH visible AS (\n'
        '        UPDATE contacts SET business_context_id = $2 RETURNING *\n'
        '    )\n'
        '    SELECT 1\n'
        '"""\n',
    )
    _, new_mutations = _classify(tmp_path)
    assert [f.operation for f in new_mutations] == ["UPDATE"]


def test_every_write_in_one_literal_is_counted(tmp_path: Path) -> None:
    """One literal can hold several statements.

    `scan_file` used `search`, reporting only the first match per pattern, so a
    second write added inside an existing literal in an allow-listed module
    would never surface as inventory drift -- the same hole the multiset
    comparison closed for identical keys. `scan_sql_file` already iterated.
    """
    _write(
        tmp_path,
        "atlas_brain/services/crm_provider.py",
        'SQL = """\n'
        "    INSERT INTO contacts (a) VALUES ($1);\n"
        "    INSERT INTO contacts (b) VALUES ($2);\n"
        '"""\n',
    )
    findings, _ = MOD.scan_tree(tmp_path)
    assert len([f for f in findings if f.operation == "INSERT"]) == 2


def test_a_second_write_in_an_existing_literal_is_drift(tmp_path: Path) -> None:
    """The reason the count matters: it is what makes drift observable."""
    provider = _seed_provider(
        tmp_path,
        'SQL = """\n    INSERT INTO contacts (a) VALUES ($1);\n"""\n',
    )
    assert MOD.main(["--root", str(tmp_path)]) == 0

    provider.write_text(
        'SQL = """\n'
        "    INSERT INTO contacts (a) VALUES ($1);\n"
        "    INSERT INTO contacts (b) VALUES ($2);\n"
        '"""\n',
        encoding="utf-8",
    )
    assert MOD.main(["--root", str(tmp_path)]) == 1


def test_nested_tests_directory_does_not_exempt(tmp_path: Path) -> None:
    """Only the authoritative `tests/` root exempts.

    This rule has been wrong twice: first keyed on basename, which exempted 13
    real production modules named `test_*.py`; then on "any ancestor directory
    named tests", which exempted `atlas_brain/services/tests/evil.py`.
    """
    _write(
        tmp_path,
        "atlas_brain/services/tests/evil.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    blocking, _ = _classify(tmp_path)
    assert [f.path for f in blocking] == ["atlas_brain/services/tests/evil.py"]


def test_authoritative_test_root_still_exempts(tmp_path: Path) -> None:
    """Both sides: real fixtures under tests/ must stay exempt, at any depth."""
    _write(
        tmp_path,
        "tests/fixtures/deep/sample.py",
        'SQL = "INSERT INTO contacts (full_name) VALUES ($1)"\n',
    )
    blocking, new_mutations = _classify(tmp_path)
    assert blocking == []
    assert new_mutations == []


def test_each_match_reports_its_own_source_line(tmp_path: Path) -> None:
    """A multi-line literal must not report every match at its opening line.

    Beyond readability, the (operation, line, snippet) dedup key depended on
    surrounding text happening to differ; with a shared line the two statements
    below could collapse and under-count the inventory.
    """
    _write(
        tmp_path,
        "atlas_brain/services/crm_provider.py",
        'SQL = """\n'
        "    INSERT INTO contacts (a) VALUES ($1);\n"
        "    INSERT INTO contacts (a) VALUES ($1);\n"
        '"""\n',
    )
    findings, _ = MOD.scan_tree(tmp_path)
    lines = sorted(f.line for f in findings if f.operation == "INSERT")
    assert lines == [2, 3], f"expected the statements' own lines, got {lines}"
