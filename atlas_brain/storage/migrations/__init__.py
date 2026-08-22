"""
Database migrations for Atlas Brain.

Tracks applied migrations in `schema_migrations` table to avoid re-running.
"""

import asyncio
import hashlib
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("atlas.storage.migrations")

MIGRATIONS_DIR = Path(__file__).parent


def _parse_migration_identity(filename: str) -> tuple[int, str]:
    """Extract the numeric prefix and canonical migration name."""
    prefix = filename.split("_", 1)[0]
    match = re.match(r"\d+", prefix)
    version = int(match.group()) if match else 0
    return version, filename.removesuffix(".sql")


def _find_duplicate_migration_prefixes(migration_files: list[Path]) -> dict[int, list[str]]:
    duplicates: dict[int, list[str]] = {}
    seen: dict[int, list[str]] = {}
    for migration_file in migration_files:
        version, _ = _parse_migration_identity(migration_file.name)
        seen.setdefault(version, []).append(migration_file.name)
    for version, names in seen.items():
        if version > 0 and len(names) > 1:
            duplicates[version] = sorted(names)
    return duplicates


async def _ensure_migrations_table(executor) -> None:
    """Create the migrations tracking table if it doesn't exist.

    ``executor`` is a pool OR a single acquired connection; both expose
    execute/fetch/fetchval. run_migrations passes a connection so the whole
    run needs exactly one, which is what makes a min=max=1 pool safe.
    """
    await executor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            content_sha256 VARCHAR(64),
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    await executor.execute(
        "ALTER TABLE schema_migrations "
        "ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)"
    )


async def _get_applied_migrations(executor) -> set[str]:
    """Get set of already applied migration names (e.g. '025_temporal_patterns')."""
    rows = await executor.fetch("SELECT name FROM schema_migrations")
    return {row["name"] for row in rows}


async def _record_migration(
    executor,
    filename: str,
    content_sha256: str,
    *,
    fill_newly_self_recorded_digest: bool = False,
) -> None:
    """Record that a migration has been applied.

    ``executor`` is a pool or a single acquired connection (see
    _ensure_migrations_table). A migration may insert its own legacy ledger row
    in SQL. Only the runner that just executed a pending file may fill that
    newly-created row's digest; existing historical rows remain untouched.
    """
    version, name = _parse_migration_identity(filename)
    existing_version = await executor.fetchval(
        "SELECT version FROM schema_migrations WHERE name = $1",
        name,
    )
    if existing_version is not None:
        if fill_newly_self_recorded_digest:
            await executor.execute(
                "UPDATE schema_migrations SET content_sha256 = $2 "
                "WHERE name = $1 AND content_sha256 IS DISTINCT FROM $2",
                name,
                content_sha256,
            )
            persisted_digest = await executor.fetchval(
                "SELECT content_sha256 FROM schema_migrations WHERE name = $1",
                name,
            )
            if persisted_digest != content_sha256:
                raise RuntimeError(
                    "self-recorded migration did not persist its expected "
                    f"content SHA-256: {name}"
                )
        return

    record_version = version
    conflicting_name = await executor.fetchval(
        "SELECT name FROM schema_migrations WHERE version = $1",
        version,
    )
    if conflicting_name and conflicting_name != name:
        record_version = await executor.fetchval(
            """
            SELECT CASE
                WHEN COALESCE(MIN(version), 0) < 0 THEN MIN(version) - 1
                ELSE -1
            END
            FROM schema_migrations
            """
        )
        logger.warning(
            "Migration version collision for %s on prefix %d with existing %s; "
            "recording under synthetic version %d",
            name,
            version,
            conflicting_name,
            record_version,
        )

    await executor.execute(
        "INSERT INTO schema_migrations (version, name, content_sha256) "
        "VALUES ($1, $2, $3)",
        record_version,
        name,
        content_sha256,
    )


@dataclass(frozen=True)
class MigrationContentIntegrityReport:
    """Read-only migration-source identity classification for one run."""

    verified: tuple[str, ...]
    legacy_unverified: tuple[str, ...]
    mismatched: tuple[str, ...]
    missing_source: tuple[str, ...]


class PendingMigrationContentIntegrityError(RuntimeError):
    """Raised before pending SQL when provenance evidence remains unresolved."""


def _migration_content_sha256(source: bytes) -> str:
    """Return the deterministic identity for exactly one migration source."""
    return hashlib.sha256(source).hexdigest()


async def migration_content_integrity_report(
    executor,
    migration_files: Collection[Path],
) -> MigrationContentIntegrityReport:
    """Classify stored migration identities without altering ledger rows."""
    source_by_name = {path.stem: path for path in migration_files}
    rows = await executor.fetch("SELECT name, content_sha256 FROM schema_migrations")
    verified: list[str] = []
    legacy_unverified: list[str] = []
    mismatched: list[str] = []
    missing_source: list[str] = []

    for row in rows:
        name = row["name"]
        migration_file = source_by_name.get(name)
        if migration_file is None:
            missing_source.append(name)
            continue

        try:
            source = migration_file.read_bytes()
        except OSError:
            # This phase is diagnostic-only. A deployment with no readable
            # packaged source still has no evidence to verify, but it must not
            # lose startup availability merely because the new report ran.
            missing_source.append(name)
            continue

        recorded_digest = row["content_sha256"]
        if recorded_digest is None:
            legacy_unverified.append(name)
            continue

        current_digest = _migration_content_sha256(source)
        if current_digest == recorded_digest:
            verified.append(name)
        else:
            mismatched.append(name)

    return MigrationContentIntegrityReport(
        verified=tuple(sorted(verified)),
        legacy_unverified=tuple(sorted(legacy_unverified)),
        mismatched=tuple(sorted(mismatched)),
        missing_source=tuple(sorted(missing_source)),
    )


async def _migration_content_integrity_report(
    executor,
    migration_files: Collection[Path],
) -> MigrationContentIntegrityReport:
    """Backward-compatible internal alias for the canonical read-only report."""
    return await migration_content_integrity_report(executor, migration_files)


def _log_migration_content_integrity(report: MigrationContentIntegrityReport) -> None:
    """Expose evidence without making historical records a startup blocker."""
    if report.legacy_unverified:
        logger.info(
            "Migration content identity is unavailable for %d legacy ledger rows",
            len(report.legacy_unverified),
        )
    if report.mismatched or report.missing_source:
        logger.error(
            "Migration content integrity mismatch: mismatched=%s missing_source=%s",
            ", ".join(report.mismatched) or "none",
            ", ".join(report.missing_source) or "none",
        )


async def _unresolved_pending_migration_content_evidence(
    executor,
    migration_files: Collection[Path],
    report: MigrationContentIntegrityReport,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return evidence that still cannot admit a new migration application.

    Generic content reports intentionally keep historical discrepancies visible.
    A current attestation may satisfy admission only for its own reviewed name;
    it never changes the underlying report or treats unavailable source bytes as
    verified.
    """
    evidence_names = frozenset(report.mismatched) | frozenset(report.missing_source)
    if not evidence_names:
        return report.mismatched, report.missing_source

    from .reconciliation import (
        attest_known_historical_migration_reconciliations,
        known_historical_migration_reconciliation_names,
        known_historical_missing_source_reconciliation_names,
    )

    mismatched_candidate_names = (
        frozenset(report.mismatched)
        & known_historical_migration_reconciliation_names()
    )
    missing_source_candidate_names = (
        frozenset(report.missing_source)
        & known_historical_missing_source_reconciliation_names()
    )
    candidate_names = mismatched_candidate_names | missing_source_candidate_names
    if not candidate_names:
        return report.mismatched, report.missing_source

    try:
        attestations = await attest_known_historical_migration_reconciliations(
            executor,
            migration_files,
            candidate_names=candidate_names,
        )
    except Exception as exc:
        raise PendingMigrationContentIntegrityError(
            "Refusing to apply pending migrations because known historical "
            "migration evidence could not be attested for "
            f"{','.join(sorted(candidate_names))}"
        ) from exc

    attested_mismatched_names = frozenset(
        attestation.migration_name
        for attestation in attestations
        if attestation.migration_name in mismatched_candidate_names
        and attestation.status == "attested"
    )
    attested_missing_source_names = frozenset(
        attestation.migration_name
        for attestation in attestations
        if attestation.migration_name in missing_source_candidate_names
        and attestation.status == "attested"
    )
    return (
        tuple(
            name
            for name in report.mismatched
            if name not in attested_mismatched_names
        ),
        tuple(
            name
            for name in report.missing_source
            if name not in attested_missing_source_names
        ),
    )


# Cluster-wide advisory key serializing migration runs. Concurrent first
# starts (main app + standalone MCP servers, or two replicas) otherwise
# snapshot the pending set independently and race the check-then-insert in
# _record_migration against the unique schema_migrations.version constraint.
_MIGRATIONS_ADVISORY_LOCK_KEY = 0x41544C41  # "ATLA"
_MIGRATIONS_LOCK_POLL_SECONDS = 0.25
_ATOMIC_BOOKKEEPING_MARKER = "-- atlas: atomic-bookkeeping"


def _requires_atomic_bookkeeping(sql: str) -> bool:
    """Return whether SQL and its ledger identity must commit together.

    Most Atlas migrations remain deliberately autocommit: several use ``CREATE
    INDEX CONCURRENTLY``. A migration whose safety depends on its privilege or
    ownership changes committing with its ``schema_migrations`` record may opt
    into this narrow execution mode with the first non-empty SQL line. A direct
    insert into the ledger also requires it: otherwise its SQL can commit a
    legacy-looking row before the runner attaches the source digest.
    """
    first_line = next((line.strip() for line in sql.splitlines() if line.strip()), "")
    return (
        first_line == _ATOMIC_BOOKKEEPING_MARKER
        or _contains_executable_self_recording_insert(sql)
    )


def _executable_sql(sql: str, *, preserve_quoted_identifiers: bool = False) -> str:
    """Mask comments and literals so recognizers see executable SQL only."""
    code: list[str] = []
    i = 0
    single_quote = False
    double_quote = False
    line_comment = False
    block_comment = False
    dollar_quote: str | None = None

    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if line_comment:
            if ch == "\n":
                line_comment = False
                code.append(ch)
            else:
                code.append(" ")
            i += 1
            continue

        if block_comment:
            code.append(" ")
            if ch == "*" and nxt == "/":
                code.append(" ")
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if dollar_quote is not None:
            if sql.startswith(dollar_quote, i):
                code.extend(" " * len(dollar_quote))
                i += len(dollar_quote)
                dollar_quote = None
            else:
                code.append("\n" if ch == "\n" else " ")
                i += 1
            continue

        if single_quote:
            code.append("\n" if ch == "\n" else " ")
            if ch == "'" and nxt == "'":
                code.append(" ")
                i += 2
                continue
            if ch == "'":
                single_quote = False
            i += 1
            continue

        if double_quote:
            code.append(ch if preserve_quoted_identifiers else ("\n" if ch == "\n" else " "))
            if ch == '"' and nxt == '"':
                code.append(nxt if preserve_quoted_identifiers else " ")
                i += 2
                continue
            if ch == '"':
                double_quote = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            code.extend((" ", " "))
            line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            code.extend((" ", " "))
            block_comment = True
            i += 2
            continue

        if ch == "'":
            code.append(" ")
            single_quote = True
            i += 1
            continue

        if ch == '"':
            code.append(ch if preserve_quoted_identifiers else " ")
            double_quote = True
            i += 1
            continue

        if ch == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[i:])
            if match:
                dollar_quote = match.group(0)
                code.extend(" " * len(dollar_quote))
                i += len(dollar_quote)
                continue

        code.append(ch)
        i += 1

    return "".join(code)


def _contains_executable_concurrently(sql: str) -> bool:
    """Return whether SQL uses CONCURRENTLY outside comments and literals."""
    return bool(re.search(r"\bCONCURRENTLY\b", _executable_sql(sql), re.IGNORECASE))


_SELF_RECORDING_INSERT_RE = re.compile(
    r"(?i:\bINSERT\s+INTO\s+(?:ONLY\s+)?)"
    r"(?:(?:\"[A-Za-z_][A-Za-z0-9_$]*\"|[A-Za-z_][A-Za-z0-9_$]*)\s*\.\s*)?"
    r"(?:(?i:schema_migrations)|\"schema_migrations\")(?![A-Za-z0-9_$])",
)


def _contains_executable_self_recording_insert(sql: str) -> bool:
    """Return whether migration SQL directly inserts its ledger row.

    The existing explicit marker remains the escape hatch for a migration that
    performs equivalent bookkeeping through dynamic SQL. Direct inserts are
    recognized automatically so legacy self-recording files get the same
    rollback-safe SQL-plus-digest unit without rewriting their historical text.
    Unquoted target identifiers use PostgreSQL's case-folding rules; quoted
    targets must be the exact lowercase ledger identifier.
    """
    return bool(
        _SELF_RECORDING_INSERT_RE.search(
            _executable_sql(sql, preserve_quoted_identifiers=True)
        )
    )


def _split_sql_statements(sql: str) -> list[str]:
    """Split migration SQL into statements without entering a transaction.

    asyncpg sends one ``execute()`` string as one extended-query unit. PostgreSQL
    rejects ``CREATE/DROP INDEX CONCURRENTLY`` when those statements are batched
    with other SQL, so non-atomic migrations run statement-by-statement on the
    already-acquired autocommit connection. The splitter is intentionally small
    but respects the constructs used by packaged migrations: single/double
    quotes, line/block comments, and dollar-quoted PL/pgSQL bodies.
    """
    statements: list[str] = []
    current: list[str] = []
    i = 0
    single_quote = False
    double_quote = False
    line_comment = False
    block_comment = False
    dollar_quote: str | None = None

    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""

        if line_comment:
            current.append(ch)
            if ch == "\n":
                line_comment = False
            i += 1
            continue

        if block_comment:
            current.append(ch)
            if ch == "*" and nxt == "/":
                current.append(nxt)
                block_comment = False
                i += 2
            else:
                i += 1
            continue

        if dollar_quote is not None:
            if sql.startswith(dollar_quote, i):
                current.append(dollar_quote)
                i += len(dollar_quote)
                dollar_quote = None
            else:
                current.append(ch)
                i += 1
            continue

        if single_quote:
            current.append(ch)
            if ch == "'" and nxt == "'":
                current.append(nxt)
                i += 2
                continue
            if ch == "'":
                single_quote = False
            i += 1
            continue

        if double_quote:
            current.append(ch)
            if ch == '"' and nxt == '"':
                current.append(nxt)
                i += 2
                continue
            if ch == '"':
                double_quote = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            current.extend((ch, nxt))
            line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            current.extend((ch, nxt))
            block_comment = True
            i += 2
            continue

        if ch == "'":
            current.append(ch)
            single_quote = True
            i += 1
            continue

        if ch == '"':
            current.append(ch)
            double_quote = True
            i += 1
            continue

        if ch == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", sql[i:])
            if match:
                dollar_quote = match.group(0)
                current.append(dollar_quote)
                i += len(dollar_quote)
                continue

        current.append(ch)
        if ch == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        i += 1

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


async def run_migrations(
    pool,
    *,
    migrations_dir: Path | None = None,
    only: Collection[str] | None = None,
) -> None:
    """
    Run all pending migrations.

    Only runs migrations that haven't been applied yet.
    Tracks applied migrations in schema_migrations table.

    Concurrency: the whole run holds a SESSION-level advisory lock on one
    acquired connection, so simultaneous entrants (another replica, or the
    standalone MCP servers starting alongside the main app) queue at the lock
    and then re-snapshot, finding nothing pending.

    Two properties depend on doing it this way rather than with a transaction:

    * **Exactly one connection.** Every statement -- the lock, the bookkeeping,
      the migration SQL -- runs on the same acquired connection. A transaction
      that then reached back to the pool would deadlock a deployment configured
      with ``min_pool_size == max_pool_size == 1``.
    * **No open transaction.** Several migrations use ``CREATE INDEX
      CONCURRENTLY``, which Postgres refuses inside a transaction block. A
      session-level lock leaves the connection in autocommit, so they still run.

    Args:
        pool: The database pool to run migrations against
        migrations_dir: override for tests; defaults to the packaged dir
        only: restrict the run to these migration stems (e.g.
            ``{"350_scoped_mailbox_credentials"}``). A component that depends on
            one specific table can apply exactly that prerequisite instead of
            the whole chain. The full chain is NOT fresh-applicable -- 076+
            reference an out-of-band ``product_metadata`` table that no
            migration creates -- so a component which needs a later, otherwise
            self-contained migration cannot get it by running everything.
    """
    directory = migrations_dir if migrations_dir is not None else MIGRATIONS_DIR
    conn = await pool.acquire()
    try:
        # POLL with try-lock instead of blocking in pg_advisory_lock.
        #
        # A blocking waiter sits inside an open (implicit) transaction for as
        # long as it waits. CREATE INDEX CONCURRENTLY -- used by five packaged
        # migrations -- must wait for every concurrent transaction on the table
        # to drain before it can finish. So a blocked waiter and a holder
        # running CONCURRENTLY wait on each other and PostgreSQL kills one with
        # a deadlock. Two replicas starting together is enough to trigger it.
        #
        # try-lock returns immediately, so between polls this connection holds
        # no transaction and the holder's CONCURRENTLY build can complete.
        waited = 0.0
        while not await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", _MIGRATIONS_ADVISORY_LOCK_KEY
        ):
            await asyncio.sleep(_MIGRATIONS_LOCK_POLL_SECONDS)
            waited += _MIGRATIONS_LOCK_POLL_SECONDS
            if waited % 10 < _MIGRATIONS_LOCK_POLL_SECONDS:
                logger.info(
                    "Waiting %.0fs for another process to finish migrations",
                    waited,
                )
        try:
            await _ensure_migrations_table(conn)

            # Snapshot under the lock, so a queued entrant re-reads AFTER the
            # winner recorded its work.
            applied = await _get_applied_migrations(conn)

            migration_catalog = sorted(directory.glob("*.sql"))
            integrity_report = await migration_content_integrity_report(
                conn,
                migration_catalog,
            )
            _log_migration_content_integrity(integrity_report)
            migration_files = migration_catalog
            if only is not None:
                requested = set(only)
                migration_files = [
                    f for f in migration_files if f.stem in requested
                ]
                missing = requested - {f.stem for f in migration_files}
                if missing:
                    raise FileNotFoundError(
                        f"requested migrations not found: {sorted(missing)}"
                    )

            if not migration_files:
                logger.info("No migration files found")
                return

            pending = [f for f in migration_files if f.stem not in applied]

            if not pending:
                logger.debug("All %d migrations already applied", len(migration_files))
                return

            unresolved_mismatched, unresolved_missing_source = (
                await _unresolved_pending_migration_content_evidence(
                    conn,
                    migration_catalog,
                    report=integrity_report,
                )
            )
            if unresolved_mismatched or unresolved_missing_source:
                raise PendingMigrationContentIntegrityError(
                    "Refusing to apply pending migrations with unresolved "
                    "migration-content evidence: "
                    f"mismatched={','.join(unresolved_mismatched) or 'none'} "
                    f"missing_source={','.join(unresolved_missing_source) or 'none'} "
                    f"pending={','.join(migration_file.stem for migration_file in pending)}"
                )

            logger.info("Running %d pending migrations (of %d total)", len(pending), len(migration_files))

            for migration_file in pending:
                logger.info("Running migration: %s", migration_file.name)

                source = migration_file.read_bytes()
                content_sha256 = _migration_content_sha256(source)
                sql = source.decode("utf-8")

                try:
                    if _requires_atomic_bookkeeping(sql):
                        # A marked migration may not contain concurrently-run
                        # DDL. Its database effects and ledger row are one
                        # rollback-safe unit, closing the otherwise possible
                        # crash window between migration SQL and bookkeeping.
                        if _contains_executable_concurrently(sql):
                            raise RuntimeError(
                                "atomic-bookkeeping migration cannot use CONCURRENTLY"
                            )
                        async with conn.transaction():
                            await conn.execute(sql)
                            await _record_migration(
                                conn,
                                migration_file.name,
                                content_sha256,
                                fill_newly_self_recorded_digest=True,
                            )
                    elif _contains_executable_concurrently(sql):
                        for statement in _split_sql_statements(sql):
                            await conn.execute(statement)
                        await _record_migration(
                            conn,
                            migration_file.name,
                            content_sha256,
                            fill_newly_self_recorded_digest=True,
                        )
                    else:
                        await conn.execute(sql)
                        await _record_migration(
                            conn,
                            migration_file.name,
                            content_sha256,
                            fill_newly_self_recorded_digest=True,
                        )
                    logger.info("Migration %s completed successfully", migration_file.name)
                except Exception as e:
                    logger.error("Migration %s failed: %s", migration_file.name, e)
                    raise
        finally:
            await conn.execute(
                "SELECT pg_advisory_unlock($1)", _MIGRATIONS_ADVISORY_LOCK_KEY
            )
    finally:
        await pool.release(conn)


async def check_schema_exists(pool) -> bool:
    """
    Check if the database schema has been initialized.

    Returns:
        True if schema exists, False otherwise
    """
    try:
        result = await pool.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'sessions'
            )
            """
        )
        return result
    except Exception:
        return False
