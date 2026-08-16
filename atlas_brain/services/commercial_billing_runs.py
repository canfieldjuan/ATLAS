"""Durable, pre-approval EOM commercial billing-run snapshots.

The companion candidate service is deliberately pure.  This service is the
smallest separate write boundary that can retain the exact preview an operator
reviewed and later prove whether its source evidence still matches.  It never
creates invoices, PDFs, Gmail drafts, email, service markers, or sent state.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional
from uuid import UUID, uuid4

import asyncpg

from ..storage.database import DatabasePool, get_db_pool
from ..storage.exceptions import DatabaseOperationError, DatabaseUnavailableError
from .commercial_billing_candidates import (
    CommercialBillingCandidateService,
    CommercialBillingCandidatesUnavailableError,
    CommercialBillingCandidatesValidationError,
    get_commercial_billing_candidate_service,
    parse_billing_period,
)


_RUN_SOURCE = "eom_admin"
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_CANDIDATE_KEY_LENGTH = 512
_MAX_CANDIDATES = 500
_MAX_REASON_LENGTH = 1000
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_DECISIONS = frozenset({"included", "excluded"})
_DATABASE_UNAVAILABLE_ERRORS = (
    DatabaseOperationError,
    DatabaseUnavailableError,
    asyncpg.PostgresConnectionError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
    asyncpg.AdminShutdownError,
    asyncpg.CrashShutdownError,
    asyncpg.UndefinedTableError,
    asyncpg.UndefinedColumnError,
    asyncpg.InvalidSchemaNameError,
    asyncpg.InsufficientPrivilegeError,
    asyncpg.InvalidAuthorizationSpecificationError,
)


class CommercialBillingRunError(Exception):
    """Base error for the durable commercial billing-run contract."""

    code = "commercial_billing_run_error"


class CommercialBillingRunValidationError(CommercialBillingRunError):
    """The request or generated preview cannot safely form a durable run."""

    code = "invalid_commercial_billing_run"


class CommercialBillingRunNotFoundError(CommercialBillingRunError):
    """The requested durable review run does not exist."""

    code = "commercial_billing_run_not_found"


class CommercialBillingRunConflictError(CommercialBillingRunError):
    """One operation key was reused with different durable intent."""

    code = "commercial_billing_run_idempotency_conflict"


class CommercialBillingRunUnavailableError(CommercialBillingRunError):
    """The durable review store cannot safely serve this request."""

    code = "commercial_billing_runs_unavailable"


@dataclass(frozen=True)
class _SnapshotCandidate:
    candidate_key: str
    source_fingerprint: str
    snapshot: dict[str, Any]
    display_order: int


@dataclass(frozen=True)
class _NormalizedPreview:
    billing_period: str
    contract_version: int
    calendar_id: str | None
    candidates: tuple[_SnapshotCandidate, ...]
    snapshot_fingerprint: str


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CommercialBillingRunUnavailableError(
            "Commercial billing candidate evidence is not JSON-safe"
        ) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing snapshot evidence is invalid"
            ) from exc
    if not isinstance(value, Mapping):
        raise CommercialBillingRunUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive boundary
        raise CommercialBillingRunUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        ) from exc
    if not isinstance(normalized, dict):  # pragma: no cover - json object invariant
        raise CommercialBillingRunUnavailableError(
            "Commercial billing snapshot evidence is invalid"
        )
    return normalized


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _reject_database_unsafe_text(value: str, field: str) -> str:
    """Reject text asyncpg/PostgreSQL cannot represent before a transaction."""

    if "\x00" in value:
        raise CommercialBillingRunValidationError(
            f"{field} contains a database-unsafe NUL character"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CommercialBillingRunValidationError(
            f"{field} contains UTF-8-unencodable text"
        ) from exc
    return value


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _validate_idempotency_key(value: str) -> str:
    if not isinstance(value, str):
        raise CommercialBillingRunValidationError("Idempotency key is required")
    key = _reject_database_unsafe_text(value, "Idempotency key").strip()
    if not key or len(key) > _MAX_IDEMPOTENCY_KEY_LENGTH:
        raise CommercialBillingRunValidationError(
            "Idempotency key must contain 1 to 128 characters"
        )
    return key


def _validate_candidate_key(value: Any) -> str:
    candidate_key = _text(value)
    if candidate_key is None:
        raise CommercialBillingRunValidationError("Candidate key is required")
    normalized = _reject_database_unsafe_text(candidate_key, "Candidate key").strip()
    if not normalized or len(normalized) > _MAX_CANDIDATE_KEY_LENGTH:
        raise CommercialBillingRunValidationError(
            "Candidate key must contain 1 to 512 characters"
        )
    return normalized


def _validate_source_fingerprint(value: Any) -> str:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise CommercialBillingRunValidationError("Source fingerprint is invalid")
    return value


def _required_text(value: Any, field: str, *, limit: int) -> str:
    text = _text(value)
    if text is None:
        raise CommercialBillingRunValidationError(f"{field} is required")
    normalized = _reject_database_unsafe_text(text, field).strip()
    if not normalized or len(normalized) > limit:
        raise CommercialBillingRunValidationError(
            f"{field} must contain 1 to {limit} characters"
        )
    return normalized


async def lock_commercial_billing_candidate_identity(
    conn: Any,
    *,
    candidate_key: str,
    source_fingerprint: str,
) -> None:
    """Serialize candidate review with approval, including rolling upgrades.

    The namespace deliberately matches the deployed approval lock.  Keeping it
    stable lets the new review writer serialize against an older approval
    process during a rolling application deployment as well as current code.
    """

    await conn.fetchval(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        "commercial-billing-approval:"
        f"candidate:{candidate_key}:{source_fingerprint}",
    )


async def lock_commercial_billing_run_candidate(
    conn: Any,
    *,
    billing_run_id: UUID,
    candidate_key: str,
) -> Any | None:
    """Lock one immutable review snapshot before its decision or approval."""

    return await conn.fetchrow(
        """
        SELECT run.billing_period, candidate.source_fingerprint, candidate.snapshot
        FROM commercial_billing_runs AS run
        JOIN commercial_billing_run_candidates AS candidate
          ON candidate.billing_run_id = run.id
        WHERE run.id = $1 AND candidate.candidate_key = $2
        FOR UPDATE OF candidate
        """,
        billing_run_id,
        candidate_key,
    )


def _review_decision_view(row: Any | None) -> dict[str, Any]:
    if row is None:
        return {
            "decidedAt": None,
            "decidedBy": None,
            "decision": "included",
            "isExplicit": False,
            "reason": None,
            "revision": 0,
        }
    return {
        "decidedAt": _timestamp(row["decided_at"]),
        "decidedBy": row["decided_by"],
        "decision": row["decision"],
        "id": str(row["id"]),
        "isExplicit": True,
        "reason": row["reason"],
        "revision": row["revision"],
    }


def _normalize_preview(preview: Any, *, billing_period: str) -> _NormalizedPreview:
    if not isinstance(preview, Mapping):
        raise CommercialBillingRunUnavailableError(
            "Commercial billing candidate evidence is invalid"
        )
    if preview.get("billingPeriod") != billing_period:
        raise CommercialBillingRunUnavailableError(
            "Commercial billing candidate period does not match the requested run"
        )
    contract_version = preview.get("contractVersion")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version <= 0
    ):
        raise CommercialBillingRunUnavailableError(
            "Commercial billing candidate contract version is invalid"
        )
    calendar_id = preview.get("calendarId")
    if calendar_id is not None and not isinstance(calendar_id, str):
        raise CommercialBillingRunUnavailableError(
            "Commercial billing calendar evidence is invalid"
        )
    raw_candidates = preview.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) > _MAX_CANDIDATES:
        raise CommercialBillingRunUnavailableError(
            "Commercial billing candidate evidence is invalid"
        )

    candidates: list[_SnapshotCandidate] = []
    seen_keys: set[str] = set()
    for display_order, raw_candidate in enumerate(raw_candidates):
        snapshot = _json_object(raw_candidate)
        candidate_key = _text(snapshot.get("candidateKey"))
        source_fingerprint = _text(snapshot.get("sourceFingerprint"))
        if (
            candidate_key is None
            or candidate_key != candidate_key.strip()
            or not candidate_key
            or len(candidate_key) > _MAX_CANDIDATE_KEY_LENGTH
            or source_fingerprint is None
            or _FINGERPRINT_PATTERN.fullmatch(source_fingerprint) is None
            or snapshot.get("billingPeriod") != billing_period
            or not isinstance(snapshot.get("blockers"), list)
        ):
            raise CommercialBillingRunUnavailableError(
                "Commercial billing candidate evidence is invalid"
            )
        if candidate_key in seen_keys:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing candidate keys must be unique"
            )
        seen_keys.add(candidate_key)
        candidates.append(
            _SnapshotCandidate(
                candidate_key=candidate_key,
                source_fingerprint=source_fingerprint,
                snapshot=snapshot,
                display_order=display_order,
            )
        )

    snapshot_fingerprint = _fingerprint(
        {
            "billingPeriod": billing_period,
            "calendarId": calendar_id,
            "contractVersion": contract_version,
            "candidates": [
                {
                    "candidateKey": candidate.candidate_key,
                    "sourceFingerprint": candidate.source_fingerprint,
                }
                for candidate in sorted(candidates, key=lambda item: item.candidate_key)
            ],
        }
    )
    return _NormalizedPreview(
        billing_period=billing_period,
        contract_version=contract_version,
        calendar_id=calendar_id,
        candidates=tuple(candidates),
        snapshot_fingerprint=snapshot_fingerprint,
    )


class CommercialBillingRunService:
    """Own durable snapshots and read-only freshness reconciliation."""

    def __init__(
        self,
        *,
        pool: Optional[DatabasePool] = None,
        candidate_service_loader: Callable[[], CommercialBillingCandidateService] = (
            get_commercial_billing_candidate_service
        ),
    ) -> None:
        self._configured_pool = pool
        self._candidate_service_loader = candidate_service_loader

    @property
    def pool(self) -> DatabasePool:
        pool = self._configured_pool or get_db_pool()
        if not pool.is_initialized:
            raise CommercialBillingRunUnavailableError("Commercial billing database unavailable")
        return pool

    async def create_run(
        self,
        *,
        billing_period: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Snapshot one pure preview; matching retries return the original run."""

        period = parse_billing_period(billing_period).label
        key = _validate_idempotency_key(idempotency_key)
        actor_text = _text(actor)
        if actor_text is None or not actor_text.strip() or len(actor_text) > 128:
            raise CommercialBillingRunValidationError("Authenticated actor is required")
        request_fingerprint = _fingerprint({"billingPeriod": period})

        try:
            async with self.pool.transaction() as conn:
                await self._lock_operation_key(conn, key)
                existing = await self._find_by_operation_key(conn, key)
                if existing is not None:
                    self._assert_request_fingerprint(existing, request_fingerprint)
                    return {
                        "billingRun": await self._run_view(conn, existing["id"]),
                        "replayed": True,
                    }

            preview = await self._load_preview(period)

            async with self.pool.transaction() as conn:
                await self._lock_operation_key(conn, key)
                existing = await self._find_by_operation_key(conn, key)
                if existing is not None:
                    self._assert_request_fingerprint(existing, request_fingerprint)
                    return {
                        "billingRun": await self._run_view(conn, existing["id"]),
                        "replayed": True,
                    }

                run_id = uuid4()
                now = datetime.now(timezone.utc)
                created = await conn.fetchrow(
                    """
                    INSERT INTO commercial_billing_runs (
                        id, billing_period, calendar_id, state,
                        candidate_contract_version, snapshot_fingerprint, source,
                        idempotency_key, request_fingerprint, created_by,
                        created_at, updated_at
                    )
                    VALUES ($1, $2, $3, 'draft', $4, $5, $6, $7, $8, $9, $10, $10)
                    ON CONFLICT (source, idempotency_key) DO NOTHING
                    RETURNING id
                    """,
                    run_id,
                    preview.billing_period,
                    preview.calendar_id,
                    preview.contract_version,
                    preview.snapshot_fingerprint,
                    _RUN_SOURCE,
                    key,
                    request_fingerprint,
                    actor_text.strip(),
                    now,
                )
                if created is None:
                    existing = await self._find_by_operation_key(conn, key)
                    if existing is None:
                        raise CommercialBillingRunConflictError(
                            "Billing-run idempotency conflict could not be reconciled"
                        )
                    self._assert_request_fingerprint(existing, request_fingerprint)
                    return {
                        "billingRun": await self._run_view(conn, existing["id"]),
                        "replayed": True,
                    }
                for candidate in preview.candidates:
                    await conn.execute(
                        """
                        INSERT INTO commercial_billing_run_candidates (
                            id, billing_run_id, candidate_key, source_fingerprint,
                            display_order, snapshot, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                        """,
                        uuid4(),
                        run_id,
                        candidate.candidate_key,
                        candidate.source_fingerprint,
                        candidate.display_order,
                        _canonical_json(candidate.snapshot),
                        now,
                    )
                return {
                    "billingRun": await self._run_view(conn, run_id),
                    "replayed": False,
                }
        except (
            CommercialBillingRunError,
            CommercialBillingCandidatesUnavailableError,
            CommercialBillingCandidatesValidationError,
        ):
            raise
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def set_candidate_review_decision(
        self,
        *,
        billing_run_id: UUID,
        candidate_key: str,
        expected_source_fingerprint: str,
        decision: str,
        reason: str,
        idempotency_key: str,
        actor: str,
    ) -> dict[str, Any]:
        """Append one include/exclude decision without creating financial state."""

        if not isinstance(billing_run_id, UUID):
            raise CommercialBillingRunValidationError("Billing run id is invalid")
        selected = _validate_candidate_key(candidate_key)
        fingerprint = _validate_source_fingerprint(expected_source_fingerprint)
        if not isinstance(decision, str) or decision not in _REVIEW_DECISIONS:
            raise CommercialBillingRunValidationError(
                "Review decision must be included or excluded"
            )
        decision_reason = _required_text(reason, "Reason", limit=_MAX_REASON_LENGTH)
        key = _validate_idempotency_key(idempotency_key)
        decided_by = _required_text(actor, "Authenticated actor", limit=128)
        request_fingerprint = _fingerprint(
            {
                "billingRunId": str(billing_run_id),
                "candidateKey": selected,
                "sourceFingerprint": fingerprint,
                "decision": decision,
                "reason": decision_reason,
            }
        )

        try:
            async with self.pool.transaction() as conn:
                await self._lock_review_decision_operation_key(conn, key)
                existing = await self._find_review_decision_by_operation_key(conn, key)
                if existing is not None:
                    self._assert_review_decision_request(existing, request_fingerprint)
                    return {
                        "reviewDecision": _review_decision_view(existing),
                        "replayed": True,
                    }

                await lock_commercial_billing_candidate_identity(
                    conn,
                    candidate_key=selected,
                    source_fingerprint=fingerprint,
                )
                stored = await lock_commercial_billing_run_candidate(
                    conn,
                    billing_run_id=billing_run_id,
                    candidate_key=selected,
                )
                if stored is None:
                    raise CommercialBillingRunNotFoundError(
                        "Commercial billing candidate not found"
                    )
                if stored["source_fingerprint"] != fingerprint:
                    raise CommercialBillingRunConflictError(
                        "Commercial billing candidate fingerprint does not match review evidence"
                    )
                approved = await self._find_approval_for_candidate_identity(
                    conn,
                    candidate_key=selected,
                    source_fingerprint=fingerprint,
                )
                if approved is not None:
                    raise CommercialBillingRunConflictError(
                        "Approved commercial billing candidates cannot be reviewed again"
                    )

                revision = await self._next_review_decision_revision(
                    conn,
                    candidate_key=selected,
                    source_fingerprint=fingerprint,
                )
                created = await conn.fetchrow(
                    """
                    INSERT INTO commercial_billing_candidate_review_decisions (
                        id, billing_run_id, candidate_key, source_fingerprint,
                        revision, decision, reason, source, idempotency_key,
                        request_fingerprint, decided_by, decided_at, created_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $12
                    )
                    RETURNING id, billing_run_id, candidate_key, source_fingerprint,
                              revision, decision, reason, request_fingerprint,
                              decided_by, decided_at
                    """,
                    uuid4(),
                    billing_run_id,
                    selected,
                    fingerprint,
                    revision,
                    decision,
                    decision_reason,
                    _RUN_SOURCE,
                    key,
                    request_fingerprint,
                    decided_by,
                    datetime.now(timezone.utc),
                )
                if created is None:  # pragma: no cover - INSERT RETURNING invariant
                    raise CommercialBillingRunUnavailableError(
                        "Commercial billing review decision could not be reconciled"
                    )
                return {
                    "reviewDecision": _review_decision_view(created),
                    "replayed": False,
                }
        except CommercialBillingRunError:
            raise
        except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError) as exc:
            raise CommercialBillingRunConflictError(
                "Commercial billing review decision could not be reconciled"
            ) from exc
        except asyncpg.PostgresError as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def list_runs(
        self,
        *,
        billing_period: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return a bounded list of durable draft-run summaries."""

        period = (
            parse_billing_period(billing_period).label
            if billing_period is not None
            else None
        )
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise CommercialBillingRunValidationError("limit must be between 1 and 100")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise CommercialBillingRunValidationError("offset must be nonnegative")
        try:
            rows = await self.pool.fetch(
                """
                SELECT
                    run.id,
                    run.billing_period,
                    run.calendar_id,
                    run.state,
                    run.candidate_contract_version,
                    run.snapshot_fingerprint,
                    run.created_by,
                    run.created_at,
                    run.updated_at,
                    COUNT(candidate.id)::int AS candidate_count,
                    COALESCE(
                        SUM(
                            CASE
                                WHEN jsonb_array_length(candidate.snapshot -> 'blockers') > 0
                                THEN 1 ELSE 0
                            END
                        ),
                        0
                    )::int AS blocked_candidate_count
                FROM commercial_billing_runs AS run
                LEFT JOIN commercial_billing_run_candidates AS candidate
                    ON candidate.billing_run_id = run.id
                WHERE ($1::varchar IS NULL OR run.billing_period = $1)
                GROUP BY run.id
                ORDER BY run.created_at DESC, run.id DESC
                LIMIT $2 OFFSET $3
                """,
                period,
                limit,
                offset,
            )
            return {
                "items": [self._run_summary(row) for row in rows],
                "limit": limit,
                "offset": offset,
            }
        except CommercialBillingRunError:
            raise
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def get_run(self, run_id: UUID) -> dict[str, Any]:
        """Return one immutable snapshot with all stored candidate evidence."""

        try:
            return await self._run_view(self.pool, run_id)
        except CommercialBillingRunError:
            raise
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

    async def reconcile_run(self, run_id: UUID) -> dict[str, Any]:
        """Compare stored evidence with a fresh pure preview without writing."""

        try:
            stored = await self._run_view(self.pool, run_id)
            current = await self._load_preview(stored["billingPeriod"])
        except (
            CommercialBillingRunError,
            CommercialBillingCandidatesUnavailableError,
            CommercialBillingCandidatesValidationError,
        ):
            raise
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing database unavailable"
            ) from exc

        current_by_key = {
            candidate.candidate_key: candidate for candidate in current.candidates
        }
        stored_by_key = {
            candidate["candidateKey"]: candidate for candidate in stored["candidates"]
        }
        changes: list[dict[str, Any]] = []
        for stored_candidate in stored["candidates"]:
            candidate_key = stored_candidate["candidateKey"]
            current_candidate = current_by_key.pop(candidate_key, None)
            if current_candidate is None:
                status = "missing"
                current_fingerprint = None
            elif current_candidate.source_fingerprint == stored_candidate["sourceFingerprint"]:
                status = "unchanged"
                current_fingerprint = current_candidate.source_fingerprint
            else:
                status = "changed"
                current_fingerprint = current_candidate.source_fingerprint
            changes.append(
                {
                    "candidateKey": candidate_key,
                    "currentSourceFingerprint": current_fingerprint,
                    "status": status,
                    "storedSourceFingerprint": stored_candidate["sourceFingerprint"],
                }
            )
        for candidate in current.candidates:
            if candidate.candidate_key not in stored_by_key:
                changes.append(
                    {
                        "candidateKey": candidate.candidate_key,
                        "currentSourceFingerprint": candidate.source_fingerprint,
                        "status": "new",
                        "storedSourceFingerprint": None,
                    }
                )
        is_stale = (
            current.snapshot_fingerprint != stored["snapshotFingerprint"]
            or any(change["status"] != "unchanged" for change in changes)
        )
        return {
            "billingPeriod": stored["billingPeriod"],
            "billingRunId": stored["id"],
            "candidateChanges": changes,
            "currentSnapshotFingerprint": current.snapshot_fingerprint,
            "isStale": is_stale,
            "snapshotFingerprint": stored["snapshotFingerprint"],
        }

    async def _load_preview(self, billing_period: str) -> _NormalizedPreview:
        try:
            candidate_service = self._candidate_service_loader()
            preview = await candidate_service.preview(billing_period=billing_period)
        except (CommercialBillingCandidatesUnavailableError, CommercialBillingCandidatesValidationError):
            raise
        except _DATABASE_UNAVAILABLE_ERRORS as exc:
            raise CommercialBillingCandidatesUnavailableError(
                "Commercial billing candidate evidence is unavailable"
            ) from exc
        except Exception as exc:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing candidate evidence is unavailable"
            ) from exc
        return _normalize_preview(preview, billing_period=billing_period)

    @staticmethod
    async def _lock_operation_key(conn: Any, key: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-run-create:{_RUN_SOURCE}:{key}",
        )

    @staticmethod
    async def _lock_review_decision_operation_key(conn: Any, key: str) -> None:
        await conn.fetchval(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"commercial-billing-run-review-decision:{_RUN_SOURCE}:{key}",
        )

    @staticmethod
    async def _find_by_operation_key(conn: Any, key: str) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT id, request_fingerprint
            FROM commercial_billing_runs
            WHERE source = $1 AND idempotency_key = $2
            FOR UPDATE
            """,
            _RUN_SOURCE,
            key,
        )

    @staticmethod
    def _assert_request_fingerprint(row: Any, expected: str) -> None:
        if row["request_fingerprint"] != expected:
            raise CommercialBillingRunConflictError(
                "Idempotency key was already used with a different billing period"
            )

    @staticmethod
    async def _find_review_decision_by_operation_key(conn: Any, key: str) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT id, billing_run_id, candidate_key, source_fingerprint,
                   revision, decision, reason, request_fingerprint,
                   decided_by, decided_at
            FROM commercial_billing_candidate_review_decisions
            WHERE source = $1 AND idempotency_key = $2
            FOR UPDATE
            """,
            _RUN_SOURCE,
            key,
        )

    @staticmethod
    def _assert_review_decision_request(row: Any, expected: str) -> None:
        if row["request_fingerprint"] != expected:
            raise CommercialBillingRunConflictError(
                "Idempotency key was already used with a different review decision"
            )

    @staticmethod
    async def _find_approval_for_candidate_identity(
        conn: Any,
        *,
        candidate_key: str,
        source_fingerprint: str,
    ) -> Any | None:
        return await conn.fetchrow(
            """
            SELECT id
            FROM commercial_billing_candidate_approvals
            WHERE candidate_key = $1 AND source_fingerprint = $2
            """,
            candidate_key,
            source_fingerprint,
        )

    @staticmethod
    async def _next_review_decision_revision(
        conn: Any,
        *,
        candidate_key: str,
        source_fingerprint: str,
    ) -> int:
        latest = await conn.fetchval(
            """
            SELECT COALESCE(MAX(revision), 0)
            FROM commercial_billing_candidate_review_decisions
            WHERE candidate_key = $1
              AND source_fingerprint = $2
            """,
            candidate_key,
            source_fingerprint,
        )
        if isinstance(latest, bool) or not isinstance(latest, int) or latest < 0:
            raise CommercialBillingRunUnavailableError(
                "Commercial billing review decision history is invalid"
            )
        return latest + 1

    async def _run_view(self, executor: Any, run_id: UUID) -> dict[str, Any]:
        row = await executor.fetchrow(
            """
            SELECT
                id,
                billing_period,
                calendar_id,
                state,
                candidate_contract_version,
                snapshot_fingerprint,
                created_by,
                created_at,
                updated_at
            FROM commercial_billing_runs
            WHERE id = $1
            """,
            run_id,
        )
        if row is None:
            raise CommercialBillingRunNotFoundError("Commercial billing run not found")
        candidate_rows = await executor.fetch(
            """
            SELECT candidate_key, source_fingerprint, display_order, snapshot
            FROM commercial_billing_run_candidates
            WHERE billing_run_id = $1
            ORDER BY display_order ASC, candidate_key ASC
            """,
            run_id,
        )
        decision_rows = await executor.fetch(
            """
            SELECT DISTINCT ON (decision.candidate_key, decision.source_fingerprint)
                   decision.id, decision.candidate_key, decision.source_fingerprint,
                   decision.revision, decision.decision, decision.reason,
                   decision.decided_by, decision.decided_at
            FROM commercial_billing_candidate_review_decisions AS decision
            JOIN commercial_billing_run_candidates AS candidate
              ON candidate.candidate_key = decision.candidate_key
             AND candidate.source_fingerprint = decision.source_fingerprint
            WHERE candidate.billing_run_id = $1
            ORDER BY decision.candidate_key, decision.source_fingerprint,
                     decision.revision DESC
            """,
            run_id,
        )
        latest_decision_by_candidate = {
            (decision["candidate_key"], decision["source_fingerprint"]): decision
            for decision in decision_rows
        }
        candidates = [_json_object(candidate["snapshot"]) for candidate in candidate_rows]
        for candidate, row_candidate in zip(candidates, candidate_rows):
            if (
                candidate.get("candidateKey") != row_candidate["candidate_key"]
                or candidate.get("sourceFingerprint")
                != row_candidate["source_fingerprint"]
            ):
                raise CommercialBillingRunUnavailableError(
                    "Commercial billing snapshot evidence is invalid"
                )
            candidate["reviewDecision"] = _review_decision_view(
                latest_decision_by_candidate.get(
                    (row_candidate["candidate_key"], row_candidate["source_fingerprint"])
                )
            )
        return {
            "billingPeriod": row["billing_period"],
            "calendarId": row["calendar_id"],
            "candidateContractVersion": row["candidate_contract_version"],
            "candidates": candidates,
            "createdAt": _timestamp(row["created_at"]),
            "createdBy": row["created_by"],
            "id": str(row["id"]),
            "snapshotFingerprint": row["snapshot_fingerprint"],
            "state": row["state"],
            "summary": {
                "blockedCandidateCount": sum(
                    1 for candidate in candidates if candidate["blockers"]
                ),
                "candidateCount": len(candidates),
            },
            "updatedAt": _timestamp(row["updated_at"]),
        }

    @staticmethod
    def _run_summary(row: Any) -> dict[str, Any]:
        return {
            "billingPeriod": row["billing_period"],
            "calendarId": row["calendar_id"],
            "candidateContractVersion": row["candidate_contract_version"],
            "createdAt": _timestamp(row["created_at"]),
            "createdBy": row["created_by"],
            "id": str(row["id"]),
            "snapshotFingerprint": row["snapshot_fingerprint"],
            "state": row["state"],
            "summary": {
                "blockedCandidateCount": row["blocked_candidate_count"],
                "candidateCount": row["candidate_count"],
            },
            "updatedAt": _timestamp(row["updated_at"]),
        }


def get_commercial_billing_run_service() -> CommercialBillingRunService:
    """Build a fresh request service around the shared database pool."""

    return CommercialBillingRunService()


__all__ = [
    "CommercialBillingRunConflictError",
    "CommercialBillingRunError",
    "CommercialBillingRunNotFoundError",
    "CommercialBillingRunService",
    "CommercialBillingRunUnavailableError",
    "CommercialBillingRunValidationError",
    "get_commercial_billing_run_service",
    "lock_commercial_billing_candidate_identity",
    "lock_commercial_billing_run_candidate",
]
