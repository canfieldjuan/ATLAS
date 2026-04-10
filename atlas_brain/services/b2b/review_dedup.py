from __future__ import annotations

import re
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from typing import Any


def _coerce_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or value is None:
        numeric = default
    elif isinstance(value, int):
        numeric = value
    elif isinstance(value, float):
        if value != value:
            numeric = default
        else:
            numeric = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            numeric = default
        else:
            try:
                numeric = int(text)
            except ValueError:
                try:
                    numeric = int(float(text))
                except ValueError:
                    numeric = default
    else:
        numeric = default
    if minimum is not None and numeric < minimum:
        return minimum
    return numeric


def _coerce_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or value is None:
        numeric = default
    elif isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            numeric = default
        else:
            try:
                numeric = float(text)
            except ValueError:
                numeric = default
    else:
        numeric = default
    if numeric != numeric:
        numeric = default
    if minimum is not None and numeric < minimum:
        return minimum
    return numeric


def normalize_review_text_for_hash(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text.lower()


def make_review_text_payload(
    summary: Any,
    review_text: Any,
    pros: Any = None,
    cons: Any = None,
) -> str:
    parts = [
        normalize_review_text_for_hash(summary),
        normalize_review_text_for_hash(review_text),
        normalize_review_text_for_hash(pros),
        normalize_review_text_for_hash(cons),
    ]
    return "\n".join(part for part in parts if part)


def normalize_review_date_key(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc).date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).date().isoformat()
    except ValueError:
        pass
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def normalize_reviewer_key(value: Any) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    return normalized or None


def normalize_reviewer_stem_key(value: Any, *, stem_length: int) -> str | None:
    reviewer_key = normalize_reviewer_key(value)
    if not reviewer_key:
        return None
    return reviewer_key[: _coerce_int(stem_length, 5, minimum=1)]


def normalize_rating_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric != numeric:
        return ""
    return f"{numeric:.1f}"


def make_cross_source_identity_key(
    vendor_name: str,
    reviewer_name: Any,
    reviewed_at: Any,
    rating: Any = None,
) -> str | None:
    reviewer_key = normalize_reviewer_key(reviewer_name)
    reviewed_key = normalize_review_date_key(reviewed_at)
    vendor_key = str(vendor_name or "").strip().lower()
    if not vendor_key or not reviewer_key or not reviewed_key:
        return None
    rating_key = normalize_rating_key(rating)
    return f"{vendor_key}|{reviewer_key}|{reviewed_key}|{rating_key}"


def review_text_similarity(left_payload: str, right_payload: str) -> float:
    if not left_payload or not right_payload:
        return 0.0
    return float(SequenceMatcher(None, left_payload, right_payload).ratio())


def _normalize_imported_at_rank(raw: Any) -> float:
    if raw is None:
        return float("inf")
    if isinstance(raw, datetime):
        value = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return float("inf")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return float("inf")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


async def load_cross_source_review_candidates(
    pool,
    *,
    vendor_name: str,
    content_hash: str | None,
    identity_key: str | None,
    reviewer_stem: str | None,
    reviewed_date: str | None,
    rating: Any = None,
    max_candidates: int,
    reviewer_stem_length: int,
    review_date_tolerance_days: int,
    rating_tolerance: float,
) -> list[dict[str, Any]]:
    reviewer_stem_length = _coerce_int(reviewer_stem_length, 5, minimum=1)
    review_date_tolerance_days = _coerce_int(review_date_tolerance_days, 1, minimum=0)
    rating_tolerance = _coerce_float(rating_tolerance, 1.0, minimum=0.0)
    max_candidates = _coerce_int(max_candidates, 20, minimum=1)
    if not content_hash and not identity_key and not (reviewer_stem and reviewed_date):
        return []
    rows = await pool.fetch(
        """
        SELECT id, source, source_review_id, reviewer_name, reviewed_at, rating,
               imported_at, enrichment_status, source_weight,
               cross_source_content_hash,
               cross_source_identity_key,
               summary, review_text, pros, cons
        FROM b2b_reviews
        WHERE vendor_name = $1
          AND duplicate_of_review_id IS NULL
          AND (
                ($2::text IS NOT NULL AND cross_source_content_hash = $2)
             OR ($3::text IS NOT NULL AND cross_source_identity_key = $3)
             OR (
                    $4::text IS NOT NULL
                AND $5::date IS NOT NULL
                AND reviewed_at IS NOT NULL
                AND ABS((reviewed_at AT TIME ZONE 'UTC')::date - $5::date) <= $6
                AND (
                    $7::numeric IS NULL
                    OR rating IS NULL
                    OR ABS(COALESCE(rating, 0)::numeric - $7::numeric) <= $8
                )
                AND LEFT(
                    regexp_replace(lower(COALESCE(reviewer_name, '')), '[^a-z0-9]+', '', 'g'),
                    $9
                ) = $4
             )
          )
        ORDER BY imported_at ASC, id ASC
        LIMIT $10
        """,
        vendor_name,
        content_hash,
        identity_key,
        reviewer_stem,
        reviewed_date,
        review_date_tolerance_days,
        rating,
        rating_tolerance,
        reviewer_stem_length,
        max_candidates,
    )
    return [dict(row) for row in rows]


def choose_cross_source_duplicate_survivor(
    *,
    candidates: list[dict[str, Any]],
    incoming_source: str,
    incoming_content_hash: str | None,
    incoming_identity_key: str | None,
    incoming_payload: str,
    similarity_threshold: float,
    incoming_reviewer_name: Any = None,
    incoming_reviewed_at: Any = None,
    incoming_rating: Any = None,
    loose_similarity_threshold: float = 0.9,
    reviewer_stem_length: int = 5,
    review_date_tolerance_days: int = 1,
    rating_tolerance: float = 1.0,
    require_source_difference: bool = True,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    similarity_threshold = _coerce_float(similarity_threshold, 0.82, minimum=0.0)
    loose_similarity_threshold = _coerce_float(loose_similarity_threshold, 0.9, minimum=0.0)
    reviewer_stem_length = _coerce_int(reviewer_stem_length, 5, minimum=1)
    review_date_tolerance_days = _coerce_int(review_date_tolerance_days, 1, minimum=0)
    rating_tolerance = _coerce_float(rating_tolerance, 1.0, minimum=0.0)
    ranked: list[tuple[tuple[int, int, float, str], dict[str, Any], str, dict[str, Any]]] = []
    for candidate in candidates:
        candidate_source = str(candidate.get("source") or "").strip().lower()
        if require_source_difference and candidate_source == str(incoming_source or "").strip().lower():
            continue
        candidate_hash = str(candidate.get("cross_source_content_hash") or "").strip() or None
        candidate_identity = str(candidate.get("cross_source_identity_key") or "").strip() or None
        candidate_payload = make_review_text_payload(
            candidate.get("summary"),
            candidate.get("review_text"),
            candidate.get("pros"),
            candidate.get("cons"),
        )
        exact_hash = bool(
            incoming_content_hash
            and candidate_hash
            and incoming_content_hash == candidate_hash
        )
        similarity = review_text_similarity(incoming_payload, candidate_payload)
        identity_match = bool(
            incoming_identity_key
            and candidate_identity
            and incoming_identity_key == candidate_identity
        )
        loose_reviewer_match = reviewer_keys_overlap(
            incoming_reviewer_name,
            candidate.get("reviewer_name"),
            stem_length=reviewer_stem_length,
        )
        loose_date_match = review_dates_within_tolerance(
            incoming_reviewed_at,
            candidate.get("reviewed_at"),
            tolerance_days=review_date_tolerance_days,
        )
        loose_rating_match = rating_values_within_tolerance(
            incoming_rating,
            candidate.get("rating"),
            tolerance=rating_tolerance,
        )
        if exact_hash:
            reason = "cross_source_exact_content"
        elif identity_match and similarity >= similarity_threshold:
            reason = "cross_source_identity_similarity"
        elif (
            loose_reviewer_match
            and loose_date_match
            and loose_rating_match
            and similarity >= loose_similarity_threshold
        ):
            reason = "cross_source_reviewer_date_similarity"
        else:
            continue
        status = str(candidate.get("enrichment_status") or "").strip().lower()
        status_rank = 1 if status == "enriched" else 0
        source_weight = float(candidate.get("source_weight") or 0.0)
        imported_at_rank = _normalize_imported_at_rank(candidate.get("imported_at"))
        detail = {
            "survivor_review_id": str(candidate.get("id") or ""),
            "survivor_source": candidate_source,
            "reason_code": reason,
            "content_hash_match": exact_hash,
            "identity_key_match": identity_match,
            "reviewer_overlap_match": loose_reviewer_match,
            "review_date_tolerance_match": loose_date_match,
            "rating_tolerance_match": loose_rating_match,
            "text_similarity": round(similarity, 4),
        }
        rank = (
            1 if exact_hash else 0,
            status_rank,
            source_weight,
            -imported_at_rank,
        )
        ranked.append((rank, candidate, reason, detail))
    if not ranked:
        return None, None, None
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, candidate, reason, detail = ranked[0]
    return candidate, reason, detail


def _row_source_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _rating_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def review_dates_within_tolerance(left: Any, right: Any, *, tolerance_days: int) -> bool:
    tolerance_days = _coerce_int(tolerance_days, 1, minimum=0)
    left_key = normalize_review_date_key(left)
    right_key = normalize_review_date_key(right)
    if not left_key or not right_key:
        return False
    try:
        left_day = date.fromisoformat(left_key)
        right_day = date.fromisoformat(right_key)
    except ValueError:
        return False
    return abs((left_day - right_day).days) <= tolerance_days


def rating_values_within_tolerance(left: Any, right: Any, *, tolerance: float) -> bool:
    tolerance = _coerce_float(tolerance, 1.0, minimum=0.0)
    left_value = _rating_value(left)
    right_value = _rating_value(right)
    if left_value is None or right_value is None:
        return True
    return abs(left_value - right_value) <= tolerance


def reviewer_keys_overlap(left: Any, right: Any, *, stem_length: int) -> bool:
    left_key = normalize_reviewer_key(left)
    right_key = normalize_reviewer_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    left_stem = normalize_reviewer_stem_key(left_key, stem_length=stem_length)
    right_stem = normalize_reviewer_stem_key(right_key, stem_length=stem_length)
    if not left_stem or not right_stem:
        return False
    return left_key.startswith(right_stem) or right_key.startswith(left_stem)


def _row_status_rank(value: Any) -> int:
    status = str(value or "").strip().lower()
    if status == "enriched":
        return 3
    if status == "no_signal":
        return 2
    if status == "quarantined":
        return 1
    return 0


def _row_source_weight(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric != numeric:
        return 0.0
    return numeric


def canonical_review_sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        _row_status_rank(row.get("enrichment_status")),
        _row_source_weight(row.get("source_weight")),
        -_normalize_imported_at_rank(row.get("imported_at")),
        str(row.get("id") or ""),
    )


def choose_canonical_review_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=canonical_review_sort_key)


def cluster_cross_source_duplicates(
    rows: list[dict[str, Any]],
    *,
    similarity_threshold: float,
    loose_similarity_threshold: float = 0.9,
    reviewer_stem_length: int = 5,
    review_date_tolerance_days: int = 1,
    rating_tolerance: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Return duplicate decisions for rows belonging to one vendor.

    The returned mapping is keyed by discarded review id and contains:
    - survivor_review_id
    - duplicate_reason
    - duplicate_detail
    """
    decisions: dict[str, dict[str, Any]] = {}
    if not rows:
        return decisions

    by_content_hash: dict[str, list[dict[str, Any]]] = {}
    by_identity_key: dict[str, list[dict[str, Any]]] = {}
    payload_by_id: dict[str, str] = {}

    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            continue
        payload_by_id[row_id] = make_review_text_payload(
            row.get("summary"),
            row.get("review_text"),
            row.get("pros"),
            row.get("cons"),
        )
        content_hash = str(row.get("cross_source_content_hash") or "").strip()
        identity_key = str(row.get("cross_source_identity_key") or "").strip()
        if content_hash:
            by_content_hash.setdefault(content_hash, []).append(row)
        if identity_key:
            by_identity_key.setdefault(identity_key, []).append(row)

    for content_hash, group in by_content_hash.items():
        distinct_sources = {_row_source_key(row.get("source")) for row in group if _row_source_key(row.get("source"))}
        if len(group) < 2 or len(distinct_sources) < 2:
            continue
        survivor = choose_canonical_review_row(group)
        if survivor is None:
            continue
        survivor_id = str(survivor.get("id") or "")
        survivor_source = _row_source_key(survivor.get("source"))
        for row in group:
            row_id = str(row.get("id") or "")
            if not row_id or row_id == survivor_id:
                continue
            decisions[row_id] = {
                "survivor_review_id": survivor_id,
                "duplicate_reason": "cross_source_exact_content",
                "duplicate_detail": {
                    "survivor_review_id": survivor_id,
                    "survivor_source": survivor_source,
                    "reason_code": "cross_source_exact_content",
                    "content_hash_match": True,
                    "identity_key_match": (
                        str(row.get("cross_source_identity_key") or "").strip()
                        and str(row.get("cross_source_identity_key") or "").strip()
                        == str(survivor.get("cross_source_identity_key") or "").strip()
                    ),
                    "text_similarity": 1.0,
                    "backfill_scope": "cross_source_review_dedup",
                    "content_hash": content_hash,
                },
            }

    for identity_key, group in by_identity_key.items():
        distinct_sources = {_row_source_key(row.get("source")) for row in group if _row_source_key(row.get("source"))}
        if len(group) < 2 or len(distinct_sources) < 2:
            continue
        candidates = [row for row in group if str(row.get("id") or "") not in decisions]
        if len(candidates) < 2:
            continue
        survivor = choose_canonical_review_row(candidates)
        if survivor is None:
            continue
        survivor_id = str(survivor.get("id") or "")
        survivor_source = _row_source_key(survivor.get("source"))
        survivor_payload = payload_by_id.get(survivor_id, "")
        for row in candidates:
            row_id = str(row.get("id") or "")
            if not row_id or row_id == survivor_id or row_id in decisions:
                continue
            similarity = review_text_similarity(
                payload_by_id.get(row_id, ""),
                survivor_payload,
            )
            if similarity < similarity_threshold:
                continue
            decisions[row_id] = {
                "survivor_review_id": survivor_id,
                "duplicate_reason": "cross_source_identity_similarity",
                "duplicate_detail": {
                    "survivor_review_id": survivor_id,
                    "survivor_source": survivor_source,
                    "reason_code": "cross_source_identity_similarity",
                    "content_hash_match": False,
                    "identity_key_match": True,
                    "text_similarity": round(similarity, 4),
                    "backfill_scope": "cross_source_review_dedup",
                    "identity_key": identity_key,
                },
            }

    by_reviewer_stem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in decisions:
            continue
        reviewer_stem = normalize_reviewer_stem_key(
            row.get("reviewer_name"),
            stem_length=reviewer_stem_length,
        )
        if reviewer_stem:
            by_reviewer_stem.setdefault(reviewer_stem, []).append(row)

    for reviewer_stem, group in by_reviewer_stem.items():
        distinct_sources = {_row_source_key(row.get("source")) for row in group if _row_source_key(row.get("source"))}
        if len(group) < 2 or len(distinct_sources) < 2:
            continue
        candidates = [row for row in group if str(row.get("id") or "") not in decisions]
        if len(candidates) < 2:
            continue
        survivor = choose_canonical_review_row(candidates)
        if survivor is None:
            continue
        survivor_id = str(survivor.get("id") or "")
        survivor_source = _row_source_key(survivor.get("source"))
        survivor_payload = payload_by_id.get(survivor_id, "")
        for row in candidates:
            row_id = str(row.get("id") or "")
            if not row_id or row_id == survivor_id or row_id in decisions:
                continue
            similarity = review_text_similarity(
                payload_by_id.get(row_id, ""),
                survivor_payload,
            )
            if similarity < loose_similarity_threshold:
                continue
            if not review_dates_within_tolerance(
                row.get("reviewed_at"),
                survivor.get("reviewed_at"),
                tolerance_days=review_date_tolerance_days,
            ):
                continue
            if not rating_values_within_tolerance(
                row.get("rating"),
                survivor.get("rating"),
                tolerance=rating_tolerance,
            ):
                continue
            decisions[row_id] = {
                "survivor_review_id": survivor_id,
                "duplicate_reason": "cross_source_reviewer_date_similarity",
                "duplicate_detail": {
                    "survivor_review_id": survivor_id,
                    "survivor_source": survivor_source,
                    "reason_code": "cross_source_reviewer_date_similarity",
                    "content_hash_match": False,
                    "identity_key_match": False,
                    "reviewer_overlap_match": True,
                    "review_date_tolerance_match": True,
                    "rating_tolerance_match": True,
                    "text_similarity": round(similarity, 4),
                    "backfill_scope": "cross_source_review_dedup",
                    "reviewer_stem": reviewer_stem,
                },
            }

    return decisions
