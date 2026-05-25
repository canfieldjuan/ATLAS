"""Search projection helpers for generated ticket FAQ Markdown drafts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from .campaign_ports import JsonDict
from .storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    row_to_dict,
)
from .ticket_faq_ports import TicketFAQDraft


_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class TicketFAQSearchDocument:
    """Compact searchable projection for one generated FAQ item."""

    account_id: str
    corpus_id: str
    faq_id: str
    target_id: str
    target_mode: str
    status: str
    rank: int
    topic: str
    question: str
    answer_summary: str
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    ticket_count: int = 0
    search_text: str = ""

    def as_dict(self) -> JsonDict:
        return {
            "account_id": self.account_id,
            "corpus_id": self.corpus_id,
            "faq_id": self.faq_id,
            "target_id": self.target_id,
            "target_mode": self.target_mode,
            "status": self.status,
            "rank": self.rank,
            "topic": self.topic,
            "question": self.question,
            "answer_summary": self.answer_summary,
            "source_ids": list(self.source_ids),
            "ticket_count": self.ticket_count,
        }


@dataclass(frozen=True)
class TicketFAQSearchResult:
    """A matched FAQ search document plus deterministic score."""

    document: TicketFAQSearchDocument
    score: int

    def as_dict(self) -> JsonDict:
        row = self.document.as_dict()
        row["score"] = self.score
        row.pop("search_text", None)
        return row


@dataclass(frozen=True)
class TicketFAQSearchResponse:
    """API-shaped FAQ search response envelope."""

    query: str
    results: tuple[TicketFAQSearchResult, ...]

    def as_dict(self) -> JsonDict:
        return {
            "query": self.query,
            "results": [result.as_dict() for result in self.results],
            "count": len(self.results),
        }


@dataclass(frozen=True)
class TicketFAQSearchProjectionKey:
    """Replacement key for one FAQ search projection group."""

    account_id: str
    corpus_id: str
    faq_id: str


def build_ticket_faq_search_documents(
    draft: TicketFAQDraft,
    *,
    account_id: str | None = None,
    corpus_id: str | None = None,
) -> tuple[TicketFAQSearchDocument, ...]:
    """Project a generated FAQ draft into one searchable row per FAQ item."""

    resolved_account_id = _clean(account_id) or _metadata_account_id(draft.metadata)
    resolved_corpus_id = _clean(corpus_id) or _metadata_corpus_id(draft.metadata) or draft.target_id
    documents: list[TicketFAQSearchDocument] = []
    for index, item in enumerate(draft.items, start=1):
        if not isinstance(item, Mapping):
            continue
        topic = _clean(item.get("topic"))
        question = _clean(item.get("question"))
        answer_summary = _answer_summary(item)
        search_text = _search_text(item, topic=topic, question=question, answer_summary=answer_summary)
        if not question and not search_text:
            continue
        documents.append(
            TicketFAQSearchDocument(
                account_id=resolved_account_id,
                corpus_id=resolved_corpus_id,
                faq_id=draft.id,
                target_id=draft.target_id,
                target_mode=draft.target_mode,
                status=draft.status,
                rank=_int_value(item.get("rank"), default=index),
                topic=topic,
                question=question,
                answer_summary=answer_summary,
                source_ids=_source_ids(item.get("source_ids")),
                ticket_count=_int_value(item.get("ticket_count"), default=0),
                search_text=search_text,
            )
        )
    return tuple(documents)


def build_ticket_faq_search_projection_key(
    draft: TicketFAQDraft,
    *,
    account_id: str | None = None,
    corpus_id: str | None = None,
) -> TicketFAQSearchProjectionKey:
    """Build the replacement key for a generated FAQ draft projection."""

    resolved_account_id = _clean(account_id) or _metadata_account_id(draft.metadata)
    resolved_corpus_id = _clean(corpus_id) or _metadata_corpus_id(draft.metadata) or draft.target_id
    return TicketFAQSearchProjectionKey(
        account_id=resolved_account_id,
        corpus_id=resolved_corpus_id,
        faq_id=draft.id,
    )


def search_ticket_faq_documents(
    documents: Iterable[TicketFAQSearchDocument],
    *,
    query: str,
    account_id: str,
    corpus_id: str | None = None,
    status: str | None = "approved",
    limit: int = 10,
) -> TicketFAQSearchResponse:
    """Search projected FAQ rows with tenant/corpus/status isolation."""

    normalized_query = _clean(query)
    normalized_account_id = _clean(account_id)
    query_tokens = _tokens(normalized_query)
    if not normalized_account_id or not query_tokens or limit <= 0:
        return TicketFAQSearchResponse(query=normalized_query, results=())
    matches: list[TicketFAQSearchResult] = []
    for document in documents:
        if document.account_id != normalized_account_id:
            continue
        if corpus_id is not None and document.corpus_id != corpus_id:
            continue
        if status is not None and document.status != status:
            continue
        score = _score_document(document, query_tokens)
        if score <= 0:
            continue
        matches.append(TicketFAQSearchResult(document=document, score=score))
    matches.sort(key=lambda result: (-result.score, result.document.rank, result.document.question))
    return TicketFAQSearchResponse(
        query=normalized_query,
        results=tuple(matches[:limit]),
    )


@dataclass(frozen=True)
class PostgresTicketFAQSearchRepository:
    """Postgres adapter for persisted FAQ search projection rows."""

    pool: Any

    async def replace_documents(
        self,
        documents: Sequence[TicketFAQSearchDocument],
        *,
        replace_keys: Sequence[TicketFAQSearchProjectionKey] = (),
    ) -> int:
        """Replace projected rows for each requested FAQ/corpus group."""

        groups: dict[TicketFAQSearchProjectionKey, list[TicketFAQSearchDocument]] = {
            _validate_projection_key(key): [] for key in replace_keys
        }
        for document in documents:
            _validate_search_document(document)
            key = TicketFAQSearchProjectionKey(
                account_id=document.account_id,
                corpus_id=document.corpus_id,
                faq_id=document.faq_id,
            )
            groups.setdefault(key, []).append(document)
        for key, group_documents in groups.items():
            _validate_distinct_ranks(key, group_documents)
        return await _replace_document_groups_atomic(self.pool, groups)

    async def search(
        self,
        *,
        query: str,
        account_id: str,
        corpus_id: str | None = None,
        status: str | None = "approved",
        limit: int = 10,
    ) -> TicketFAQSearchResponse:
        """Search persisted FAQ projection rows with tenant/corpus filters."""

        normalized_query = _clean(query)
        normalized_account_id = _clean(account_id)
        if not normalized_account_id or not _tokens(normalized_query) or limit <= 0:
            return TicketFAQSearchResponse(query=normalized_query, results=())
        clauses: list[str] = ["account_id = $2", "search_vector @@ search_query.query"]
        params: list[Any] = [normalized_query, normalized_account_id]
        if corpus_id is not None:
            params.append(corpus_id)
            clauses.append(f"corpus_id = ${len(params)}")
        if status is not None:
            params.append(status)
            clauses.append(f"status = ${len(params)}")
        params.append(int(limit))
        sql = (
            "WITH search_query AS ("
            "SELECT websearch_to_tsquery('english', $1) AS query"
            ") "
            "SELECT account_id, corpus_id, faq_id, target_id, target_mode, "
            "status, rank, topic, question, answer_summary, source_ids, "
            "ticket_count, search_text, "
            "(ts_rank_cd(search_vector, search_query.query) * 1000)::integer AS score "
            "FROM ticket_faq_search_documents, search_query "
            "WHERE " + " AND ".join(clauses) + " "
            "ORDER BY score DESC, rank ASC, question ASC "
            f"LIMIT ${len(params)}"
        )
        rows = await self.pool.fetch(sql, *params)
        results = tuple(_row_to_search_result(row_to_dict(row)) for row in rows)
        return TicketFAQSearchResponse(query=normalized_query, results=results)


def _score_document(document: TicketFAQSearchDocument, query_tokens: Sequence[str]) -> int:
    question_tokens = set(_tokens(document.question))
    topic_tokens = set(_tokens(document.topic))
    text_counts = Counter(_tokens(document.search_text))
    score = 0
    for token in query_tokens:
        if token in question_tokens:
            score += 4
        if token in topic_tokens:
            score += 2
        score += min(text_counts.get(token, 0), 3)
    return score


def _row_to_search_document(row: Mapping[str, Any]) -> TicketFAQSearchDocument:
    return TicketFAQSearchDocument(
        account_id=_clean(row.get("account_id")),
        corpus_id=_clean(row.get("corpus_id")),
        faq_id=_clean(row.get("faq_id")),
        target_id=_clean(row.get("target_id")),
        target_mode=_clean(row.get("target_mode")),
        status=_clean(row.get("status")),
        rank=_int_value(row.get("rank"), default=0),
        topic=_clean(row.get("topic")),
        question=_clean(row.get("question")),
        answer_summary=_clean(row.get("answer_summary")),
        source_ids=_source_ids_jsonb(row.get("source_ids")),
        ticket_count=_int_value(row.get("ticket_count"), default=0),
        search_text=_clean(row.get("search_text")),
    )


async def _replace_document_groups_atomic(
    db: Any,
    groups: Mapping[TicketFAQSearchProjectionKey, Sequence[TicketFAQSearchDocument]],
) -> int:
    transaction = getattr(db, "transaction", None)
    if callable(transaction):
        async with transaction() as connection:
            return await _replace_document_groups(connection or db, groups)
    acquire = getattr(db, "acquire", None)
    if callable(acquire):
        async with acquire() as connection:
            connection_transaction = getattr(connection, "transaction", None)
            if callable(connection_transaction):
                async with connection_transaction():
                    return await _replace_document_groups(connection, groups)
            return await _replace_document_groups(connection, groups)
    return await _replace_document_groups(db, groups)


async def _replace_document_groups(
    db: Any,
    groups: Mapping[TicketFAQSearchProjectionKey, Sequence[TicketFAQSearchDocument]],
) -> int:
    replaced = 0
    for key, documents in groups.items():
        await db.execute(
            """
            DELETE FROM ticket_faq_search_documents
             WHERE account_id = $1
               AND corpus_id = $2
               AND faq_id = $3::uuid
            """,
            key.account_id,
            key.corpus_id,
            key.faq_id,
        )
        for document in documents:
            await db.execute(
                """
                INSERT INTO ticket_faq_search_documents (
                    account_id, corpus_id, faq_id, target_id, target_mode,
                    status, rank, topic, question, answer_summary,
                    source_ids, ticket_count, search_text
                )
                VALUES (
                    $1, $2, $3::uuid, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11::jsonb, $12, $13
                )
                ON CONFLICT (account_id, corpus_id, faq_id, rank)
                DO UPDATE SET
                    target_id = EXCLUDED.target_id,
                    target_mode = EXCLUDED.target_mode,
                    status = EXCLUDED.status,
                    topic = EXCLUDED.topic,
                    question = EXCLUDED.question,
                    answer_summary = EXCLUDED.answer_summary,
                    source_ids = EXCLUDED.source_ids,
                    ticket_count = EXCLUDED.ticket_count,
                    search_text = EXCLUDED.search_text,
                    updated_at = NOW()
                """,
                document.account_id,
                document.corpus_id,
                document.faq_id,
                document.target_id,
                document.target_mode,
                document.status,
                document.rank,
                document.topic,
                document.question,
                document.answer_summary,
                json_dump_jsonb(list(document.source_ids)),
                document.ticket_count,
                document.search_text,
            )
            replaced += 1
    return replaced


def _validate_projection_key(
    key: TicketFAQSearchProjectionKey,
) -> TicketFAQSearchProjectionKey:
    if not _clean(key.account_id):
        raise ValueError("ticket FAQ search projection key requires account_id")
    if not _clean(key.corpus_id):
        raise ValueError("ticket FAQ search projection key requires corpus_id")
    if not _clean(key.faq_id):
        raise ValueError("ticket FAQ search projection key requires faq_id")
    return key


def _validate_search_document(document: TicketFAQSearchDocument) -> None:
    _validate_projection_key(
        TicketFAQSearchProjectionKey(
            account_id=document.account_id,
            corpus_id=document.corpus_id,
            faq_id=document.faq_id,
        )
    )
    if not _clean(document.status):
        raise ValueError("ticket FAQ search document requires status")
    if int(document.rank) <= 0:
        raise ValueError("ticket FAQ search document requires rank > 0")
    if int(document.ticket_count) < 0:
        raise ValueError("ticket FAQ search document requires ticket_count >= 0")


def _validate_distinct_ranks(
    key: TicketFAQSearchProjectionKey,
    documents: Sequence[TicketFAQSearchDocument],
) -> None:
    ranks: set[int] = set()
    for document in documents:
        rank = int(document.rank)
        if rank in ranks:
            raise ValueError(
                "ticket FAQ search projection requires distinct ranks "
                f"for account_id={key.account_id} corpus_id={key.corpus_id} "
                f"faq_id={key.faq_id}"
            )
        ranks.add(rank)


def _row_to_search_result(row: Mapping[str, Any]) -> TicketFAQSearchResult:
    return TicketFAQSearchResult(
        document=_row_to_search_document(row),
        score=_int_value(row.get("score"), default=0),
    )


def _search_text(item: Mapping[str, Any], *, topic: str, question: str, answer_summary: str) -> str:
    parts: list[str] = [topic, question, answer_summary]
    for key in ("answer", "when_to_contact_support"):
        parts.append(_clean(item.get(key)))
    steps = item.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        parts.extend(_clean(step) for step in steps)
    return " ".join(part for part in parts if part)


def _answer_summary(item: Mapping[str, Any]) -> str:
    for key in ("summary", "answer"):
        value = _clean(item.get(key))
        if value:
            return value
    return ""


def _source_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(_clean(item) for item in value if _clean(item)))


def _source_ids_jsonb(value: Any) -> tuple[str, ...]:
    decoded = decode_jsonb_field(value, default=[])
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        return ()
    return tuple(dict.fromkeys(_clean(item) for item in decoded if _clean(item)))


def _metadata_account_id(metadata: Mapping[str, Any]) -> str:
    scope = metadata.get("scope")
    if isinstance(scope, Mapping):
        return _clean(scope.get("account_id"))
    return _clean(metadata.get("account_id"))


def _metadata_corpus_id(metadata: Mapping[str, Any]) -> str:
    return _clean(metadata.get("corpus_id") or metadata.get("source_file_id"))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(value.lower()))


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "PostgresTicketFAQSearchRepository",
    "TicketFAQSearchDocument",
    "TicketFAQSearchProjectionKey",
    "TicketFAQSearchResponse",
    "TicketFAQSearchResult",
    "build_ticket_faq_search_documents",
    "build_ticket_faq_search_projection_key",
    "search_ticket_faq_documents",
]
