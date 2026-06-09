"""Deterministic clustering helpers for support-ticket source rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HTML_TAG_NAMES_RE = (
    r"(?:a|abbr|article|aside|b|blockquote|body|br|button|cite|code|dd|"
    r"del|div|dl|dt|em|figcaption|figure|footer|h[1-6]|header|hr|html|i|"
    r"img|ins|li|main|mark|nav|ol|p|pre|s|section|small|span|strike|strong|"
    r"sub|sup|table|tbody|td|tfoot|th|thead|time|tr|u|ul)"
)
_HTML_ATTR_RE = (
    r"(?:\s+[a-z_:][a-z0-9_:.-]*\s*=\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))"
)
_HTML_SIGNAL_RE = re.compile(
    rf"</?{_HTML_TAG_NAMES_RE}(?:{_HTML_ATTR_RE})*\s*/?>",
    re.IGNORECASE,
)
_HTML_CUSTOM_TAG_RE = re.compile(
    r"</?[a-z][a-z0-9:-]*-[a-z0-9:-]*(?:\s+[^<>]*)?/?>",
    re.IGNORECASE,
)
_TAG_FALLBACK_RE = re.compile(r"</?[^>]+>")
_COMPACT_KEY_RE = re.compile(r"[^a-z0-9]+")
_PHRASE_FOLDS = (
    (re.compile(r"\bsign[-\s]?in\b", re.IGNORECASE), "login"),
    (re.compile(r"\blog\s+in\b", re.IGNORECASE), "login"),
    (re.compile(r"\blog[-\s]?in\b", re.IGNORECASE), "login"),
    (re.compile(r"\blocked\s+out\b", re.IGNORECASE), "login"),
    (re.compile(r"\baccount\s+access\b", re.IGNORECASE), "login"),
    (
        re.compile(
            r"\b(?:can(?:not|'t)|cant|unable\s+to)\s+access\s+(?:my\s+|the\s+)?account\b",
            re.IGNORECASE,
        ),
        "login",
    ),
    (re.compile(r"\baccess\s+(?:my\s+|the\s+)?account\b", re.IGNORECASE), "login"),
    (re.compile(r"\bsingle[-\s]?sign[-\s]?on\b", re.IGNORECASE), "sso"),
    (re.compile(r"\bsingle\s+sign\s+on\b", re.IGNORECASE), "sso"),
    (re.compile(r"\be[-\s]?mail\b", re.IGNORECASE), "email"),
    (re.compile(r"\btwo[-\s]?factor\b", re.IGNORECASE), "2fa"),
)
_STOPWORDS = {
    "a",
    "able",
    "account",
    "about",
    "again",
    "agent",
    "after",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "but",
    "by",
    "can",
    "cannot",
    "cant",
    "case",
    "client",
    "could",
    "customer",
    "do",
    "does",
    "doing",
    "done",
    "find",
    "for",
    "from",
    "get",
    "getting",
    "got",
    "had",
    "has",
    "have",
    "having",
    "hello",
    "help",
    "hi",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "need",
    "needs",
    "not",
    "of",
    "on",
    "or",
    "our",
    "page",
    "please",
    "problem",
    "question",
    "request",
    "screen",
    "support",
    "team",
    "thanks",
    "thank",
    "that",
    "the",
    "their",
    "there",
    "this",
    "ticket",
    "to",
    "try",
    "trying",
    "unable",
    "use",
    "using",
    "we",
    "what",
    "when",
    "where",
    "why",
    "will",
    "with",
    "work",
    "working",
    "works",
    "would",
    "you",
}
_TOKEN_FOLDS = {
    "address": "",
    "automate": "automation",
    "automated": "automation",
    "automating": "automation",
    "billed": "billing",
    "bill": "billing",
    "bills": "billing",
    "charge": "billing",
    "charged": "billing",
    "charges": "billing",
    "cancellation": "cancel",
    "cancelled": "cancel",
    "cancelling": "cancel",
    "cancels": "cancel",
    "chart": "dashboard",
    "charts": "dashboard",
    "change": "update",
    "changed": "update",
    "changing": "update",
    "credential": "login",
    "credentials": "login",
    "download": "export",
    "downloaded": "export",
    "downloading": "export",
    "downloads": "export",
    "edit": "update",
    "edited": "update",
    "editing": "update",
    "exported": "export",
    "exporting": "export",
    "exports": "export",
    "invitation": "invite",
    "invitations": "invite",
    "invoices": "invoice",
    "integrations": "api",
    "integration": "api",
    "modifying": "update",
    "modify": "update",
    "modified": "update",
    "payments": "payment",
    "pwd": "password",
    "reporting": "report",
    "reports": "report",
    "renew": "renewal",
    "renewed": "renewal",
    "renewing": "renewal",
    "renews": "renewal",
    "resetting": "reset",
    "resets": "reset",
    "signin": "login",
    "signins": "login",
    "saml": "sso",
    "updated": "update",
    "updating": "update",
    "webhook": "api",
    "webhooks": "api",
}
_SINGLE_TOKEN_CLUSTER_LABELS = {
    "billing",
    "cancel",
    "email",
    "export",
    "api",
    "invite",
    "invoice",
    "login",
    "password",
    "payment",
    "refund",
    "subscription",
}
_LOW_SIGNAL_ANCHOR_TOKENS = {
    "arrive",
    "auth",
    "authenticate",
    "authenticated",
    "authenticating",
    "authentication",
    "broken",
    "error",
    "failed",
    "failure",
    "missing",
    "never",
    "new",
    "out",
    "report",
    "return",
    "same",
    "update",
}
_EXPLICIT_LABEL_KEYS = ("pain_category", "category", "intent", "topic")
_TEXT_KEYS = (
    "source_title",
    "title",
    "subject",
    "ticket_subject",
    "question",
    "text",
    "description",
    "message",
    "body",
    "content",
    "summary",
)


@dataclass(frozen=True)
class SupportTicketClusterHint:
    """Stable row-level cluster hint."""

    key: str
    label: str
    source: str
    tokens: frozenset[str] = frozenset()


@dataclass
class _ClusterBucket:
    key: str
    label: str
    source: str
    count: int = 0
    token_sets: list[frozenset[str]] = field(default_factory=list)
    token_counts: Counter[str] = field(default_factory=Counter)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            self._skip_depth += 1
        self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def support_ticket_plain_text(value: Any) -> str:
    """Return compact readable text from plain or common HTML ticket bodies."""

    raw = str(value or "")
    if not raw.strip():
        return ""
    text = raw
    if not _looks_like_html(text):
        decoded = unescape(text)
        if not _looks_like_html(decoded):
            return _compact(decoded)
        text = decoded
    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
        parsed = "".join(parser.parts)
    except Exception:
        parsed = _TAG_FALLBACK_RE.sub(" ", text)
    return _compact(parsed)


def _looks_like_html(text: str) -> bool:
    return bool(
        _HTML_SIGNAL_RE.search(text)
        or _HTML_CUSTOM_TAG_RE.search(text)
    )


def support_ticket_tokens(value: Any) -> frozenset[str]:
    text = support_ticket_plain_text(value).lower()
    for pattern, replacement in _PHRASE_FOLDS:
        text = pattern.sub(replacement, text)
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        token = _TOKEN_FOLDS.get(raw, raw)
        if not token:
            continue
        if token in _STOPWORDS:
            continue
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        token = _TOKEN_FOLDS.get(token, token)
        if not token or token in _STOPWORDS or len(token) < 2:
            continue
        tokens.add(token)
    return frozenset(tokens)


def support_ticket_cluster_hint(row: Mapping[str, Any]) -> SupportTicketClusterHint | None:
    """Derive a deterministic cluster hint for one support-ticket row."""

    existing = support_ticket_plain_text(row.get("support_ticket_cluster"))
    if existing:
        key = support_ticket_plain_text(row.get("support_ticket_cluster_key"))
        return SupportTicketClusterHint(
            key=key or f"cluster:{_compact_key(existing)}",
            label=existing,
            source=support_ticket_plain_text(row.get("support_ticket_cluster_source")) or "provided",
        )

    explicit = _first_text(row, _EXPLICIT_LABEL_KEYS)
    if explicit:
        return SupportTicketClusterHint(
            key=f"explicit:{_compact_key(explicit)}",
            label=explicit,
            source="explicit",
        )

    tokens = _row_tokens(row)
    if len(tokens) < 2:
        if len(tokens) == 1:
            token = next(iter(tokens))
            if token in _SINGLE_TOKEN_CLUSTER_LABELS:
                return SupportTicketClusterHint(
                    key=f"keyword:{_compact_key(token)}",
                    label=token,
                    source="keyword",
                    tokens=tokens,
                )
        return None
    label = _label_from_tokens(tokens)
    return SupportTicketClusterHint(
        key=f"tokens:{_compact_key(label)}",
        label=label,
        source="token_set",
        tokens=tokens,
    )


def assign_support_ticket_clusters(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return rows annotated with stable deterministic support-ticket clusters."""

    buckets: list[_ClusterBucket] = []
    assignments: list[_ClusterBucket | None] = []
    hints = tuple(support_ticket_cluster_hint(row) for row in rows)
    token_row_counts = _token_row_counts(hints)

    for hint in hints:
        if hint is None:
            assignments.append(None)
            continue
        bucket = _bucket_for_hint(
            buckets,
            hint,
            token_row_counts=token_row_counts,
        )
        bucket.count += 1
        if hint.tokens:
            bucket.token_sets.append(hint.tokens)
            bucket.token_counts.update(hint.tokens)
        assignments.append(bucket)

    out: list[dict[str, Any]] = []
    for row, bucket in zip(rows, assignments, strict=False):
        next_row = dict(row)
        if bucket is not None:
            label = _bucket_label(bucket)
            next_row["support_ticket_cluster"] = label
            next_row["support_ticket_cluster_key"] = _bucket_key(bucket, label)
            next_row["support_ticket_cluster_source"] = bucket.source
        out.append(next_row)
    return tuple(out)


def support_ticket_cluster_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return bounded top-cluster counts for preview/diagnostics output."""

    annotated = _ensure_clustered(rows)
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    uncategorized_count = 0
    for row in annotated:
        label = support_ticket_plain_text(row.get("support_ticket_cluster"))
        if not label:
            uncategorized_count += 1
            continue
        key = support_ticket_plain_text(row.get("support_ticket_cluster_key")) or label.lower()
        labels.setdefault(key, label)
        counts[key] += 1

    clusters = [
        {"label": labels[key], "count": count}
        for key, count in counts.most_common(max(1, limit))
    ]
    shown_count = sum(int(item["count"]) for item in clusters)
    remaining_count = sum(counts.values()) - shown_count
    if remaining_count > 0:
        clusters.append({"label": "remaining", "count": remaining_count})
    if uncategorized_count > 0:
        clusters.append({"label": "uncategorized", "count": uncategorized_count})
    return clusters


def support_ticket_cluster_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return compact quality counters for pre-check diagnostics."""

    annotated = _ensure_clustered(rows)
    counts: Counter[str] = Counter()
    uncategorized_count = 0
    for row in annotated:
        label = support_ticket_plain_text(row.get("support_ticket_cluster"))
        if not label:
            uncategorized_count += 1
            continue
        key = support_ticket_plain_text(row.get("support_ticket_cluster_key")) or label.lower()
        counts[key] += 1
    cluster_counts = tuple(counts.values())
    return {
        "clustered_row_count": sum(cluster_counts),
        "uncategorized_row_count": uncategorized_count,
        "cluster_count": len(cluster_counts),
        "singleton_cluster_count": sum(1 for count in cluster_counts if count == 1),
        "largest_cluster_count": max(cluster_counts, default=0),
    }


def _ensure_clustered(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if any(support_ticket_plain_text(row.get("support_ticket_cluster")) for row in rows):
        return tuple(dict(row) for row in rows)
    return assign_support_ticket_clusters(rows)


def _bucket_for_hint(
    buckets: list[_ClusterBucket],
    hint: SupportTicketClusterHint,
    *,
    token_row_counts: Counter[str],
) -> _ClusterBucket:
    if hint.source != "token_set":
        for bucket in buckets:
            if bucket.key == hint.key:
                return bucket
        bucket = _ClusterBucket(key=hint.key, label=hint.label, source=hint.source)
        buckets.append(bucket)
        return bucket

    match = _matching_token_bucket(
        buckets,
        hint.tokens,
        token_row_counts=token_row_counts,
    )
    if match is not None:
        bucket, anchor = match
        if anchor:
            bucket.key = f"anchor:{_compact_key(anchor)}"
            bucket.label = anchor
            bucket.source = "token_anchor"
        return bucket
    bucket = _ClusterBucket(key=hint.key, label=hint.label, source=hint.source)
    buckets.append(bucket)
    return bucket


def _matching_token_bucket(
    buckets: Sequence[_ClusterBucket],
    tokens: frozenset[str],
    *,
    token_row_counts: Counter[str],
) -> tuple[_ClusterBucket, str] | None:
    best_overlap: tuple[float, int, _ClusterBucket] | None = None
    best_anchor: tuple[int, int, str, _ClusterBucket] | None = None
    for bucket in buckets:
        if bucket.source not in {"token_set", "token_anchor"}:
            continue
        for existing in bucket.token_sets:
            common = len(tokens & existing)
            if common >= 2:
                overlap = common / max(1, min(len(tokens), len(existing)))
                if overlap >= 0.6:
                    score = (overlap, common, bucket)
                    if best_overlap is None or score[:2] > best_overlap[:2]:
                        best_overlap = score
                    continue
            anchor = _shared_anchor(tokens & existing, token_row_counts)
            if anchor:
                score = (-token_row_counts[anchor], -len(anchor), anchor, bucket)
                if best_anchor is None or score[:3] < best_anchor[:3]:
                    best_anchor = score
    if best_overlap is not None:
        return best_overlap[2], ""
    if best_anchor is not None:
        return best_anchor[3], best_anchor[2]
    return None


def _bucket_label(bucket: _ClusterBucket) -> str:
    if bucket.source == "token_anchor":
        return bucket.label
    if bucket.source != "token_set" or not bucket.token_counts:
        return bucket.label
    return _label_from_counter(bucket.token_counts)


def _bucket_key(bucket: _ClusterBucket, label: str) -> str:
    if bucket.source == "token_set":
        return f"tokens:{_compact_key(label)}"
    return bucket.key


def _row_tokens(row: Mapping[str, Any]) -> frozenset[str]:
    parts: list[str] = []
    for key in _TEXT_KEYS:
        value = support_ticket_plain_text(row.get(key))
        if value:
            parts.append(value)
    source_id = support_ticket_plain_text(row.get("source_id"))
    source_title = support_ticket_plain_text(row.get("source_title"))
    if source_title and source_title != source_id:
        parts.append(source_title)
    return support_ticket_tokens(" ".join(part for part in parts if part))


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = support_ticket_plain_text(row.get(key))
        if value:
            return value
    return ""


def _label_from_tokens(tokens: frozenset[str]) -> str:
    return " ".join(sorted(tokens)[:4])


def _label_from_counter(counts: Counter[str]) -> str:
    selected = [
        token
        for token, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    return " ".join(sorted(selected))


def _shared_anchor(tokens: frozenset[str], token_row_counts: Counter[str]) -> str:
    candidates = [
        token
        for token in tokens
        if _is_anchor_candidate(token, token_row_counts)
    ]
    if not candidates:
        return ""
    return min(candidates, key=lambda token: (-token_row_counts[token], -len(token), token))


def _is_anchor_candidate(token: str, token_row_counts: Counter[str]) -> bool:
    return (
        token_row_counts[token] >= 2
        and token not in _LOW_SIGNAL_ANCHOR_TOKENS
        and not token.isdigit()
    )


def _token_row_counts(
    hints: Sequence[SupportTicketClusterHint | None],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for hint in hints:
        if hint is None or hint.source != "token_set":
            continue
        counts.update(hint.tokens)
    return counts


def _compact(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _compact_key(value: Any) -> str:
    return _COMPACT_KEY_RE.sub("-", support_ticket_plain_text(value).lower()).strip("-")


__all__ = [
    "SupportTicketClusterHint",
    "assign_support_ticket_clusters",
    "support_ticket_cluster_hint",
    "support_ticket_cluster_quality",
    "support_ticket_cluster_summary",
    "support_ticket_plain_text",
    "support_ticket_tokens",
]
