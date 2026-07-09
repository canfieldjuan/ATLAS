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
_LEADING_BRACKETED_METADATA_RE = re.compile(r"^(?:\[[^\]\n]{1,80}\]\s*)+")
_HTML_TAG_NAMES_RE = (
    r"(?:a|abbr|article|aside|b|body|br|button|cite|code|dd|"
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
# Excluded tags mentioned MID-PROSE ("How do I add <script> to the page?",
# "How do I write <blockquote>hello</blockquote> in the editor?") are
# customer wording, not markup. A body that STARTS with an excluded tag
# ("<script>alert(1)", "<blockquote>quoted prior reply") is markup intent:
# script-only bodies exclude, quote-to-EOF bodies are all-quote. Inside a
# document detected via other tags, exclusion applies as usual.
_HTML_EXCLUDED_TAG_AT_START_RE = re.compile(
    r"\A\s*(?:(?:\[[^\]\n]{1,80}\]|<!doctype[^>]*>|<!--.*?-->"
    r"|</?(?:html|head|body|meta|title|link)[^>]*>)\s*)*"
    r"<(script|style|blockquote)\b",
    re.IGNORECASE | re.DOTALL,
)
_HTML_EXCLUDED_TAG_RE = re.compile(
    r"</?(script|style|blockquote)\b[^>]*>",
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
    (re.compile(r"\bidentity\s+provider\b", re.IGNORECASE), "sso"),
    (re.compile(r"\bazure\s+ad\b", re.IGNORECASE), "sso"),
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
    "idp": "sso",
    "okta": "sso",
    "onelogin": "sso",
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
# Token-set clustering compares each row's tokens against every prior
# token set, which is quadratic in row count (measured ~6.7s at 2k rows and
# ~40 minutes extrapolated at 35k on real long-form text, #1454). Above this
# many token-set rows the preview is skipped and reported instead of running.
# The legacy submit path capped uploads at 1,000 rows, so this threshold
# never skips an input shape the path previously clustered.
MAX_TOKEN_SET_CLUSTER_ROWS = 2000

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


# Block-level members of the SAME tag families _HTML_SIGNAL_RE recognizes,
# so HTML detection and line extraction cannot drift apart. These tags carry
# the message structure (paragraph/break/list/table/heading/quote boundaries)
# that line-based hygiene downstream keys on.
_BLOCK_TAG_NAMES = frozenset({
    "article", "aside", "blockquote", "body", "br", "dd", "div", "dl", "dt",
    "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "html", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "tbody", "tfoot", "thead", "tr", "ul",
})
# Bodies excluded from customer text: script/style are markup machinery;
# blockquote is the HTML-native quoted-prior-message marker.
_EXCLUDED_CONTENT_TAGS = frozenset({"script", "style", "blockquote"})


class _HTMLTextExtractor(HTMLParser):
    """Extract text with line boundaries at block tags.

    Block-level tags emit newlines so downstream line-based hygiene sees the
    structure rich HTML encodes in tags; inline tags emit spaces as before.
    script/style bodies (and blockquote bodies when ``exclude_blockquote``)
    are excluded. Excluded content is buffered, not discarded: if a
    script/style scope never closes (malformed HTML puts the parser in CDATA
    mode and would otherwise swallow the rest of the ticket), the buffered
    text is recovered tag-stripped at EOF so customer text is never lost. An
    unclosed blockquote stays excluded -- a quote running to end-of-message
    is a quote, not data loss.
    """

    def __init__(self, *, exclude_blockquote: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_stack: list[str] = []
        self._pending: list[str] = []
        self._excluded = {"script", "style"}
        if exclude_blockquote:
            self._excluded = _EXCLUDED_CONTENT_TAGS

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in self._excluded and not self._skip_stack:
            # The excluded tag's own boundary follows the block rule too.
            self.parts.append("\n" if lowered in _BLOCK_TAG_NAMES else " ")
            self._skip_stack.append(lowered)
            return
        if self._skip_stack:
            if lowered in self._excluded:
                self._skip_stack.append(lowered)
            return
        self.parts.append("\n" if lowered in _BLOCK_TAG_NAMES else " ")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if self._skip_stack:
            if lowered == self._skip_stack[-1]:
                self._skip_stack.pop()
                if lowered in {"script", "style"}:
                    # CDATA may have swallowed outer scopes' close tags
                    # (<blockquote><script>x</blockquote></script>): unwind
                    # scopes whose close tags sit in the buffer outside code
                    # literals, then drop the buffer -- a closed scope's
                    # content is excluded and must never leak into a later
                    # malformed scope's recovery.
                    self._unwind_swallowed_scopes("".join(self._pending))
                    self._pending.clear()
                if not self._skip_stack:
                    self.parts.append(
                        "\n" if lowered in _BLOCK_TAG_NAMES else " "
                    )
            return
        self.parts.append("\n" if lowered in _BLOCK_TAG_NAMES else " ")

    def _unwind_swallowed_scopes(self, buffered: str) -> None:
        if not self._skip_stack or not buffered:
            return
        masked = _code_literal_regions(buffered)
        available: dict[str, int] = {}
        while self._skip_stack:
            scope = self._skip_stack[-1]
            if scope not in available:
                available[scope] = sum(
                    1
                    for match in re.finditer(
                        rf"</{scope}\s*>", buffered, re.IGNORECASE
                    )
                    if not any(
                        lo <= match.start() <= hi for lo, hi in masked
                    )
                )
            if available[scope] <= 0:
                break
            available[scope] -= 1
            self._skip_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            # Buffer whenever the parser may be in CDATA mode (an open
            # script/style is always innermost -- CDATA parses no tags), so a
            # malformed script nested in a blockquote can still recover the
            # swallowed tail. Blockquote-only scopes are never buffered.
            if self._skip_stack[-1] in {"script", "style"}:
                self._pending.append(data)
            return
        self.parts.append(data)

    def finalize(self) -> str:
        self.close()
        if self._skip_stack and self._pending:
            # Unclosed script/style swallowed trailing content via CDATA
            # mode. Everything before the first recognized HTML signal in
            # the buffer is script/CSS machinery and stays excluded; the
            # remainder is real markup the parser never saw -- re-extract it
            # so customer text is recovered without admitting machinery.
            # An unclosed blockquote is a quote to EOF and stays excluded.
            if self._skip_stack[-1] in {"script", "style"}:
                buffered = "".join(self._pending)
                # The buffer may hold swallowed close tags of outer scopes;
                # unwind them first. If a non-CDATA scope (an unclosed
                # blockquote) still remains, the tail is quoted content and
                # stays excluded.
                self._skip_stack.pop()
                self._unwind_swallowed_scopes(buffered)
                if any(
                    scope not in {"script", "style"}
                    for scope in self._skip_stack
                ):
                    return "".join(self.parts)
                start = _first_markup_outside_code_literals(buffered)
                if start is not None:
                    recovered = _extract_html_text(
                        buffered[start:],
                        exclude_blockquote="blockquote" in self._excluded,
                    )
                    if recovered.strip():
                        # The scope-open boundary was already emitted per the
                        # block rule; the recovered extraction carries its own
                        # boundaries. Do not invent one here.
                        self.parts.append(recovered)
        return "".join(self.parts)


def _first_markup_outside_code_literals(buffered: str) -> int | None:
    """First HTML-signal offset outside string literals and comments.

    Script/CSS code embeds markup in quoted templates ('<p>x</p>'),
    template literals, regex literals, and comments; those are code, not
    lost ticket text. Only markup in plain code context marks where real
    trailing HTML began.
    """

    masked = _code_literal_regions(buffered)
    candidates = [
        match.start()
        for pattern in (
            _HTML_SIGNAL_RE,
            _HTML_CUSTOM_TAG_RE,
            _HTML_EXCLUDED_TAG_RE,
        )
        for match in pattern.finditer(buffered)
    ]
    for position in sorted(candidates):
        if not any(lo <= position <= hi for lo, hi in masked):
            return position
    return None


_JS_EXPRESSION_KEYWORDS = frozenset({
    "return", "typeof", "case", "in", "of", "delete", "void", "throw",
    "do", "else", "yield", "await", "instanceof", "new",
})


def _code_literal_regions(buffered: str) -> list[tuple[int, int]]:
    """Mask string/template/regex literals and comments in script text.

    Regex-literal handling is deliberately bounded: a slash opens a regex
    only in expression position (not after a value: alphanumerics, closing
    brackets/braces, closed literals, or postfix ++/--), a slash inside a
    character class does not close it, and a candidate that reaches a
    newline was division all along (JS regex literals cannot span lines),
    so a misread slash can never mask to EOF.
    """

    masked: list[tuple[int, int]] = []
    state: str | None = None
    start = 0
    in_char_class = False
    i = 0
    length = len(buffered)
    prev = ""
    prev2 = ""
    word = ""
    prev_word = ""
    while i < length:
        ch = buffered[i]
        nxt = buffered[i + 1] if i + 1 < length else ""
        if state is None:
            if ch in "'\"`":
                state, start = ch, i
            elif ch == "/" and nxt == "/":
                state, start = "//", i
                i += 1
            elif ch == "/" and nxt == "*":
                state, start = "/*", i
                i += 1
            elif ch == "<" and buffered[i:i + 4] == "<!--":
                state, start = "<!--", i
                i += 3
            elif ch == "/" and (
                (word or prev_word) in _JS_EXPRESSION_KEYWORDS
                or (
                    # "</tag>" (slash ADJACENT to <) is markup; a spaced
                    # "< /" is a less-than operator before a regex.
                    not (prev == "<" and i > 0 and buffered[i - 1] == "<")
                    and prev not in ")]}\"'`"
                    and not prev.isalnum()
                    and not (prev in "+-" and prev2 == prev)
                )
            ):
                # Expression-position slash: candidate regex literal.
                state, start = "re", i
                in_char_class = False
            if state is None:
                if ch.isspace():
                    # Whitespace completes a word (x in /re/): remember it
                    # for the keyword check without polluting the buffer.
                    if word:
                        prev_word, word = word, ""
                elif ch.isalnum() or ch in "$_":
                    # A word opened by "." is a property access, never an
                    # expression keyword (obj.return / 2 is division).
                    word = word + ch if word else ("." + ch if prev == "." else ch)
                    prev2, prev = prev, ch
                else:
                    word = ""
                    prev_word = ""
                    prev2, prev = prev, ch
        elif state == "re":
            if ch == "\\":
                i += 1
            elif ch == "\n":
                # JS regex literals cannot span lines: this was division.
                state = None
                prev2, prev = "", "/"
            elif ch == "[":
                in_char_class = True
            elif ch == "]":
                in_char_class = False
            elif ch == "/" and not in_char_class:
                masked.append((start, i))
                state = None
                # A closed regex is a value: a following slash is division.
                prev2, prev = "", ")"
        elif state in "'\"`":
            if ch == "\\":
                i += 1
            elif ch == state:
                masked.append((start, i))
                state = None
                # A closed literal is a value: a following slash is division.
                prev2, prev = "", ch
        elif state == "//":
            if ch == "\n":
                masked.append((start, i))
                state = None
        elif state == "/*":
            if ch == "*" and nxt == "/":
                masked.append((start, i + 1))
                state = None
                i += 1
        elif state == "<!--":
            if buffered[i:i + 3] == "-->":
                masked.append((start, i + 2))
                state = None
                i += 2
        i += 1
    if state == "re":
        # Unclosed single-line regex candidate at EOF: treat as division,
        # masking nothing, rather than swallowing the buffer.
        pass
    elif state is not None:
        # Unterminated string/comment runs to EOF: everything after it is
        # still code context, never recoverable ticket text.
        masked.append((start, length - 1))
    return masked


def _extract_html_text(text: str, *, exclude_blockquote: bool) -> str:
    parser = _HTMLTextExtractor(exclude_blockquote=exclude_blockquote)
    parser.feed(text)
    return parser.finalize()


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
    try:
        parsed = _extract_html_text(text, exclude_blockquote=True)
        compacted = _compact(parsed)
        if compacted:
            return compacted
        # An all-quote body would otherwise turn a previously admitted row
        # empty; keep the unexcluded extraction rather than losing the row.
        parsed = _extract_html_text(text, exclude_blockquote=False)
    except Exception:
        parsed = _TAG_FALLBACK_RE.sub(" ", text)
    return _compact(parsed)


def support_ticket_plain_text_lines(value: Any) -> str:
    """Return readable text with line boundaries preserved.

    The line-preserving seam for downstream line-based hygiene (scalar
    history signature/quote handling and the junk/auto-reply gate): block
    tags become newlines, each line is whitespace-compacted, and empty lines
    are dropped. Plain-text input keeps its own newlines. Quoted
    ``blockquote`` bodies and script/style bodies are excluded with no
    all-quote fallback -- an all-quote body is genuinely empty of new
    customer text for hygiene purposes.
    """

    raw = str(value or "")
    if not raw.strip():
        return ""
    text = raw
    if not _looks_like_html(text):
        decoded = unescape(text)
        if not _looks_like_html(decoded):
            return _compact_lines(decoded)
        text = decoded
    try:
        parsed = _extract_html_text(text, exclude_blockquote=True)
    except Exception:
        parsed = _TAG_FALLBACK_RE.sub(" ", text)
    return _compact_lines(parsed)


def _compact_lines(text: str) -> str:
    lines = [_compact(line) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _looks_like_html(text: str) -> bool:
    return bool(
        _HTML_SIGNAL_RE.search(text)
        or _HTML_EXCLUDED_TAG_AT_START_RE.search(text)
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
        token = _strip_plural_suffix(token)
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
    *,
    max_token_set_rows: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return rows annotated with stable deterministic support-ticket clusters."""

    annotated, _diagnostics = assign_support_ticket_clusters_with_diagnostics(
        rows,
        max_token_set_rows=max_token_set_rows,
    )
    return annotated


def assign_support_ticket_clusters_with_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_token_set_rows: int | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Cluster rows and report whether the token-set preview was skipped.

    Token-set hints (rows without an explicit category) require pairwise
    token comparison that is quadratic in row count, so above
    ``max_token_set_rows`` those rows are deliberately left uncategorized
    instead of silently wedging the worker (#1454). Explicit, provided, and
    keyword hints always cluster; they use cheap key-equality bucketing.
    """

    if max_token_set_rows is None:
        max_token_set_rows = MAX_TOKEN_SET_CLUSTER_ROWS
    buckets: list[_ClusterBucket] = []
    assignments: list[_ClusterBucket | None] = []
    hints = tuple(support_ticket_cluster_hint(row) for row in rows)
    token_row_counts = _token_row_counts(hints)
    token_set_row_count = sum(
        1 for hint in hints if hint is not None and hint.source == "token_set"
    )
    skip_token_set_preview = token_set_row_count > max_token_set_rows

    for hint in hints:
        if hint is None:
            assignments.append(None)
            continue
        if skip_token_set_preview and hint.source == "token_set":
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
    diagnostics: dict[str, Any] = {
        "token_set_row_count": token_set_row_count,
        "max_token_set_rows": max_token_set_rows,
        "cluster_preview_skipped": skip_token_set_preview,
    }
    return tuple(out), diagnostics


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
            parts.append(_strip_leading_ticket_metadata(value))
    source_id = support_ticket_plain_text(row.get("source_id"))
    source_title = support_ticket_plain_text(row.get("source_title"))
    if source_title and source_title != source_id:
        parts.append(_strip_leading_ticket_metadata(source_title))
    return support_ticket_tokens(" ".join(part for part in parts if part))


def _strip_plural_suffix(token: str) -> str:
    if token.endswith(("ss", "us", "is", "as")):
        return token
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _strip_leading_ticket_metadata(value: str) -> str:
    return _compact(_LEADING_BRACKETED_METADATA_RE.sub("", value))


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
    "MAX_TOKEN_SET_CLUSTER_ROWS",
    "SupportTicketClusterHint",
    "assign_support_ticket_clusters",
    "assign_support_ticket_clusters_with_diagnostics",
    "support_ticket_cluster_hint",
    "support_ticket_cluster_quality",
    "support_ticket_cluster_summary",
    "support_ticket_plain_text",
    "support_ticket_plain_text_lines",
    "support_ticket_tokens",
]
