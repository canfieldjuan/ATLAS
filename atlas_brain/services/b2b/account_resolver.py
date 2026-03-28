"""B2B account resolution: resolve anonymous review authors to named companies.

Pure, synchronous module with no DB or LLM calls. Reads review dicts,
applies a priority-ordered pipeline of deterministic extractors, and returns
a ResolutionResult with evidence.

Resolution priority:
  1. Direct structured fields (reviewer_company, enrichment company_name)
  2. Bio/title regex extraction ("CTO at Acme", "Engineer @ Stripe")
  3. Source-specific metadata (Reddit flair, Quora credentials, etc.)
  4. Enrichment-extracted company_name (LLM-derived, lower confidence)
"""

from __future__ import annotations

import html as _html
import json
import logging
import re
from dataclasses import dataclass, field as dc_field
from typing import Any
from urllib.parse import urlparse

from ...services.company_normalization import normalize_company_name

logger = logging.getLogger("atlas.services.b2b.account_resolver")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ResolutionSignal:
    signal_type: str
    value: str
    confidence: float
    source_field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "value": self.value,
            "confidence": self.confidence,
            "source_field": self.source_field,
        }


@dataclass
class ExcludedCandidate:
    name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


@dataclass
class ResolutionResult:
    resolved_company_name: str | None = None
    normalized_company_name: str | None = None
    confidence_score: float = 0.0
    confidence_label: str = "unresolved"
    resolution_method: str = "none"
    signals: list[ResolutionSignal] = dc_field(default_factory=list)
    excluded_candidates: list[ExcludedCandidate] = dc_field(default_factory=list)

    def to_evidence_json(self) -> dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "excluded_candidates": [e.to_dict() for e in self.excluded_candidates],
        }


# ---------------------------------------------------------------------------
# Bio regex patterns
# ---------------------------------------------------------------------------

_BIO_COMPANY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "CTO at Acme Corp" / "Engineer at Stripe"
    (re.compile(
        r"\b(?:CEO|CTO|CIO|CISO|CFO|COO|VP|SVP|EVP|Director|Manager|Lead|Head|"
        r"Principal|Sr\.?|Senior|Staff|Engineer|Dev|Developer|Analyst|Consultant|"
        r"Architect|PM|Product\s+Manager|Designer|Admin|Sysadmin|DBA)\s+"
        r"(?:at|@)\s+([A-Z][A-Za-z0-9\s&.'-]{2,40}?)(?:\s*[,.(|]|\s+and\b|\s+but\b|\s+since|\s+for\s+\d|\s*$)",
        re.IGNORECASE,
    ), "title_at_company"),

    # "I work at Acme" / "I work for Acme" / "we are at Acme" / "we use this at Acme"
    # NOTE: `work|worked` allows `at|for`; `am|was|are|use\s+\w+` only allows `at`
    # to prevent false positives like "we use X for task management" → "task management".
    # Negative lookahead blocks article+determiner phrases like "a company that...",
    # "an IT department", "the business", "our team" — these are descriptions, not names.
    (re.compile(
        r"\b(?:I|we)\s+"
        r"(?:(?:work|worked)\s+(?:at|for)|(?:am|was|are|use\s+\w+(?:\s+\w+)?)\s+at)\s+"
        r"(?!a\s|an\s|the\s|my\s|our\s|your\s|their\s|this\s|that\s|some\s|any\s)"
        r"([A-Z][A-Za-z0-9\s&.'-]{2,40}?)(?:\s*[,.(|]|\s+and\b|\s+but\b|\s+since|\s+for\s+\d|\s*$)",
        re.IGNORECASE,
    ), "work_at_company"),

    # "Engineer @ Stripe"
    (re.compile(
        r"\b\w+(?:\s+\w+){0,2}\s*@\s*([A-Z][A-Za-z0-9\s&.'-]{2,30}?)(?:\s*[,.(|]|\s*$)",
        re.IGNORECASE,
    ), "handle_at_company"),

    # "Founder, Acme Inc" / "Co-founder, Acme"
    (re.compile(
        r"\b(?:Founder|Co-?founder|Owner)\s*[,;:]\s+"
        r"([A-Z][A-Za-z0-9\s&.'-]{2,40}?)(?:\s*[,.(|]|\s*$)",
        re.IGNORECASE,
    ), "founder_of_company"),

    # "Acme Corp | Senior Engineer"
    (re.compile(
        r"^([A-Z][A-Za-z0-9\s&.'-]{2,40}?)\s*\|\s*\w+",
        re.IGNORECASE,
    ), "company_pipe_title"),

    # "Former employee at Acme"
    (re.compile(
        r"\bformer(?:ly)?\s+(?:\w+\s+){0,2}(?:at|of|from)\s+"
        r"([A-Z][A-Za-z0-9\s&.'-]{2,40}?)(?:\s*[,.(|]|\s*$)",
        re.IGNORECASE,
    ), "former_at_company"),
]

# Common role words that should NOT be treated as company names
_ROLE_WORDS = frozenset({
    "engineer", "developer", "manager", "director", "lead", "head",
    "analyst", "consultant", "architect", "designer", "admin",
    "sysadmin", "devops", "qa", "tester", "intern", "associate",
    "specialist", "coordinator", "officer", "president", "founder",
    "ceo", "cto", "cio", "cfo", "coo", "vp", "svp", "evp",
})

# Terms that are definitively NOT company names regardless of extraction context.
# Exact-match against the cleaned, lowercased capture.
_REJECT_COMPANY_TERMS = frozenset({
    # Email providers (captured by handle_at_company pattern)
    "gmail", "yahoo", "hotmail", "outlook", "proton", "icloud", "aol",
    # Generic single-word captures from work_at_company / employment_claim
    "support", "work", "tasks", "accounting", "email", "meetings",
    "ticketing", "data", "reports", "tech", "them", "it", "hr",
    "business", "company", "team", "organization", "department", "dept",
    "school", "university", "startup", "agency", "firm", "corp", "inc",
    # Generic multi-word phrases that slip through
    "email marketing", "various purposes", "customer support",
    "project management", "task management", "ticket management",
    "workflow management", "thought management", "customer relationship management",
    "automated project management", "advanced use cases",
    "all our hr materials", "all our employee-related information",
    "warrants the cost", "is expiring",
    "meetings and scheduling", "my new company",
    # Founder pattern false positives
    "entrepreneur", "freelancer", "self-employed", "indie hacker",
    # Reddit flair false positives
    "moderator", "mod", "admin", "helper", "verified",
})


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _clean_extracted_name(raw: str) -> str:
    """Clean up regex-extracted company name."""
    name = raw.strip()
    # Remove trailing role-like words
    words = name.split()
    while words and words[-1].lower() in _ROLE_WORDS:
        words.pop()
    name = " ".join(words).strip()
    # Remove trailing punctuation
    name = re.sub(r"[,;:.|-]+$", "", name).strip()
    return name


def _extract_from_bio_regex(text: str, field_name: str) -> ResolutionSignal | None:
    """Apply bio regex patterns to a text field. Returns first match."""
    if not text or len(text) < 5:
        return None
    for pattern, signal_type in _BIO_COMPANY_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = _clean_extracted_name(m.group(1))
            if raw and len(raw) >= 2 and raw.lower() not in _REJECT_COMPANY_TERMS:
                return ResolutionSignal(
                    signal_type=signal_type,
                    value=raw,
                    confidence=0.6,
                    source_field=field_name,
                )
    return None


def _extract_from_reviewer_company(review: dict) -> ResolutionSignal | None:
    """Priority 1: Direct reviewer_company field from parser."""
    company = (review.get("reviewer_company") or "").strip()
    if not company:
        return None
    return ResolutionSignal(
        signal_type="reviewer_company_field",
        value=company,
        confidence=0.9,
        source_field="reviewer_company",
    )


def _extract_from_enrichment(review: dict) -> ResolutionSignal | None:
    """Priority 4: LLM-extracted company_name from enrichment."""
    enrichment = review.get("enrichment")
    if not enrichment:
        return None
    if isinstance(enrichment, str):
        try:
            enrichment = json.loads(enrichment)
        except (json.JSONDecodeError, TypeError):
            return None
    ctx = enrichment.get("reviewer_context")
    if not isinstance(ctx, dict):
        return None
    company = (ctx.get("company_name") or "").strip()
    if not company:
        return None
    return ResolutionSignal(
        signal_type="enrichment_company_name",
        value=company,
        confidence=0.35,
        source_field="enrichment.reviewer_context.company_name",
    )


def _extract_from_reddit_metadata(review: dict) -> list[ResolutionSignal]:
    """Priority 3a: Reddit flair and employment claims."""
    signals: list[ResolutionSignal] = []
    raw_meta = review.get("raw_metadata")
    if not isinstance(raw_meta, dict):
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except (json.JSONDecodeError, TypeError):
                return signals
        else:
            return signals

    # Reddit flair: often "Company Name | Role" or "Role @ Company"
    flair = (raw_meta.get("author_flair_text") or "").strip()
    if flair:
        # Try "Company | Role" format
        if "|" in flair:
            parts = flair.split("|", 1)
            candidate = parts[0].strip()
            if candidate and candidate.lower() not in _ROLE_WORDS and len(candidate) >= 3:
                signals.append(ResolutionSignal(
                    signal_type="reddit_flair_company",
                    value=candidate,
                    confidence=0.65,
                    source_field="raw_metadata.author_flair_text",
                ))
        # Try bio regex on flair text
        bio_signal = _extract_from_bio_regex(flair, "raw_metadata.author_flair_text")
        if bio_signal:
            bio_signal.confidence = 0.6
            signals.append(bio_signal)

    # Employment claim from insider evidence extraction
    if raw_meta.get("employment_claim"):
        # The review text itself may contain "I work at X"
        text = (review.get("review_text") or "")[:500]
        bio_signal = _extract_from_bio_regex(text, "review_text")
        if bio_signal:
            bio_signal.confidence = 0.55
            bio_signal.signal_type = "reddit_employment_claim"
            signals.append(bio_signal)

    return signals


def _extract_from_quora_metadata(review: dict) -> list[ResolutionSignal]:
    """Priority 3c: Quora author credentials in reviewer_title."""
    signals: list[ResolutionSignal] = []
    title = (review.get("reviewer_title") or "").strip()
    if not title:
        return signals
    source = review.get("source", "")
    if source != "quora":
        return signals
    # Quora credentials: "Software Engineer at Google (2015-present)"
    bio_signal = _extract_from_bio_regex(title, "reviewer_title")
    if bio_signal:
        bio_signal.confidence = 0.65
        bio_signal.signal_type = "quora_credentials"
        signals.append(bio_signal)
    return signals


def _extract_from_producthunt_metadata(review: dict) -> list[ResolutionSignal]:
    """Priority 3d: ProductHunt user headline in reviewer_title."""
    signals: list[ResolutionSignal] = []
    title = (review.get("reviewer_title") or "").strip()
    if not title:
        return signals
    source = review.get("source", "")
    if source != "producthunt":
        return signals
    bio_signal = _extract_from_bio_regex(title, "reviewer_title")
    if bio_signal:
        bio_signal.confidence = 0.6
        bio_signal.signal_type = "producthunt_headline"
        signals.append(bio_signal)
    return signals


def _extract_from_title_bio(review: dict) -> list[ResolutionSignal]:
    """Priority 2: Bio regex on reviewer_title for any source."""
    signals: list[ResolutionSignal] = []
    title = (review.get("reviewer_title") or "").strip()
    if title:
        sig = _extract_from_bio_regex(title, "reviewer_title")
        if sig:
            signals.append(sig)
    return signals


def _extract_from_review_text(review: dict) -> list[ResolutionSignal]:
    """Priority 2b: Bio regex on review_text and summary (any source).

    Only scans the first 500 chars of review_text to avoid noise from
    longer posts. Lower confidence than title/flair since review text
    is noisier.
    """
    signals: list[ResolutionSignal] = []
    # Summary first (shorter, more likely to self-identify)
    summary = (review.get("summary") or "").strip()
    if summary:
        sig = _extract_from_bio_regex(summary, "summary")
        if sig:
            sig.confidence = 0.5
            signals.append(sig)
    # First 500 chars of review text
    text = (review.get("review_text") or "")[:500].strip()
    if text:
        sig = _extract_from_bio_regex(text, "review_text")
        if sig:
            sig.confidence = 0.45
            signals.append(sig)
    return signals


def _extract_from_hackernews_metadata(review: dict) -> list[ResolutionSignal]:
    """Priority 3b: HackerNews-specific signals.

    HN parser stores author username in reviewer_name. The "about" field
    is not fetched yet, but review text often contains self-identification
    ("at my company X", "we use this at Y").
    """
    signals: list[ResolutionSignal] = []
    source = (review.get("source") or "").lower()
    if source != "hackernews":
        return signals
    # Scan review text for company mentions (HN posts are often first-person)
    text = (review.get("review_text") or "")[:500]
    if text:
        sig = _extract_from_bio_regex(text, "review_text")
        if sig:
            sig.confidence = 0.45
            sig.signal_type = "hackernews_text_company"
            signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def _apply_guardrails(
    candidate: str,
    vendor_name: str,
    blocked_names: set[str] | None = None,
    *,
    trust_direct: bool = False,
) -> ExcludedCandidate | None:
    """Check if a candidate should be excluded. Returns None if it passes.

    trust_direct=True skips the domain_like check for signals that come
    directly from the reviewer (reviewer_company_field, github_profile_company).
    Companies like 'kore.ai' or 'Purplle.com' are real brand names and should
    not be blocked when the reviewer explicitly declared them.
    All other checks (vendor match, blocked, generic, too_short) still apply.
    """
    if not candidate or not candidate.strip():
        return ExcludedCandidate(name=candidate or "", reason="empty")

    normalized = normalize_company_name(candidate)
    if not normalized:
        return ExcludedCandidate(name=candidate, reason="empty_after_normalization")

    vendor_norm = normalize_company_name(vendor_name)
    if normalized == vendor_norm:
        return ExcludedCandidate(name=candidate, reason="incumbent_vendor")

    if blocked_names and normalized in blocked_names:
        return ExcludedCandidate(name=candidate, reason="blocked_name")

    # Generic descriptors
    _GENERIC = {
        "company", "mycompany", "my company", "ourcompany", "our company",
        "customer", "client", "startup", "enterprise", "organization",
        "agency", "firm", "business", "employer",
    }
    if normalized in _GENERIC:
        return ExcludedCandidate(name=candidate, reason="generic_descriptor")

    # Too short
    if len(normalized) < 2:
        return ExcludedCandidate(name=candidate, reason="too_short")

    # Looks like a domain — skipped for direct reviewer declarations
    if not trust_direct:
        if re.match(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", normalized) and " " not in normalized:
            return ExcludedCandidate(name=candidate, reason="domain_like")

    return None


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------

def _compute_confidence(signals: list[ResolutionSignal]) -> tuple[float, str]:
    """Compute composite confidence from collected signals.

    Returns (score, label).
    """
    if not signals:
        return 0.0, "unresolved"

    # Use the best single signal as base
    best = max(signals, key=lambda s: s.confidence)
    score = best.confidence

    # Boost if multiple signals agree on the same company
    if len(signals) > 1:
        normalized_values = {normalize_company_name(s.value) for s in signals}
        if len(normalized_values) == 1:
            # All signals agree
            score = min(score + 0.1 * (len(signals) - 1), 0.95)

    if score >= 0.8:
        return round(score, 2), "high"
    if score >= 0.5:
        return round(score, 2), "medium"
    if score >= 0.3:
        return round(score, 2), "low"
    return round(score, 2), "unresolved"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def resolve_review(
    review: dict[str, Any],
    *,
    vendor_name: str,
    blocked_names: set[str] | None = None,
) -> ResolutionResult:
    """Run the full resolution pipeline on a single review.

    Applies extractors in priority order, collects signals, applies guardrails,
    picks the highest-confidence passing candidate.
    """
    all_signals: list[ResolutionSignal] = []
    excluded: list[ExcludedCandidate] = []

    # Collect signals from all extractors (priority order)
    extractors: list[ResolutionSignal | None] = [
        # Priority 1: Direct structured fields
        _extract_from_reviewer_company(review),
    ]
    # Priority 2: Bio/title regex (any source)
    extractors.extend(_extract_from_title_bio(review))
    # Priority 2b: Review text and summary (any source)
    extractors.extend(_extract_from_review_text(review))
    # Priority 3: Source-specific metadata
    source = (review.get("source") or "").lower()
    if source == "reddit":
        extractors.extend(_extract_from_reddit_metadata(review))
        # Reddit profile data injected by task
        reddit_profile = review.get("_reddit_profile")
        if isinstance(reddit_profile, dict):
            sig = extract_from_reddit_profile(reddit_profile)
            if sig:
                extractors.append(sig)
    elif source == "quora":
        extractors.extend(_extract_from_quora_metadata(review))
    elif source == "producthunt":
        extractors.extend(_extract_from_producthunt_metadata(review))
    elif source == "hackernews":
        extractors.extend(_extract_from_hackernews_metadata(review))
        # HN profile data injected by task
        hn_profile = review.get("_hn_profile")
        if isinstance(hn_profile, dict):
            sig = extract_from_hn_profile(hn_profile)
            if sig:
                extractors.append(sig)
    elif source == "github":
        # GitHub profile data injected by task
        gh_profile = review.get("_gh_profile")
        if isinstance(gh_profile, dict):
            extractors.extend(extract_from_github_profile(gh_profile))
    # Priority 4: Enrichment company_name (low confidence, LLM-derived)
    extractors.append(_extract_from_enrichment(review))

    # Filter None values
    raw_signals = [s for s in extractors if s is not None]

    # Apply guardrails to each signal
    # Signals from direct reviewer declarations bypass the domain_like check
    # because brands like 'kore.ai' or 'Purplle.com' are real company names.
    _TRUST_DIRECT_TYPES = {"reviewer_company_field", "github_profile_company"}
    passing: list[ResolutionSignal] = []
    for sig in raw_signals:
        exclusion = _apply_guardrails(
            sig.value, vendor_name, blocked_names,
            trust_direct=sig.signal_type in _TRUST_DIRECT_TYPES,
        )
        if exclusion:
            excluded.append(exclusion)
        else:
            passing.append(sig)

    if not passing:
        return ResolutionResult(
            signals=[s for s in raw_signals],
            excluded_candidates=excluded,
        )

    # Pick best passing signal
    best = max(passing, key=lambda s: s.confidence)

    # Collect all signals that agree with the best
    best_norm = normalize_company_name(best.value)
    agreeing = [s for s in passing if normalize_company_name(s.value) == best_norm]

    score, label = _compute_confidence(agreeing)

    return ResolutionResult(
        resolved_company_name=best.value,
        normalized_company_name=best_norm or None,
        confidence_score=score,
        confidence_label=label,
        resolution_method=best.signal_type,
        signals=raw_signals,
        excluded_candidates=excluded,
    )


# ---------------------------------------------------------------------------
# Async profile fetchers (called by task before sync resolver)
# ---------------------------------------------------------------------------

_HN_USER_API = "https://hacker-news.firebaseio.com/v0/user/{username}.json"
_GITHUB_USER_API = "https://api.github.com/users/{username}"
_REDDIT_USER_API = "https://www.reddit.com/user/{username}/about.json"
_REDDIT_USER_AGENT = "python:atlas.b2b.resolver:v1.0 (by /u/atlas_bot)"

# Domains that are platforms/tools, not company identifiers
_PLATFORM_DOMAINS = frozenset({
    "github.com", "gitlab.com", "bitbucket.org",
    "twitter.com", "x.com", "linkedin.com", "facebook.com", "instagram.com",
    "youtube.com", "keybase.io", "medium.com", "substack.com", "notion.so",
    "t.co", "bit.ly", "tinyurl.com", "ycombinator.com", "news.ycombinator.com",
    "stackoverflow.com", "reddit.com", "quora.com", "producthunt.com",
    "google.com", "apple.com", "microsoft.com", "amazon.com", "cloudflare.com",
    "oreilly.com", "npmjs.com", "pypi.org", "rubygems.org",
})


def _domain_to_company_candidate(url: str) -> str | None:
    """Extract a company name candidate from a URL.

    Returns the SLD (second-level domain) capitalised, or None if the URL
    points to a known platform, a personal-looking domain, or is too short.

    Examples:
        https://stripe.com   -> "Stripe"
        https://webiphany.com -> "Webiphany"
        https://twitter.com  -> None  (platform)
        https://metachris.com -> None (looks personal)
    """
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or parsed.path).lower().split(":")[0]
    except Exception:
        return None

    host = re.sub(r"^www\.", "", host)
    if not host or host in _PLATFORM_DOMAINS:
        return None

    # Extract SLD (e.g. "stripe" from "stripe.com")
    parts = host.split(".")
    name = parts[0] if parts else ""
    if len(name) < 3:
        return None

    # Skip if the name looks like a personal handle (email-like chars, digits only)
    if re.match(r"^[0-9]+$", name):
        return None

    return name.capitalize()


async def fetch_hn_profile(
    username: str,
    http_client: Any = None,
    *,
    timeout: float = 10.0,
) -> dict[str, str]:
    """Fetch HackerNews user profile. Returns {about, company_from_about, profile_urls}.

    HN API is public, no auth needed. The 'about' field is HTML with entities
    (e.g. &#x2F; for /) — we decode entities before stripping tags so that
    URLs and bio text are fully readable for regex and domain extraction.
    """
    if not username or not username.strip():
        return {}
    import httpx
    client = http_client or httpx.AsyncClient(timeout=timeout)
    close_after = http_client is None
    try:
        url = _HN_USER_API.format(username=username.strip())
        resp = await client.get(url)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        about_raw = (data.get("about") or "").strip()
        if not about_raw:
            return {}
        # Decode HTML entities FIRST (&#x2F; -> /, &amp; -> &, etc.)
        # Must happen before tag stripping so URLs become parseable.
        about_decoded = _html.unescape(about_raw)
        # Extract URLs from decoded text before removing tags
        profile_urls = re.findall(r'https?://[^\s<>"\']+', about_decoded)
        # Strip HTML tags, collapse whitespace
        about_clean = re.sub(r"<[^>]+>", " ", about_decoded)
        about_clean = re.sub(r"\s+", " ", about_clean).strip()
        result: dict = {"about": about_clean}
        if profile_urls:
            result["profile_urls"] = profile_urls[:5]
        # Try to extract company from bio text
        sig = _extract_from_bio_regex(about_clean, "hn_profile_about")
        if sig:
            result["company_from_about"] = sig.value
        return result
    except Exception:
        logger.debug("HN profile fetch failed for %s", username, exc_info=True)
        return {}
    finally:
        if close_after:
            await client.aclose()


async def fetch_github_profile(
    username: str,
    http_client: Any = None,
    *,
    timeout: float = 10.0,
) -> dict[str, str]:
    """Fetch GitHub user profile. Returns {company, bio, blog}.

    GitHub API is public (60 req/hr unauthenticated). The 'company' field
    is explicitly set by users and is the strongest signal.
    """
    if not username or not username.strip():
        return {}
    import httpx
    client = http_client or httpx.AsyncClient(timeout=timeout)
    close_after = http_client is None
    try:
        url = _GITHUB_USER_API.format(username=username.strip())
        resp = await client.get(url)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        result: dict[str, str] = {}
        company = (data.get("company") or "").strip()
        # GitHub company field often starts with @ for org handles
        if company:
            company = company.lstrip("@").strip()
            if company:
                result["company"] = company
        bio = (data.get("bio") or "").strip()
        if bio:
            result["bio"] = bio
        blog = (data.get("blog") or "").strip()
        if blog:
            result["blog"] = blog
        return result
    except Exception:
        logger.debug("GitHub profile fetch failed for %s", username, exc_info=True)
        return {}
    finally:
        if close_after:
            await client.aclose()


def extract_from_hn_profile(profile: dict[str, str]) -> ResolutionSignal | None:
    """Extract company signal from a fetched HN profile.

    Priority:
      1. company_from_about — bio regex matched a company name in the about text
      2. bio regex re-run on about (redundant guard, handles edge cases)
      3. profile_urls — domain of first non-platform URL in the about field
    """
    company = (profile.get("company_from_about") or "").strip()
    if company:
        return ResolutionSignal(
            signal_type="hn_profile_about",
            value=company,
            confidence=0.65,
            source_field="hn_user_profile.about",
        )
    # Re-run bio regex on cleaned about text
    about = (profile.get("about") or "").strip()
    if about:
        sig = _extract_from_bio_regex(about, "hn_user_profile.about")
        if sig:
            sig.confidence = 0.6
            sig.signal_type = "hn_profile_about"
            return sig
    # Fall back: try to derive a company name from profile URLs in the about field.
    # Many HN users list their company website (e.g. https://stripe.com) without
    # any textual "at X" marker.
    for url in (profile.get("profile_urls") or []):
        candidate = _domain_to_company_candidate(url)
        if candidate:
            return ResolutionSignal(
                signal_type="hn_profile_url_domain",
                value=candidate,
                confidence=0.45,
                source_field="hn_user_profile.about_urls",
            )
    return None


def extract_from_github_profile(profile: dict[str, str]) -> list[ResolutionSignal]:
    """Extract company signals from a fetched GitHub profile."""
    signals: list[ResolutionSignal] = []
    # GitHub company field is the strongest signal
    company = (profile.get("company") or "").strip()
    if company:
        signals.append(ResolutionSignal(
            signal_type="github_profile_company",
            value=company,
            confidence=0.8,
            source_field="github_user_profile.company",
        ))
    # Bio might also contain employer info
    bio = (profile.get("bio") or "").strip()
    if bio:
        sig = _extract_from_bio_regex(bio, "github_user_profile.bio")
        if sig:
            sig.confidence = 0.6
            sig.signal_type = "github_profile_bio"
            signals.append(sig)
    return signals


async def fetch_reddit_profile(
    username: str,
    http_client: Any = None,
    *,
    timeout: float = 10.0,
) -> dict[str, str]:
    """Fetch Reddit user profile. Returns {bio, profile_urls}.

    Reddit public API — no auth required. Sets a descriptive User-Agent to
    comply with Reddit's API terms (otherwise returns 429/403).
    Respects Reddit's rate limit; callers should use a semaphore.
    """
    if not username or not username.strip():
        return {}
    import httpx
    headers = {"User-Agent": _REDDIT_USER_AGENT}
    client = http_client or httpx.AsyncClient(timeout=timeout, headers=headers)
    close_after = http_client is None
    try:
        url = _REDDIT_USER_API.format(username=username.strip())
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        user_data = data.get("data") or {}
        result: dict[str, str] = {}
        # Public description lives under subreddit (user's profile page)
        subreddit = user_data.get("subreddit") or {}
        bio_raw = (subreddit.get("public_description") or "").strip()
        if bio_raw:
            # Decode HTML entities, strip tags
            bio_decoded = _html.unescape(bio_raw)
            profile_urls = re.findall(r'https?://[^\s<>"\']+', bio_decoded)
            bio_clean = re.sub(r"<[^>]+>", " ", bio_decoded)
            bio_clean = re.sub(r"\s+", " ", bio_clean).strip()
            if bio_clean:
                result["bio"] = bio_clean
            if profile_urls:
                result["profile_urls"] = profile_urls[:5]
        return result
    except Exception:
        logger.debug("Reddit profile fetch failed for %s", username, exc_info=True)
        return {}
    finally:
        if close_after:
            await client.aclose()


def extract_from_reddit_profile(profile: dict[str, str]) -> ResolutionSignal | None:
    """Extract company signal from a fetched Reddit profile.

    Priority:
      1. Bio regex match on public_description text
      2. Domain of first non-platform URL in bio
    """
    bio = (profile.get("bio") or "").strip()
    if bio:
        sig = _extract_from_bio_regex(bio, "reddit_user_profile.bio")
        if sig:
            sig.confidence = 0.55
            sig.signal_type = "reddit_profile_bio"
            return sig
    for url in (profile.get("profile_urls") or []):
        candidate = _domain_to_company_candidate(url)
        if candidate:
            return ResolutionSignal(
                signal_type="reddit_profile_url_domain",
                value=candidate,
                confidence=0.4,
                source_field="reddit_user_profile.bio_urls",
            )
    return None
