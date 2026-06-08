"""Brand-voice profile helpers for content-ops LLM generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .campaign_ports import JsonDict, TenantScope


_MAX_DESCRIPTOR_COUNT = 8
_MAX_DESCRIPTOR_CHARS = 120
_MAX_EXEMPLAR_COUNT = 3
_MAX_EXEMPLAR_CHARS = 1200
_MAX_BANNED_TERM_CHARS = 120
_SUPPORTED_POV = {
    "first_person": "first_person",
    "first": "first_person",
    "we": "first_person",
    "second_person": "second_person",
    "second": "second_person",
    "you": "second_person",
    "third_person": "third_person",
    "third": "third_person",
}
_PLAIN_READING_LEVELS = frozenset({"plain", "simple", "accessible"})
_CONCISE_READING_LEVELS = frozenset({"concise", "short"})


@dataclass(frozen=True)
class BrandVoiceProfile:
    """Request-time brand voice profile for one tenant/account."""

    id: str = ""
    account_id: str = ""
    name: str = ""
    descriptors: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    banned_terms: tuple[str, ...] = ()
    preferred_pov: str | None = None
    reading_level: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def has_guidance(self) -> bool:
        return bool(
            self.descriptors
            or self.exemplars
            or self.banned_terms
            or self.preferred_pov
            or self.reading_level
        )


def brand_voice_profile_from_mapping(
    value: Mapping[str, Any] | BrandVoiceProfile | None,
    *,
    scope: TenantScope | None = None,
    profile_id: str | None = None,
) -> BrandVoiceProfile | None:
    """Normalize an inline profile and fail closed on cross-account drift."""

    if value is None:
        if _clean(profile_id):
            raise ValueError("brand_voice_profile_id requires inputs.brand_voice")
        return None
    if isinstance(value, BrandVoiceProfile):
        _validate_profile_id(value.id, profile_id)
        _validate_scope(value, scope)
        return value
    if not isinstance(value, Mapping):
        raise ValueError("brand_voice must be an object")

    inline_profile_id = _clean(value.get("id") or value.get("profile_id"))
    _validate_profile_id(inline_profile_id, profile_id)
    profile = BrandVoiceProfile(
        id=inline_profile_id or _clean(profile_id),
        account_id=_clean(value.get("account_id") or getattr(scope, "account_id", "")),
        name=_clean(value.get("name") or value.get("label")),
        descriptors=_clean_sequence(
            value.get("descriptors")
            or value.get("descriptor")
            or value.get("tone")
            or value.get("voice"),
            limit=_MAX_DESCRIPTOR_COUNT,
            char_limit=_MAX_DESCRIPTOR_CHARS,
        ),
        exemplars=_exemplars(value.get("exemplars") or value.get("samples")),
        banned_terms=_clean_sequence(
            value.get("banned_terms") or value.get("forbidden_terms"),
            limit=20,
            char_limit=_MAX_BANNED_TERM_CHARS,
        ),
        preferred_pov=_preferred_pov(value.get("preferred_pov") or value.get("pov")),
        reading_level=_clean(value.get("reading_level")),
        metadata=dict(value.get("metadata") or {}),
    )
    _validate_scope(profile, scope)
    return profile if profile.has_guidance() or profile.id else None


def apply_brand_voice_to_system_prompt(
    prompt: str,
    profile: BrandVoiceProfile | None,
) -> str:
    """Inject brand-voice guidance into the grounded system prompt."""

    block = brand_voice_prompt_block(profile)
    if "{brand_voice}" in prompt:
        return prompt.replace("{brand_voice}", block)
    if not block:
        return prompt
    return f"{prompt.rstrip()}\n\n{block}"


def brand_voice_prompt_block(profile: BrandVoiceProfile | None) -> str:
    if profile is None or not profile.has_guidance():
        return ""
    lines = [
        "## Brand voice",
        "Use this profile as style guidance only. It changes wording, rhythm, "
        "and tone; it must not add facts, claims, evidence, or source details. "
        "Do not omit or alter grounded claims to satisfy the voice profile.",
    ]
    if profile.name:
        lines.append(f"Profile: {profile.name}")
    if profile.descriptors:
        lines.append("Descriptors: " + ", ".join(profile.descriptors))
    if profile.preferred_pov:
        lines.append(f"Preferred POV: {profile.preferred_pov}")
    if profile.reading_level:
        lines.append(f"Reading level: {profile.reading_level}")
    if profile.banned_terms:
        lines.append("Avoid these terms: " + ", ".join(profile.banned_terms))
    if profile.exemplars:
        lines.append("Style exemplars:")
        for index, exemplar in enumerate(profile.exemplars, start=1):
            lines.append(f"{index}. {exemplar}")
    return "\n".join(lines)


def brand_voice_result_metadata(
    parsed: Mapping[str, Any],
    profile: BrandVoiceProfile | None,
) -> dict[str, Any]:
    if profile is None:
        return dict(parsed)
    return {
        **dict(parsed),
        "_brand_voice_profile": brand_voice_profile_metadata(profile),
        "_brand_voice_audit": brand_voice_audit(parsed, profile),
    }


def brand_voice_profile_metadata(profile: BrandVoiceProfile) -> JsonDict:
    return {
        key: value
        for key, value in {
            "id": profile.id,
            "account_id": profile.account_id,
            "name": profile.name,
            "descriptors": list(profile.descriptors),
            "preferred_pov": profile.preferred_pov,
            "reading_level": profile.reading_level,
        }.items()
        if value not in ("", None, [], {})
    }


def brand_voice_audit(parsed: Mapping[str, Any], profile: BrandVoiceProfile) -> JsonDict:
    text = _flatten_text(parsed)
    lower_text = text.lower()
    banned_terms = [
        term for term in profile.banned_terms if term and term.lower() in lower_text
    ]
    warnings: list[str] = []
    pov_warning = _pov_warning(lower_text, profile.preferred_pov)
    if pov_warning:
        warnings.append(pov_warning)
    reading_warning = _reading_level_warning(text, profile.reading_level)
    if reading_warning:
        warnings.append(reading_warning)
    return {
        "passed": not banned_terms and not warnings,
        "banned_terms": banned_terms,
        "warnings": warnings,
    }


def brand_voice_quality_blockers(parsed: Mapping[str, Any]) -> tuple[str, ...]:
    """Return fail-visible blocker codes for failed brand-voice audits."""

    audit = parsed.get("_brand_voice_audit")
    if not isinstance(audit, Mapping) or audit.get("passed") is True:
        return ()
    blockers: list[str] = []
    for warning in audit.get("warnings") or ():
        warning_text = _clean(warning)
        if warning_text:
            blockers.append(f"brand_voice:{warning_text}")
    for term in audit.get("banned_terms") or ():
        term_text = _clean(term)
        if term_text:
            blockers.append(f"brand_voice:banned_term:{term_text}")
    return tuple(dict.fromkeys(blockers))


def _validate_scope(profile: BrandVoiceProfile, scope: TenantScope | None) -> None:
    expected = _clean(getattr(scope, "account_id", ""))
    if expected and profile.account_id and expected != profile.account_id:
        raise ValueError("brand_voice.account_id does not match tenant scope")


def _validate_profile_id(inline_id: str, profile_id: str | None) -> None:
    expected = _clean(profile_id)
    if expected and inline_id and expected != inline_id:
        raise ValueError("brand_voice.id does not match brand_voice_profile_id")


def _preferred_pov(value: Any) -> str | None:
    text = _clean(value).lower().replace("-", "_").replace(" ", "_")
    return _SUPPORTED_POV.get(text)


def _exemplars(value: Any) -> tuple[str, ...]:
    items: list[str] = []
    raw_items: Sequence[Any]
    if value is None:
        raw_items = ()
    elif isinstance(value, str):
        raw_items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_items = value
    else:
        raw_items = ()
    for item in raw_items:
        if isinstance(item, Mapping):
            text = _clean(item.get("text") or item.get("content") or item.get("body"))
        else:
            text = _clean(item)
        if text and text not in items:
            items.append(text[:_MAX_EXEMPLAR_CHARS])
        if len(items) >= _MAX_EXEMPLAR_COUNT:
            break
    return tuple(items)


def _clean_sequence(
    value: Any,
    *,
    limit: int,
    char_limit: int | None = None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_items = (value,) if isinstance(value, str) else value
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (bytes, bytearray)):
        return ()
    items: list[str] = []
    for item in raw_items:
        text = _clean(item)
        if char_limit is not None:
            text = text[:char_limit]
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return tuple(items)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(
            _flatten_text(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def _pov_warning(lower_text: str, preferred_pov: str | None) -> str:
    if not preferred_pov:
        return ""
    has_you = bool(re.search(r"\b(you|your|yours)\b", lower_text))
    has_we = bool(re.search(r"\b(we|our|ours|us)\b", lower_text))
    if preferred_pov == "second_person" and not has_you:
        return "preferred_pov_second_person_not_detected"
    if preferred_pov == "first_person" and not has_we:
        return "preferred_pov_first_person_not_detected"
    if preferred_pov == "third_person" and (has_you or has_we):
        return "preferred_pov_third_person_mixed"
    return ""


def _reading_level_warning(text: str, reading_level: str | None) -> str:
    level = _clean(reading_level).lower()
    if level not in _PLAIN_READING_LEVELS and level not in _CONCISE_READING_LEVELS:
        return ""
    sentences = [
        sentence for sentence in re.split(r"[.!?]+", text) if sentence.strip()
    ] or [text]
    word_count = sum(len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences)
    average_words = word_count / max(1, len(sentences))
    if level in _CONCISE_READING_LEVELS and average_words > 18:
        return "reading_level_concise_exceeded"
    if level in _PLAIN_READING_LEVELS and average_words > 24:
        return "reading_level_plain_exceeded"
    return ""


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


__all__ = [
    "BrandVoiceProfile",
    "apply_brand_voice_to_system_prompt",
    "brand_voice_audit",
    "brand_voice_quality_blockers",
    "brand_voice_profile_from_mapping",
    "brand_voice_profile_metadata",
    "brand_voice_prompt_block",
    "brand_voice_result_metadata",
]
