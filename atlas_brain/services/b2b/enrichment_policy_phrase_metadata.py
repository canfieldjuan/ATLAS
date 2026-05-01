from __future__ import annotations


PHRASE_METADATA_FIELDS: tuple[str, ...] = (
    "specific_complaints",
    "pricing_phrases",
    "feature_gaps",
    "quotable_phrases",
    "recommendation_language",
    "positive_aspects",
)
PHRASE_SUBJECT_VALUES: tuple[str, ...] = (
    "subject_vendor", "alternative", "self", "third_party", "unclear",
)
PHRASE_POLARITY_VALUES: tuple[str, ...] = (
    "negative", "positive", "mixed", "unclear",
)
PHRASE_ROLE_VALUES: tuple[str, ...] = (
    "primary_driver", "supporting_context", "passing_mention", "unclear",
)
PHRASE_UNCLEAR = "unclear"
