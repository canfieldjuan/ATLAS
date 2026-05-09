from __future__ import annotations

from extracted_content_pipeline.services._parse_retry_helpers import (
    accumulate_usage,
    clip_invalid_response,
    parse_attempt_limit,
    retry_prompt_with_invalid_response,
)


def test_clip_invalid_response_strips_and_caps_text() -> None:
    assert clip_invalid_response("  abcdef  ", limit=4) == "abcd"


def test_clip_invalid_response_returns_cleaned_text_when_within_limit() -> None:
    assert clip_invalid_response("  ok  ", limit=10) == "ok"


def test_parse_attempt_limit_counts_initial_attempt_plus_retries() -> None:
    assert parse_attempt_limit(0) == 1
    assert parse_attempt_limit(1) == 2
    assert parse_attempt_limit(2) == 3


def test_parse_attempt_limit_clamps_negative_values_to_initial_attempt() -> None:
    assert parse_attempt_limit(-3) == 1


def test_retry_prompt_with_invalid_response_returns_base_without_prior() -> None:
    assert retry_prompt_with_invalid_response(
        "Base prompt",
        prior_invalid_response="",
        instruction="Return JSON.",
    ) == "Base prompt"


def test_retry_prompt_with_invalid_response_appends_instruction_and_excerpt() -> None:
    assert retry_prompt_with_invalid_response(
        "Base prompt",
        prior_invalid_response="{bad",
        instruction="Return JSON.",
    ) == "Base prompt\n\nReturn JSON. Previous response excerpt:\n{bad"


def test_accumulate_usage_adds_numeric_fields() -> None:
    assert accumulate_usage(
        {"input_tokens": 10, "output_tokens": 5},
        {"input_tokens": 2, "output_tokens": 3},
    ) == {"input_tokens": 12, "output_tokens": 8}


def test_accumulate_usage_preserves_bool_and_metadata_fields() -> None:
    assert accumulate_usage(
        {"cached": False, "model": "first"},
        {"cached": True, "model": "second", "provider": "host"},
    ) == {"cached": True, "model": "second", "provider": "host"}


def test_accumulate_usage_ignores_non_mapping_usage() -> None:
    assert accumulate_usage({"input_tokens": 10}, None) == {"input_tokens": 10}
