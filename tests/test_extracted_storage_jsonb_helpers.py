"""Tests for the shared JSONB / asyncpg helpers extracted in
PR-ContentAssets-Consistency-1.

Locks the contract so all four content-asset adapters (campaigns,
reports, landing pages, sales briefs) share one tested implementation
of the helpers they previously copy-pasted.
"""

from __future__ import annotations

from collections import OrderedDict

from extracted_content_pipeline.storage._jsonb_helpers import (
    decode_jsonb_field,
    json_dump_jsonb,
    parse_command_tag,
    row_to_dict,
)


# ---- json_dump_jsonb --------------------------------------------------------


def test_json_dump_jsonb_serializes_dict_compactly() -> None:
    out = json_dump_jsonb({"a": 1, "b": "hello"})
    assert out == '{"a":1,"b":"hello"}'


def test_json_dump_jsonb_handles_none_as_empty_object() -> None:
    """Callers passing ``meta or None`` shouldn't have to special-case None."""
    assert json_dump_jsonb(None) == "{}"


def test_json_dump_jsonb_falls_back_to_str_for_non_serializable() -> None:
    """``default=str`` lets datetime / UUID etc. round-trip without explicit handling."""

    class _Custom:
        def __str__(self) -> str:
            return "custom-value"

    out = json_dump_jsonb({"x": _Custom()})
    assert out == '{"x":"custom-value"}'


# ---- row_to_dict ------------------------------------------------------------


def test_row_to_dict_passes_through_plain_dict() -> None:
    src = {"a": 1, "b": 2}
    out = row_to_dict(src)
    assert out == src
    assert out is not src  # defensive copy


def test_row_to_dict_coerces_ordered_mapping() -> None:
    src = OrderedDict([("a", 1)])
    assert row_to_dict(src) == {"a": 1}


def test_row_to_dict_returns_empty_for_non_mapping_non_iterable() -> None:
    """Records that don't iterate as key-value pairs fall back to {}."""
    # A string fails dict() coercion (it's iterable but not as kv pairs).
    assert row_to_dict("not_a_record") == {}


# ---- decode_jsonb_field -----------------------------------------------------


def test_decode_jsonb_field_passes_through_pre_decoded_dict() -> None:
    """asyncpg with the json codec installed delivers JSONB pre-decoded."""
    assert decode_jsonb_field({"k": "v"}, default=[]) == {"k": "v"}


def test_decode_jsonb_field_passes_through_pre_decoded_list() -> None:
    assert decode_jsonb_field(["a", "b"], default={}) == ["a", "b"]


def test_decode_jsonb_field_decodes_string_form() -> None:
    assert decode_jsonb_field('{"k":"v"}', default=[]) == {"k": "v"}


def test_decode_jsonb_field_falls_back_to_default_on_malformed_string() -> None:
    """Malformed JSON strings fall back to default rather than raising."""
    assert decode_jsonb_field("not_json{", default={"fallback": True}) == {"fallback": True}


def test_decode_jsonb_field_falls_back_to_default_on_none() -> None:
    assert decode_jsonb_field(None, default=[]) == []


# ---- parse_command_tag ------------------------------------------------------


def test_parse_command_tag_returns_true_on_update_1() -> None:
    assert parse_command_tag("UPDATE 1") is True


def test_parse_command_tag_returns_false_on_update_0() -> None:
    """Wrong-id / wrong-tenant misses surface as False."""
    assert parse_command_tag("UPDATE 0") is False


def test_parse_command_tag_returns_true_on_higher_row_count() -> None:
    """Bulk updates (UPDATE N where N>1) also count as a hit."""
    assert parse_command_tag("UPDATE 5") is True


def test_parse_command_tag_treats_none_as_success() -> None:
    """Test fakes / alt drivers that return None default to True."""
    assert parse_command_tag(None) is True


def test_parse_command_tag_treats_unknown_string_as_success() -> None:
    """'OK' (some drivers) defaults to True so the parse never crashes."""
    assert parse_command_tag("OK") is True


def test_parse_command_tag_handles_other_command_kinds() -> None:
    """``DELETE 2`` / ``INSERT 0 3`` parse the trailing integer."""
    assert parse_command_tag("DELETE 2") is True
    assert parse_command_tag("INSERT 0 3") is True
    assert parse_command_tag("DELETE 0") is False
