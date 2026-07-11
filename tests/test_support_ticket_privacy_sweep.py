"""Adversarial grammar sweep for the support-ticket privacy predicates.

Generates the marker vocabulary from its semantic families (stems x key
structure x audiences x value shapes x containers) in BOTH error directions,
so the closed rule is exercised across the whole space instead of
per-reviewer-finding spellings. Every case is deterministic.
"""

from __future__ import annotations

import itertools

import pytest

from extracted_content_pipeline.support_ticket_privacy import (
    support_ticket_comment_is_private,
    support_ticket_row_is_private,
)

_PRIVATE_STEMS = ("private", "internal", "hidden", "confidential", "restricted")
_PRIVATE_AUDIENCES = ("agent", "agents", "staff", "admin", "team", "support")
_PUBLIC_AUDIENCES = (
    "customer", "customers", "user", "users", "requester", "client", "viewers",
)
_TRUE_VALUES = (True, "true", "yes", 1, "1", "on")
_FALSE_VALUES = (False, "false", "no", 0, "0.0", "off")


def _private_assertion_cases() -> list[tuple[str, object]]:
    cases = []
    for stem, value in itertools.product(_PRIVATE_STEMS, _TRUE_VALUES):
        for form in (stem, f"is_{stem}", f"{stem}_flag", f"is{stem}"):
            cases.append((form, value))
    return cases


@pytest.mark.parametrize(("key", "value"), _private_assertion_cases())
def test_sweep_private_assertion_keys_fail_closed(key: str, value: object) -> None:
    assert support_ticket_comment_is_private({key: value, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "value"),
    [(stem, value) for stem in _PRIVATE_STEMS for value in _FALSE_VALUES],
)
def test_sweep_private_assertion_false_passes(key: str, value: object) -> None:
    assert support_ticket_comment_is_private({key: value, "body": "x"}) is False


@pytest.mark.parametrize(
    "key",
    [f"visible_to_{aud}" for aud in _PRIVATE_AUDIENCES]
    + [f"{aud}_visible" for aud in _PRIVATE_AUDIENCES]
    + [f"visible_{aud}" for aud in _PRIVATE_AUDIENCES]
    + [f"{aud}_only" for aud in _PRIVATE_AUDIENCES]
    + [f"hidden_from_{aud}" for aud in _PUBLIC_AUDIENCES],
)
def test_sweep_private_audience_visibility_keys_fail_closed(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is True


@pytest.mark.parametrize("key", [f"visible_to_{aud}" for aud in _PUBLIC_AUDIENCES])
def test_sweep_public_audience_visibility(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is False
    assert support_ticket_comment_is_private({key: False, "body": "x"}) is True


@pytest.mark.parametrize(
    "key",
    ["not_public", "not_visible", "not_customer_facing", "non_public",
     "not_publicly_visible"],
)
def test_sweep_negated_public_keys_fail_closed(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is True


def test_sweep_negated_private_key_passes() -> None:
    assert support_ticket_comment_is_private(
        {"not_private": True, "body": "x"}
    ) is False


_PRIVATE_PHRASES = (
    "internal", "private", "agents only", "internal only",
    "for internal use only", "restricted to agents", "private to team",
    "agent_note", "staff note", "internal_response", "support staff only",
    "agent-only", "staffonly",
)
_PUBLIC_PHRASES = (
    "public", "customer", "external", "visible to customer", "published",
    "everyone",
)
_LABEL_KEYS = (
    "visibility", "privacy", "audience", "access", "comment_visibility",
    "privacy_label",
)


@pytest.mark.parametrize(
    ("key", "phrase"),
    list(itertools.product(_LABEL_KEYS, _PRIVATE_PHRASES)),
)
def test_sweep_private_label_phrases_fail_closed(key: str, phrase: str) -> None:
    assert support_ticket_comment_is_private({key: phrase, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "phrase"),
    list(itertools.product(
        ("visibility", "privacy", "audience", "access"), _PUBLIC_PHRASES,
    )),
)
def test_sweep_public_label_phrases_pass(key: str, phrase: str) -> None:
    assert support_ticket_comment_is_private({key: phrase, "body": "x"}) is False


@pytest.mark.parametrize(
    "kind_value",
    ["private_note", "internal_comment", "agent_note", "staff_reply",
     "private_response", "support_agent_note"],
)
def test_sweep_private_kind_values_fail_closed(kind_value: str) -> None:
    assert support_ticket_comment_is_private(
        {"type": kind_value, "body": "x"}
    ) is True


@pytest.mark.parametrize(
    "kind_value",
    ["Comment", "question", "customer_note", "customer_reply", "incident"],
)
def test_sweep_public_kind_values_pass(kind_value: str) -> None:
    assert support_ticket_comment_is_private(
        {"type": kind_value, "body": "x"}
    ) is False


@pytest.mark.parametrize("stem", ["private", "internal", "hidden"])
def test_sweep_object_wrapped_polarity(stem: str) -> None:
    assert support_ticket_comment_is_private(
        {stem: {"value": True}, "body": "x"}
    ) is True
    assert support_ticket_comment_is_private(
        {stem: {"value": False}, "body": "x"}
    ) is False


@pytest.mark.parametrize("label", ["internal", "private", "agents only"])
def test_sweep_object_label_subfields_fail_closed(label: str) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": {"name": label}, "body": "x"}
    ) is True


def test_sweep_object_public_forms_pass() -> None:
    assert support_ticket_comment_is_private(
        {"public": {"value": True}, "body": "x"}
    ) is False
    assert support_ticket_comment_is_private(
        {"visibility": {"name": "public", "value": True}, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("internal_id", "x-1"),
        ("private_ip", "10.0.0.1"),
        ("publication_date", "2026-01-01"),
        ("hidden_count", 3),
        ("internal_status", "open"),
        ("has_access", True),
        ("has_access", 2),
        ("has_access", "2e3"),
        ("has_access", float("inf")),
        ("hasAccess", -7),
        ("has-access", {"value": 3}),
        ("has access", ["internal", 2]),
        ("HAS_ACCESS", {"nested": {"value": False}}),
        ("support_ticket_id", "T-1"),
        ("end_date", "2026-01-01"),
        ("from_email", "a@b.example"),
        ("external_id", "z"),
        ("team_id", "t-9"),
        ("client_version", "1.2.3"),
        ("user_agent", "Mozilla"),
        ("customer_id", "c-1"),
        ("admin_url", "https://example.test"),
    ],
)
def test_sweep_data_columns_are_never_markers(column: str, value: object) -> None:
    assert support_ticket_row_is_private({"body": "x", column: value}) is False


@pytest.mark.parametrize(
    "column",
    ["public_comment", "public_comments", "private_note", "internal_notes",
     "agent_note", "staff_notes"],
)
def test_sweep_content_columns_do_not_reject_rows(column: str) -> None:
    assert support_ticket_row_is_private(
        {"body": "x", column: "free text content here"}
    ) is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("private_note", {}),
        ("public_comment", {}),
    ],
)
def test_sweep_empty_structured_content_columns_fail_closed(
    column: str,
    value: object,
) -> None:
    assert support_ticket_row_is_private({"body": "x", column: value}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"private_note": {"foo": "bar"}},
        {"internal_notes": {"metadata": "billing role"}},
        {"agent_note": {"label": "billing role"}},
        {"staff_reply": {"value": "billing role"}},
        {"public_comment": {"custom": "customer role"}},
        {"public_comments": {"wrapper": {"foo": "bar"}}},
        {"private_notes": [{"foo": "bar"}]},
        {"internal_comments": [{"value": "billing role"}]},
        {"agent_messages": ({"label": "billing role"},)},
        {"public_comments": [{"metadata": {"foo": "bar"}}]},
    ],
)
def test_sweep_non_content_structured_note_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"private_note": {"body": "agent note text"}},
        {"internal_notes": {"text": "agent note text", "foo": "bar"}},
        {"agent_note": {"content": "agent note text", "metadata": "v2"}},
        {
            "public_comment": {
                "plain_body": "customer comment text",
                "value": "billing role",
            }
        },
        {"staff_reply": {"message": "agent reply text"}},
        {"public_comments": {"description": "customer comment text"}},
    ],
)
def test_sweep_body_bearing_note_containers_retain_row_carveout(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is False


_PARITY_VALUES = (
    "private",
    "internal",
    "true",
    "false",
    "public",
    "requester",
    "billing role",
    "2e3",
    "Infinity",
    True,
    False,
    2,
)

_PUBLIC_FAMILY_VALUE_STEMS = (
    ("public", "public"),
    ("visible", "visible"),
    ("external", "external"),
    ("published", "published"),
    ("customer_facing", "customer facing"),
    ("client_facing", "client facing"),
    ("user_facing", "user facing"),
    ("public_facing", "public facing"),
    ("visibility", "visible"),
    ("privacy", "public"),
    ("confidentiality", "public"),
)

_NEGATED_PUBLIC_FAMILY_VALUES = tuple(
    (key, value)
    for key, stem in _PUBLIC_FAMILY_VALUE_STEMS
    for value in (
        f"not {stem}",
        f"NOT {stem.upper()}",
        f"never {stem}",
        f"NEVER {stem.upper()}",
        f"un{stem.replace(' ', '')}",
        f"UN{stem.replace(' ', '').upper()}",
    )
)

_PUBLIC_FLAG_KEYS = (
    "published",
    "customer_facing",
    "client_facing",
    "user_facing",
    "public_facing",
)
_STRICT_LABEL_KEYS = ("visibility", "privacy", "confidentiality")
_DEFAULT_CLOSED_TEXT_KEYS = (
    "private",
    "public",
    *_PUBLIC_FLAG_KEYS,
    *_STRICT_LABEL_KEYS,
)
_NEUTRAL_DATA_KEYS = ("access", "audience", "type", "kind")
_SEMANTIC_PUBLIC_TOKENS = tuple(dict.fromkeys(
    [value for _, value in _PUBLIC_FAMILY_VALUE_STEMS]
    + [*_PUBLIC_AUDIENCES, "end user"]
))
_SEMANTIC_PRIVATE_TOKENS = tuple(dict.fromkeys(
    [*_PRIVATE_STEMS, *_PRIVATE_AUDIENCES]
))
_UNRECOGNIZED_CONTEXT_TEMPLATES = (
    "kept {}",
    "no longer {}",
    "withheld from {}",
    "not made {}",
    "open to {}",
    "shared with {}",
)
_UNRECOGNIZED_CONTEXT_VALUES = tuple(dict.fromkeys(
    template.format(token)
    for template, token in itertools.product(
        _UNRECOGNIZED_CONTEXT_TEMPLATES,
        _SEMANTIC_PUBLIC_TOKENS + _SEMANTIC_PRIVATE_TOKENS,
    )
))


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
def test_sweep_unprefixed_public_flag_keys_fail_closed(key: str) -> None:
    compact_key = key.replace("_", "")
    assert support_ticket_comment_is_private({
        f"un{compact_key}": True,
        "body": "x",
    }) is True
    assert support_ticket_comment_is_private({
        f"un{compact_key}": False,
        "body": "x",
    }) is False


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
@pytest.mark.parametrize("suffix", ["status", "label"])
def test_sweep_public_flag_label_status_aliases_reuse_family_policy(
    key: str,
    suffix: str,
) -> None:
    aliases = (
        f"{key}_{suffix}",
        f"{key.replace('_', '')}{suffix}",
    )
    for alias in aliases:
        assert support_ticket_comment_is_private({
            alias: "not published",
            "body": "x",
        }) is True
        assert support_ticket_comment_is_private({
            alias: "2026-07-09",
            "body": "x",
        }) is False
        for prefix in ("un", "not", "non"):
            assert support_ticket_comment_is_private({
                f"{prefix}{key.replace('_', '')}_{suffix}": True,
                "body": "x",
            }) is True
            assert support_ticket_comment_is_private({
                f"{prefix}{key.replace('_', '')}{suffix}": True,
                "body": "x",
            }) is True


_RECOGNIZED_FAILING_CLOSED_VALUES = (
    *_UNRECOGNIZED_CONTEXT_VALUES,
    "2026-02-30",
    "2026-13-01",
    "2026-02-30Z",
    "2026-13-01+0000",
    "2026-13-01+00:00",
    "Infinity",
    "NaN",
    "unpublished",
    "notpublished",
    "nonpublished",
)


def _semantic_container_shapes(value: str) -> tuple[object, ...]:
    return (
        value,
        {"value": value},
        [value],
        (value,),
        {value},
        frozenset({value}),
        [[value]],
        {"value": [value]},
    )


def _reference_spellings(words: tuple[str, ...]) -> tuple[str, ...]:
    if len(words) == 1:
        return words
    camel = words[0] + "".join(word.title() for word in words[1:])
    return tuple(dict.fromkeys((
        " ".join(words),
        "_".join(words),
        "-".join(words),
        camel,
        "".join(words),
    )))


_REFERENCE_KEYS = {
    "private": tuple(dict.fromkeys(
        spelling
        for words in (("private",), ("internal", "flag"), ("not", "public"))
        for spelling in _reference_spellings(words)
    )),
    "public": tuple(dict.fromkeys(
        spelling
        for words in (("public",), ("is", "public"), ("external",))
        for spelling in _reference_spellings(words)
    )),
    "public_flag": tuple(dict.fromkeys(
        spelling
        for words in (
            ("published",),
            ("customer", "facing"),
            ("public", "facing", "status"),
        )
        for spelling in _reference_spellings(words)
    )),
    "strict": tuple(dict.fromkeys(
        spelling
        for words in (
            ("visibility",),
            ("privacy", "label"),
            ("confidentiality",),
        )
        for spelling in _reference_spellings(words)
    )),
    "value": tuple(dict.fromkeys(
        spelling
        for words in (("access",), ("audience", "label"))
        for spelling in _reference_spellings(words)
    )),
    "kind": ("type", "kind"),
}

_REFERENCE_ATOMS = {
    "public": ("public",),
    "public_phrase": _reference_spellings(("visible", "to", "customer")),
    "strict_public": ("requester", "client"),
    "private": (
        "private",
        *_reference_spellings(("internal", "and", "private")),
        *_reference_spellings(("agents", "and", "staff", "only")),
    ),
    "recognized_context": _reference_spellings(("not", "publicly", "visible")),
    "unknown": _reference_spellings(("billing", "role")),
    "true": (True, "true", "1"),
    "false": (False, "false", "0"),
    "exponent_true": ("1e0", "+1e+0"),
    "malformed_numeric": ("2e3", "-7"),
    "nonfinite": ("NaN", "sNaN42", "qNaN", "-qNaN7", "Infinity"),
    "publication_date": (
        "2026-07-09",
        "2026-07-09Z",
        "2026-07-09T12:34+0000",
        "2026-07-09T12:34+00:00",
    ),
    "finite_number": (5,),
    "invalid_date": (
        "2026-02-30",
        "2026-13-01Z",
        "2026-02-30x",
        "2026-2-3",
        "2026-13-1",
        "2026-02-30 junk",
    ),
}


def _reference_expected_private(family: str, atom: str) -> bool:
    if atom == "public":
        return False
    if atom == "public_phrase":
        return family == "private"
    if atom == "strict_public":
        return family in {"private", "public"}
    if atom in {"private", "recognized_context"}:
        return family not in {"value", "kind"} or atom == "private"
    if atom == "unknown":
        return family in {"private", "public", "public_flag", "strict"}
    if atom == "true":
        return family in {"private", "strict"}
    if atom == "false":
        return family in {"public", "public_flag", "strict"}
    if atom == "exponent_true":
        return family in {"private", "strict", "value"}
    if atom == "malformed_numeric":
        return family in {"private", "public", "strict", "value"}
    if atom == "nonfinite":
        return family != "kind"
    if atom == "publication_date":
        return family in {"private", "public", "strict"}
    if atom == "finite_number":
        return family in {"private", "public", "strict", "value"}
    if atom == "invalid_date":
        return family in {"private", "public", "public_flag", "strict"}
    raise AssertionError(f"unmodeled reference atom: {atom}")


def test_sweep_generated_reference_model_matches_supported_cross_product() -> None:
    wrappers = (
        lambda value: value,
        lambda value: {"value": value},
        lambda value: [value],
        lambda value: {"value": [value]},
    )
    for family, keys in _REFERENCE_KEYS.items():
        for atom, raw_values in _REFERENCE_ATOMS.items():
            expected = _reference_expected_private(family, atom)
            for key, raw_value, wrap in itertools.product(
                keys,
                raw_values,
                wrappers,
            ):
                marker = {"body": "x", key: wrap(raw_value)}
                assert support_ticket_comment_is_private(marker) is expected, (
                    family,
                    atom,
                    key,
                    raw_value,
                    marker,
                )

    closed_keys = tuple(
        (family, key)
        for family in ("public", "public_flag", "strict")
        for key in _REFERENCE_KEYS[family]
    )
    for family, key in closed_keys:
        for atom in ("private", "recognized_context", "nonfinite", "invalid_date"):
            for raw_value in _REFERENCE_ATOMS[atom]:
                for shaped in (
                    {"label": "public", "value": raw_value},
                    {"public": True, "value": raw_value},
                    ["public", raw_value],
                    {"value": ["public", raw_value]},
                ):
                    assert support_ticket_comment_is_private({
                        "body": "x",
                        key: shaped,
                    }) is True, (family, atom, key, raw_value, shaped)

        for raw_value in _REFERENCE_ATOMS["publication_date"]:
            for shaped in (
                {"label": "public", "value": raw_value},
                {"public": True, "value": raw_value},
                ["public", raw_value],
                {"value": ["public", raw_value]},
            ):
                assert support_ticket_comment_is_private({
                    "body": "x",
                    key: shaped,
                }) is False, (family, "publication_date", key, raw_value, shaped)

        for raw_value in _REFERENCE_ATOMS["finite_number"]:
            expected = family != "public_flag"
            for shaped in (
                {"label": "public", "value": raw_value},
                {"public": True, "value": raw_value},
                ["public", raw_value],
                {"value": ["public", raw_value]},
            ):
                assert support_ticket_comment_is_private({
                    "body": "x",
                    key: shaped,
                }) is expected, (family, "finite_number", key, raw_value, shaped)

    for key in ("privateNote", "publicComments"):
        assert support_ticket_row_is_private({
            "body": "x",
            key: ["customer question", "follow-up"],
        }) is False
        assert support_ticket_row_is_private({
            "body": "x",
            key: ["private"],
        }) is True


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS + _STRICT_LABEL_KEYS)
@pytest.mark.parametrize("value", _RECOGNIZED_FAILING_CLOSED_VALUES)
def test_sweep_public_evidence_cannot_mask_recognized_failing_values(
    key: str,
    value: str,
) -> None:
    for shaped_value in (
        {"label": "public", "value": value},
        {"value": value, "label": "public"},
        ["public", value],
        {"value": ["public", value]},
        {"value": {"label": "public", "value": value}},
    ):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is True


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS + _STRICT_LABEL_KEYS)
@pytest.mark.parametrize("value", ["billing role", "release cohort"])
def test_sweep_public_evidence_keeps_genuinely_neutral_metadata(
    key: str,
    value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        key: {"label": "public", "value": value},
    }) is False


@pytest.mark.parametrize(
    "key",
    ["private_note", "internal_notes", "public_comment", "public_comments"],
)
@pytest.mark.parametrize("value", _PARITY_VALUES)
def test_sweep_content_singletons_match_scalar_policy(
    key: str,
    value: object,
) -> None:
    expected = support_ticket_row_is_private({"body": "x", key: value})

    for wrapped in ([value], (value,), {value}, frozenset({value})):
        assert support_ticket_row_is_private({
            "body": "x",
            key: wrapped,
        }) is expected


@pytest.mark.parametrize(
    "marker",
    [
        {"private_note": ["private", "customer question"]},
        {"internal_notes": ["true", "customer question"]},
        {"public_comment": ["false", "customer question"]},
        {"public_comments": ["2e3", "customer question"]},
    ],
)
def test_sweep_content_sequences_classify_each_string_before_carveout(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize("key", ["private_note", "public_comments"])
def test_sweep_content_sequences_retain_free_text_carveout(key: str) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        key: ["customer question", "follow-up"],
    }) is False


@pytest.mark.parametrize("key", ["access", "audience"])
@pytest.mark.parametrize("value", ["0e0", "1e0", "+1e+0", "-0e-2"])
def test_sweep_boolish_exponent_numerics_retain_malformed_provenance(
    key: str,
    value: str,
) -> None:
    for shaped_value in (value, {"value": value}, [value], [[value]]):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is True


@pytest.mark.parametrize("key", ["private_note", "internal_notes"])
@pytest.mark.parametrize("value", ["0e0", "1e0"])
def test_sweep_content_sequences_reject_boolish_exponent_numerics(
    key: str,
    value: str,
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        key: [value],
    }) is True


@pytest.mark.parametrize(("key", "value"), _NEGATED_PUBLIC_FAMILY_VALUES)
def test_sweep_negated_public_family_vocabulary_fails_closed(
    key: str,
    value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        key: value,
    }) is True


@pytest.mark.parametrize(("key", "value"), _PUBLIC_FAMILY_VALUE_STEMS)
def test_sweep_positive_public_family_vocabulary_still_admits(
    key: str,
    value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        key: value,
    }) is False


@pytest.mark.parametrize("key", _DEFAULT_CLOSED_TEXT_KEYS)
@pytest.mark.parametrize("value", _UNRECOGNIZED_CONTEXT_VALUES)
def test_sweep_closed_families_reject_unrecognized_text_by_construction(
    key: str,
    value: str,
) -> None:
    for shaped_value in _semantic_container_shapes(value):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is True


@pytest.mark.parametrize("key", _NEUTRAL_DATA_KEYS)
@pytest.mark.parametrize("value", _UNRECOGNIZED_CONTEXT_VALUES)
def test_sweep_data_families_retain_neutral_text_by_construction(
    key: str,
    value: str,
) -> None:
    for shaped_value in _semantic_container_shapes(value):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is False


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS + _STRICT_LABEL_KEYS)
@pytest.mark.parametrize(
    "value",
    ["public", "customer", "requester", "visible to customer", "end user"],
)
def test_sweep_closed_families_retain_recognized_public_values(
    key: str,
    value: str,
) -> None:
    for shaped_value in _semantic_container_shapes(value):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is False


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
@pytest.mark.parametrize(
    "value",
    [
        "2026-07-09",
        "2026-07-09Z",
        "2026-07-09+0000",
        "2026-07-09+00:00",
        "2026-07-09T12:34:56Z",
        5,
        "2e3",
    ],
)
def test_sweep_public_flags_admit_explicit_date_and_number_data_shapes(
    key: str,
    value: object,
) -> None:
    for shaped_value in (value, {"value": value}, [value], (value,)):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is False


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
@pytest.mark.parametrize(
    "value",
    [
        "Infinity",
        "NaN",
        "2026-99-99",
        "2026-02-30",
        "2026-99-99Z",
        "2026-02-30+0000",
        "2026-02-30+00:00",
    ],
)
def test_sweep_public_flags_reject_invalid_data_shapes(
    key: str,
    value: str,
) -> None:
    for shaped_value in (value, {"value": value}, [value], (value,)):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is True


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
@pytest.mark.parametrize(
    "value",
    [
        "Infinity", "-Infinity", "NaN", "+sNaN42",
        float("inf"), float("-inf"), float("nan"),
    ],
)
def test_sweep_public_flags_reject_nonfinite_data_beside_public_evidence(
    key: str,
    value: object,
) -> None:
    for shaped_value in (
        {"label": "public", "value": value},
        {"name": "requester", "status": value},
        ["public", value],
        {"value": ["public", value]},
    ):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is True


@pytest.mark.parametrize("key", _PUBLIC_FLAG_KEYS)
@pytest.mark.parametrize(
    "value",
    ["2026-07-09", "2026-07-09Z", "2026-07-09+00:00", 5, "2e3"],
)
def test_sweep_public_flags_keep_valid_data_beside_public_evidence(
    key: str,
    value: object,
) -> None:
    for shaped_value in (
        {"label": "public", "value": value},
        ["public", value],
        [value, "public"],
        {"value": ["public", value]},
    ):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: shaped_value,
        }) is False


@pytest.mark.parametrize(
    "key",
    _PUBLIC_FLAG_KEYS,
)
def test_sweep_public_flag_date_values_still_admit(key: str) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        key: "2026-07-09",
    }) is False


@pytest.mark.parametrize("key", ["access", "audience", "type", "kind"])
@pytest.mark.parametrize(
    "value",
    ["not published", "never published", "unpublished"],
)
def test_sweep_negated_public_vocabulary_does_not_widen_to_data_families(
    key: str,
    value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        key: value,
    }) is False


@pytest.mark.parametrize("placeholder", ["", "   ", None])
@pytest.mark.parametrize(
    ("key", "public_value"),
    [
        ("public_comments", "customer question"),
        ("private_note", "customer question"),
        ("visibility", "requester"),
        ("public", "public"),
    ],
)
def test_sweep_absent_sequence_elements_do_not_override_public_evidence(
    key: str,
    public_value: str,
    placeholder: object,
) -> None:
    for value in (
        [public_value, placeholder],
        [placeholder, public_value],
        (public_value, placeholder),
    ):
        assert support_ticket_row_is_private({"body": "x", key: value}) is False


@pytest.mark.parametrize("placeholder", ["", "   ", None])
@pytest.mark.parametrize(
    ("key", "neutral_value"),
    [
        ("published", "2026-07-09"),
        ("access", "billing role"),
        ("audience", "partner workspace"),
    ],
)
def test_sweep_absent_sequence_elements_preserve_scalar_neutral_outcome(
    key: str,
    neutral_value: str,
    placeholder: object,
) -> None:
    expected = support_ticket_row_is_private({
        "body": "x",
        key: neutral_value,
    })
    assert expected is False

    for value in (
        [neutral_value, placeholder],
        [placeholder, neutral_value],
        (neutral_value, placeholder),
        {neutral_value, placeholder},
        frozenset({neutral_value, placeholder}),
    ):
        assert support_ticket_row_is_private({"body": "x", key: value}) is expected


@pytest.mark.parametrize("placeholder", ["", "   ", None])
@pytest.mark.parametrize(
    ("key", "scalar_value", "expected"),
    [
        ("private", "public", False),
        ("private", "private", True),
        ("public", "public", False),
        ("public", "private", True),
        ("published", "2026-07-09", False),
        ("published", "kept private", True),
        ("visibility", "requester", False),
        ("visibility", "opaque producer status", True),
        ("access", "billing role", False),
        ("access", "private", True),
        ("type", "billing role", False),
        ("type", "private", True),
        ("private_note", "customer question", False),
        ("private_note", "true", True),
    ],
)
def test_sweep_blank_is_identity_across_every_family_and_container(
    key: str,
    scalar_value: str,
    expected: bool,
    placeholder: object,
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        key: scalar_value,
    }) is expected

    wrapped_values: list[object] = [
        [scalar_value, placeholder],
        [placeholder, scalar_value],
        (scalar_value, placeholder),
        {scalar_value, placeholder},
        frozenset({scalar_value, placeholder}),
        [[scalar_value], [placeholder]],
        [[placeholder], [scalar_value]],
    ]
    if key != "private_note":
        wrapped_values.extend([
            {"value": [scalar_value, placeholder]},
            {"value": [[scalar_value], [placeholder]]},
        ])
    for value in wrapped_values:
        assert support_ticket_row_is_private({"body": "x", key: value}) is expected


@pytest.mark.parametrize("placeholder", [[], (), set(), frozenset(), [[], ()]])
@pytest.mark.parametrize(
    ("key", "scalar_value", "expected"),
    [
        ("private", "public", False),
        ("public", "public", False),
        ("published", "2026-07-09", False),
        ("visibility", "requester", False),
        ("access", "billing role", False),
        ("type", "billing role", False),
        ("private_note", "customer question", False),
    ],
)
def test_sweep_empty_nested_sequences_are_identity_across_every_family(
    key: str,
    scalar_value: str,
    expected: bool,
    placeholder: object,
) -> None:
    for value in ([scalar_value, placeholder], [placeholder, scalar_value]):
        assert support_ticket_row_is_private({"body": "x", key: value}) is expected

    if key != "private_note":
        assert support_ticket_row_is_private({
            "body": "x",
            key: {"value": [scalar_value, placeholder]},
        }) is expected


@pytest.mark.parametrize(
    "placeholder",
    [
        {"value": ""},
        {"value": []},
        {"value": {"value": None}},
        {"label": ["", None]},
    ],
)
@pytest.mark.parametrize(
    ("key", "scalar_value"),
    [
        ("visibility", "requester"),
        ("published", "2026-07-09"),
        ("access", "billing role"),
        ("type", "billing role"),
    ],
)
def test_sweep_placeholder_only_marker_wrappers_are_sequence_identity(
    key: str,
    scalar_value: str,
    placeholder: dict[str, object],
) -> None:
    for value in ([scalar_value, placeholder], [placeholder, scalar_value]):
        assert support_ticket_row_is_private({"body": "x", key: value}) is False

    assert support_ticket_row_is_private({
        "body": "x",
        key: [scalar_value, {}],
    }) is True


@pytest.mark.parametrize(
    ("key", "values"),
    [
        ("published", ["2026-07-09", "2026-07-10"]),
        ("access", ["billing role", "partner workspace"]),
        ("audience", ["billing role", "partner workspace"]),
    ],
)
def test_sweep_multiple_substantive_neutral_values_remain_fail_closed(
    key: str,
    values: list[str],
) -> None:
    assert support_ticket_row_is_private({"body": "x", key: values}) is True


@pytest.mark.parametrize("placeholder", ["", "   ", None])
@pytest.mark.parametrize(
    ("key", "private_value"),
    [
        ("private_note", "private"),
        ("public_comments", "false"),
        ("visibility", "internal"),
        ("visibility", "2e3"),
        ("published", "private"),
    ],
)
def test_sweep_absent_sequence_elements_do_not_mask_private_evidence(
    key: str,
    private_value: str,
    placeholder: object,
) -> None:
    for value in (
        [private_value, placeholder],
        [placeholder, private_value],
        (private_value, placeholder),
    ):
        assert support_ticket_row_is_private({"body": "x", key: value}) is True


@pytest.mark.parametrize("key", ["visibility", "public_comments"])
def test_sweep_all_absent_sequence_elements_match_absent_scalar(key: str) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        key: ["", None, "   "],
    }) is False


@pytest.mark.parametrize(
    "key",
    ["private", "public", "visibility", "access", "type", "published"],
)
@pytest.mark.parametrize("value", _PARITY_VALUES)
def test_sweep_marker_singletons_match_scalar_and_value_wrapper_policy(
    key: str,
    value: object,
) -> None:
    expected = support_ticket_comment_is_private({"body": "x", key: value})

    for wrapped in (
        [value],
        (value,),
        {value},
        frozenset({value}),
        {"value": value},
    ):
        assert support_ticket_comment_is_private({
            "body": "x",
            key: wrapped,
        }) is expected


@pytest.mark.parametrize(
    "marker",
    [
        {"public": {"name": "public", "value": "2e3"}},
        {"public": {"label": "public", "status": "Infinity"}},
        {"public": {"public": True, "value": 5}},
        {"visible": {"name": "public", "value": "-1e2"}},
        {"visible": {"label": "requester", "status": "NaN"}},
        {"visible": {"public": True, "value": float("inf")}},
        {"external": {"name": "public", "value": "+4E-2"}},
        {"external": {"label": "client", "status": "-Infinity"}},
        {"external": {"public": True, "value": -7}},
    ],
)
def test_sweep_public_families_reject_malformed_numeric_provenance(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


def test_sweep_malformed_data_column_families_retain_scalar_policy() -> None:
    for key in ("published", "customer_facing", "type", "kind"):
        scalar = support_ticket_comment_is_private({"body": "x", key: "2e3"})
        wrapped = support_ticket_comment_is_private({
            "body": "x",
            key: {"value": "2e3"},
        })
        assert wrapped is scalar


@pytest.mark.parametrize(
    "marker",
    [
        {"metadata": {"private": True}},
        {"visibility": {"name": "public", "metadata": {"private": True}}},
        {
            "public_comments": [
                {"body": "customer text", "metadata": {"private": True}}
            ]
        },
    ],
)
def test_sweep_unknown_key_subtrees_remain_opaque(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"visibility": {"name": "public", "privacy": True}},
        {"public_comments": [{"body": "private text", "privacy": True}]},
    ],
)
def test_sweep_recognized_nested_privacy_still_fails_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"private_note": [{"value": True}]},
        {"internal_notes": ({"private": True},)},
        {"public_comment": [{"value": False}]},
        {"public_comments": [{"hidden": True}, {"body": "customer text"}]},
        {"public_comments": [True, "customer text"]},
        {
            "public_comments": [
                {"visibility": "requester", "body": "public text"},
                {"hidden": True, "body": "private text"},
            ]
        },
    ],
)
def test_sweep_sequence_wrapped_structured_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"private_notes": ["agent note text"]},
        {"private_notes": [{"body": "agent note text"}]},
        {"private_notes": ["agent note text", {"body": "more context"}]},
        {"public_comments": ["customer comment text"]},
        {"public_comments": [{"body": "customer comment text"}]},
        {"public_comments": ["customer comment text", {"body": "more context"}]},
    ],
)
def test_sweep_sequence_content_containers_retain_row_carveout(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({**marker, "body": "x"}) is False


@pytest.mark.parametrize("public_marker", [{"public": True}, {"private": False}])
@pytest.mark.parametrize(
    "content",
    ["customer question", {"body": "customer question"}],
)
def test_sweep_public_marker_composes_with_admitted_content(
    public_marker: dict[str, object],
    content: object,
) -> None:
    assert support_ticket_row_is_private({
        "public_comments": [content, public_marker],
    }) is False


@pytest.mark.parametrize("alias", ["private_note", "internal_notes"])
@pytest.mark.parametrize(
    "content_key",
    [
        "body", "message", "text", "content", "description",
        "plain_body", "html_body",
    ],
)
@pytest.mark.parametrize(
    "empty_content",
    ["", "   ", None, [], ["", None], {}, {"text": ""}],
)
def test_sweep_content_carveout_requires_nonblank_text(
    alias: str,
    content_key: str,
    empty_content: object,
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        alias: {content_key: empty_content},
    }) is True


@pytest.mark.parametrize(
    "alias",
    ["private_note", "internal_notes", "public_comment"],
)
@pytest.mark.parametrize("content_key", ["body", "html_body", "content"])
@pytest.mark.parametrize("empty_content", ["<br>", "<p></p>", "&nbsp;"])
def test_sweep_content_carveout_requires_rendered_text(
    alias: str,
    content_key: str,
    empty_content: str,
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        alias: {content_key: empty_content},
    }) is True

    assert support_ticket_row_is_private({
        "body": "x",
        alias: {content_key: "<p>customer text</p>"},
    }) is False


@pytest.mark.parametrize(
    "rendered_empty",
    ["&amp;", "&lt;", "&copy;", "&amp;&amp;", "...", "<b>&copy;</b>"],
)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_sweep_rendered_empty_content_wrappers_are_sequence_identity(
    rendered_empty: str,
    depth: int,
) -> None:
    placeholder: object = rendered_empty
    for index in range(depth):
        key = "body" if index % 2 == 0 else "text"
        placeholder = {key: placeholder}

    assert support_ticket_row_is_private({
        "public_comments": [
            {"body": "customer question"},
            placeholder,
        ],
    }) is False


@pytest.mark.parametrize("rendered_content", ["A", "&#65;", "customer?"])
def test_sweep_rendered_alphanumeric_content_remains_substantive(
    rendered_content: str,
) -> None:
    assert support_ticket_row_is_private({
        "public_comments": [
            {"body": "customer question"},
            {"body": rendered_content},
        ],
    }) is False


@pytest.mark.parametrize("alias", ["private_note", "internal_notes"])
@pytest.mark.parametrize(
    "content",
    [
        {"body": "agent note text"},
        {"body": ["", {"text": "agent note text"}]},
    ],
)
def test_sweep_content_carveout_keeps_nested_nonblank_text(
    alias: str,
    content: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        alias: content,
    }) is False


@pytest.mark.parametrize(
    "alias",
    ["private_note", "internal_notes", "public_comments"],
)
@pytest.mark.parametrize("content_key", ["body", "text", "content", "message"])
@pytest.mark.parametrize(
    "privacy_marker",
    [
        {"private": True},
        {"hidden": True},
        {"privacy": True},
        {"public": False},
    ],
)
def test_sweep_nested_content_privacy_markers_dominate_carveout(
    alias: str,
    content_key: str,
    privacy_marker: dict[str, object],
) -> None:
    nested = {**privacy_marker, "text": "private nested content"}
    for content_value in (nested, [nested], {"content": [nested]}):
        assert support_ticket_row_is_private({
            "body": "x",
            alias: {content_key: content_value},
        }) is True


@pytest.mark.parametrize("alias", ["private_note", "public_comments"])
@pytest.mark.parametrize("public_marker", [{"public": True}, {"private": False}])
def test_sweep_nested_content_public_markers_keep_genuine_text(
    alias: str,
    public_marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        alias: {"body": {**public_marker, "text": "customer content"}},
    }) is False


@pytest.mark.parametrize(
    "alias",
    ["private_note", "public_comment", "public_comments"],
)
@pytest.mark.parametrize("marker_key", ["status", "value", "label", "name"])
def test_sweep_structured_failing_markers_dominate_content_carveout(
    alias: str,
    marker_key: str,
) -> None:
    assert support_ticket_row_is_private({
        "body": "x",
        alias: {"body": "customer text", marker_key: "kept private"},
    }) is True

    assert support_ticket_row_is_private({
        "body": "x",
        alias: "customer says kept private for later",
    }) is False


@pytest.mark.parametrize("alias", ["private_note", "public_comment"])
@pytest.mark.parametrize("marker_key", ["status", "value", "label", "name"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_sweep_recognized_failing_markers_dominate_at_every_content_depth(
    alias: str,
    marker_key: str,
    depth: int,
) -> None:
    nested: object = {
        "text": "PRIVATE SENTINEL",
        marker_key: "kept private",
    }
    for index in range(depth):
        key = "body" if index % 2 == 0 else "content"
        nested = {key: nested}

    assert support_ticket_row_is_private({alias: nested}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"visibility": ["public", "billing role"]},
        {"public": [True, "billing role"]},
        {"private": [False, "billing role"]},
    ],
)
def test_sweep_mixed_sequence_markers_preserve_unresolved_provenance(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize("audience", ["requester", "client", "end user"])
def test_sweep_strict_public_audiences_survive_sequence_wrappers(
    audience: str,
) -> None:
    assert support_ticket_comment_is_private({
        "visibility": [audience],
        "body": "x",
    }) is False


@pytest.mark.parametrize("audience", ["requester", "client", "end user"])
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_sweep_raw_strict_sequence_booleans_stay_ambiguous(
    audience: str,
    value: bool,
    reverse: bool,
) -> None:
    sequence = [value, audience] if reverse else [audience, value]

    assert support_ticket_comment_is_private({
        "visibility": sequence,
        "body": "x",
    }) is True


@pytest.mark.parametrize("audience", ["requester", "client", "end user"])
@pytest.mark.parametrize("value", [True, False])
def test_sweep_generic_strict_sequence_booleans_remain_neutral(
    audience: str,
    value: bool,
) -> None:
    assert support_ticket_comment_is_private({
        "visibility": {"label": audience, "value": [[value]]},
        "body": "x",
    }) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"visibility": {"private": {"value": True}}},
        {"privacy": {"hidden": {"value": True}}},
        {"visibility": {"confidential": {"value": True}}},
        {"privacy": {"restricted": {"value": True}}},
        {"visibility": {"public": {"value": False}}},
        {"privacy": {"publicly_visible": {"value": False}}},
        {"visibility": {"hidden_from_customer": {"value": True}}},
    ],
)
def test_sweep_nested_assertion_objects_preserve_private_polarity(
    marker: dict[str, object],
) -> None:
    row = {**marker, "body": "private text"}

    assert support_ticket_comment_is_private(row) is True
    assert support_ticket_row_is_private(row) is True


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize(("value", "expected_private"), [(True, True), (False, False)])
def test_sweep_neutral_object_wrappers_inherit_private_polarity(
    depth: int,
    value: bool,
    expected_private: bool,
) -> None:
    nested: object = value
    for _ in range(depth):
        nested = {"value": nested}

    assert support_ticket_comment_is_private(
        {"private": nested, "body": "x"}
    ) is expected_private


@pytest.mark.parametrize(
    ("marker", "expected_private"),
    [
        ({"visibility": {"private": "client"}}, True),
        ({"visibility": {"public": "client"}}, True),
        ({"visibility": {"visibility": "client"}}, False),
    ],
)
def test_sweep_nested_classified_labels_use_their_own_family_policy(
    marker: dict[str, object],
    expected_private: bool,
) -> None:
    assert support_ticket_comment_is_private(
        {**marker, "body": "x"}
    ) is expected_private


def test_sweep_nested_classified_neutral_is_not_promoted_to_public() -> None:
    assert support_ticket_comment_is_private(
        {"visibility": {"published": "2026-07-09"}, "body": "x"}
    ) is True


@pytest.mark.parametrize(
    ("outer_key", "subfield", "neutral_value"),
    [
        ("visibility", "access", "end user"),
        ("visibility", "audience", "client"),
        ("published", "type", "2026-07-09"),
        ("customer_facing", "kind", "2e3"),
    ],
)
def test_sweep_locally_neutral_subfields_export_no_positive_provenance(
    outer_key: str,
    subfield: str,
    neutral_value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        outer_key: {subfield: neutral_value},
    }) is True

    assert support_ticket_comment_is_private({
        "body": "x",
        outer_key: {"label": "public", subfield: neutral_value},
    }) is False


def test_sweep_private_and_neutral_nested_subfields_fail_closed() -> None:
    assert support_ticket_comment_is_private({
        "visibility": {"private": True, "type": "billing role"},
        "body": "x",
    }) is True


@pytest.mark.parametrize(
    ("outer_key", "public_label"),
    [("visibility", "requester"), ("published", "public")],
)
@pytest.mark.parametrize("subfield", ["access", "audience", "type", "kind"])
@pytest.mark.parametrize(
    "neutral_value",
    ["kept private", "withheld from public", "billing role"],
)
def test_sweep_neutral_subfield_policy_precedes_outer_family_policy(
    outer_key: str,
    public_label: str,
    subfield: str,
    neutral_value: str,
) -> None:
    assert support_ticket_comment_is_private({
        "body": "x",
        outer_key: {"label": public_label, subfield: neutral_value},
    }) is False

    assert support_ticket_comment_is_private({
        "body": "x",
        outer_key: {"label": public_label, subfield: "private"},
    }) is True


@pytest.mark.parametrize("key", ["privacy", "confidentiality"])
@pytest.mark.parametrize("value", [True, False])
def test_sweep_boolean_strict_label_wrappers_fail_closed(
    key: str,
    value: bool,
) -> None:
    assert support_ticket_comment_is_private(
        {key: {"value": value}, "body": "x"}
    ) is True


@pytest.mark.parametrize("key", ["access", "audience", "type", "kind"])
@pytest.mark.parametrize("value", [True, False])
def test_sweep_categorical_boolean_wrappers_match_scalar_policy(
    key: str,
    value: bool,
) -> None:
    expected = support_ticket_comment_is_private({key: value, "body": "x"})

    for wrapped in (
        {"value": value},
        [value],
        (value,),
        {"value": [value]},
        {"value": (value,)},
    ):
        assert support_ticket_comment_is_private({
            key: wrapped,
            "body": "x",
        }) is expected


@pytest.mark.parametrize("strict_key", ["privacy", "confidentiality"])
@pytest.mark.parametrize("value", [True, False])
def test_sweep_classified_strict_booleans_conflict_with_public_evidence(
    strict_key: str,
    value: bool,
) -> None:
    assert support_ticket_comment_is_private({
        "visibility": {"label": "public", strict_key: value},
        "body": "x",
    }) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"published": {"value": False}},
        {"customer_facing": {"value": False}},
        {"client_facing": {"status": False}},
        {"user_facing": {"public": False}},
        {"public_facing": {"is_public": False}},
        {"published": {}},
        {"customer_facing": {"label": "public", "value": False}},
    ],
)
def test_sweep_object_wrapped_false_public_flags_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"published": {"value": "2026-07-09"}},
        {"customer_facing": {"value": True}},
    ],
)
def test_sweep_object_wrapped_public_flags_keep_public_values(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"published": "private"},
        {"customer_facing": "confidential"},
        {"user_facing": "internal only"},
        {"published": {"value": "private"}},
    ],
)
def test_sweep_public_flags_honor_explicit_private_text(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "label"),
    [
        ("access", "internal and private"),
        ("audience", "agents and staff only"),
        ("type", "staff and agent reply"),
        ("access", "restricted or internal"),
        ("audience", "support and admins only"),
        ("kind", "private or confidential response"),
    ],
)
def test_sweep_private_value_conjunctions_fail_closed(key: str, label: str) -> None:
    assert support_ticket_comment_is_private({key: label, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "wrapped_label"),
    [
        ("access", {"label": "billing role"}),
        ("audience", {"label": "enterprise admins"}),
        ("type", {"name": "question"}),
        ("access", {"name": "workspace owners"}),
        ("audience", {"status": "trial accounts"}),
        ("kind", {"label": "incident"}),
    ],
)
def test_sweep_neutral_object_labels_match_scalar_fail_open_policy(
    key: str,
    wrapped_label: dict[str, str],
) -> None:
    scalar_label = next(iter(wrapped_label.values()))

    assert support_ticket_comment_is_private(
        {key: wrapped_label, "body": "x"}
    ) is False
    assert support_ticket_comment_is_private(
        {key: scalar_label, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"access": 5},
        {"audience": -7},
        {"access": {"value": 5}},
        {"access": "2e3"},
        {"audience": "-1e2"},
        {"access": {"value": "2e3"}},
        {"audience": {"label": "+4E-2"}},
        {"access": float("inf")},
        {"audience": float("-inf")},
        {"access": {"value": float("nan")}},
        {"access": "Infinity"},
        {"audience": "-Infinity"},
        {"access": {"value": "NaN"}},
        {"audience": {"label": "+sNaN42"}},
    ],
)
def test_sweep_value_labels_reject_malformed_numeric_markers(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "marker",
    [
        {"access": "gizmo"},
        {"audience": "xyz123"},
    ],
)
def test_sweep_value_labels_retain_producer_defined_neutral_text(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"visibility": {"name": "public", "value": "2e3"}},
        {"privacy": {"label": "customer", "value": "Infinity"}},
        {"confidentiality": {"name": "requester", "value": 5}},
    ],
)
def test_sweep_strict_labels_reject_malformed_numeric_provenance(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({**marker, "body": "x"}) is True


@pytest.mark.parametrize(
    "label",
    [
        "requester",
        "client",
        "end user",
        "requesters",
        "clients",
        "endusers",
        "viewers",
    ],
)
def test_sweep_strict_labels_admit_public_audiences(label: str) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": label, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"name": "requester", "value": "billing role"},
        {"value": "billing role", "name": "requester"},
    ],
)
def test_sweep_strict_label_object_is_mapping_order_independent(
    marker: dict[str, str],
) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": marker, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    "marker",
    [
        {"name": "public", "value": "billing role"},
        {"value": "billing role", "name": "public"},
    ],
)
def test_sweep_explicit_public_label_ignores_neutral_metadata(
    marker: dict[str, str],
) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": marker, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    "label",
    ["visible to end user", "visible to end users", "public for end users"],
)
def test_sweep_end_user_visibility_phrases_are_public(label: str) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": label, "body": "x"}
    ) is False


@pytest.mark.parametrize("private_key", ["private", "hidden", "internal"])
@pytest.mark.parametrize("label", ["end user", "visible to end user"])
def test_sweep_public_audience_phrases_do_not_invert_private_polarity(
    private_key: str,
    label: str,
) -> None:
    assert support_ticket_comment_is_private({
        private_key: label,
        "body": "x",
    }) is True
    assert support_ticket_comment_is_private({
        "visibility": {private_key: label},
        "body": "x",
    }) is True


@pytest.mark.parametrize("key", ["access", "audience", "type", "kind"])
def test_sweep_public_audience_phrases_retain_data_family_semantics(
    key: str,
) -> None:
    assert support_ticket_comment_is_private({
        key: "end user",
        "body": "x",
    }) is False


def test_sweep_explicit_public_labels_retain_private_field_override() -> None:
    assert support_ticket_comment_is_private({
        "private": "public",
        "body": "x",
    }) is False
    assert support_ticket_comment_is_private({
        "visibility": {"public": "end user"},
        "body": "x",
    }) is False


def test_sweep_public_audience_label_does_not_invert_private_assertion() -> None:
    assert support_ticket_comment_is_private(
        {"private": "client", "body": "x"}
    ) is True
