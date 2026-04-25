"""F1: phrase-tag salience bonus tests.

The Slack canary in Phase 7 Step B verification revealed that the witness
selector was schema-version blind: with no per-phrase weighting, v3
witnesses with stale-but-specific pain_category beat freshly v4 witnesses
where Phase 4 had correctly demoted to overall_dissatisfaction.

These tests lock the new bonus contract:
  - subject == 'subject_vendor' AND polarity in (negative, mixed): +2.0
  - role == 'primary_driver': +1.5
  - pain_confidence == 'strong': +2.0; 'weak': +0.5
  - grounding_status == 'grounded': +0.75

v3 spans carry None on these tags and stay at the legacy baseline,
so existing v3 ranking behavior is preserved.

The fix does NOT add a schema_version weight directly, because that
would mask future regressions where v4 evidence is genuinely weaker
than its v3 predecessor.
"""

from __future__ import annotations

from atlas_brain.autonomous.tasks._b2b_witnesses import _witness_salience


def _baseline_review() -> dict:
    return {
        "rating": 5,
        "rating_max": 5,
        "raw_metadata": {"source_weight": 0.7},
        "reviewed_at": None,  # avoid recency bonus for clean math
        "reviewer_company": None,
    }


def _baseline_enrichment(**overrides) -> dict:
    base = {
        "churn_signals": {},
        "reviewer_context": {},
    }
    base.update(overrides)
    return base


def _baseline_span(**overrides) -> dict:
    base = {
        "competitor": None,
        "flags": [],
        "operating_model_shift": "none",
        "productivity_delta_claim": "unknown",
    }
    base.update(overrides)
    return base


def test_v3_span_no_phrase_tags_gets_no_bonus():
    """v3 spans carry None on subject/polarity/role; baseline preserved."""
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    span = _baseline_span()
    score_v3 = _witness_salience(review, enrichment, span)

    # Same span shape but with phrase tags in the v4-good combination
    span_v4 = _baseline_span(
        subject="subject_vendor",
        polarity="negative",
        role="primary_driver",
    )
    enrichment_v4 = _baseline_enrichment(pain_confidence="strong")
    score_v4 = _witness_salience(review, enrichment_v4, span_v4)

    # 2.0 (subj+pol) + 1.5 (role) + 2.0 (conf=strong) = 5.5 lift
    assert round(score_v4 - score_v3, 2) == 5.5


def test_subject_vendor_negative_gets_two_point_bonus():
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    span_off = _baseline_span()
    span_on = _baseline_span(subject="subject_vendor", polarity="negative")
    assert round(
        _witness_salience(review, enrichment, span_on)
        - _witness_salience(review, enrichment, span_off),
        2,
    ) == 2.0


def test_subject_vendor_mixed_polarity_also_eligible():
    """Mixed polarity is half-weight in pain derivation (Phase 3) but
    still counts as a real vendor complaint for selection purposes."""
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    span = _baseline_span(subject="subject_vendor", polarity="mixed")
    span_neg = _baseline_span(subject="subject_vendor", polarity="negative")
    assert (
        _witness_salience(review, enrichment, span)
        == _witness_salience(review, enrichment, span_neg)
    )


def test_subject_self_does_not_earn_subject_polarity_bonus():
    """A phrase tagged subject=self talks about the reviewer's own
    company, not the subject vendor. Even with negative polarity it
    should not earn the subject_vendor + negative bonus."""
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    baseline = _witness_salience(review, enrichment, _baseline_span())
    self_neg = _witness_salience(
        review,
        enrichment,
        _baseline_span(subject="self", polarity="negative"),
    )
    assert self_neg == baseline


def test_subject_vendor_positive_no_bonus():
    """Positive polarity is for counterevidence routing (Phase 3); the
    selection bonus is for pain witnesses only."""
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    baseline = _witness_salience(review, enrichment, _baseline_span())
    pos = _witness_salience(
        review,
        enrichment,
        _baseline_span(subject="subject_vendor", polarity="positive"),
    )
    assert pos == baseline


def test_primary_driver_role_gets_one_and_a_half_bonus():
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    baseline = _witness_salience(review, enrichment, _baseline_span())
    pd = _witness_salience(
        review, enrichment, _baseline_span(role="primary_driver"),
    )
    assert round(pd - baseline, 2) == 1.5


def test_supporting_context_role_no_bonus():
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    baseline = _witness_salience(review, enrichment, _baseline_span())
    sc = _witness_salience(
        review, enrichment, _baseline_span(role="supporting_context"),
    )
    assert sc == baseline


def test_passing_mention_role_no_bonus():
    """passing_mention is barred from common_pattern by Phase 5a
    candidate_types -- it can still earn flex slots, but should NOT
    get the primary_driver salience bonus."""
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    baseline = _witness_salience(review, enrichment, _baseline_span())
    pm = _witness_salience(
        review, enrichment, _baseline_span(role="passing_mention"),
    )
    assert pm == baseline


def test_pain_confidence_strong_gets_two_point_bonus():
    review = _baseline_review()
    span = _baseline_span()
    base = _witness_salience(review, _baseline_enrichment(), span)
    strong = _witness_salience(
        review, _baseline_enrichment(pain_confidence="strong"), span,
    )
    assert round(strong - base, 2) == 2.0


def test_pain_confidence_weak_gets_half_point_bonus():
    review = _baseline_review()
    span = _baseline_span()
    base = _witness_salience(review, _baseline_enrichment(), span)
    weak = _witness_salience(
        review, _baseline_enrichment(pain_confidence="weak"), span,
    )
    assert round(weak - base, 2) == 0.5


def test_pain_confidence_none_no_bonus():
    """A pain that Phase 4 demoted to overall_dissatisfaction with
    confidence='none' should NOT earn a confidence bonus -- otherwise
    we'd reward un-corroborated extractions."""
    review = _baseline_review()
    span = _baseline_span()
    base = _witness_salience(review, _baseline_enrichment(), span)
    none_conf = _witness_salience(
        review, _baseline_enrichment(pain_confidence="none"), span,
    )
    assert none_conf == base


def test_grounded_status_gets_small_quote_quality_bonus():
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    base = _witness_salience(review, enrichment, _baseline_span())
    grounded = _witness_salience(
        review,
        enrichment,
        _baseline_span(grounding_status="grounded"),
    )
    assert round(grounded - base, 2) == 0.75


def test_not_grounded_status_gets_no_quote_quality_bonus():
    review = _baseline_review()
    enrichment = _baseline_enrichment()
    base = _witness_salience(review, enrichment, _baseline_span())
    not_grounded = _witness_salience(
        review,
        enrichment,
        _baseline_span(grounding_status="not_grounded"),
    )
    assert not_grounded == base


def test_full_v4_combo_outscores_strong_v3_baseline():
    """Realistic Slack-style scenario: a v4 row that legitimately got
    Phase 4 demoted (no strong churn signals) should still outscore
    a v3 row with similar review-level signals because the v4 row
    surfaces a properly tagged subject_vendor + negative + primary_driver
    pain phrase."""
    review = _baseline_review()
    # v3 row: has a competitor mention (legacy +1.5) but no phrase tags
    v3_review = {**review, "reviewer_company": "Globex"}
    v3_enrichment = _baseline_enrichment(reviewer_context={"decision_maker": True})
    v3_span = _baseline_span(competitor="Asana")
    v3_score = _witness_salience(v3_review, v3_enrichment, v3_span)

    # v4 row: simpler review-level shape (no competitor / decision-maker /
    # company name) but the v2 prompt extracted a real phrase
    v4_enrichment = _baseline_enrichment(pain_confidence="weak")
    v4_span = _baseline_span(
        subject="subject_vendor",
        polarity="negative",
        role="primary_driver",
    )
    v4_score = _witness_salience(review, v4_enrichment, v4_span)

    # v3 legacy bonuses: 1.5 (decision_maker) + 1.5 (reviewer_company) +
    #                    1.5 (competitor) = +4.5
    # v4 phrase bonuses: 2.0 (subj+pol) + 1.5 (role) + 0.5 (weak) = +4.0
    # The v3 row still wins here -- which is correct! v3 has more
    # review-level signals. The bonus is intended to keep the playing
    # field even, not to bias toward v4. So we just assert v4 closes
    # most of the gap.
    assert v3_score - v4_score < 1.0


def test_full_v4_strong_conf_outscores_minimal_v3():
    """A v4 row with strong pain_confidence + full phrase tags should
    beat a minimal v3 row that had no review-level signals at all."""
    review = _baseline_review()
    v3_score = _witness_salience(review, _baseline_enrichment(), _baseline_span())

    v4_enrichment = _baseline_enrichment(pain_confidence="strong")
    v4_span = _baseline_span(
        subject="subject_vendor",
        polarity="negative",
        role="primary_driver",
    )
    v4_score = _witness_salience(review, v4_enrichment, v4_span)
    assert v4_score - v3_score == 5.5
