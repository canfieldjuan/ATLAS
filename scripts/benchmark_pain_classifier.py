#!/usr/bin/env python3
"""Benchmark pain classification against the hand-labeled baseline corpus.

Reads the fixture produced by sample_pain_benchmark_corpus.py, compares the
stored pipeline output to human labels, and emits stratified metrics per
bucket and globally. Read-only; does not modify pipeline code.

What this measures right now (Phase 0 baseline):
  - Primary pain accuracy: stored pain_category vs. expected_primary_pain
  - Pricing false-positive rate: stored=pricing when human says it's not
  - Pricing recall: we catch real pricing complaints when they are labeled so
  - pricing_is_driver calibration: how human-flagged driver status lines up
    with classifier output
  - Phrase-level ground truth distribution: subject / polarity / role /
    grounded counts across all labeled phrases

After Phases 1-4 ship, re-run this against the same fixture to see the
improvement. The benchmark logic itself does not change between phases.

Usage:
  cd /home/juan-canfield/Desktop/Atlas
  source .venv/bin/activate
  python scripts/benchmark_pain_classifier.py
  # reads tests/fixtures/pain_classification_baseline.json by default

  python scripts/benchmark_pain_classifier.py --output /tmp/baseline_report.json
  # also write machine-readable metrics

  python scripts/benchmark_pain_classifier.py --strict
  # fail if any entries still have unfilled human_labels
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / ".env.local", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_pain_classifier")

_DEFAULT_FIXTURE = Path("tests/fixtures/pain_classification_baseline.json")

_REVIEW_LEVEL_LABEL_KEYS: tuple[str, ...] = (
    "expected_primary_pain",
    "pricing_is_driver",
    "has_subject_attribution_issue",
    "has_polarity_trap",
    "expected_verbatim_witness_count",
)
_PHRASE_LEVEL_LABEL_KEYS: tuple[str, ...] = (
    "subject",
    "polarity",
    "role",
    "grounded_in_source_text",
)


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"fixture not found at {path}. Run sample_pain_benchmark_corpus.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"fixture at {path} has no entries")
    return entries


def _label_is_unfilled(value: Any) -> bool:
    """A label counts as unfilled if it is None, or an empty string (notes
    fields are permitted to be empty and are not checked here)."""
    return value is None


def _count_unlabeled(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-bucket counts of entries with ANY unfilled required label
    (review-level or any phrase-level)."""
    by_bucket: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = str(entry.get("bucket") or "unknown")
        slot = by_bucket.setdefault(bucket, {"total": 0, "unlabeled": 0})
        slot["total"] += 1
        review_labels = entry.get("review_level_labels") or {}
        if any(_label_is_unfilled(review_labels.get(k)) for k in _REVIEW_LEVEL_LABEL_KEYS):
            slot["unlabeled"] += 1
            continue
        phrases = entry.get("extracted_phrases") or []
        phrase_missing = False
        for phrase in phrases:
            labels = phrase.get("human_labels") or {}
            if any(_label_is_unfilled(labels.get(k)) for k in _PHRASE_LEVEL_LABEL_KEYS):
                phrase_missing = True
                break
        if phrase_missing:
            slot["unlabeled"] += 1
    return by_bucket


def _fully_labeled(entry: dict[str, Any]) -> bool:
    review_labels = entry.get("review_level_labels") or {}
    if any(_label_is_unfilled(review_labels.get(k)) for k in _REVIEW_LEVEL_LABEL_KEYS):
        return False
    for phrase in entry.get("extracted_phrases") or []:
        labels = phrase.get("human_labels") or {}
        if any(_label_is_unfilled(labels.get(k)) for k in _PHRASE_LEVEL_LABEL_KEYS):
            return False
    return True


def _review_level_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute review-level metrics. Only considers fully-labeled entries."""
    labeled = [e for e in entries if _fully_labeled(e)]
    total = len(labeled)
    if total == 0:
        return {
            "labeled_count": 0,
            "note": "no fully-labeled entries -- fill in human_labels to produce metrics",
        }

    correct_primary = 0
    pricing_stored = 0
    pricing_stored_and_correct = 0  # stored says pricing, human says pricing
    pricing_expected = 0
    pricing_expected_caught = 0  # human says pricing, stored also says pricing
    pricing_driver_flag_true = 0
    pricing_driver_flag_true_and_stored_pricing = 0
    pricing_driver_flag_false_but_stored_pricing = 0

    pain_confusion: Counter[tuple[str, str]] = Counter()

    for e in labeled:
        stored = str(e.get("stored_enrichment", {}).get("pain_category") or "").strip()
        labels = e.get("review_level_labels") or {}
        expected = str(labels.get("expected_primary_pain") or "").strip()
        pain_confusion[(stored, expected)] += 1

        if stored == expected:
            correct_primary += 1

        if stored == "pricing":
            pricing_stored += 1
            if expected == "pricing":
                pricing_stored_and_correct += 1

        if expected == "pricing":
            pricing_expected += 1
            if stored == "pricing":
                pricing_expected_caught += 1

        flag = bool(labels.get("pricing_is_driver"))
        if flag:
            pricing_driver_flag_true += 1
            if stored == "pricing":
                pricing_driver_flag_true_and_stored_pricing += 1
        else:
            if stored == "pricing":
                pricing_driver_flag_false_but_stored_pricing += 1

    def _rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator > 0 else None

    return {
        "labeled_count": total,
        "primary_pain_accuracy": _rate(correct_primary, total),
        "primary_pain_correct": correct_primary,
        "pricing_stored_count": pricing_stored,
        "pricing_expected_count": pricing_expected,
        "pricing_false_positive_rate": _rate(
            pricing_stored - pricing_stored_and_correct, pricing_stored
        ),
        "pricing_recall": _rate(pricing_expected_caught, pricing_expected),
        "pricing_driver_flag_true_count": pricing_driver_flag_true,
        "pricing_driver_alignment_rate": _rate(
            pricing_driver_flag_true_and_stored_pricing, pricing_driver_flag_true
        ),
        "pricing_stored_without_driver_flag": pricing_driver_flag_false_but_stored_pricing,
        "pain_confusion_top": pain_confusion.most_common(10),
    }


def _phrase_level_distribution(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate phrase-level label distributions. Only considers phrases
    from fully-labeled entries so we have consistent ground truth."""
    subject_counts: Counter[str] = Counter()
    polarity_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    grounded_true = 0
    grounded_false = 0
    total_phrases = 0

    pricing_phrase_subjects: Counter[str] = Counter()
    pricing_phrase_polarities: Counter[str] = Counter()
    pricing_phrase_subject_vendor_negative = 0
    pricing_phrase_count = 0

    for entry in entries:
        if not _fully_labeled(entry):
            continue
        for phrase in entry.get("extracted_phrases") or []:
            labels = phrase.get("human_labels") or {}
            subject = labels.get("subject")
            polarity = labels.get("polarity")
            role = labels.get("role")
            grounded = labels.get("grounded_in_source_text")
            if subject is None or polarity is None or role is None or grounded is None:
                continue
            total_phrases += 1
            subject_counts[str(subject)] += 1
            polarity_counts[str(polarity)] += 1
            role_counts[str(role)] += 1
            if bool(grounded):
                grounded_true += 1
            else:
                grounded_false += 1
            if phrase.get("source_field") == "pricing_phrases":
                pricing_phrase_count += 1
                pricing_phrase_subjects[str(subject)] += 1
                pricing_phrase_polarities[str(polarity)] += 1
                if str(subject) == "subject_vendor" and str(polarity) == "negative":
                    pricing_phrase_subject_vendor_negative += 1

    def _rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator > 0 else None

    return {
        "labeled_phrase_count": total_phrases,
        "subject_breakdown": dict(subject_counts),
        "polarity_breakdown": dict(polarity_counts),
        "role_breakdown": dict(role_counts),
        "grounded_in_source_text": {
            "true": grounded_true,
            "false": grounded_false,
            "true_rate": _rate(grounded_true, total_phrases),
        },
        "pricing_phrases": {
            "count": pricing_phrase_count,
            "subject_breakdown": dict(pricing_phrase_subjects),
            "polarity_breakdown": dict(pricing_phrase_polarities),
            "subject_vendor_and_negative_count": pricing_phrase_subject_vendor_negative,
            "would_survive_layers_1_and_2_rate": _rate(
                pricing_phrase_subject_vendor_negative, pricing_phrase_count
            ),
        },
    }


def _by_bucket(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        buckets.setdefault(str(e.get("bucket") or "unknown"), []).append(e)
    return buckets


def _phrase_grounding_pipeline_vs_human(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Baseline the grounding check the proposed Phase 1b will ship.

    Current pipeline has no explicit phrase-grounding output; we simulate it
    with a raw case-insensitive substring check of phrase text against
    (summary + ' ' + review_text). Compare to human `grounded_in_source_text`
    labels. This measures how far a no-normalization grounding check gets us
    before Phase 1b's normalized-grounding helper lands.
    """
    true_positive = 0   # pipeline True, human True
    false_positive = 0  # pipeline True, human False
    true_negative = 0   # pipeline False, human False
    false_negative = 0  # pipeline False, human True
    unlabeled = 0
    total = 0

    for e in entries:
        review_text = str(e.get("review_text") or "")
        summary = str(e.get("summary") or "")
        source_blob = (summary + " " + review_text).lower() if summary else review_text.lower()
        for phrase in e.get("extracted_phrases") or []:
            total += 1
            text = str(phrase.get("text") or "").strip().lower()
            human = (phrase.get("human_labels") or {}).get("grounded_in_source_text")
            if human is None:
                unlabeled += 1
                continue
            pipeline = bool(text) and text in source_blob
            if pipeline and human:
                true_positive += 1
            elif pipeline and not human:
                false_positive += 1
            elif not pipeline and not human:
                true_negative += 1
            else:
                false_negative += 1

    labeled = total - unlabeled
    correct = true_positive + true_negative

    def _rate(num: int, denom: int) -> float | None:
        return round(num / denom, 4) if denom > 0 else None

    return {
        "total_phrases": total,
        "labeled_phrases": labeled,
        "unlabeled_phrases": unlabeled,
        "pipeline_accuracy": _rate(correct, labeled),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "note": (
            "Pipeline uses raw substring match today. Phase 1b adds normalized "
            "grounding (whitespace, unicode punctuation, markdown) which should "
            "convert some false_negative cases to true_positive."
        ),
    }


def _phrase_gate_survival_by_field(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each phrase source_field, report what fraction of labeled phrases
    would survive Layers 1+2 (subject=subject_vendor AND polarity in
    {negative, mixed}). Shows which phrase buckets will shrink most when
    gates land.

    Only considers fully-labeled phrases. Counts are NOT accuracy - there is
    no pipeline prediction to compare to yet; this is ground-truth
    distribution per phrase type.
    """
    by_field: dict[str, dict[str, int]] = {}

    for e in entries:
        for phrase in e.get("extracted_phrases") or []:
            field = str(phrase.get("source_field") or "unknown")
            labels = phrase.get("human_labels") or {}
            subject = labels.get("subject")
            polarity = labels.get("polarity")
            if subject is None or polarity is None:
                continue
            slot = by_field.setdefault(field, {
                "labeled": 0,
                "subject_vendor": 0,
                "negative_or_mixed": 0,
                "would_survive_layer_1_and_2": 0,
            })
            slot["labeled"] += 1
            if str(subject) == "subject_vendor":
                slot["subject_vendor"] += 1
            if str(polarity) in ("negative", "mixed"):
                slot["negative_or_mixed"] += 1
            if str(subject) == "subject_vendor" and str(polarity) in ("negative", "mixed"):
                slot["would_survive_layer_1_and_2"] += 1

    def _rate(num: int, denom: int) -> float | None:
        return round(num / denom, 4) if denom > 0 else None

    for field, slot in by_field.items():
        slot["subject_vendor_rate"] = _rate(slot["subject_vendor"], slot["labeled"])
        slot["negative_or_mixed_rate"] = _rate(slot["negative_or_mixed"], slot["labeled"])
        slot["would_survive_rate"] = _rate(slot["would_survive_layer_1_and_2"], slot["labeled"])
    return by_field


async def _fetch_witness_audit(
    review_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Look up ACTIVE stored witnesses for each review_id.

    Witnesses are snapshotted per (vendor_name, analysis_window_days,
    schema_version, as_of_date); a single review can produce rows on many
    dates as synthesis re-runs. For an honest baseline of "what the UI shows
    today", we take only the latest as_of_date per (vendor, window, schema)
    tuple. Historical snapshots are excluded.

    Returns rows grouped by review_id. Skipped entirely when
    --skip-witness-audit is passed.
    """
    from atlas_brain.storage.database import close_database, get_db_pool, init_database

    if not review_ids:
        return {}

    await init_database()
    pool = get_db_pool()
    try:
        conn = await pool.acquire()
        try:
            rows = await conn.fetch(
                """
                WITH latest AS (
                    SELECT vendor_name, analysis_window_days, schema_version,
                           MAX(as_of_date) AS as_of_date
                    FROM b2b_vendor_witnesses
                    WHERE review_id = ANY($1::text[])
                    GROUP BY vendor_name, analysis_window_days, schema_version
                )
                SELECT w.review_id, w.witness_id, w.excerpt_text, w.pain_category,
                       w.witness_type, w.selection_reason, w.vendor_name,
                       w.as_of_date, w.analysis_window_days
                FROM b2b_vendor_witnesses w
                JOIN latest l
                  ON l.vendor_name = w.vendor_name
                 AND l.analysis_window_days = w.analysis_window_days
                 AND l.schema_version = w.schema_version
                 AND l.as_of_date = w.as_of_date
                WHERE w.review_id = ANY($1::text[])
                """,
                review_ids,
            )
        finally:
            await pool.release(conn)
    finally:
        await close_database()

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        rid = str(r["review_id"])
        grouped.setdefault(rid, []).append(dict(r))
    return grouped


def _witness_grounding_metrics(
    entries: list[dict[str, Any]],
    witnesses_by_review: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """For each review, count how many stored witnesses have excerpt_text that
    is NOT a case-insensitive substring of the full review_text. This is the
    plan's 'witness over-count' baseline: high numbers indicate witnesses
    whose quoted excerpts cannot be verified against source text."""
    reviews_with_witnesses = 0
    total_witnesses = 0
    ungrounded_witnesses = 0
    over_count_vs_expected = 0
    expected_total = 0
    human_expected_count_total = 0

    for e in entries:
        rid = str(e.get("review_id") or "")
        review_text_lower = str(e.get("review_text") or "").lower()
        witnesses = witnesses_by_review.get(rid) or []
        if witnesses:
            reviews_with_witnesses += 1
        for w in witnesses:
            total_witnesses += 1
            excerpt = str(w.get("excerpt_text") or "").strip().lower()
            if not excerpt or not review_text_lower:
                ungrounded_witnesses += 1
                continue
            if excerpt not in review_text_lower:
                ungrounded_witnesses += 1

        labels = e.get("review_level_labels") or {}
        expected_value = labels.get("expected_verbatim_witness_count")
        if isinstance(expected_value, int):
            human_expected_count_total += expected_value
            expected_total += 1
            # Over-count: today's witnesses exceed what the annotator said
            # could legitimately be quoted.
            if len(witnesses) > expected_value:
                over_count_vs_expected += len(witnesses) - expected_value

    def _rate(num: int, denom: int) -> float | None:
        return round(num / denom, 4) if denom > 0 else None

    return {
        "total_reviews": len(entries),
        "reviews_with_witnesses": reviews_with_witnesses,
        "total_witnesses": total_witnesses,
        "ungrounded_witnesses": ungrounded_witnesses,
        "ungrounded_rate": _rate(ungrounded_witnesses, total_witnesses),
        "human_labeled_reviews": expected_total,
        "human_expected_witness_count_total": human_expected_count_total,
        "over_count_vs_human_expected": over_count_vs_expected,
    }


def _format_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Pain Classification Benchmark -- Phase 0 Baseline")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Total entries in fixture: {report['total_entries']}")
    lines.append(f"Fully labeled: {report['fully_labeled']}")
    lines.append(f"Unlabeled: {report['total_entries'] - report['fully_labeled']}")
    lines.append("")
    lines.append("Labeling progress per bucket:")
    for bucket, counts in report["label_progress"].items():
        lines.append(f"  {bucket:22s}: {counts['unlabeled']:3d}/{counts['total']:3d} unlabeled")
    lines.append("")

    if report["fully_labeled"] == 0:
        lines.append("No fully-labeled entries yet. Fill in human_labels to see")
        lines.append("human-dependent metrics. DB-backed witness grounding shown below.")
        lines.append("")
        wg = report["global"].get("witness_grounding")
        if wg is not None:
            lines.append("-" * 72)
            lines.append("Global witness grounding audit (latest snapshot, no labels needed)")
            lines.append("-" * 72)
            lines.append(f"  reviews covered:               {wg['total_reviews']}")
            lines.append(f"  reviews with stored witnesses: {wg['reviews_with_witnesses']}")
            lines.append(f"  total stored witnesses:        {wg['total_witnesses']}")
            lines.append(f"  ungrounded witnesses:          {wg['ungrounded_witnesses']}")
            lines.append(f"  ungrounded_rate:               {wg['ungrounded_rate']}")

        deferred = report.get("deferred_benchmarks") or []
        if deferred:
            lines.append("")
            lines.append("-" * 72)
            lines.append("Deferred benchmarks (need pipeline changes before they can be measured)")
            lines.append("-" * 72)
            for item in deferred:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    lines.append("-" * 72)
    lines.append("Global review-level metrics")
    lines.append("-" * 72)
    g = report["global"]["review_level"]
    lines.append(f"  labeled_count:                    {g['labeled_count']}")
    lines.append(f"  primary_pain_accuracy:            {g['primary_pain_accuracy']}")
    lines.append(f"  pricing_stored_count:             {g['pricing_stored_count']}")
    lines.append(f"  pricing_expected_count:           {g['pricing_expected_count']}")
    lines.append(f"  pricing_false_positive_rate:      {g['pricing_false_positive_rate']}")
    lines.append(f"  pricing_recall:                   {g['pricing_recall']}")
    lines.append(f"  pricing_driver_flag_true_count:   {g['pricing_driver_flag_true_count']}")
    lines.append(f"  pricing_driver_alignment_rate:    {g['pricing_driver_alignment_rate']}")
    lines.append(f"  pricing_stored_without_driver:    {g['pricing_stored_without_driver_flag']}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-bucket review-level metrics")
    lines.append("-" * 72)
    for bucket, metrics in report["by_bucket"].items():
        rl = metrics["review_level"]
        lines.append(f"  [{bucket}]")
        lines.append(f"    labeled:               {rl.get('labeled_count')}")
        lines.append(f"    pain accuracy:         {rl.get('primary_pain_accuracy')}")
        lines.append(f"    pricing FPR:           {rl.get('pricing_false_positive_rate')}")
        lines.append(f"    pricing recall:        {rl.get('pricing_recall')}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Global phrase-level distribution (ground truth)")
    lines.append("-" * 72)
    p = report["global"]["phrase_level"]
    lines.append(f"  labeled_phrase_count:        {p['labeled_phrase_count']}")
    lines.append(f"  subject:     {p['subject_breakdown']}")
    lines.append(f"  polarity:    {p['polarity_breakdown']}")
    lines.append(f"  role:        {p['role_breakdown']}")
    lines.append(f"  grounded:    {p['grounded_in_source_text']}")
    pp = p.get("pricing_phrases") or {}
    lines.append("")
    lines.append(f"  pricing-phrase slice:")
    lines.append(f"    count:                                 {pp.get('count')}")
    lines.append(f"    subjects:                              {pp.get('subject_breakdown')}")
    lines.append(f"    polarities:                            {pp.get('polarity_breakdown')}")
    lines.append(f"    subject_vendor AND negative count:     {pp.get('subject_vendor_and_negative_count')}")
    lines.append(f"    would_survive_layers_1_2_rate:         {pp.get('would_survive_layers_1_and_2_rate')}")

    wg = report["global"].get("witness_grounding")
    if wg is not None:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Global witness grounding audit (DB-backed, latest snapshot only)")
        lines.append("-" * 72)
        lines.append(f"  reviews covered:               {wg['total_reviews']}")
        lines.append(f"  reviews with stored witnesses: {wg['reviews_with_witnesses']}")
        lines.append(f"  total stored witnesses:        {wg['total_witnesses']}")
        lines.append(f"  ungrounded witnesses:          {wg['ungrounded_witnesses']}")
        lines.append(f"  ungrounded_rate:               {wg['ungrounded_rate']}")
        lines.append(f"  human-labeled reviews:         {wg['human_labeled_reviews']}")
        lines.append(f"  over_count_vs_human_expected:  {wg['over_count_vs_human_expected']}")

    ga = report["global"].get("grounding_accuracy") or {}
    lines.append("")
    lines.append("-" * 72)
    lines.append("Phrase grounding: pipeline (raw substring) vs human labels")
    lines.append("-" * 72)
    lines.append(f"  total_phrases:       {ga.get('total_phrases')}")
    lines.append(f"  labeled_phrases:     {ga.get('labeled_phrases')}")
    lines.append(f"  unlabeled_phrases:   {ga.get('unlabeled_phrases')}")
    lines.append(f"  pipeline_accuracy:   {ga.get('pipeline_accuracy')}")
    lines.append(f"  confusion:           TP={ga.get('true_positive')} FP={ga.get('false_positive')} "
                 f"TN={ga.get('true_negative')} FN={ga.get('false_negative')}")

    gs = report["global"].get("gate_survival_by_field") or {}
    if gs:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Gate-survival by phrase field (ground truth only)")
        lines.append("-" * 72)
        for field, slot in sorted(gs.items()):
            lines.append(
                f"  {field:30s} labeled={slot['labeled']:4d} "
                f"subject_vendor_rate={slot['subject_vendor_rate']} "
                f"neg_or_mixed_rate={slot['negative_or_mixed_rate']} "
                f"would_survive_rate={slot['would_survive_rate']}"
            )

    deferred = report.get("deferred_benchmarks") or []
    if deferred:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Deferred benchmarks (need pipeline changes before they can be measured)")
        lines.append("-" * 72)
        for item in deferred:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _build_report(
    entries: list[dict[str, Any]],
    witnesses_by_review: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    fully_labeled_count = sum(1 for e in entries if _fully_labeled(e))
    label_progress = _count_unlabeled(entries)

    global_review = _review_level_metrics(entries)
    global_phrase = _phrase_level_distribution(entries)
    global_grounding_accuracy = _phrase_grounding_pipeline_vs_human(entries)
    global_gate_survival = _phrase_gate_survival_by_field(entries)

    by_bucket_metrics: dict[str, dict[str, Any]] = {}
    for bucket, bucket_entries in _by_bucket(entries).items():
        by_bucket_metrics[bucket] = {
            "review_level": _review_level_metrics(bucket_entries),
            "phrase_level": _phrase_level_distribution(bucket_entries),
            "grounding_accuracy": _phrase_grounding_pipeline_vs_human(bucket_entries),
            "gate_survival_by_field": _phrase_gate_survival_by_field(bucket_entries),
        }
        if witnesses_by_review is not None:
            by_bucket_metrics[bucket]["witness_grounding"] = _witness_grounding_metrics(
                bucket_entries, witnesses_by_review
            )

    report: dict[str, Any] = {
        "total_entries": len(entries),
        "fully_labeled": fully_labeled_count,
        "label_progress": label_progress,
        "global": {
            "review_level": global_review,
            "phrase_level": global_phrase,
            "grounding_accuracy": global_grounding_accuracy,
            "gate_survival_by_field": global_gate_survival,
        },
        "by_bucket": by_bucket_metrics,
        "deferred_benchmarks": [
            "pipeline-vs-human subject accuracy (requires Phase 1a tags)",
            "pipeline-vs-human polarity accuracy (requires Phase 1a tags)",
            "pipeline-vs-human role accuracy (requires Phase 1a tags)",
            "normalized grounding accuracy (requires Phase 1b helper)",
        ],
    }
    if witnesses_by_review is not None:
        report["global"]["witness_grounding"] = _witness_grounding_metrics(
            entries, witnesses_by_review
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=_DEFAULT_FIXTURE,
        help=f"Path to labeled fixture (default {_DEFAULT_FIXTURE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write machine-readable JSON report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any fixture entries still have unfilled labels.",
    )
    parser.add_argument(
        "--skip-witness-audit",
        action="store_true",
        help="Skip the DB-backed witness grounding audit (fixture-only mode).",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    entries = _load_fixture(args.fixture)

    witnesses_by_review: dict[str, list[dict[str, Any]]] | None = None
    if not args.skip_witness_audit:
        review_ids = [str(e.get("review_id")) for e in entries if e.get("review_id")]
        try:
            witnesses_by_review = await _fetch_witness_audit(review_ids)
        except Exception as exc:
            logger.warning(
                "witness audit failed (%s); proceeding with fixture-only metrics",
                exc,
            )
            witnesses_by_review = None

    report = _build_report(entries, witnesses_by_review)

    text = _format_text_report(report)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("wrote JSON report to %s", args.output)

    if args.strict and report["fully_labeled"] < report["total_entries"]:
        logger.error(
            "--strict: %d of %d entries still unlabeled",
            report["total_entries"] - report["fully_labeled"],
            report["total_entries"],
        )
        return 2
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
