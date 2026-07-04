"""Fit runner: judge stored candidates, or evaluate a model against the
harness corpus (v2 S6, #1931 -- final arc slice).

``judge_fit_once`` selects prequalified candidates from the store, judges
each through the S5 client, guards the output (S3), and persists a fit
review (S4). It reuses ``list_candidates``' deterministic keyword+score
gate so below-threshold posts never reach the model, honours a
per-run call cap, skips already-reviewed candidates (unless ``refresh``
and the input changed), and contains per-candidate transport failures so
one bad call never aborts the pass.

``run_eval_cases`` points the SAME prompt builder + client at the S1
fixture corpus and emits harness-gradeable prediction envelopes -- the
one ruler grades a live model with zero adapter code, no store writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .fit import build_fit_prompt
from .fit_client import FitClientError, OpenAICompatibleJudgeClient
from .fit_guard import guard_fit_decision
from .store import ListeningStore, fit_input_hash


@dataclass
class FitRunStats:
    judged: int = 0
    blocked: int = 0
    skipped: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)


def _candidate_prompt(candidate) -> tuple[dict, ...]:
    # Only the CONTENT fields the fit decision is about, so they match the
    # staleness hash exactly. Volatile engagement metrics
    # (reddit_score/num_comments) and the derived keyword_score are
    # deliberately excluded: an upvote must not make a fit review stale.
    return build_fit_prompt(
        title=candidate.title,
        subreddit=candidate.subreddit,
        body=candidate.body_excerpt,
        matched_topics=candidate.matched_topics,
    )


def _candidate_input_hash(candidate) -> str:
    return fit_input_hash(
        post_id=candidate.post_id,
        subreddit=candidate.subreddit,
        title=candidate.title,
        body_excerpt=candidate.body_excerpt,
        matched_topics=candidate.matched_topics,
    )


def judge_fit_once(
    store: ListeningStore,
    client: OpenAICompatibleJudgeClient,
    *,
    now: int,
    min_final_score: float,
    max_calls: int,
    prompt_version: str,
    refresh: bool = False,
    pace_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> FitRunStats:
    """Judge up to ``max_calls`` prequalified 'new' candidates."""
    stats = FitRunStats()
    if max_calls <= 0:
        return stats
    # The deterministic keyword+score gate runs first: below-threshold
    # candidates are never fetched, so they never reach the model. Ordered
    # by final_score DESC, so the strongest candidates are judged first.
    candidates = store.list_candidates(status="new", min_final_score=min_final_score)
    for candidate in candidates:
        if stats.calls >= max_calls:
            break
        input_hash = _candidate_input_hash(candidate)
        existing = store.get_fit_review(candidate.post_id)
        if existing is not None and (
            not refresh
            or (
                existing.input_hash == input_hash
                and existing.prompt_version == prompt_version
            )
        ):
            # Already judged; under --refresh, re-judge only when the content
            # inputs OR the prompt version changed (a prompt bump makes an
            # older review stale even if the Reddit content is unchanged).
            stats.skipped += 1
            continue
        if stats.calls and pace_seconds > 0:
            sleep(pace_seconds)
        try:
            decision, meta = client.judge(_candidate_prompt(candidate))
        except FitClientError as exc:
            # Transport/HTTP failure: contained like the poller/tracker do
            # -- recorded, the call counts against the cap, pass continues.
            stats.calls += 1
            stats.errors.append(f"{candidate.post_id}: {exc}")
            continue
        stats.calls += 1
        stats.input_tokens += meta.input_tokens
        stats.output_tokens += meta.output_tokens
        if decision is None:
            # Malformed model output: recorded as data, no review persisted.
            stats.errors.append(f"{candidate.post_id}: {meta.parse_error}")
            continue
        guarded = guard_fit_decision(decision)
        store.upsert_fit_review(
            post_id=candidate.post_id,
            verdict=decision.verdict,
            reason=decision.reason,
            angle=decision.angle,
            risk_flags=decision.risk_flags,
            guard_ok=guarded.ok,
            guard_codes=guarded.codes,
            source="model",
            model_id=meta.model_id,
            prompt_version=prompt_version,
            input_hash=input_hash,
            reviewed_at=now,
        )
        stats.judged += 1
        if not guarded.ok:
            stats.blocked += 1
    return stats


def run_eval_cases(
    client: OpenAICompatibleJudgeClient,
    cases: tuple[dict, ...],
    *,
    prompt_version: str,
) -> list[dict]:
    """Judge each fixture case's candidate and return harness prediction
    envelopes keyed by case_id. No store writes -- the same prompt+client
    the runner uses, graded by the S1 harness against the same corpus."""
    envelopes: list[dict] = []
    for case in cases:
        candidate = case["candidate"]
        messages = build_fit_prompt(
            title=candidate.get("title", ""),
            subreddit=candidate.get("subreddit", ""),
            body=candidate.get("body", ""),
            matched_topics=tuple(candidate.get("matched_topics", ())),
        )
        prediction: dict | None
        parse_error: str | None
        try:
            decision, meta = client.judge(messages)
        except FitClientError:
            # Could not evaluate this case: record as a gradeable failure
            # (a closed parse-error code), not a crash.
            prediction, parse_error, model_id = None, "model_http_error", client.model_id
        else:
            model_id = meta.model_id
            if decision is None:
                prediction, parse_error = None, meta.parse_error
            else:
                prediction = {
                    "verdict": decision.verdict,
                    "reason": decision.reason,
                    "angle": decision.angle,
                    "risk_flags": list(decision.risk_flags),
                }
                parse_error = None
        envelopes.append(
            {
                "case_id": case["case_id"],
                "prediction": prediction,
                "model_id": model_id,
                "prompt_version": prompt_version,
                "parse_error": parse_error,
            }
        )
    return envelopes
