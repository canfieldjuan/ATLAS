"""Fit runner + digest integration + judge-fit CLI (v2 S6, #1931 final).

Real everything except the HTTP/model boundary (the injectable transport):
real store, real prompt builder, real guard, real parser, real digest
writer, real CLI main() in-process, real S1 harness for the closing
round-trip proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_reddit.fit_client import FitClientError, OpenAICompatibleJudgeClient
from atlas_reddit.fit_eval import evaluate_predictions, load_cases, load_predictions
from atlas_reddit.fit_runner import judge_fit_once, run_eval_cases
from atlas_reddit.store import ListeningStore

NOW = 1_751_600_000


def _ok_body(verdict="yes", reason="Repeat questions despite docs.",
             angle="Ask what the ticket history shows.", flags=(),
             prompt_tokens=100, completion_tokens=30) -> tuple[int, str]:
    prediction = {"verdict": verdict, "reason": reason, "angle": angle,
                  "risk_flags": list(flags)}
    return 200, json.dumps({
        "choices": [{"message": {"content": json.dumps(prediction)}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


class FakeTransport:
    def __init__(self, responses=None, default=None) -> None:
        self.responses = list(responses or [])
        self.default = default if default is not None else _ok_body()
        self.calls = 0

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        item = self.responses.pop(0) if self.responses else self.default
        if isinstance(item, Exception):
            raise item
        return item


def _client(transport) -> OpenAICompatibleJudgeClient:
    return OpenAICompatibleJudgeClient(
        backend="local", base_url="http://127.0.0.1:1234/v1", model="m",
        api_key="", timeout_seconds=5.0, transport=transport,
    )


@pytest.fixture()
def store(tmp_path: Path):
    with ListeningStore(tmp_path / "listening.db") as s:
        yield s


def _seed(store: ListeningStore, post_id: str, *, score: float = 3.0,
          body: str = "We have docs but users still ask.", status: str = "new") -> None:
    store.upsert_candidate(
        post_id=post_id, subreddit="CustomerSuccess", title="Docs vs product",
        url="https://www.reddit.com/r/CustomerSuccess/x/", author="u",
        created_utc=NOW - 3600, reddit_score=5, num_comments=3,
        keyword_score=score, final_score=score,
        matched_topics=("repeat-tickets",), observed_at=NOW - 3600,
        body_excerpt=body,
    )
    if status != "new":
        store.set_candidate_status(post_id, status)


def _run(store, transport, **overrides):
    kwargs = dict(now=NOW, min_final_score=1.0, max_calls=25,
                  prompt_version="fit.v1")
    kwargs.update(overrides)
    return judge_fit_once(store, _client(transport), **kwargs)


# -- judge_fit_once --------------------------------------------------------


def test_judges_new_candidates_and_persists_model_reviews(store) -> None:
    _seed(store, "t3_a")
    stats = _run(store, FakeTransport())
    assert stats.judged == 1 and stats.calls == 1
    review = store.get_fit_review("t3_a")
    assert review.verdict == "yes" and review.source == "model"
    assert review.guard_ok is True and review.model_id == "m"
    assert (stats.input_tokens, stats.output_tokens) == (100, 30)


def test_below_threshold_never_reaches_the_model(store) -> None:
    """The deterministic keyword+score gate runs first: a below-threshold
    candidate is never fetched, so the model is never called for it."""
    _seed(store, "t3_low", score=0.5)   # below fit_min_score=1.0
    _seed(store, "t3_high", score=3.0)
    transport = FakeTransport()
    stats = _run(store, transport, min_final_score=1.0)
    assert transport.calls == 1  # only the above-threshold candidate
    assert store.get_fit_review("t3_low") is None
    assert store.get_fit_review("t3_high") is not None
    assert stats.judged == 1


def test_max_calls_cap_is_enforced(store) -> None:
    for i in range(4):
        _seed(store, f"t3_{i}")
    transport = FakeTransport()
    stats = _run(store, transport, max_calls=2)
    assert transport.calls == 2 and stats.judged == 2


def test_max_calls_zero_makes_no_calls(store) -> None:
    _seed(store, "t3_a")
    transport = FakeTransport()
    stats = _run(store, transport, max_calls=0)
    assert transport.calls == 0 and stats.judged == 0


def test_already_reviewed_candidates_are_skipped(store) -> None:
    _seed(store, "t3_a")
    _run(store, FakeTransport())
    transport = FakeTransport()
    stats = _run(store, transport)
    assert transport.calls == 0 and stats.skipped == 1


def test_refresh_rejudges_only_when_inputs_changed(store) -> None:
    _seed(store, "t3_a", body="original body")
    _run(store, FakeTransport())
    # same inputs + refresh -> still skipped
    t1 = FakeTransport()
    assert _run(store, t1, refresh=True).skipped == 1
    assert t1.calls == 0
    # inputs change -> refresh re-judges
    _seed(store, "t3_a", body="a very different body now")
    t2 = FakeTransport()
    stats = _run(store, t2, refresh=True)
    assert t2.calls == 1 and stats.judged == 1
    # but without --refresh, a changed candidate is NOT re-judged
    _seed(store, "t3_a", body="changed yet again")
    t3 = FakeTransport()
    assert _run(store, t3).skipped == 1 and t3.calls == 0


def test_guard_blocked_output_persisted_flagged_and_redacted(store) -> None:
    _seed(store, "t3_a")
    body = _ok_body(angle="An audit guarantees a 40% ticket reduction.")
    stats = _run(store, FakeTransport(default=body))
    assert stats.judged == 1 and stats.blocked == 1
    review = store.get_fit_review("t3_a")
    assert review.guard_ok is False
    assert review.reason == "" and review.angle is None  # redacted
    assert "GUARANTEED_DEFLECTION" in review.guard_codes


def test_transport_failure_is_contained_pass_continues(store) -> None:
    _seed(store, "t3_a")
    _seed(store, "t3_b")
    transport = FakeTransport(
        responses=[FitClientError("fit judge HTTP 429"), _ok_body()]
    )
    stats = _run(store, transport)
    assert transport.calls == 2
    assert stats.judged == 1 and len(stats.errors) == 1
    assert "429" in stats.errors[0]


def test_malformed_model_output_recorded_no_review(store) -> None:
    _seed(store, "t3_a")
    bad = (200, json.dumps({
        "choices": [{"message": {"content": "the thread looks fit to me"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }))
    stats = _run(store, FakeTransport(default=bad))
    assert stats.judged == 0 and len(stats.errors) == 1
    assert "model_output_invalid_json" in stats.errors[0]
    assert store.get_fit_review("t3_a") is None


# -- run_eval_cases: the closing "same ruler" proof ------------------------


def test_eval_envelopes_grade_through_the_real_harness(tmp_path: Path) -> None:
    """The arc's closing property: the runner's eval output feeds the S1
    harness with ZERO adapter code -- the same ruler grades a live model."""
    cases = [
        {
            "case_id": "obvious_fit",
            "category": "obvious_fit",
            "candidate": {
                "post_id": "t3_x", "subreddit": "CustomerSuccess",
                "title": "Repeat questions despite docs",
                "body": "KB exists but users still ask the same things.",
                "matched_topics": ["repeat-tickets"],
            },
            "expected_verdicts": ["yes"],
            "required_reason_terms": ["repeat"],
        }
    ]
    good = _ok_body(
        verdict="yes",
        reason="They describe repeat questions despite documentation.",
        angle="Ask what the ticket history shows about which questions recur.",
    )
    envelopes = run_eval_cases(
        _client(FakeTransport(default=good)), tuple(cases), prompt_version="fit.v1"
    )
    cases_path = tmp_path / "cases.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    cases_path.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    preds_path.write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in envelopes),
        encoding="utf-8",
    )
    loaded = load_cases(cases_path)
    graded = evaluate_predictions(
        loaded, load_predictions(preds_path, frozenset(c.case_id for c in loaded))
    )
    assert graded.ok  # a live model's output graded by the same corpus


def test_eval_transport_failure_is_a_gradeable_envelope() -> None:
    cases = ({"case_id": "c1", "candidate": {"title": "t", "subreddit": "s"}},)
    envelopes = run_eval_cases(
        _client(FakeTransport(default=FitClientError("boom"))),
        cases, prompt_version="fit.v1",
    )
    assert envelopes[0]["prediction"] is None
    assert envelopes[0]["parse_error"] == "model_http_error"


# -- digest integration ----------------------------------------------------


def test_digest_renders_guard_ok_fit_line_and_hides_blocked(store, tmp_path) -> None:
    from atlas_reddit.digest import write_digest

    _seed(store, "t3_ok")
    _seed(store, "t3_blocked")
    _run(store, FakeTransport())  # judges t3_ok (and t3_blocked, both clean)
    # force a blocked review on t3_blocked
    store.upsert_fit_review(
        post_id="t3_blocked", verdict="yes", reason="cut tickets 40%",
        angle="ROI story", risk_flags=(), guard_ok=False,
        guard_codes=("GUARANTEED_DEFLECTION",), source="model", model_id="m",
        prompt_version="fit.v1", input_hash="h", reviewed_at=NOW,
    )
    content = write_digest(store, digest_dir=tmp_path / "d", generated_on="2026-07-03").read_text()
    assert "fit: yes" in content
    assert "Ask what the ticket history shows" in content  # t3_ok angle
    # the blocked review's post shows no fit line (no verdict text for it)
    assert content.count("fit: yes") == 1


# -- judge-fit CLI ---------------------------------------------------------


def test_cli_judge_fit_backend_off_exits_two(tmp_path, monkeypatch, capsys) -> None:
    from atlas_reddit.__main__ import main

    for var in ("ATLAS_REDDIT_FIT_BACKEND", "ATLAS_REDDIT_FIT_BASE_URL",
                "ATLAS_REDDIT_FIT_MODEL", "ATLAS_REDDIT_FIT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    code = main(["judge-fit", "--db", str(tmp_path / "s.db")])
    assert code == 2
    assert "backend is off" in capsys.readouterr().err


def _patch_client(monkeypatch, transport) -> None:
    import atlas_reddit.__main__ as m

    monkeypatch.setattr(m, "build_judge_client", lambda settings: _client(transport))


def test_cli_judge_fit_real_mode_persists(tmp_path, monkeypatch, capsys) -> None:
    from atlas_reddit.__main__ import main

    db = tmp_path / "s.db"
    with ListeningStore(db) as store:
        _seed(store, "t3_a")
    _patch_client(monkeypatch, FakeTransport())
    code = main(["judge-fit", "--db", str(db)])
    assert code == 0
    assert "judged=1" in capsys.readouterr().out
    with ListeningStore(db) as store:
        assert store.get_fit_review("t3_a").source == "model"


def test_cli_judge_fit_eval_mode_writes_envelopes(tmp_path, monkeypatch, capsys) -> None:
    from atlas_reddit.__main__ import main

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps({"case_id": "c1", "candidate": {"title": "t", "subreddit": "s"}}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "preds.jsonl"
    _patch_client(monkeypatch, FakeTransport())
    code = main([
        "judge-fit", "--eval-cases", str(cases), "--predictions-output", str(out),
    ])
    assert code == 0
    envelope = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert envelope["case_id"] == "c1"
    assert envelope["prediction"]["verdict"] == "yes"
    assert envelope["prompt_version"] == "fit.v1"
