# PR-Reddit-Fit-Runner

## Why this slice exists

Sixth and FINAL slice of the approved Reddit Listening v2 arc (#1931
comment 4872154794). It closes the loop: the fit RUNNER judges prequalified
stored candidates through the S5 client, guards (S3) and persists (S4) each
verdict, and the digest renders guard-passed advisories; an eval mode points
the SAME prompt + client at the S1 fixture corpus and emits harness-gradeable
envelopes, so the ruler that opened the arc grades a live model with zero
adapter code. After this merges the arc is complete.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Vertical slice

1. `atlas_reddit/fit_runner.py`: `judge_fit_once` (select new candidates
   above `fit_min_score` -- the deterministic gate that keeps below-threshold
   posts away from the model -- judge up to `fit_max_calls_per_run`, skip
   already-reviewed unless `--refresh` and the `input_hash` changed, contain
   per-candidate `FitClientError`, tally token usage) + `run_eval_cases`
   (judge fixture candidates -> harness prediction envelopes, no store
   writes) + `FitRunStats`.
2. `atlas_reddit/digest.py`: `render_digest(..., fit_reviews={})` renders a
   fit line per candidate from guard-passed reviews (all model text through
   `_sanitize_inline`); `write_digest` queries
   `list_fit_reviews(post_ids, only_guard_ok=True)`.
3. `atlas_reddit/__main__.py`: `judge-fit` command (real backend via
   `build_judge_client`) and `judge-fit --eval-cases <cases.jsonl>
   --predictions-output <path>` eval mode.
4. `tests/test_atlas_reddit_fit_runner.py`: runner both sides, the
   below-threshold/cap/skip/refresh/containment/malformed probes, the digest
   guard-ok-only rendering, the CLI (backend-off, real mode, eval mode), and
   the closing round-trip proof (eval output grades through the real harness).

### Review Contract

- Acceptance criteria:
  - [ ] Below-threshold candidates NEVER reach the model (the
        `list_candidates` score gate; asserted by the fake transport's call
        count); the `max_calls` cap is enforced; `max_calls=0` makes no calls.
  - [ ] New candidates are judged, guarded, and persisted as `source=model`
        reviews with token usage tallied; guard-blocked output persists
        flagged + REDACTED.
  - [ ] Already-reviewed candidates are skipped; `--refresh` re-judges ONLY
        when the `input_hash` changed (probed all three: same+refresh,
        changed+refresh, changed+no-refresh).
  - [ ] A per-candidate transport failure is contained (recorded, pass
        continues); malformed model output is recorded with its parse code
        and no review is persisted.
  - [ ] The digest renders a fit line only for guard-passed reviews (through
        the sanitizer); guard-blocked reviews never surface; a digest with no
        fit data is unchanged (existing tests stay green).
  - [ ] `judge-fit` exits 2 when the backend is off; real mode persists and
        exits 0/1; eval mode writes envelopes and exits 0/2.
  - [ ] CLOSING PROOF: eval-mode envelopes grade through the REAL S1 harness
        (`scripts/evaluate_atlas_reddit_fit.py`) with zero adapter code.
- Reachability proof (#1952): `python -m atlas_reddit judge-fit` persists
  model reviews (observable store rows) and `judge-fit --eval-cases ...
  --predictions-output p.jsonl` writes envelopes that the harness CLI then
  grades -- both exercised in-process and end-to-end in Verification.
- Affected surfaces: new runner module, digest render/query (additive), one
  CLI command, one test file. No Reddit-auth, config, or store-schema change.
- Risk areas: the deterministic gate must actually precede model calls
  (probed via call count); redaction of blocked reviews at render time
  (guard-ok-only query); cap/skip/refresh interaction (each probed).
- Reviewer rules triggered: R1, R2 (runner both sides + gate-before-call +
  containment both directions), R8 (idempotent skip + refresh-on-change),
  R11 (zero new dependencies), R12 (test auto-enrolls via the glob), R14
  (reachability + the closing harness round-trip named above).
- Test-adapter posture (#1934 real-adapters rule): only the HTTP/model
  boundary is faked (injectable transport); real store, prompt builder,
  guard, parser, digest writer, CLI, and the real S1 harness.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/digest.py`
- `atlas_reddit/fit_runner.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Fit-Runner.md`
- `plans/archive/PR-Reddit-Fit-Client.md`
- `tests/test_atlas_reddit_fit_runner.py`

## Mechanism

`judge_fit_once` fetches `list_candidates(status='new',
min_final_score=fit_min_score)` (ordered by score DESC, so the strongest are
judged first and below-threshold posts are never fetched), then for each: if
a review exists and either `--refresh` is off or the recomputed `input_hash`
matches, skip; otherwise build the prompt, call the client, and on success
guard + `upsert_fit_review(source='model')`. A `FitClientError` is recorded
and the pass continues; a `None` decision records its parse code without
persisting. The loop stops after `max_calls` model calls. `run_eval_cases`
runs the same prompt + client over fixture candidates and returns envelopes
`{case_id, prediction|null, model_id, prompt_version, parse_error}` -- a
transport failure becomes `parse_error='model_http_error'` so every case is
gradeable. The digest renders a `fit:` line only from guard-passed reviews,
sanitized like all Reddit text.

## Intentional

- **The deterministic gate is the score filter, not a runner flag**: reusing
  `list_candidates(min_final_score=...)` means "no model call below threshold"
  is structurally true and provable by call count, not by trusting new logic.
- **Transport failures are contained, malformed output is data**: the runner
  mirrors the poller/tracker/purge containment -- one bad candidate never
  aborts the pass -- and a parse failure is recorded, never persisted.
- **The digest queries guard-ok reviews only**: unsafe model text the guard
  blocked (already redacted at rest in S4) also never reaches render.
- **Eval mode writes no store rows**: it grades a model against the corpus,
  which is a measurement, not ingestion.
- **The same ruler closes the arc**: eval envelopes feed the S1 harness with
  no adapter, so "the harness is the ruler" holds for live models too.

## Deferred

- Scheduling (an autonomous task cadence for poll/track/purge/judge-fit/
  digest) and any live-model quality tuning are operator decisions beyond
  the approved arc. Live-use credentials (Reddit token, a fit backend key)
  remain unminted -- runbook-documented, operator-provided.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_runner.py -q`:
  15 passed (runner both sides; below-threshold/cap/skip/refresh/containment/
  malformed; digest guard-ok rendering; CLI backend-off/real/eval; the
  closing harness round-trip).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 552 passed.
- Reachability + closing proof: build eval-mode envelopes from a fake-transport
  client, then `python scripts/evaluate_atlas_reddit_fit.py --cases ...
  --predictions <emitted> --fail-on-eval-fail` grades them (1/1, exit 0).
- ASCII byte-scan on changed Python files: clean. No-write probe over the new
  modules: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 109 |
| `atlas_reddit/digest.py` | 29 |
| `atlas_reddit/fit_runner.py` | 180 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Fit-Runner.md` | 138 |
| `plans/archive/PR-Reddit-Fit-Client.md` | 0 |
| `tests/test_atlas_reddit_fit_runner.py` | 311 |
| **Total** | **770** |
