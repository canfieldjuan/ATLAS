# PR-Reddit-Listening-Config-Scoring

## Why this slice exists

First implementation slice (S1) of the approved #1934 arc (Fable 5 builder
trial; product epic #1931): a read-only Reddit listening tool for the
Resolution Audit. S1 lays the two foundations everything downstream consumes:
the operator-curated watchlist (subreddits + topic phrase clusters + scoring
knobs) and the deterministic keyword scorer that ranks candidate threads. No
network, no database, no LLM in this slice.

Diff-budget note: the total lands over the 400 LOC soft cap because a first
slice of a new package carries one-time arc scaffolding (package init, CI
workflow enrollment, sample watchlist, this plan) and because the watchlist
parser is guard-shaped, so AGENTS.md 3i requires a negative fixture per
detection branch. The code under test is ~330 LOC; the rest is mandated test
coverage and scaffolding that later slices reuse rather than repeat.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Vertical slice

1. New top-level package `atlas_reddit/` (modeled on `atlas_comms/`,
   deliberately outside `atlas_brain`'s import graph).
2. Typed env settings (`ATLAS_REDDIT_*` via pydantic-settings; no raw
   `os.environ`) with the watchlist path as the only S1 field.
3. Fail-closed TOML watchlist loader (stdlib `tomllib`): unknown keys,
   bad versions, invalid subreddit names, out-of-range or bool-typed
   weights, duplicates, and malformed/missing files all raise
   `WatchlistError` rather than being defaulted or skipped.
4. Pure deterministic keyword scorer: case-insensitive word-boundary
   phrase matching; a post scores 0 unless a topic phrase matches;
   help-signal and question bonuses only amplify topical posts; topic
   weight counts once per topic regardless of phrase count.
5. Committed `atlas_reddit/watchlist.sample.toml` (illustrative, no secrets,
   no usernames) plus a test asserting the shipped sample always parses.
6. CI enrollment in the same PR: new path-filtered
   `.github/workflows/atlas_reddit_checks.yml` running
   `tests/test_atlas_reddit_config.py` and `tests/test_atlas_reddit_scoring.py`
   via the tests/test_atlas_reddit_*.py glob so future slices auto-enroll.

### Review Contract

- Acceptance criteria:
  - [ ] `parse_watchlist`/`load_watchlist` accept the documented shape and
        reject every invalid shape with a `WatchlistError` naming the
        offending key (negative fixture per detection branch).
  - [ ] Weight/bonus bounds probed both sides: weight 0 and >10 rejected,
        10.0 and 0.0001 accepted; bonus 0 accepted, negative and >5
        rejected; TOML `true` (bool-is-int trap) rejected as a number.
  - [ ] `score_post` is pure and deterministic; zero-gate holds (no topic
        match -> 0.0 even with question mark + help signals).
  - [ ] Word-boundary matching proven ("sla" does not match "island" or
        "translate"); documented no-stemming behavior has a test.
  - [ ] Public-API contract holds under adversarial input: non-dict
        documents raise `WatchlistError` (never bare AttributeError);
        regex-metachar phrases ("c++", "(deflection)") match literally;
        emoji-adjacent phrases still match (second side of the
        word-boundary lookarounds).
  - [ ] The committed sample watchlist parses through the real loader.
  - [ ] New tests run in PR CI via the new path-filtered workflow.
- Affected surfaces: new standalone package + tests + one new CI workflow.
  No existing Atlas surface is imported, touched, or re-exported.
- Risk areas: none live yet (no network, no DB, no secrets); main risk is
  validator false-accept/false-reject, covered by the boundary probes above.
- Reviewer rules triggered: R1, R2 (guard-shaped validator: failure branches
  + near-miss fixtures), R10, R11 (zero new dependencies; tomllib over
  undeclared-transitive pyyaml; typed settings), R12 (CI enrollment ships in
  this PR).
- Test-adapter posture (#1934 real-adapters rule): zero mocks in this slice.
  Scoring tests build watchlists through the real `parse_watchlist`; config
  tests do real file I/O under pytest `tmp_path` (an allowed temp-filesystem
  boundary); the committed sample is validated through the real loader.

### Files touched

- `.github/workflows/atlas_reddit_checks.yml`
- `atlas_reddit/__init__.py`
- `atlas_reddit/config.py`
- `atlas_reddit/scoring.py`
- `atlas_reddit/watchlist.sample.toml`
- `plans/PR-Reddit-Listening-Config-Scoring.md`
- `tests/test_atlas_reddit_config.py`
- `tests/test_atlas_reddit_scoring.py`

## Mechanism

`atlas_reddit/config.py` holds both config surfaces. Env settings are a
pydantic-settings class (`env_prefix="ATLAS_REDDIT_"`); S1 exposes only
`watchlist_path`, defaulting under the gitignored `data/` tree. The watchlist
itself is TOML parsed with stdlib `tomllib` into frozen dataclasses
(`SubredditEntry`, `Topic`, `Watchlist`). Validation is centralized in
`parse_watchlist(dict) -> Watchlist` (pure, unit-testable without files);
`load_watchlist(Path)` adds file I/O and wraps `FileNotFoundError` /
`TOMLDecodeError` into `WatchlistError` so callers have one error surface.
Every check rejects rather than defaults: unknown keys at any level, version
!= 1, subreddit names failing Reddit's `[A-Za-z0-9][A-Za-z0-9_]{2,20}` shape,
weights outside (0, 10], bonuses outside [0, 5], bools where numbers belong
(TOML `true` would otherwise coerce to 1.0), and case-insensitive duplicates.

`atlas_reddit/scoring.py` is a pure function `score_post(title, body,
subreddit_weight, watchlist) -> ScoreBreakdown`. Phrases compile to
`(?<!\w)escaped(?!\w)` patterns (lru_cached), matched against casefolded
`title\nbody`. If no topic matches, the result is 0.0 and bonuses are not
evaluated. Otherwise: `total = subreddit_weight * (sum(topic weights with a
hit) + help_signal_bonus if any signal hit + question_bonus if '?' present)`,
rounded to 4 places. The breakdown carries matched topics/phrases so the S3
digest can show why a thread surfaced.

## Intentional

- **TOML over the issue-sketch YAML**: PyYAML is importable here only as a
  transitive dependency (not declared in requirements.txt), and CI images run
  Python 3.11+ where `tomllib` is stdlib. Zero new dependencies beats format
  preference; the sketch's intent (human-edited watchlist file) is preserved.
- **Watchlist in a file, not SQLite** (approved arc amendment c): one
  human-editable source of truth; the S2 database holds runtime state only.
- **Topic weight counts once per topic, not per matched phrase**: phrase
  counts reward keyword-stuffed posts, not relevant ones. Matched phrases are
  still reported for digest display.
- **Zero-gate on bonuses**: an off-topic question with help language must not
  surface; bonuses amplify topical posts only.
- **No stemming/fuzzy matching**: plural/variant forms are explicit phrase
  entries. Deterministic and inspectable beats clever for a v1 radar; the
  sample file documents the behavior.
- **`active` flag and `list_type` from the #1931 sketch omitted**: dead
  fields until the S4 poller exists; unknown-key rejection means adding them
  later is an explicit, reviewed change.
- **Synchronous stdlib code, no async**: this is a local single-user CLI
  tool, not a brain service; the async-first convention targets `atlas_brain`
  I/O paths. Nothing here does I/O except reading one config file.
- **Package placement `atlas_reddit/` at repo root**: matches the
  `atlas_comms`/`atlas_edge` standalone-service precedent; keeps
  brain-startup and extracted-checks import graphs untouched.

Root-cause notes (per the #1934 retroactive fix rule; both found and fixed
pre-push during implementation):

- **Empty `help_signals = []` was rejected by the parser.** Root cause: the
  shared phrase-list validator conflated "present but empty" (coherent
  config: the bonus feature unused) with invalid, because topic phrases
  genuinely may not be empty. Fixed at the parser (`allow_empty` for
  help_signals only), not by adjusting the failing test fixtures; a
  dedicated boundary test locks the semantic.
- **`parse_watchlist` crashed with bare AttributeError/TypeError on
  non-dict input.** Root cause: the public validator assumed its own
  caller's contract instead of checking it; fixed with a type guard at the
  function boundary so every invalid input surfaces as `WatchlistError`.

Review-fix notes (Codex review of 26b3a878a; both findings verified real and
fixed at root in this PR):

- **Topic duplicate check compared untrimmed names while `Topic` stored
  stripped ones** (check/store mismatch class): " ticket-deflection "
  slipped past "ticket-deflection", stored as an identical name, and
  `score_post` would double-count that topic's weight. Fixed by normalizing
  once before the duplicate check so validation, dedupe, and storage all see
  one value; class-proofed with the cited case plus unseen
  whitespace/case/newline variants.
- **Subreddit name regex used `match()` with a `$` anchor**, which tolerates
  a trailing newline: "SaaS\n" passed validation, then
  `subreddit_weight("SaaS")` missed and a future poller would fetch an
  invalid subreddit. Fixed with `fullmatch()`; class-proofed with
  trailing/leading/internal newline and carriage-return probes.

## Deferred

- S2 SQLite store (candidates/tracked_threads/replies/purge_log; PRAGMA
  user_version migrations).
- S3 Markdown digest + `python -m atlas_reddit` CLI (adds `digest_dir` and
  `db_path` settings fields when they become real).
- S4 PRAW read-only poller: verify the current Reddit OAuth + PRAW auth path
  against live docs BEFORE auth code (operator-required gate); fail-closed
  scope assertion (read/identity/history only); setup runbook; praw
  dependency justification.
- S5 reply tracker; S6 deletion-compliance purge job.
- LLM `judge_fit` pass: beyond this arc entirely (claim-registry-gated,
  OpenRouter + local backend behind one port).
- Hard filters (unseen/dismissed/fresh/text-post) live with the poller (S4),
  where the post metadata exists.

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_config.py` and
  `tests/test_atlas_reddit_scoring.py`: 88 passed (happy path, per-branch
  negative fixtures, boundary values both sides, bool-typed-number traps,
  duplicates, mixed valid/invalid collections, near-miss zero-scores,
  word-boundary probes, regex-metachar phrases, emoji-adjacent matches,
  non-dict input contract, committed-sample-parses, env override).
- bash `scripts/check_ascii_python.sh`: passed (no non-ASCII in new .py).
- python `scripts/audit_workflow_security_posture.py` over .github/workflows:
  passed (new workflow clean; pre-existing WARNs on other files unchanged).
- python `scripts/sync_pr_plan.py` on this plan: Files touched + diff table
  regenerated from the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_reddit_checks.yml` | 31 |
| `atlas_reddit/__init__.py` | 9 |
| `atlas_reddit/config.py` | 276 |
| `atlas_reddit/scoring.py` | 104 |
| `atlas_reddit/watchlist.sample.toml` | 115 |
| `plans/PR-Reddit-Listening-Config-Scoring.md` | 212 |
| `tests/test_atlas_reddit_config.py` | 365 |
| `tests/test_atlas_reddit_scoring.py` | 263 |
| **Total** | **1375** |
