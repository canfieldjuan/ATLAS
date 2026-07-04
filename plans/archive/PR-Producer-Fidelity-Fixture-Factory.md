# PR-Producer-Fidelity-Fixture-Factory

## Why this slice exists

Fable-arc codification slice 2 (from the #1943 review of the #1934 arc; slice
1 was the #1944 diff-budget gate). Root cause of the S6 mass-deletion P1: the
producer's id contract (PrawListingSource / PrawHistorySource emit Reddit
fullnames, t3_/t1_) and the consumers' expectations (store rows, purge kind
regexes) are two independent encodings connected only by prose, so consumer
tests could hand-seed bare ids the pipeline never emits and stay green while
production drifted. The arc's fix added one probe test
(test_real_poller_mapping_stores_fullnames) but left fixture construction
hand-rolled at every consumer site -- the drift class is still open. This
slice fixes that root: a shared factory derives consumer-test fixtures from
the REAL producer mappings and seeds stores through the REAL consumers
(poll_once / track_once), so a consumer test physically cannot seed a shape
the pipeline never emits.

Diff-budget overage (562 added lines vs the 400 cap): the factory, its
enforcing fidelity tests, and the S6-site (purge tests) adoption are one
indivisible slice. The factory without adoption reproduces the S6 failure
mode -- fixtures exist, nothing uses them -- and the fidelity tests are the
slice's acceptance evidence. Splitting would ship an unused fixture library
and defer the enforcement that is the point.

## Scope (this PR)

Ownership lane: Workflow/process
Slice phase: Robust testing

1. `tests/atlas_reddit_fixtures.py` -- praw-shaped doubles encoding the praw
   fullname contract exactly once; contextmanagers running the real Praw*
   sources over a stubbed praw module; seeding helpers running the real
   poll_once / track_once.
2. `tests/test_atlas_reddit_fixture_fidelity.py` -- enforcing tests:
   producer output locksteps with the purge kind regexes; pipeline-seeded
   rows purge cleanly end to end; the factory rejects non-producer ids.
3. Purge-test seed helpers adopt the factory; corrupt-row simulations move
   to explicitly named `_seed_raw_*` helpers.
4. One workflow paths line so factory edits re-trigger the reddit checks.

### Files touched

- `tests/atlas_reddit_fixtures.py`
- `tests/test_atlas_reddit_fixture_fidelity.py`
- `tests/test_atlas_reddit_purge.py`
- `.github/workflows/atlas_reddit_checks.yml`
- `plans/PR-Producer-Fidelity-Fixture-Factory.md`

### Review Contract

Acceptance criteria (reviewer checks one-by-one):

1. The factory never hand-writes a stored id: candidate ids come from the
   real `PrawListingSource.fetch_new` mapping, reply ids from the real
   `PrawHistorySource.fetch_thread_replies` mapping, and the praw
   `{kind}_{id}` fullname contract is encoded in exactly one place (the
   praw doubles).
2. Store seeding runs the real consumers (`poll_once`, `track_once`) and
   fails loudly (assert) when a fixture is silently filtered.
3. The fidelity test imports BOTH real sides (producer path and purge
   `_KIND_RE`) -- lockstep, not a copied regex.
4. Purge-test `_seed_candidate` / `_seed_reply` reject non-producer shapes;
   only `_seed_raw_*` (used by the four corrupt-row sites) can seed
   malformed ids, and they say so.
5. `python -m pytest tests/test_atlas_reddit_*.py -q` passes.

Affected surfaces: `tests/` plus one CI workflow paths line. No production
code changes; `atlas_reddit/` is untouched.

Risk areas: praw doubles drifting from praw's API (the fullname contract is
pinned by a fidelity test; wider transport fidelity is documented as out of
the doubles' scope); poll_once-based seeding coupling to scoring admission
(mitigated by a fixed probe phrase plus the admitted-count assert).

Reviewer rule IDs triggered: R2 (failure-branch fixtures -- corrupt-row raw
path stays covered); R14 (checked-out PR-head verification). No other
path-glob row in `docs/REVIEWER_RULES.md` matches a tests+workflow diff.

## Mechanism

The factory stubs the praw MODULE (generalizing the `_stub_praw_with`
pattern the purge tests established) and instantiates the real
`PrawListingSource` / `PrawHistorySource`, so the production mapping lines
-- fullname passthrough, permalink-to-url, int casts, own-author filtering,
`removeprefix("t1_")` refresh lookups -- execute under test. Seeding drives
the real `poll_once` (watchlist with a fixed probe phrase, admitted count
asserted) and `track_once` (own-comment discovery, then reply admission),
so every stored row was produced by the pipeline. The fidelity test then
asserts producer output satisfies purge's `_KIND_RE` per table -- the one
place the two contracts are mechanically joined.

## Intentional

- The four corrupt-row purge tests keep seeding malformed ids on purpose,
  through `_seed_raw_*` helpers named for it: "never delete on a data-shape
  mismatch" is a real branch that needs real bad rows.
- The praw doubles mimic only the attributes the real sources read; they
  are not a praw emulator. Their one load-bearing contract (fullname ==
  kind_id) is pinned by a fidelity test.
- Store/tracker/poller/digest test files are not converted here; the purge
  tests are the S6 site and the adoption template.

## Deferred

- Factory adoption in `tests/test_atlas_reddit_store.py` / `_tracker.py` /
  `_poller.py` / `_digest.py` (mechanical follow-up on this template),
  including own-submission history support in `real_history_source`.
- Repo-wide trusted-base-ref execution for gate scripts (#1944 waiver 18).

Parked hardening: none.

## Verification

Commands run from the repo root:

- `python -m pytest tests/test_atlas_reddit_*.py -q` -- **322 passed**
  (full reddit suite: new fidelity tests + factory-backed purge tests +
  all pre-existing files).
- `bash scripts/local_pr_review.sh --current-pr-body-file <pr-body.md>` --
  all checks PASS, 0 failed.
- `python scripts/check_diff_budget.py --additions 562 --body-file
  <pr-body.md>` -- overage carried by the line-anchored override in the PR
  body (see Why this slice exists). Result: PASS with override.

## Estimated diff size

| File | LOC (added) |
|---|---:|
| `tests/atlas_reddit_fixtures.py` | 279 |
| `tests/test_atlas_reddit_fixture_fidelity.py` | 110 |
| `tests/test_atlas_reddit_purge.py` | 37 |
| `.github/workflows/atlas_reddit_checks.yml` | 1 |
| `plans/PR-Producer-Fidelity-Fixture-Factory.md` | 135 |
| **Total** | **562** |
