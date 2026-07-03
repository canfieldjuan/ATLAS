# PR-Reddit-Fit-Store

## Why this slice exists

Fourth slice of the approved Reddit Listening v2 arc (#1931 comment
4872154794), after S1 (harness), S2 (contract + prompt), S3 (guard). This
is where fit output becomes durable: schema v5 adds a bounded post-body
column and a side table for advisory verdicts, and a manual import CLI runs
predictions through the REAL S2 parser + S3 guard before persisting. No
model calls, no network -- the model client is S5. Over the diff cap: a
schema migration must land with the wind-back migration-test updates it
forces (three pre-existing tests) plus both-sides persistence probes; the
runtime code core is ~200 lines.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Vertical slice

1. `atlas_reddit/store.py`: SCHEMA_VERSION 4 -> 5; a v4->v5 migration rung
   (ALTER candidates ADD `body_excerpt` + CREATE `candidate_fit_reviews`);
   the fresh-v5 DDL; `FitReview` dataclass; `upsert_fit_review` /
   `get_fit_review` / `list_fit_reviews`; the `fit_input_hash` helper; a
   `_require_str_tuple` fail-closed helper.
2. `atlas_reddit/poller.py`: persist a bounded, whitespace-collapsed
   `body_excerpt` (<= `MAX_BODY_EXCERPT_CHARS`) through the real poll path.
3. `atlas_reddit/__main__.py`: `import-fit` command -- JSONL of
   `{post_id, prediction}` through `parse_fit_decision` + `guard_fit_decision`
   -> `upsert_fit_review` (source=manual); partial-error exit contract.
4. `tests/test_atlas_reddit_fit_store.py`: migration, body_excerpt through
   the fixture factory, upsert idempotence, fail-closed enums/flags,
   guard-rejected redaction, purge CASCADE, FK rejection, list filters,
   input_hash, and the import-fit CLI (both exit paths).
5. Wind-back migration-test updates in `tests/test_atlas_reddit_purge.py`
   and `tests/test_atlas_reddit_tracker.py`: a store rewound to an old
   version must drop the v5 additions to be faithful, and the ladder now
   walks to v5.
6. Housekeeping (separate first commit): archive the merged S3 plan.

### Review Contract

- Acceptance criteria:
  - [ ] A real v4 store opens at v5 with data intact and `body_excerpt`
        defaulting to empty; the fresh-v5 DDL matches the migration result.
  - [ ] The poller persists a bounded, whitespace-collapsed body_excerpt
        through the REAL poll path (fixture factory, not a hand-built row).
  - [ ] `upsert_fit_review` is idempotent by post_id; unknown verdict,
        source, non-string flags/codes, and non-bool guard_ok all fail
        closed with StoreError.
  - [ ] **Guard-rejected rows persist flagged + REDACTED**: guard_ok=0 keeps
        verdict/codes/provenance but reason='' and angle='' (probed at the
        raw-row level), enforced by both the method and the table CHECK.
  - [ ] **Purge CASCADE**: purging a reviewed candidate removes its review
        in the same transaction (FK ON DELETE CASCADE); a review for an
        unknown candidate fails closed as StoreError, not raw IntegrityError.
  - [ ] `list_fit_reviews` filters by post_ids subset and only_guard_ok;
        `fit_input_hash` is stable and input-sensitive.
  - [ ] `import-fit` persists valid predictions (guard-blocked ones flagged
        + redacted), reports partial errors and exits 1, exits 2 on a
        missing file; the whole path uses the real parser + guard.
- Reachability proof (#1952): the real CLI `python -m atlas_reddit
  import-fit <preds.jsonl> --db <db>` persists rows and returns
  0/1/2; exercised in-process by the CLI tests and reproduced in
  Verification.
- Affected surfaces: store schema + fit methods, poller body wiring, one
  CLI command, one new test file, three migration-test updates, plans
  housekeeping. No config, digest, network, or credential surface.
- Risk areas: **data at rest** (body_excerpt widens third-party text
  stored locally -- operator-approved; bounded + purge/tombstone-covered +
  guard redaction keep it inside the deletion-compliance story); migration
  fidelity (wind-back fixtures made faithful); FK/redaction correctness
  (probed both directions).
- Reviewer rules triggered: R1, R2 (store guard-shaped: both sides +
  redaction both directions + FK both directions), R8 (idempotent upsert +
  purge replay via CASCADE), R11 (zero new dependencies), R12 (tests
  auto-enroll via the glob), R14 (reachability named above).
- Test-adapter posture: real SQLite, real migrations, real parser + guard,
  real CLI main() in-process, candidates seeded through the real poll path.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/poller.py`
- `atlas_reddit/store.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Fit-Store.md`
- `plans/archive/PR-Reddit-Fit-Guard.md`
- `tests/test_atlas_reddit_fit_store.py`
- `tests/test_atlas_reddit_purge.py`
- `tests/test_atlas_reddit_tracker.py`

## Mechanism

The v4->v5 rung is one sequential ladder step: `ALTER TABLE candidates ADD
COLUMN body_excerpt TEXT NOT NULL DEFAULT ''` and a
`candidate_fit_reviews` table whose `post_id` is the PK and an FK to
candidates `ON DELETE CASCADE`, with a `CHECK (guard_ok = 1 OR (reason = ''
AND angle = ''))` so a rejected row can never carry text. `upsert_fit_review`
validates enums/flags with the store's fail-closed helpers, redacts
reason/angle when guard_ok is False, and wraps the FK IntegrityError (an
unknown candidate) into StoreError so the CLI error family catches it. The
poller collapses whitespace and caps `post.selftext` at
`MAX_BODY_EXCERPT_CHARS`. `import-fit` reads JSONL line by line: malformed
JSON, missing post_id, unknown candidate, or a FitParseError are recorded
as per-line warnings (partial-error exit 1), while parse-ok predictions go
through the guard and persist -- guard-blocked ones flagged and redacted.

## Intentional

- **body_excerpt is bounded and purge-covered** (operator-approved,
  2026-07-03): a fit prompt needs the body, the runner reads from the
  store, and re-fetching live at judge time would add a second Reddit
  boundary and break offline runner tests. Capped + whitespace-collapsed +
  covered by the existing purge/tombstone path exactly like replies.body.
- **Redaction lives in the store, keyed on the guard's decision**: the S3
  guard decides, the store enforces guard_ok=0 -> empty text (method +
  CHECK). Unsafe text the guard caught never lands in SQLite.
- **FK CASCADE, not application-level cleanup**: deletion compliance
  extends to fit output for free -- the review dies with its candidate in
  the purge transaction.
- **input_hash now, --refresh later**: the hash is persisted so S6 can
  detect a stale review honestly; the refresh flag is an S6 concern.
- **Manual import before the model client**: predictions become
  gradeable/persistable with zero model calls, so the store contract is
  proven before S5 wires a live backend.

## Deferred

- S5 OpenAI-compatible judge client (`ATLAS_REDDIT_FIT_*`) -> S6 fit runner
  + digest integration (selection reuse, --refresh via input_hash, digest
  renders guard_ok rows only), per the approved arc.

Parked hardening: none.

## Verification

- `.venv/bin/python -m pytest tests/test_atlas_reddit_fit_store.py -q`: 19
  passed (migration, body_excerpt via the real poll path, idempotence,
  fail-closed enums/flags, redaction at the raw-row level, purge CASCADE,
  FK rejection, list filters, input_hash, import-fit both exits).
- Full package suite `.venv/bin/python -m pytest
  tests/test_atlas_reddit_*.py -q`: 518 passed (incl. the faithful v1/v2/v3
  wind-back migrations and the no-write probe over the new code).
- Reachability: `python -m atlas_reddit import-fit <preds.jsonl> --db <db>`
  persists reviews and returns 0 on success, 1 on partial errors, 2 on a
  missing file.
- ASCII byte-scan on changed Python files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 85 |
| `atlas_reddit/poller.py` | 14 |
| `atlas_reddit/store.py` | 253 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Fit-Store.md` | 145 |
| `plans/archive/PR-Reddit-Fit-Guard.md` | 0 |
| `tests/test_atlas_reddit_fit_store.py` | 341 |
| `tests/test_atlas_reddit_purge.py` | 12 |
| `tests/test_atlas_reddit_tracker.py` | 2 |
| **Total** | **855** |
