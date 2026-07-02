# PR-Reddit-Listening-Praw-Poller

## Why this slice exists

Fourth slice (S4) of the approved #1934 arc (Fable 5 builder trial; product
epic #1931), continued under the arc-continuation protocol after S3 merged as
2f8eaa0c5. S4 is the first networked component: the read-only PRAW poller
that fills the S1-S3 pipeline with live candidates. It lands behind the
operator-required doc-verification gate -- the current Reddit OAuth + PRAW
auth path was verified against live documentation on 2026-07-02 BEFORE any
auth code was written (findings recorded in Mechanism and in
`docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`).

Compliance is the product here: scopes are exactly identity/history/read and
the tool fails closed on anything more (including the all-scopes wildcard),
the wrapper's public surface is read-only by construction, a static probe
greps the package for Reddit write-API usage, the User-Agent follows the
verified required format, and polling paces well inside the verified 60
requests/minute OAuth budget.

Diff-budget note: over the 400 LOC soft cap for the arc's standing reasons
plus S4-specific weight: the operator runbook (app registration + scoped
token mint) ships in this slice per the arc, and the trial rules demand
adversarial coverage of the auth boundary (scope guard both sides, wildcard
refusal, credential absence, forbidden-write static probe). Code under test
is ~330 LOC; the rest is runbook, mandated coverage, and this plan.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Vertical slice

1. `atlas_reddit/reddit_client.py`: transport-neutral ListingPost +
   ListingSource protocol; PrawListingSource (lazy praw import,
   check_for_updates=False so PRAW cannot phone home) that validates
   granted scopes fail-closed at construction; pure validate_scopes
   guard; build_user_agent in the verified required format.
2. `atlas_reddit/poller.py`: poll_once -- listings -> hard filters
   (text/self posts only, freshness window) -> real scorer -> replay-safe
   store upserts; injectable clock and sleep; one failing subreddit is
   surfaced in stats without aborting the pass.
3. CLI `python -m atlas_reddit poll` with validated knobs; RedditAuthError
   / WatchlistError / StoreError / OSError all map to the exit-2-stderr
   operator contract.
4. Settings: creds as SecretStr fields (ATLAS_REDDIT_CLIENT_ID,
   ATLAS_REDDIT_CLIENT_SECRET, ATLAS_REDDIT_REFRESH_TOKEN,
   ATLAS_REDDIT_USERNAME) plus bounded poller knobs (freshness window,
   per-subreddit limit, pace seconds, score floor).
5. praw>=7.8.1 pinned in `requirements.txt` (lazy-imported; the test
   suite never needs it installed).
6. `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`: script-app registration, the
   documented scoped-refresh-token mint (authorization-code flow,
   duration permanent, three scopes preset), env contract with
   placeholders only, verification steps, revocation.
7. Housekeeping (separate first commit): archive the merged S3 plan doc
   and regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] validate_scopes fails closed both ways: exact set and subsets
        pass; any excess scope, the wildcard, and the empty set are
        refused with the runbook named in the error.
  - [ ] Missing credentials raise before any praw import (proven: the
        test environment does not have praw installed).
  - [ ] The wrapper's public surface is exactly fetch_new +
        granted_scopes, and a static probe finds zero Reddit write-API
        attribute calls anywhere in the package.
  - [ ] The User-Agent matches the verified required format; invalid
        usernames are refused.
  - [ ] poll_once admits topical fresh text posts into the real store via
        the real scorer; link posts, stale posts, and below-floor posts
        are skipped with per-reason counters; freshness and score floors
        are inclusive at the boundary.
  - [ ] Pacing sleeps exactly n-1 times between subreddits (never before
        the first); zero pace never sleeps; one failing subreddit is
        recorded and does not abort the pass.
  - [ ] Re-polling preserves triage state (a dismissed candidate stays
        dismissed while volatile fields refresh) -- proven through the
        real store.
  - [ ] CLI poll without credentials or watchlist exits 2 with a clean
        stderr message; nonsense knobs are usage errors.
  - [ ] New tests run in PR CI via the existing path-filtered glob
        workflow.
- Affected surfaces: new client/poller modules, CLI subcommand, settings
  fields, one dependency pin, one new runbook doc; one plans/ housekeeping
  move. No existing Atlas surface touched.
- Risk areas: the auth boundary (fail-closed scope guard, both sides
  probed); credential handling (SecretStr, env-only, placeholders in
  docs); no data migration; live network behavior is deferred to the
  operator smoke in the runbook because tests fake the transport boundary.
- Reviewer rules triggered: R1, R2 (guard-shaped auth boundary: failure
  branches both sides + static forbidden-write probe), R3
  (credentials/scopes: fail-closed, SecretStr, no logging of secrets),
  R8 (replay-safe upserts re-proven through the poller path), R10, R11
  (praw dependency justified here; typed settings), R12 (CI enrollment
  via the existing glob).
- Test-adapter posture (#1934 real-adapters rule): the Reddit API is the
  one true external boundary and is faked at the ListingSource protocol;
  everything else is real (parser-built watchlists, real SQLite stores,
  real scorer, real CLI main() in-process). Clock and sleep enter as
  arguments. praw is never imported by the suite.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/config.py`
- `atlas_reddit/poller.py`
- `atlas_reddit/reddit_client.py`
- `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`
- `plans/INDEX.md`
- `plans/PR-Reddit-Listening-Praw-Poller.md`
- `plans/archive/PR-Reddit-Listening-Digest.md`
- `requirements.txt`
- `tests/test_atlas_reddit_poller.py`

## Mechanism

Doc-verified auth path (2026-07-02; PRAW authentication docs, PRAW
refresh-token tutorial, reddit-archive OAuth2 + API wikis): script-type
apps support the authorization-code grant; scopes are requested at the
authorize endpoint and duration permanent yields a refresh token that
persists until revoked (access tokens expire hourly and PRAW refreshes
them). Password-grant scope restriction is documented nowhere, so a scoped
refresh token is the only documented way to hold exactly
identity/history/read -- which is why the runbook mints one via
reddit.auth.url(duration permanent, three scopes) then
reddit.auth.authorize(code) against a localhost redirect. At runtime
PrawListingSource constructs praw.Reddit with the refresh token and
check_for_updates=False, then immediately passes reddit.auth.scopes()
through validate_scopes, which refuses any grant outside the allowed set --
the wildcard a password-grant token carries is just a superset and fails
the same subset check. The verified User-Agent format is built by
build_user_agent from the configured username.

poll_once iterates the watchlist with an injectable sleep between
subreddits (default 2s, wide margin inside the verified 60 requests/min),
fetches newest submissions through the ListingSource protocol, keeps
self/text posts inside the freshness window, scores them with the real S1
scorer using the subreddit's watchlist weight, and admits posts at or
above the floor via the S2 store's replay-safe upsert (status and
first_seen preserved; stale windows cannot regress state). Per-reason skip
counters and per-subreddit errors come back in PollStats; the CLI prints
them and exits 1 if any subreddit failed, 2 on operator errors.

## Intentional

- **PRAW over a hand-rolled httpx client**: the arc named PRAW; the doc
  verification confirmed it carries exactly the plumbing we must not
  reinvent (hourly token refresh, X-Ratelimit header compliance). The
  dependency is pinned and lazy-imported so the test suite and CI never
  load it.
- **Scope guard is a pure function** consumed by the wrapper: the refusal
  logic is fully testable without praw or credentials; the live praw
  construction path is exercised by the operator smoke in the runbook
  (named deferral -- there are no real credentials in CI by design).
- **Storage-side filters vs surface-side filters**: the poller decides
  what is worth storing (fresh, text, above floor); seen/dismissed/
  responded filtering stays at the digest read side, where the store's
  status-preserving upsert guarantees re-polls cannot resurrect triaged
  threads. The unseen/not-dismissed hard filters from the issue are
  therefore satisfied at read time by construction, not re-implemented in
  the poller.
- **keyword_score stored as the weight-normalized score** (total divided
  by subreddit weight): keeps both the raw topical signal and the final
  weighted score inspectable, without a schema change.
- **One failing subreddit does not abort the pass**: the error is
  captured in stats and printed as a warning (exit 1), never swallowed --
  a partial radar beats none, and the failure stays visible.
- **/rising deferred**: the issue marks it optional; /new covers the
  radar's freshness contract at v1 volumes.
- **Housekeeping commit in this branch**: authorized fold-in (#1934
  comment 4860540364); kept separate to keep the S4 code diff clean.

Review-fix notes (Codex wave 1 on 2c5c56cf0; all four verified real and
fixed at root in this PR):

- **PRAW/prawcore failures escaped the CLI contract** (P2): invalid or
  expired credentials raise from praw.Reddit()/auth.scopes() as prawcore
  exceptions (invalid_grant) that only RedditAuthError was catching.
  Root cause: the auth boundary's contract is RedditAuthError, so the
  whole praw-touching block (including the lazy import itself) now maps
  into it. Probed via a stubbed praw module -- the external transport
  boundary -- exercising the REAL constructor path (invalid_grant,
  wildcard-token refusal, scoped-token acceptance, praw-absent).
- **CLI overrides bypassed the settings ceilings** (P2): the knob had two
  entry paths with different validation; a --limit-per-subreddit typo
  above 100 would multiply PRAW's paginated requests past the verified
  budget. Root cause: bounds defined only on the typed fields. Fixed with
  shared MAX_* constants consumed by BOTH the pydantic le= bounds and the
  CLI checks -- applied to all three capped knobs, not just the cited one.
- **Eager settings construction threw pydantic tracebacks** (P2): an env
  typo like ATLAS_REDDIT_FRESHNESS_HOURS=abc broke every command incl.
  digest. Root cause: settings construction is operator-input surface
  sitting outside the operator-error contract; it now maps to exit 2
  with a clean message.
- **The scope guard validated only the ceiling** (P2, the best catch): a
  token missing read passed the subset check and would fail on every
  fetch downstream. Root cause: one-sided boundary validation. Fixed
  with a required-scopes floor (default read; future sources pass their
  own), probed on both sides.

## Deferred

- Live-credential smoke: operator runs the runbook (app registration +
  token mint + first poll) once creds exist; the scope guard proves
  itself on the first live run. Not automatable in CI by design (no
  secrets in CI).
- S5 reply tracker (uses identity/history scopes for own-comment
  discovery; the tracked_threads/replies store paths are ready).
- S6 deletion-compliance purge job (purge_log is ready; the 48h window
  logic lands there; until S6, usage stays manual-run proof posture).
- /rising listings, scheduled polling, digest delivery: beyond the arc's
  manual-run boundary.

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_poller.py` plus the existing
  `tests/test_atlas_reddit_digest.py`, `tests/test_atlas_reddit_store.py`,
  `tests/test_atlas_reddit_config.py`, and
  `tests/test_atlas_reddit_scoring.py`: 255 passed (scope guard both
  sides incl. wildcard and empty; missing-creds-before-praw-import;
  read-only public surface; package-wide forbidden-write static probe;
  UA format + invalid usernames; poller admit/skip counters; inclusive
  freshness and floor boundaries; n-1 pacing and zero-pace; failing
  subreddit isolation; triage-state preservation across re-polls through
  the real store; per-subreddit fetch limit; CLI no-creds / missing
  watchlist / nonsense knobs; wave-1 probes: missing-read floor both
  sides, required-override, stubbed-praw constructor paths, CLI ceiling
  rejections, env-typo operator errors for every command). This line is
  the single
  verification-count source; the PR body mirrors it.
- ASCII byte-scan on the five changed Python files: clean.
- python `scripts/sync_pr_plan.py` on this plan: tables regenerated from
  the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 102 |
| `atlas_reddit/config.py` | 56 |
| `atlas_reddit/poller.py` | 97 |
| `atlas_reddit/reddit_client.py` | 176 |
| `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md` | 114 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Praw-Poller.md` | 254 |
| `plans/archive/PR-Reddit-Listening-Digest.md` | 0 |
| `requirements.txt` | 1 |
| `tests/test_atlas_reddit_poller.py` | 510 |
| **Total** | **1313** |
