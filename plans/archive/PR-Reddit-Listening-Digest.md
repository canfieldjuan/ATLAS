# PR-Reddit-Listening-Digest

## Why this slice exists

Third slice (S3) of the approved #1934 arc (Fable 5 builder trial; product
epic #1931), continued under the arc-continuation protocol after S2 merged as
d42eccc74. S3 completes the thinnest end-to-end product path: with config
(S1), scoring (S1), and state (S2) in place, the digest is the surface Juan
actually reads -- a daily Markdown file with the ranked radar and a warm
replies section -- plus the `python -m atlas_reddit` manual run command the
arc promised. After this slice the full local pipeline is demonstrable from
fixture data with zero network, which is exactly why the approved arc moved
the digest ahead of the live poller (amendment a). No network, no LLM.

This branch carries one housekeeping commit: the merged S2 plan's archive
move, folded in per #1934 comment 4860540364 step 3 (direct main pushes
forbidden).

Diff-budget note: over the 400 LOC soft cap for the arc's standing reasons --
external Reddit text crossing into rendered Markdown is a small injection
surface, and the trial's adversarial-test rules require hostile-title,
structure-forging, and CLI-boundary probes alongside the happy paths. Code
under test is ~250 LOC; the rest is mandated coverage and this plan.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Vertical slice

1. `atlas_reddit/digest.py`: pure render_digest(candidates, replies,
   generated_on) -- no I/O, no clock, deterministic -- and
   write_digest(store, digest_dir, generated_on, limit, min_final_score)
   writing a dated YYYY-MM-DD markdown file under digest_dir; same-day
   regeneration overwrites.
2. Radar section: ranked status=new candidates with title link, subreddit,
   score, matched topics (the "why it surfaced"), posted date, comment and
   reddit-score counts; explicit empty state.
3. Warm replies section: unseen replies-to-me from S2 state with excerpted
   bodies; explicit stub wording for the empty case until S5 populates it.
4. Markdown-injection hardening: external titles/bodies are
   whitespace-collapsed and link-metachar-escaped so a hostile title cannot
   break out of its link or forge digest structure.
5. `atlas_reddit/__main__.py`: `python -m atlas_reddit digest` with --db,
   --digest-dir, --date (validated YYYY-MM-DD; defaults to today UTC -- the
   only place the wall clock enters), --limit (must be >= 1), --min-score;
   StoreError surfaces as exit code 2 with the message on stderr.
6. `digest_dir` typed setting (`ATLAS_REDDIT_DIGEST_DIR`).
7. Housekeeping (separate first commit): archive the merged S2 plan doc and
   regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] render_digest is pure and deterministic; both sections render
        populated and empty states.
  - [ ] Radar shows only status=new candidates, ranked by score, honoring
        limit and inclusive min-score; triaged (responded/dismissed)
        candidates are excluded.
  - [ ] Warm replies shows only unseen replies-to-me; seen replies are
        excluded; long bodies are excerpted.
  - [ ] Hostile external text cannot forge Markdown: a `](http://evil)`
        title stays escaped inside its link; embedded newlines cannot
        inject headings.
  - [ ] CLI end-to-end on a real store writes the dated file; malformed
        dates, non-positive limits, and unknown/missing commands exit 2; a
        newer-schema store fails closed through the CLI with the store's
        message on stderr.
  - [ ] Same-day regeneration overwrites (one file per day, idempotent).
  - [ ] New tests run in PR CI via the existing path-filtered glob workflow.
- Affected surfaces: new digest + CLI modules, one settings field, tests;
  one plans/ housekeeping move. No existing Atlas surface touched.
- Risk areas: Markdown injection from external Reddit text (covered by the
  escaping probes); no network, no secrets, no DB schema change.
- Reviewer rules triggered: R1, R2 (renderer/CLI failure branches +
  hostile-input fixtures), R9 (user-facing output: populated/empty/error
  states covered), R10, R11 (digest_dir typed field; zero new
  dependencies), R12 (CI enrollment via the existing glob).
- Test-adapter posture (#1934 real-adapters rule): zero mocks. Digest tests
  seed real SQLite stores through the real store API; CLI tests invoke the
  real main() in-process with argv. Dates are explicit data; the wall clock
  exists only in the CLI default.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/config.py`
- `atlas_reddit/digest.py`
- `atlas_reddit/store.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Listening-Digest.md`
- `plans/archive/PR-Reddit-Listening-Sqlite-Store.md`
- `tests/test_atlas_reddit_digest.py`
- `tests/test_atlas_reddit_store.py`

## Mechanism

render_digest builds the Markdown from already-fetched rows: a numbered
radar list ordered as the store returns it (final_score DESC), each entry a
sanitized-title link with subreddit, %g-formatted score, matched topics, and
the post's UTC date derived from created_utc (data, not clock). Sanitization
collapses all whitespace runs (so embedded newlines cannot inject headings)
and escapes link metacharacters via str.translate before external text is
placed inside link syntax. Reply bodies pass the same sanitizer plus a
140-char excerpt.

write_digest queries the real store -- list_candidates(status="new",
min_final_score, limit) and list_replies(only_unseen=True,
only_to_me=True) -- renders, ensures the digest dir exists, and writes the file named after
generated_on with a .md suffix, overwriting on same-day regeneration.

main(argv) in `atlas_reddit/__main__.py` parses subcommands (only digest in
S3), validates the date shape with a real strptime check, resolves defaults
from RedditListeningSettings, and is the single place the wall clock may run
(when --date is omitted). StoreError maps to exit 2 with the message on
stderr; success prints the written path and returns 0.

## Intentional

- **Radar = status "new" only**: the digest is the review queue, not an
  archive. Seen/dismissed/responded rows stay inspectable via the sqlite3
  CLI; a history view is not a v1 need.
- **Warm replies renders real S2 state now** with explicit stub wording for
  the empty case: the issue asked for a "warm replies section stub if reply
  tracker state exists" -- rendering real rows when present costs nothing
  and lets S5 light the section up without touching the renderer.
- **Whitespace-collapse + link-metachar escaping instead of a Markdown
  library**: the digest is read by a human in a text editor; a two-regex
  sanitizer is auditable and dependency-free, and the hostile cases (link
  breakout, heading injection) are exactly what the tests probe. Full
  CommonMark escaping is an abstraction larger than the problem (R10).
- **The wall clock lives only in the CLI default**: everything below main()
  takes dates/timestamps as data, keeping renderer and writer deterministic
  (same posture as S2; no clock mocking anywhere).
- **--date accepted for any valid day**: regeneration and backfill are
  operator conveniences; the store is the source of truth and the digest a
  projection of it.
- **argparse's native exit code 2 for usage errors and exit 2 for
  StoreError**: one "operator input/state problem" signal plus 0 for
  success; a custom exit-code taxonomy is overkill for a single-user CLI.
- **Housekeeping commit in this branch**: authorized fold-in (#1934 comment
  4860540364); kept as its own commit to keep the S3 code diff clean.

Review-fix notes (Codex wave 1 on 96b6499d4; all three verified real and
fixed at root in this PR):

- **Only the title was sanitized; URL, subreddit, topics, and thread ids
  interpolated raw** (P2): a hostile URL or topic could still break links
  or forge headings despite the slice's injection contract. Root cause:
  sanitization applied ad hoc per field instead of at every interpolation
  of external data. Fixed by sanitizing all fields at the render boundary,
  with URL-appropriate handling (control chars stripped, link-breaking
  punctuation percent-encoded, linkified only for http/https -- other
  schemes render as plain text).
- **OSError escaped the CLI error contract** (P2): --digest-dir at an
  existing file or unwritable path produced a traceback instead of the
  documented exit-2-with-stderr behavior. Root cause: the operator-error
  contract was implemented for only one error family. Fixed by adding
  OSError to the same handler.
- **--min-score nan/inf accepted** (P2): sqlite3 binds NaN as NULL, which
  silently empties the radar. Fixed at the shared root -- the store itself
  rejects non-finite filters for every caller -- plus a friendly argparse
  usage error at the CLI, the same two-layer posture as the id/enum
  guards. Both layers probed.

Review-fix notes (Codex wave 2 on b839bea34; both verified real and fixed
at root in this PR):

- **Corrupt/non-SQLite --db raised raw sqlite3.DatabaseError past the CLI
  contract** (P2). Root cause one level above Codex's either/or: the
  store's own contract says invalid store operations raise StoreError, so
  a corrupt file is a store-open failure -- the store now wraps
  sqlite3.Error at open, the CLI needs no new catch, and every future
  caller inherits the behavior. Probed at both layers.
- **A literal backslash before a Markdown delimiter defeated the escaper**
  (P2): input like a backslash-bracket pair rendered as escaped-backslash
  plus LIVE bracket. Root cause: escaping metacharacters without escaping
  the escape character itself. Fixed by adding the backslash to the
  translation table -- str.translate's single-pass independent mapping
  makes double-escaping impossible by construction.

Review-fix notes (Codex wave 3 on 4a87afcf1; all three verified real and
fixed at root in this PR):

- **connect-time sqlite failures escaped the StoreError contract** (P2,
  e.g. --db at a directory): the wave-2 wrap covered only the schema
  probe, not connect() itself -- a symptom-site fix. The whole open path
  now wraps; probed at store and CLI layers.
- **URL destinations missed the backslash** (P2): the wave-2 escape-char
  fix covered the inline surface only -- exactly the per-surface sanitizer
  drift risk. Backslash now percent-encodes (%5C) in URL destinations.
- **String-typed numeric metrics passed SQLite's type affinity into the
  renderer** (P2): forged headings via num_comments/reddit_score, and a
  string final_score crashed the :g formatter. True root: the store's
  write boundary was type-loose on numerics, same class as its id/bool
  guards. Fixed with int/finite-number guards across ALL numeric write
  params (metrics, scores, and every timestamp on candidates, replies,
  threads, and purge log), not just the two cited fields. The prior
  malformed-reply dual-layer probe was updated accordingly (API guard
  StoreError + schema NOT NULL IntegrityError both probed).

## Deferred

- S4 PRAW read-only poller (doc-verify auth path FIRST; read/identity/
  history fail-closed; populates candidates for real).
- S5 reply tracker (populates tracked_threads/replies; the warm replies
  section then carries live data).
- S6 deletion-compliance purge job.
- Digest delivery (ntfy/email) and any scheduling: beyond the arc's
  manual-run boundary.
- Radar pagination or day-over-day diffing: not needed at single-user
  volumes.

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_digest.py` plus the existing
  `tests/test_atlas_reddit_store.py`, `tests/test_atlas_reddit_config.py`,
  and `tests/test_atlas_reddit_scoring.py`: 209 passed (renderer
  populated/empty/deterministic, ranked-with-why, hostile-title link
  breakout escaped, newline heading-injection collapsed, long-body excerpt,
  dated-file write, same-day overwrite, status-filter exclusion, limit +
  inclusive min-score, seen-reply exclusion, CLI end-to-end on a real
  store, malformed dates, non-positive limits, unknown/missing commands,
  newer-schema fail-closed through the CLI with stderr message; wave-1
  probes: hostile URL percent-encoding, non-http scheme not linkified,
  hostile subreddit/topic/thread-id sanitization, output-path OSError exit
  2, non-finite min-score rejected at CLI and store layers; wave-2 probes:
  backslash-pairing escape defeat, corrupt-db fail-closed at store and
  CLI; wave-3 probes: db-path-at-directory both layers, trailing-backslash
  URL, parametrized non-numeric metrics rejected at the write boundary,
  non-int timestamps across all write paths). This line
  is the single verification-count source; the PR body mirrors it.
- ASCII byte-scan on the four changed Python files: clean.
- Live demo: `python -m atlas_reddit` digest against a scratch store wrote
  the dated digest file (fresh store, empty-state content) -- the
  end-to-end vertical-slice artifact.
- python `scripts/sync_pr_plan.py` on this plan: tables regenerated from
  the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 123 |
| `atlas_reddit/config.py` | 7 |
| `atlas_reddit/digest.py` | 151 |
| `atlas_reddit/store.py` | 55 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Digest.md` | 255 |
| `plans/archive/PR-Reddit-Listening-Sqlite-Store.md` | 0 |
| `tests/test_atlas_reddit_digest.py` | 498 |
| `tests/test_atlas_reddit_store.py` | 65 |
| **Total** | **1157** |
