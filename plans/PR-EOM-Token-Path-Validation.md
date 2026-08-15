# PR-EOM-Token-Path-Validation

## Why this slice exists

Two reachable defects in Google credential resolution, both found by Codex
review on #2355 and deferred into #2359. Both were reproduced against this head
before being fixed:

**1. A set-but-EMPTY `ATLAS_TOOLS_GOOGLE_TOKEN_FILE` resolves to a directory.**
`Path("")` is `.`, so `resolve_token_file_path("")` returned the repo ROOT:

```
resolve_token_file_path('') -> /…/Atlas-worktrees/eom-token-path-validation
is a DIRECTORY (the repo root): True
store would read/write        : /…/Atlas-worktrees/eom-token-path-validation
```

The service then tries to read a directory.

**Correction to an earlier draft of this plan.** It also claimed the setup
script would bind the same value and fail with `IsADirectoryError` after the
OAuth flow. That is NOT wired at this head: `scripts/setup_google_oauth.py`
hardcodes its own `TOKEN_FILE` and never reads `google_token_file` or calls
`resolve_token_file_path` — verified by grep at this head, 0 references. The
claim was carried over from #2359's issue text, which describes the state during
#2355 before that surface was reverted. The setup entrypoint is wired in a later
#2359 slice; this slice is service-side only.

**2. The missing-file remedy names a file the service will not read.** Under an
explicit override, `locate_token_file` correctly refuses legacy discovery — but
the warning still advertised the stable default AND claimed "Legacy locations
searched", neither of which is true on that path. Reproduced with an explicit
override pointing at an absent `account-b` credential: the service reported it
would read `/tmp/…/account-b/google_tokens.json` while telling the operator the
stable default is `DEFAULT_TOKEN_FILE` and listing legacy paths
that were never consulted. An operator who follows that restores a file the
process ignores, prolonging exactly the outage the message exists to shorten.

Tracking issue: #2359 (split from #2355). This is slice 1 of that issue and is
deliberately narrow — see Deferred.

### Problem-derived contract

- Root cause: resolution and provenance both trusted the configured string
  without validating it, and the recovery message was written for one branch
  (the unconfigured default) while being emitted on both.
- Correct fix must touch/change: `resolve_token_file_path` so a blank or
  non-string setting cannot become a filesystem path;
  `token_path_was_explicitly_configured` so provenance agrees with resolution
  about what "configured" means; and `_load`'s warning so the remedy names the
  path this process will actually read.
- Must not change: the stable default itself; legacy discovery for the
  unconfigured default; the refusal to borrow a legacy credential under an
  explicit override; the `.env` fallback; the RESTART guidance on every recovery
  message; and no credential VALUE may be logged.

## Scope (this PR)

Ownership lane: eom-ops/google-token-resolution
Slice phase: Vertical slice
Max files: 3

1. **One invariant, checked on the filesystem: an unusable configured path
   yields NO credential.** `configured_token_path_problem()` returns a reason
   when the path is (a) explicitly set but blank, or (b) resolves to an existing
   DIRECTORY. `get_credentials` then fails closed — no token file, no `.env`
   fallback — and logs the reason. Wrong account is worse than no account.
2. The directory test asks the OS, not the spelling, so `.`, `./`, `<dir>/..`
   and any absolute directory are one defect rather than four.
3. An explicitly blank override stays EXPLICIT provenance and is treated as
   INVALID. Reclassifying it as unconfigured would re-admit legacy discovery
   exactly when a secret mount or deployment substitution has failed.
4. `.strip()` is used only to DETECT blankness; a nonblank value reaches
   `Path()` verbatim, so a filename with leading/trailing whitespace is not
   silently redirected.
5. The missing-file warning branches on provenance: under an override it names
   that exact path and says legacy locations are deliberately not searched.

### Review Contract

- Acceptance criteria:
  1. An unusable configured path yields NO credential — neither a token file
     nor the `.env` fallback — and logs why. Settled by
     `tests/test_google_token_resolution.py::test_relative_directory_aliases_yield_no_credential`,
     `::test_absolute_directory_yields_no_credential`,
     `::test_a_traversal_alias_through_an_existing_dir_is_caught` and
     `::test_explicitly_blank_override_fails_closed`.
  2. An explicitly blank override keeps EXPLICIT provenance and is invalid, so
     it cannot re-admit legacy discovery — settled by
     `::test_a_blank_setting_is_still_explicit_provenance` and
     `::test_explicitly_blank_override_fails_closed`, which stages BOTH a legacy
     credential and an `.env` fallback and asserts neither is used. A blank that
     was never configured is simply the default —
     `::test_blank_is_acceptable_when_NOT_explicitly_configured`.
  2b. A nonblank path is never silently trimmed — settled by
     `::test_trailing_whitespace_in_an_absolute_path_is_preserved` and
     `::test_whitespace_bearing_paths_are_not_silently_trimmed`.
  2c. A valid configured path still works — settled by
     `::test_a_valid_configured_path_reports_no_problem`.
  2d. Validity is re-checked at POINT OF USE, never from a constructor
     snapshot. A long-lived singleton outlives filesystem changes — a
     secret-volume remount or operator repair can replace the target between
     construction and first use — so a cached verdict would let a
     merely-absent-at-startup path that later became a DIRECTORY fall through
     to the `.env` token. Settled by
     `::test_path_validity_is_rechecked_at_point_of_use`, with
     `::test_a_path_repaired_after_construction_is_honoured` holding the other
     direction so revalidation cannot get stuck at "broken".
  2d-bis. The path VALIDATED is the path READ. Legacy discovery can make the
     selected path differ from the configured one, so a legacy candidate must
     be a FILE (not merely exist) and the selected path is revalidated at point
     of use — settled by `::test_a_legacy_directory_candidate_is_never_selected`,
     `::test_the_selected_path_is_revalidated_not_just_the_configured_one`, with
     `::test_a_legacy_file_candidate_is_still_selected` proving the tightened
     check does not break real legacy discovery.
  2f. The guard is enforced at ONE point (`_load`), so every caller is covered
     — including the write path, which previously had no check. Settled by
     `::test_rotation_refuses_to_persist_through_an_unusable_path`,
     `::test_every_load_call_site_is_guarded` and
     `::test_repaired_path_recovers_after_a_refusal`.
  2e. Health agrees with behaviour on an unusable path — settled by
     `::test_health_status_reports_unconfigured_when_the_path_is_invalid` and
     `::test_health_status_is_unchanged_for_a_valid_path`.
  3. Under an explicit override the remedy names THAT path, says legacy
     discovery is deliberately skipped, and does NOT name the default or claim
     a search that did not happen — settled by
     `::test_explicit_override_recovery_names_that_path_not_the_default`.
  4. The unconfigured branch is unchanged — settled by
     `::test_default_path_recovery_still_names_default_and_legacy`.
  5. Every recovery message still requires a restart — settled by
     `::test_every_recovery_message_requires_a_restart` (unchanged from #2355)
     and asserted again in both new branch tests.
  6. Legacy discovery, override refusal and precedence are unchanged — settled
     by the pre-existing `::test_legacy_in_repo_file_is_still_found_and_warns`,
     `::test_explicit_override_never_borrows_a_legacy_credential` and
     `::test_primary_wins_over_legacy_when_both_exist`, all still passing.
- Reachability proof: `GoogleTokenStore` is built by `get_google_token_store()`
  and consumed by `atlas_brain/tools/calendar.py::_refresh_token` and
  `atlas_brain/services/calendar_provider.py::_refresh_token`. The observable
  state is the resolved `token_file_path` and the emitted log records.
- Affected surfaces: Google credential path resolution and the operator-facing
  recovery message for Calendar and Gmail. No HTTP route, no schema, no
  migration, no change to which credential is selected when the path is valid.
- Risk areas: changing resolution could move a working deployment's path
  (mitigated: only blank/non-string inputs change behaviour, and blank
  previously produced a directory, which cannot have been a working
  configuration); a provenance change could silently re-enable legacy discovery
  for a real override (mitigated by
  `::test_a_blank_setting_is_still_explicit_provenance` and the unchanged
  override-refusal test).
- Reviewer rules triggered: R1, R2, R3, R14.
  - **R1** — operator-facing behaviour change in a failure path.
  - **R2 (boundary probe)** — this is a validator change. Boundary probe, both
    sides: VALID paths still admit (`::test_a_valid_configured_path_reports_no_problem`,
    `::test_absolute_path_is_honoured_unchanged`,
    `::test_trailing_whitespace_in_an_absolute_path_is_preserved`) and INVALID
    paths refuse (`::test_relative_directory_aliases_yield_no_credential`,
    `::test_absolute_directory_yields_no_credential`,
    `::test_a_traversal_alias_through_an_existing_dir_is_caught`,
    `::test_explicitly_blank_override_fails_closed`). Each was negative-probed
    by injecting the removal of the guard and confirming failure.
  - **R3 (credential isolation)** — this selects an OAuth credential. The
    isolation property is that an unusable configuration can never fall through
    to a DIFFERENT account: `::test_explicitly_blank_override_fails_closed`
    stages both a legacy credential and an `.env` fallback and asserts NEITHER
    is used.
  - **R14** — `resolve_token_file_path` / `configured_token_path_problem` are
    an admission boundary for a filesystem path.

### Boundary-change enumeration

The seam is the admission of the configured token-path string.

- Replaced-path behaviour: previously EVERY string was passed to `Path()`.
  `""` and whitespace became `.` → the repo root directory; a non-string raised.
  Now blank and non-string resolve to `DEFAULT_TOKEN_FILE`; every other value is
  unchanged (absolute verbatim, `~` expanded, relative anchored to the
  checkout).
- Guard-relevant fields: the configured `google_token_file` string, and whether
  `google_token_file` appears in `settings.tools.model_fields_set`.
- Caller × input shape: `GoogleTokenStore.__init__` (via
  `get_google_token_store()`), `locate_token_file`, and
  `token_path_was_explicitly_configured`. Inputs are operator config, not
  request data.

Closure declaration for the **configured-path** input set:

1. **Closed or open? — CLOSED**, four shapes exhaust it: blank/non-string,
   absolute, `~`-prefixed, relative.
2. **Where does membership come from? — ENUMERATED** in
   `resolve_token_file_path` / `configured_token_path_problem` and asserted per
   shape by `::test_explicitly_blank_override_fails_closed`,
   `::test_blank_is_acceptable_when_NOT_explicitly_configured`,
   `::test_absolute_path_is_honoured_unchanged`,
   `::test_user_home_shorthand_expands` and
   `::test_relative_path_anchors_to_the_repo_root`.
3. **Out-of-set behaviour — none remains**: after the blank/non-string guard,
   `Path.is_absolute` is total, so every input resolves to a file path. The
   safety property is asserted by `::test_absolute_directory_yields_no_credential`.

Closure declaration for the **provenance** decision:

1. **Closed or open? — CLOSED**: the field was supplied, or it was not.
   Provenance is deliberately independent of the VALUE. A set-but-blank setting
   is configured-and-INVALID, never unconfigured — that separation is the whole
   point, because an empty env var is what a failed secret mount looks like and
   reclassifying it would re-admit legacy discovery exactly then.
2. **Where does membership come from? — DERIVED** from
   `settings.tools.model_fields_set` alone. Validity is a separate question,
   answered by `configured_token_path_problem()`, so path RESOLUTION (which
   always yields a usable `Path`) and path ACCEPTABILITY (which can refuse)
   never have to agree on one overloaded notion of "configured".
3. **Out-of-set behaviour — fail to NOT-configured.** Any exception reading the
   settings object returns `False`, so a settings problem degrades to legacy
   discovery rather than blocking authentication.

### Deployed-config probing

`ATLAS_TOOLS_GOOGLE_TOKEN_FILE` is **not set** in the deployed `.env`, so this
deployment runs the unconfigured-default branch and neither defect is reachable
there today. Both were reproduced directly against this head rather than
inferred — the empty-path resolution and the override-branch warning are quoted
verbatim in "Why this slice exists". The deployed service currently loads from
`DEFAULT_TOKEN_FILE` after the #2355 migration, which this slice
does not touch.

### Files touched

- `atlas_brain/services/google_oauth.py`
- `plans/PR-EOM-Token-Path-Validation.md`
- `tests/test_google_token_resolution.py`

## Mechanism

**ONE enforcement point.** Validity is checked inside `_load()` — the single
place the credential path is consumed — and every caller acts on the verdict it
returns. This is the structural answer to why the same defect kept reappearing
through a new door each review round: the check used to be duplicated per public
method, so the path VALIDATED was not always the path USED, and
`persist_refresh_token` reached `_load()` with **no check at all** — a rotation
into an unusable path lost the freshly issued Google token silently, because
`_save()` caught the `OSError` and logged it. Enforcing at the consumption point
covers present and future callers by construction rather than by remembering.
The check runs BEFORE the `_loaded` short circuit, so a path that turns bad after
a successful load is still caught, and a bad verdict is never cached, so an
operator repair takes effect.


`resolve_token_file_path` now normalises before it resolves: a non-`str` becomes
`""`, and a blank string is replaced with `DEFAULT_TOKEN_FILE` before `Path()`
sees it. That is the whole of defect 1 — the previous code's failure was that
`Path("")` is a legal path (`.`) rather than an error, so nothing downstream had
a reason to complain until a read or write hit a directory.

`token_path_was_explicitly_configured` deliberately does NOT inspect the value —
it reports only whether the field was supplied. An earlier revision of this
slice added a non-blank test there, which was wrong: it reclassified a blank
override as unconfigured and so re-admitted legacy discovery exactly when a
secret mount or deployment substitution had failed, risking authentication as a
stale unrelated account. Validity is now a separate concern
(`configured_token_path_problem`), so resolution can always return a usable
`Path` while acceptability can still refuse.

The `_load` warning is now two messages selected by provenance rather than one
message that was only true on one branch. This matters because the two branches
have *opposite* remedies: on the default branch the operator should move the
credential to the stable default, while on an override branch that same action
puts the file somewhere the process will never look.

## Intentional

- **Blank falls back rather than raising.** An exception here would take down
  credential resolution for a misconfiguration that has an obvious safe
  interpretation, and it would surface at import time rather than as an
  actionable log line. Falling back to the documented default and continuing to
  emit the (now branch-correct) recovery warning keeps the failure diagnosable.
- **No change to which credential wins when the path is valid.** Precedence,
  legacy discovery and override refusal are all untouched; the pre-existing
  tests for them still pass unmodified.
- **The deployed install is unaffected** — it does not set the override, and its
  path is neither blank nor a directory.

## Deferred

Parking predicate: the remaining #2359 items are non-blocking because none is
reachable from a valid configuration on this deployment, and each belongs to a
different surface than configured-path admission.

- `_save()` symlink dereferencing so a rotation through a compatibility symlink
  lands on the real file (#2359). Its own slice: it changes write semantics, not
  path admission.
- `scripts/setup_google_oauth.py` read-legacy / write-primary split, its
  operator documentation, and executing (non-grep) tests for it (#2359).
- The supported migration runbook, including the stop-service invariant (#2359).
- Completing the store's caller enumeration across the six known callers
  (#2359).

Parked hardening: ATLAS #2359.

## Verification

All counts re-run at this head.

- `python -m pytest tests/test_google_token_resolution.py -q` — **53 passed**
- Every consumer of the changed store — `test_google_token_resolution.py`,
  `test_calendar_import_rerun.py`, `test_eom_live_calendar_import.py`,
  `test_eom_scoped_gmail_credentials.py`, `test_leads_intake.py` — **200 passed, 1 skipped**
- **Both defects reproduced against this head BEFORE fixing**, quoted verbatim
  in "Why this slice exists": `resolve_token_file_path('')` returned the repo
  root and the stdlib `Path.is_dir` predicate was `True`; the override-branch warning named the stable
  default and claimed "Legacy locations searched" while the service reported it
  would read the explicit path.
- **Negative probes**, each asserting the file actually changed first, each
  reverted (restored state **42 passed**):
  | Injected defect | Result |
  |---|---|
  | directory check removed (the class defect returns) | 5 failed |
  | explicitly-blank no longer fails closed | 1 failed |
  | whitespace stripped from nonblank paths again | 2 failed |
  | blank reclassified as unconfigured (the design I had wrong) | 2 failed |
  | recovery message stops branching on provenance | 1 failed |
  | `get_status` stops honouring the invalid path | 2 failed |
  | validity reverted to a constructor snapshot | 2 failed |
  | legacy candidate check back to the stdlib existence check | 1 failed |
  | selected-path revalidation removed | 1 failed |
  | write path unguarded again | 1 failed |
  | enforcement removed from `_load` (the single point) | **12 failed** |
- `ruff check` on the changed module and test file: findings identical to the
  `origin/main` baseline; none introduced.
- `python -m py_compile` on the changed module — OK.
- `git diff --check` — clean.
- **HERMETICITY proved in a clean checkout.** `data/` has ZERO tracked files, so
  a `data/..` alias resolves to nothing on a fresh clone — a test relying on it
  passed here only because this worktree has an untracked `data/`. The suite was
  re-run in a fresh `origin/main` worktree with no untracked `data/`: **53
  passed** (re-proved after each round, at current `origin/main`). The non-hermetic case was removed; the traversal shape is covered by
  a test that builds its own directory under `tmp_path`.
- No credential value appears in any changed log statement.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/google_oauth.py` | 211 |
| `plans/PR-EOM-Token-Path-Validation.md` | 347 |
| `tests/test_google_token_resolution.py` | 462 |
| **Total** | **1020** |
