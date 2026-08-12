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

The service then tries to read a directory, and `scripts/setup_google_oauth.py`
binds the same value as its write target and fails with `IsADirectoryError`
only AFTER the operator has completed the interactive OAuth flow.

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

1. A blank (empty/whitespace) or non-string configured path falls back to
   `DEFAULT_TOKEN_FILE` instead of resolving to the repo root directory.
2. Provenance agrees: a set-but-blank setting is NOT an explicit override, so it
   does not suppress legacy discovery for a value that names nothing.
3. The missing-file warning branches on provenance. Under an override it names
   that exact path and states that legacy locations are deliberately not
   searched; unconfigured, it keeps naming the default and the searched legacy
   locations.

### Review Contract

- Acceptance criteria:
  1. Blank and non-string configured paths resolve to the stable default and
     never to a directory — settled by
     `tests/test_google_token_resolution.py::test_blank_configured_path_falls_back_to_the_default`,
     `::test_non_string_configured_path_falls_back_to_the_default` and
     `::test_store_with_a_blank_path_never_targets_a_directory`.
  2. Provenance treats a blank setting as unconfigured, and a real value as an
     override — settled by `::test_blank_setting_is_not_an_explicit_override`
     and `::test_a_real_value_is_still_an_explicit_override`.
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
  `::test_a_real_value_is_still_an_explicit_override` and the unchanged
  override-refusal test).
- Reviewer rules triggered: R1, R14. R1 for the operator-facing behaviour change
  in a failure path. R14 declared deliberately: `resolve_token_file_path` is an
  admission/normalisation boundary for a filesystem path.

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
   `resolve_token_file_path` and asserted per shape by
   `::test_blank_configured_path_falls_back_to_the_default`,
   `::test_non_string_configured_path_falls_back_to_the_default`,
   `::test_absolute_path_is_honoured_unchanged`,
   `::test_user_home_shorthand_expands` and
   `::test_relative_path_anchors_to_the_repo_root`.
3. **Out-of-set behaviour — none remains**: after the blank/non-string guard,
   `Path.is_absolute` is total, so every input resolves to a file path. The
   safety property is that no input can resolve to a DIRECTORY, asserted by
   `::test_store_with_a_blank_path_never_targets_a_directory`.

Closure declaration for the **provenance** decision:

1. **Closed or open? — CLOSED**: configured-and-non-blank, or not.
2. **Where does membership come from? — DERIVED** from
   `settings.tools.model_fields_set` plus a non-blank check, in one function, so
   provenance cannot disagree with resolution about what "configured" means.
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

`resolve_token_file_path` now normalises before it resolves: a non-`str` becomes
`""`, and a blank string is replaced with `DEFAULT_TOKEN_FILE` before `Path()`
sees it. That is the whole of defect 1 — the previous code's failure was that
`Path("")` is a legal path (`.`) rather than an error, so nothing downstream had
a reason to complain until a read or write hit a directory.

`token_path_was_explicitly_configured` gained the same non-blank test. Provenance
and resolution have to agree: if resolution treats blank as "use the default",
provenance must not treat it as an override, or a setting that names nothing
would suppress legacy discovery and leave an upgrading install with neither its
file nor the fallback.

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

- `python -m pytest tests/test_google_token_resolution.py -q` — **42 passed**
- Every consumer of the changed store — `test_google_token_resolution.py`,
  `test_calendar_import_rerun.py`, `test_eom_live_calendar_import.py`,
  `test_eom_scoped_gmail_credentials.py`, `test_leads_intake.py` — **189 passed, 1 skipped**
- **Both defects reproduced against this head BEFORE fixing**, quoted verbatim
  in "Why this slice exists": `resolve_token_file_path('')` returned the repo
  root and the stdlib `Path.is_dir` predicate was `True`; the override-branch warning named the stable
  default and claimed "Legacy locations searched" while the service reported it
  would read the explicit path.
- **Negative probes**, each asserting the file actually changed first, each
  reverted (restored state **42 passed**):
  | Injected defect | Result |
  |---|---|
  | blank path resolves to the repo root again | 6 failed |
  | blank setting counted as an explicit override | 2 failed |
  | recovery message stops branching on provenance | 1 failed |
- `ruff check` on the changed module and test file: findings identical to the
  `origin/main` baseline; none introduced.
- `python -m py_compile` on the changed module — OK.
- `git diff --check` — clean.
- No credential value appears in any changed log statement.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/google_oauth.py` | 64 |
| `plans/PR-EOM-Token-Path-Validation.md` | 248 |
| `tests/test_google_token_resolution.py` | 118 |
| **Total** | **430** |
