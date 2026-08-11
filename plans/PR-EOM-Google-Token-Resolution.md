# PR-EOM-Google-Token-Resolution

## Why this slice exists

Google Calendar was down in production for five days — 2026-08-05 09:31 to
2026-08-10 19:05 — and the log actively pointed at the wrong fix.

The API runs as a systemd user unit whose `WorkingDirectory` is whichever git
worktree is currently deployed. `ToolsConfig.google_token_file` defaults to the
**relative** `data/google_tokens.json`, and `GoogleTokenStore.__init__` did
`Path(token_file_path)` — resolved against the process CWD. On 2026-08-05 the
runtime moved from `worktrees/eom-receivables-runtime` to
`worktrees/atlas-runtime-main`. The old worktree had `data/google_tokens.json`
symlinked to the real file; the new one had no `data/` directory at all.

`_load()` treats an absent file as a no-op — no log line — so
`get_credentials()` silently fell through to the `.env` fallback and returned a
**different, older** refresh token. Measured against Google during the
investigation: the token-file credential refreshes fine (`expires_in=3599`),
while the `.env` credential returns `HTTP 400 invalid_grant`. The operator saw
only:

```
[CRITICAL] Calendar refresh token is INVALID (HTTP 400).
Re-run: python scripts/setup_google_oauth.py
```

Re-running that script rewrites the token **file**, which was never the problem;
the stale `.env` value would have kept winning. The message sent the operator at
the wrong file.

Gmail shares the same token file and was equally severed, but it has no `.env`
fallback, so it returned `None` and never logged a CRITICAL — which is why the
outage looked Calendar-specific and went unnoticed for five days. Calendar feeds
`calendar_import`, the dominant CRM contact-entry path.

Immediate production remediation (restoring the symlink) is already done and is
NOT what this slice ships.

**The fix is that the credential stops living in the repo.** Every in-repo
anchor — the CWD, the checkout root, the setup script's `PROJECT_ROOT` — moves
when the deployed worktree moves, because the deployed code lives inside that
worktree. No amount of "resolve the relative path better" survives that; an
earlier draft of this slice tried exactly that and would not have prevented the
outage. `ToolsConfig.google_token_file` now defaults to
`~/.config/atlas/google_tokens.json`, outside every checkout, which is already
where this deployment keeps its other service credentials. A legacy in-repo file
is still read so upgrades do not break, and using it warns with the migration
command. The setup script writes through the same resolver, so a re-auth can no
longer deposit the credential where the service never looks.

On top of that, the slice removes the SILENCE and the MISDIRECTION that turned a
severed credential into five undiagnosed days.

### Diff-budget overage — why this slice is indivisible

The runtime change is small and confined to the credential-resolution path
(`google_oauth.py`) plus the one log site that reports its failure
(`calendar.py`). The remainder is the mandatory `plans/PR-*.md` and the
regression matrix.

Splitting was considered and rejected on each seam:

- **Path fix without source reporting** still lets a stale `.env` token
  masquerade as the file token, and still tells the operator to re-run setup.
  The path fix alone would have shortened this outage but not made it legible.
- **Source reporting without the path fix** leaves resolution CWD-dependent, so
  the resolved path in the warning would itself vary with how the process was
  started — the diagnostic would be untrustworthy exactly when it is needed.
- **Either without the missing-file warning** keeps the silent step that made a
  five-day outage possible at all.

All three are one causal chain — resolve, substitute, misreport — and the tests
that prove each are the same fixtures.

### Problem-derived contract

- Root cause: a CWD-relative credential path in a service whose CWD is a
  deploy-time detail, combined with a silent cross-source fallback that
  substitutes a *different* credential and a failure message that names the
  wrong one.
- Correct fix must touch/change: path resolution in
  `atlas_brain/services/google_oauth.py` so a relative path is anchored to the
  checkout rather than the CWD — and, because that anchor still travels with a
  deployed worktree, an operator-facing warning that names the absolute-path
  remedy; `_load()` so an absent file is announced with
  its resolved path; `get_credentials()` so the credential carries which source
  supplied it; and the rejection log in `atlas_brain/tools/calendar.py` so the
  remedy names the file that actually supplied the rejected token.
- Must not change: credential PRECEDENCE (token file still wins over `.env`);
  absolute configured paths, which must stay honoured verbatim; the rotation
  persistence path (`persist_refresh_token`); the `.env` fallback continuing to
  work when no file exists; and no credential VALUE may be logged.

## Scope (this PR)

Ownership lane: eom-ops/google-token-resolution
Slice phase: Vertical slice
Max files: 6

1. Default the token file OUTSIDE the repo:
   `DEFAULT_TOKEN_FILE = "~/.config/atlas/google_tokens.json"`. This is what
   prevents recurrence.
2. `locate_token_file()`: prefer the configured/default path; fall back to the
   legacy in-repo `<checkout>/data/google_tokens.json` so upgrades keep working,
   warning with the migration command when that fallback is used.
3. `resolve_token_file_path()`: honour absolute and `~` paths verbatim; anchor
   any remaining relative path to the checkout rather than the CWD.
4. `scripts/setup_google_oauth.py` writes through the same resolver, so a
   re-auth lands where the service reads.
5. Log a WARNING naming the resolved path when the token file is absent.
6. Carry `refresh_token_source` (`"file"` / `"env"`) on `GoogleCredentials`, and
   WARN when the `.env` fallback supplies the token; the Calendar rejection
   CRITICAL names the source and the matching remedy.

### Review Contract

- Acceptance criteria:
  1. A relative token path resolves identically from any CWD — settled by
     `tests/test_google_token_resolution.py::test_relative_path_is_identical_from_two_different_cwds`
     and `::test_relative_path_anchors_to_the_repo_root`.
  2. **Recurrence is prevented**: the default credential path is outside every
     checkout and resolves identically from any deployed worktree — settled by
     `::test_default_token_file_is_outside_every_worktree` and
     `::test_default_is_identical_from_any_checkout`. WHY an in-repo anchor
     cannot achieve this is itself asserted by
     `::test_relative_path_still_moves_with_the_deployed_checkout`, and the CWD
     half by `::test_cwd_change_no_longer_moves_the_credential`.
  2b. Upgrades do not break and a re-auth lands where the service reads —
     settled by `::test_legacy_in_repo_file_is_still_found_and_warns`,
     `::test_store_itself_uses_the_legacy_fallback`,
     `::test_primary_wins_over_legacy_when_both_exist`,
     `::test_first_write_lands_in_the_stable_location` and
     `::test_setup_script_writes_where_the_service_reads`.
  3. Absolute and `~` paths are unchanged — settled by
     `::test_absolute_path_is_honoured_unchanged` and
     `::test_user_home_shorthand_expands`.
  4. An absent token file is announced with its resolved path, and a present
     one does not warn — settled by
     `::test_missing_token_file_is_logged_with_its_resolved_path`,
     `::test_missing_file_warning_names_the_stable_default` and
     `::test_present_token_file_does_not_warn_about_absence`.
  5. A credential reports which source supplied it — settled by
     `::test_file_token_is_reported_as_coming_from_the_file`,
     `::test_env_fallback_is_reported_and_warned` and
     `::test_both_services_report_their_source`.
  6. Precedence is unchanged: a token file still beats a different `.env` token
     — settled by `::test_file_token_wins_over_a_different_env_token`.
  7. No credential anywhere still returns `None` — settled by
     `::test_no_credential_anywhere_returns_none`; an unknown service is still
     rejected — `::test_unknown_service_is_rejected`.
  8. The `parents[2]` root derivation is guarded against a future file move —
     settled by `::test_repo_root_derivation_points_at_a_real_checkout`.
- Reachability proof: `GoogleTokenStore` is constructed by
  `get_google_token_store()` and consumed by
  `atlas_brain/tools/calendar.py::_refresh_token` (`store.get_credentials("calendar")`),
  the coroutine that fetches every calendar event. The observable state is the
  resolved `token_file_path`, the emitted log records, and
  `GoogleCredentials.refresh_token_source`.
- Affected surfaces: Google credential resolution for Calendar and Gmail; the
  Calendar auth-failure log line; no HTTP route, no schema, no migration.
- Risk areas: changing the resolved path could point a working deployment at a
  different file. Mitigated but NOT eliminated: absolute paths are untouched,
  and for a checkout whose CWD already equals its root the resolved path is
  unchanged. For the current deployment the anchor equals the runtime worktree,
  i.e. the same path resolved today, so this change is behaviour-neutral there
  and the operative fix remains the absolute-path config the warning names.
  Also: leaking a credential value into logs
  (mitigated: only paths, service names and the literal source word are logged);
  breaking the `.env` fallback for deployments with no token file.
- Reviewer rules triggered: R1, R14. R1 for the operator-facing behaviour
  change in a failure path. R14 declared deliberately: `resolve_token_file_path`
  is an admission/normalisation boundary for a filesystem path.

### Boundary-change enumeration

The seam is credential resolution in `GoogleTokenStore`: a CWD-relative
`Path(...)` becomes a repo-root-anchored resolution, and an unlabelled
credential gains a source label.

- Replaced-path behaviour: previously `Path("data/google_tokens.json")` resolved
  against the process CWD; now it resolves to `<repo root>/data/google_tokens.json`.
  For the deployed unit this changes the resolved path from
  `<current worktree>/data/...` to the checkout root's `data/...` — which is
  where the file has always actually lived.
- Guard-relevant fields: the configured `google_token_file` string (absolute vs
  relative vs `~`-prefixed), and the presence/absence of the file at the
  resolved path.
- Caller × input shape: one production caller,
  `tools/calendar.py::_refresh_token`, plus `get_status()` and
  `persist_refresh_token()` on the same store. Inputs are operator-configured
  strings, not request data.

Closure declaration for the **path-shape** set:

1. **Closed or open? — CLOSED**, three shapes exhaust the input: absolute,
   `~`-prefixed, and relative.
2. **Where does membership come from? — ENUMERATED** in
   `resolve_token_file_path` and asserted per shape by
   `::test_absolute_path_is_honoured_unchanged`,
   `::test_user_home_shorthand_expands` and
   `::test_relative_path_anchors_to_the_repo_root`.
3. **Out-of-set behaviour — there is none**: the stdlib `Path.is_absolute`
   predicate is total after `~` expansion, so every string resolves. The safety
   property is that resolution never consults the CWD, verified by resolving
   from two different working directories and asserting equality.

Closure declaration for the **credential-source** set:

1. **Closed or open? — CLOSED**: `"file"` or `"env"`, the only two suppliers
   `get_credentials` reads.
2. **Where does membership come from? — DERIVED** from which lookup produced
   the token, in the same expression that selects it, so the label cannot drift
   from the value it describes.
3. **Out-of-set behaviour — fail safe to `"file"`**: `calendar.py` reads the
   label with `getattr(creds, "refresh_token_source", "file")`, so a credential
   object without the attribute yields the pre-change remedy text rather than
   an AttributeError inside an error handler.

### Deployed-config probing

This slice exists because of a deployed-config divergence, so the deployed
values were probed directly rather than assumed:

- `systemctl --user cat atlas-api.service` → `WorkingDirectory=` is the runtime
  worktree, confirming the CWD the relative path was resolving against.
- The old runtime worktree still carries
  `data/google_tokens.json -> /home/juan-canfield/Desktop/Atlas/data/google_tokens.json`
  (symlink dated Jul 31 17:35); the new runtime worktree had no `data/` at all.
  Worktree birth 2026-08-05 08:30:12, first failure 09:31 — a one-hour gap.
- Both credentials were refreshed against `https://oauth2.googleapis.com/token`:
  the token-file value returned `expires_in=3599`; the `.env` value
  (`ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN`) returned `HTTP 400 invalid_grant`.
  Only digests/fingerprints were recorded, never the token values.
- `ATLAS_TOOLS_GMAIL_REFRESH_TOKEN` is unset, which is why Gmail returned `None`
  instead of a rejected credential and produced no CRITICAL.

### Files touched

- `atlas_brain/config.py`
- `atlas_brain/services/google_oauth.py`
- `atlas_brain/tools/calendar.py`
- `plans/PR-EOM-Google-Token-Resolution.md`
- `scripts/setup_google_oauth.py`
- `tests/test_google_token_resolution.py`

## Mechanism

`resolve_token_file_path()` anchors relative paths to `_REPO_ROOT`, derived from
`Path(__file__).resolve().parents[2]` — the checkout containing
`atlas_brain/`. Absolute paths return unchanged so explicit deployments and
tests keep control, and the stdlib `Path.expanduser` call runs first so `~` is not
mistaken for a relative path. `parents[2]` is a positional hop, so
`::test_repo_root_derivation_points_at_a_real_checkout` asserts the result is a
real checkout rather than trusting the count.

**What this does and does not buy, stated plainly.** Anchoring makes resolution
deterministic with respect to the process CWD. It does NOT make it stable across
deployments: when the service runs from a git worktree, this module lives inside
that worktree, so `parents[2]` is the worktree root and a relative path still
resolves under whichever worktree is deployed. Verified directly against the
deployed layout — for
`worktrees/atlas-runtime-main/atlas_brain/services/google_oauth.py`,
`parents[2]` is `worktrees/atlas-runtime-main`, which is exactly the worktree
whose `data/` was missing on 2026-08-05. Anchoring alone would therefore NOT
have prevented that outage, and this plan does not claim it would.

What removes the hazard is the pair: an absolute `ATLAS_TOOLS_GOOGLE_TOKEN_FILE`
(operator config, honoured verbatim by this function) plus the missing-file
warning that names that remedy with the resolved path. The outage becomes a
single actionable log line on first use instead of five silent days. The
residual limitation is asserted by
`::test_relative_path_still_moves_with_the_deployed_checkout` so it is recorded
in the suite rather than implied away.

`_load()` gains an `else` branch: an absent file now logs a WARNING naming the
resolved path. That single line is what distinguishes "the file is missing" from
"Google rejected the credential" — two failures with opposite remedies that
previously looked identical from the logs.

`GoogleCredentials` gains `refresh_token_source`, set in the same expression
that chooses the token, so the label cannot disagree with the value. When the
`.env` fallback supplies it, `get_credentials` warns that the two sources can be
*different credentials* — the specific surprise in this outage.

`calendar.py` then selects its remedy from that label: an `env`-sourced
rejection says the setup script will not help and points at the `.env` value; a
`file`-sourced rejection keeps the re-auth instruction and names the file. The
message also changes from "is INVALID" to "was REJECTED by Google", because the
credential's validity is Google's verdict about one specific value, not a
property of the token file.

## Intentional

- **Precedence is unchanged.** The token file still wins over `.env`. This slice
  makes the substitution visible; it does not re-rank the sources.
- **The stale `.env` value is not removed here.** `ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN`
  is operator config, not repo content, and removing it is a deployment action
  the operator owns. It is currently inert because the file wins again.
- **No credential values are logged** — only resolved paths, service names and
  the literal words `file`/`env`.
- **Pre-existing `ruff` F841 findings in both changed modules are left alone.**
  `origin/main` and this head both report 5; none are introduced here and none
  sit on a line this PR touches.
- **The immediate production fix (restoring the symlink) is already applied**
  and is deliberately not what this PR ships — the symlink is a workaround for
  the defect, and this PR removes the need for it.

## Deferred

Parking predicate: deferred items are non-blocking because Calendar and Gmail
both resolve their credentials correctly at this head, and each deferred item is
either operator config or a separate surface.

- Removing the stale `ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN` from the deployed
  `.env`. Operator action; inert today.
- A startup self-check that asserts every configured credential file exists and
  refreshes, surfacing breakage at deploy time rather than on first use. Worth
  doing, but it is a new health-check surface rather than a fix to this path.
- Gmail's silent `None` path deserves the same operator-facing warning that
  Calendar now has; Gmail's failure mode in this outage was invisible. Its log
  site is a different module and is not touched here.

Parked hardening: none.

## Verification

All counts re-run at this head.

- `python -m pytest tests/test_google_token_resolution.py -q` — **25 passed**
- Every consumer of the changed store — `test_google_token_resolution.py`,
  `test_calendar_import_rerun.py`, `test_eom_live_calendar_import.py`,
  `test_eom_scoped_gmail_credentials.py`, `test_leads_intake.py` — **172 passed, 1 skipped**
- **Negative probes**, each injected then reverted (restored state **18 passed**):
  | Injected defect | Result |
  |---|---|
  | revert the store to CWD-relative `Path(...)` (the original defect) | 1 failed |
  | delete the missing-file warning (silent again) | 1 failed |
  | hardcode `source = "file"` (hide the substitution) | 1 failed |
  | wrong `parents[]` hop, as if the module moved | 3 failed |
  | default moved back inside the repo (the original defect) | 2 failed |
  | store bypasses the legacy fallback (breaks upgrades) | 1 failed |
- **Deployed-layout check that redirected this slice.** Resolving `parents[2]`
  for `worktrees/atlas-runtime-main/atlas_brain/services/google_oauth.py` yields
  `worktrees/atlas-runtime-main` — the very worktree whose `data/` was missing.
  An in-repo anchor therefore could NOT have prevented the outage. That finding
  is why the default moved out of the repo entirely rather than being
  "anchored better"; `::test_relative_path_still_moves_with_the_deployed_checkout`
  keeps the reasoning in the suite.
- Default path resolves to `/home/juan-canfield/.config/atlas/google_tokens.json`
  — outside every checkout, alongside the deployment's other service tokens.
- `ruff check` on the two changed modules: **5 F841**, identical to the
  `origin/main` baseline (5) — none introduced, none on a touched line. The new
  test file is clean.
- `python -m py_compile` on both changed modules — OK.
- `git diff --check` — clean.
- No credential value appears in any added log statement; only resolved paths,
  service names, and the literal words `file`/`env`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 11 |
| `atlas_brain/services/google_oauth.py` | 128 |
| `atlas_brain/tools/calendar.py` | 24 |
| `plans/PR-EOM-Google-Token-Resolution.md` | 366 |
| `scripts/setup_google_oauth.py` | 15 |
| `tests/test_google_token_resolution.py` | 444 |
| **Total** | **988** |
