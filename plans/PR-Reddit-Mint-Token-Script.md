# PR-Reddit-Mint-Token-Script

## Why this slice exists

Running the Reddit Listening pipeline live needs a scoped refresh token
(`identity`/`history`/`read`), minted once via Reddit's authorization-code
flow. The procedure was only an inline snippet in
`docs/REDDIT_LISTENING_SETUP_RUNBOOK.md` -- copy/paste, easy to get the
scopes wrong. This lands it as a ready-to-run operator script that reuses
an already-registered Reddit app so no second app must be created.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/fit-eval
Slice phase: Workflow/process

1. `scripts/mint_reddit_listening_token.py`: PRAW authorization-code flow
   with the three scopes preset; resolves the app client id/secret from
   `ATLAS_REDDIT_CLIENT_ID`/`_SECRET`, falling back to the B2B scraper's
   `ATLAS_B2B_SCRAPE_REDDIT_*`; prints the refresh token + the `.env` block.
   Fail-closed on missing creds/username (exit 2).
2. `tests/test_mint_reddit_listening_token.py`: the deterministic logic --
   dotenv parsing, credential resolution + B2B fallback + precedence, the
   redirect-param parser, and the fail-closed CLI exits.

### Review Contract

- Acceptance criteria:
  - [ ] Credential resolution: explicit args > listening keys > B2B keys;
        a blank value is not "set"; missing -> `MintConfigError` -> exit 2.
  - [ ] `load_env` parses a dotenv (comments/blanks skipped, outer quotes
        stripped, `=` in values preserved); missing file -> `{}`.
  - [ ] `parse_redirect_params` extracts `code`/`state`/`error` from the raw
        HTTP request line.
  - [ ] CLI exits 2 (not a traceback) on missing creds and on missing
        username.
  - [ ] Scopes are exactly `identity`/`history`/`read` and redirect URI is
        `http://localhost:8080` (matches the read-only listening contract).
- Reachability proof (#1952): `main([...])` runs in-process in the tests up
  to the live praw/socket boundary -- the fail-closed paths return exit 2
  and the credential/param helpers return the resolved values. The live
  browser+socket mint is the operator step, out of CI by nature.
- Affected surfaces: one new script, one new test. No package/runtime code,
  no config schema change.
- Risk areas: secret handling (the script only reads existing creds and
  prints them for the operator to place; no writes to `.env`); scope
  breadth (hard-coded to the three read scopes).
- Reviewer rules triggered: R1, R2 (resolution + both fail-closed exits),
  R10 (resolve_credentials is a gate predicate -- traced: complete-pair
  only), R11 (praw already a pipeline dependency; lazy-imported), R12 (test
  auto-enrolls via the renamed glob), R13 (the wave-1 findings named defect
  classes -- cross-namespace mixing, raw-UA -- fixed at the class), R14
  (resolve_credentials / build_user_agent are validators; reachability
  proof named above).
- Test-adapter posture (#1934): the external boundary (praw, the localhost
  socket, browser approval) is the operator's live step; every deterministic
  helper is tested for real, nothing mocked.

### Files touched

- `.github/workflows/atlas_reddit_checks.yml`
- `plans/PR-Reddit-Mint-Token-Script.md`
- `scripts/mint_reddit_listening_token.py`
- `tests/test_atlas_reddit_mint_token.py`

## Mechanism

`resolve_credentials` picks the app id/secret by precedence (args, then
`ATLAS_REDDIT_*`, then `ATLAS_B2B_SCRAPE_REDDIT_*`), failing closed if
unresolved. `main` validates creds + username (exit 2 on either), then
lazy-imports praw, builds the auth URL with the three read scopes, serves a
one-shot localhost:8080 socket to catch Reddit's redirect,
`parse_redirect_params` extracts the code (with state check + error
surfacing), and `reddit.auth.authorize` exchanges it for the refresh token,
which is printed with the `.env` block. The script never writes `.env`.

## Intentional

- **Reuse the existing app**: falling back to the B2B scraper's client
  id/secret means no second Reddit app registration for the common case;
  `--client-id/--client-secret` override for a dedicated app.
- **Read-only scopes hard-coded**: `identity`/`history`/`read` only, matching
  the listening tool's fail-closed scope contract.
- **Prints, never writes**: the operator places the token; the script does
  not touch `.env` (no secret written by automation).

## Deferred

- None.

## Parked hardening

- None.

Review-fix notes (Codex wave 1; all 4 verified real, fixed at root):
- **Credential pairs resolved atomically** -- `resolve_credentials` now
  takes a COMPLETE (id, secret) pair per namespace; a partial config
  (listening id + B2B secret) falls back to a complete pair or fails
  closed, never a mismatched cross-app pair.
- **Shell env honored** -- `build_cred_source` merges exported `ATLAS_*`
  vars with `.env` (shell wins), so creds kept out of `.env` work.
- **Username validated via the production `build_user_agent`** -- rejects
  trailing newlines / invalid UA shapes instead of interpolating raw.
- **CI-enrolled** -- test renamed to `test_atlas_reddit_mint_token.py`
  (matches the workflow glob) + the script added to the workflow path
  filter, so a PR touching only it runs these tests.

## Verification

- `.venv/bin/python -m pytest tests/test_mint_reddit_listening_token.py -q`:
  15 passed (load_env parse/missing; resolution precedence + B2B fallback +
  blank-not-set + fail-closed; redirect-param code/state/error; CLI exit-2 on
  missing creds and missing username).
- ASCII byte-scan on the changed files: clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_reddit_checks.yml` | 1 |
| `plans/PR-Reddit-Mint-Token-Script.md` | 118 |
| `scripts/mint_reddit_listening_token.py` | 193 |
| `tests/test_atlas_reddit_mint_token.py` | 157 |
| **Total** | **469** |
