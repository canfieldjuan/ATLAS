# PR-Agent-Operations-Contract

## Why this slice exists

The operator explicitly requested a durable Agent Operations Contract after
repeated sessions had to rediscover how Atlas runs, tests, reaches its database,
and distinguishes local services from hosted frontends. Initial archaeology
already found misleading operational paths: the root README names a missing
root `.env.example`, tells agents to start an undefined Compose `postgres`
service, and does not explain that the nominal repository root is a shared/bare
Git root rather than a usable worktree. This is an operator-authorized process
slice under `docs/CURRENT_PRODUCT_DISCIPLINE.md`; it removes a demonstrated
workflow blocker and does not change product behavior.

This contract is likely to exceed the normal 400-LOC target because the
operator explicitly requires a machine-readable inventory, seven focused
runbooks, a discovery ledger, fresh-agent instructions, an executable
operations entrypoint, and verification coverage. Splitting those artifacts
would leave an incomplete and unverified operational path between PRs.

### Problem-derived contract

- Root cause: Atlas operational knowledge is fragmented across prose, provider
  configuration, scripts, local ignored state, and live CLI authentication;
  some prominent prose has drifted from executable configuration. There is no
  canonical capability record or stable read-only-first entrypoint, so each
  agent must repeat provider, worktree, database, test, and log discovery.
- Correct fix must touch/change: add a concise operational pointer to root
  `AGENTS.md`; add `.agent/capabilities.yaml` plus environment, deployment,
  database, testing, logs, CI, and discovery-ledger runbooks; add a thin `ops`
  command whose doctor/status/env/database/deployment/log/CI/test surfaces
  inspect reality without printing secrets or mutating production; and add
  focused automated tests for the command's discovery and safety boundaries.
- Must not change: Atlas runtime/API behavior, database schema or migrations,
  application dependencies, CI/deployment configuration, provider resources,
  customer-visible output, secrets, ignored environment files, and other
  sessions' plans/branches/PRs.

### Contract revision 1

- New evidence: the executable root Compose file has no `postgres` service,
  while both `README.md` and `CLAUDE.md` tell agents to run
  `docker compose up -d postgres`; the README also names a missing root
  `.env.example`. Leaving those entrypoints unchanged would preserve the exact
  repeated-discovery trap this slice exists to remove.
- Revised required change surface: update only the affected quick-start/testing
  paragraphs in `README.md` and `CLAUDE.md` to route agents to the operations
  contract and describe native PostgreSQL/disposable CI database reality.
- Revised non-scope: no broad README/CLAUDE cleanup and no application,
  provider, Compose, workflow, schema, or migration changes.
- Revised verification plan: documentation contract tests assert the stale
  root Compose commands and missing-template instruction are removed from the
  affected operational sections.

### Contract revision 2

- New evidence: current-head review and code tracing proved four reachable
  gaps in the new operations layer. PostgreSQL `READ ONLY` transactions still
  admit operationally mutating functions invoked by arbitrary `SELECT`; DSN
  decomposition drops libpq query parameters; the integration guard omits the
  documented legacy monthly writer URL; and doctor prints a raw origin URL that
  may contain HTTPS userinfo.
- Revised root cause: the initial implementation treated SQL statement class,
  decomposed connection coordinates, a sampled test-variable set, and a raw Git
  remote as sufficient safety boundaries. Those are weaker than the contract's
  no-mutation, exact-runtime, complete-canonical-set, and no-secret-output
  requirements.
- Revised required change surface: replace arbitrary SQL with named, fixed
  database inspections executed through Atlas's own asyncpg configuration path;
  pass the complete connection string through the child environment without
  argv exposure; admit all three canonical disposable-test URL variables; print
  only the canonical repository identifier; and update the capability/runbook
  claims plus boundary tests for those decisions.
- Revised non-scope: do not provision a database role, change application
  database configuration, add dependencies, change migrations, expose raw SQL,
  or touch provider/runtime configuration.
- Revised verification plan: add negative proof that arbitrary queries are not
  exposed, exact-DSN handoff/argv-secret tests, a complete-variable-set test,
  credential-bearing-origin redaction proof, focused tests, live fixed
  inspections, and the canonical guarded push review.

### Contract revision 3

- New evidence: current-head review found two reachable configuration-boundary
  gaps. The database environment reader hand-parses assignments instead of
  using the `python-dotenv` semantics that back `DatabaseConfig`, so quoting,
  inline comments, escapes, and interpolation can produce a different value.
  Unit mode also inherits all three canonical disposable-database URL
  variables, allowing unmarked database-backed tests to write when a URL is
  left exported from an earlier integration run.
- Revised root cause: the operations layer preserved raw connection text and
  pytest markers, but did not preserve the application's dotenv decoding or
  isolate the unit-test child process from integration-only credentials.
- Revised required change surface: parse selected database settings with the
  same checked-in `python-dotenv` dependency used by Pydantic settings; retain
  process-environment precedence and no-output/no-argv boundaries; remove every
  inherited `DATABASE_URL` / `*_DATABASE_URL` plus the disposable-database
  confirmation from unit-mode children; update capability/testing/ledger
  claims; and add boundary regression tests.
- Revised non-scope: do not source or execute shell files, print environment
  values, alter `DatabaseConfig`, change pytest markers, add dependencies,
  change integration-mode admission, or touch databases, schemas, CI, or
  provider configuration.
- Revised verification plan: focused tests prove dotenv quotes/comments,
  escapes, and interpolation match `python-dotenv`; ambient environment still
  wins; unit subprocesses omit every current and novel database-URL-shaped
  variable while preserving unrelated environment; then run compile, plan
  sync/audits, and the guarded push with GitHub retaining the full unit gate.

#### Unit database-environment closure declaration

- Membership is **OPEN** because parent-process environment names are not a
  finite enum. Membership is **DERIVED** on every unit invocation from the
  actual child-environment keys: `DATABASE_URL` and every key ending in
  `_DATABASE_URL` belong to the database-URL class. This covers the three Atlas
  integration URLs and the extracted/generic PostgreSQL test gates found under
  `tests/` without copying a closed list into the decision.
- Any recognized or novel database-URL-shaped key takes the safe default: it is
  removed before pytest starts, because skipping an unintentionally live test
  is safer than letting nominal unit mode write through inherited credentials.
  Keys outside that syntactic class remain inherited so unrelated test/runtime
  configuration is unchanged; integration mode retains its separate explicit
  URL plus disposable-database confirmation gate.

### Contract revision 4

- New evidence: current-head review proved four bounded gaps in the operations
  entrypoints. `doctor` aborts before dependencies are installed because its
  database probe imports `python-dotenv`; a configured Brain health URL is
  echoed after a successful probe; integration mode accepts option-only or
  directory-wide pytest arguments; and the capability map names a nonexistent
  runtime command.
- Revised root cause: the operations layer did not preserve its orientation
  command's dependency-free failure boundary, treated a credential-bearing URL
  as safe status detail, checked only that integration arguments were nonempty,
  and carried one unverified command label into the capability map.
- Revised required change surface: make database status probing report an
  unavailable dependency without aborting `doctor` while explicit database
  commands retain actionable failure; report configured/local Brain endpoint
  success without echoing the URL; require integration mode's first pytest
  argument to resolve to an existing Python test file under `tests/`; replace
  the nonexistent capability command; and add direct boundary regressions and
  concise durable failure notes.
- Revised non-scope: do not add a runtime command, install or vendor
  dependencies, change database configuration, broaden pytest argument parsing,
  change application health behavior, run integration tests, or touch CI,
  provider, deployment, schema, or product code.
- Revised verification plan: focused tests cover missing `python-dotenv`, a
  canary-bearing configured health URL, option-only/directory/out-of-tree versus
  file/node integration targets, and capability-command existence; then run
  compile, plan/documentation audits, live dependency-free `doctor`, and the
  guarded push while GitHub retains the full unit gate.

#### Integration target closure declaration

- Membership is **CLOSED** by the command grammar: the first argument after
  `integration` is a path, optionally followed by a pytest `::node`, whose file
  resolves under the repository's `tests/` directory, has a `.py` suffix, and
  exists as a regular file. Later arguments must be pytest option tokens; use
  `--option=value` when an option takes a value.
- Empty, option-first, directory, missing, non-Python, and out-of-tree targets
  take the safe default and are rejected before database credentials are handed
  to pytest. This closes the repo-wide fallback without inventing a general
  pytest parser.

## Scope (this PR)

Ownership lane: agent-operations
Slice phase: workflow/process
Max files: 14

1. Establish one verified, versioned operational knowledge path for a fresh
   agent without replacing existing subsystem-specific runbooks.
2. Provide a stable `./ops` discovery/inspection layer with explicit
   read-only versus mutating boundaries and prove its local behavior with
   focused tests plus safe live probes.

### Review Contract

- Acceptance criteria:
  - `./ops doctor` runs from a normal worktree without provider mutation and
    reports project, Git, runtime, test, deployment, database, CI, and useful
    command status; settled by focused subprocess tests and a live invocation.
  - `./ops env keys` emits variable names only, never values; settled by a
    fixture containing a canary secret and an assertion that the canary never
    appears in output.
  - `./ops db inspect` exposes only named, fixed inspections and arbitrary SQL
    is unavailable until a privilege-restricted inspection role exists; settled
    by the fixed registry, rejection tests, exact-DSN handoff proof, and live
    connectivity/migration inspection.
  - `.agent/capabilities.yaml` distinguishes verified, unavailable, and
    authentication-dependent capabilities and records safe verification and
    failure modes without secret values; settled by its checked-in contents
    and YAML parsing.
  - The runbooks answer all operator-requested operational questions and link
    to existing subsystem-specific sources instead of duplicating them;
    settled by the documentation contract test and cold reconstruction.
  - Root `AGENTS.md` points fresh agents at the contract, `./ops doctor`, and
    the continuous-learning rule; settled by the documentation contract test.
- Reachability proof: run the real `./ops doctor`, `./ops env keys`,
  `./ops db status`, `./ops deploy status`, `./ops ci status`, and focused test
  entrypoints and observe their redacted terminal output/exit status.
- Affected surfaces: root agent instructions, `.agent` operational metadata and
  runbooks, the root `ops` command, and focused tests for those contracts.
- Risk areas: secret disclosure, a nominally read-only database command
  admitting writes, provider-specific assumptions, stale static claims, slow
  or network-dependent doctor behavior, and accidental application/CI changes.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `./ops db inspect` named-inspection admission and
  read-only execution; integration-test database-variable admission; unit-test
  database-environment isolation; integration-test target admission;
  dependency-free database status fallback; Brain health-detail redaction; and
  `./ops env keys` secret-value suppression.
- Replaced-path behaviors: N/A - no existing runtime path is replaced.
- Guard-relevant fields: inspection name, complete PostgreSQL connection string,
  the canonical disposable-test URL set, open `*_DATABASE_URL` environment-key
  class, Git origin userinfo, environment assignment key/value split, and
  subprocess output redaction; configured Brain URL; integration target path,
  node suffix, and trailing pytest arguments.
- Caller x input shape: shell caller x fixed connectivity/migrations names;
  shell caller x arbitrary/unknown query names; test caller x each canonical
  disposable database URL independently; unit caller x current/novel database
  URL keys plus an unrelated key; Git origin x credential-bearing URL; env files
  x blank/comment/export/quoted/escaped/interpolated/canary-secret assignments;
  fresh system Python x missing `python-dotenv`; configured Brain endpoint x
  credential-bearing URL; integration caller x file/node/option-only/directory/
  missing/out-of-tree/additional-positional targets.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: database defaults come from
  `atlas_brain/storage/config.py`; the linked Vercel project comes from ignored
  `.vercel/project.json`; neither secret value is copied into version control.
- Explicit value probe: tests pass fixed inspection names, a complete DSN with
  TLS/socket parameters, each canonical disposable-test URL, a novel
  database-URL-shaped key, and fixture env paths with dotenv quoting/comments,
  escapes, and interpolation; tests also pass a credential-bearing configured
  health URL and bounded/unbounded integration targets; safe live probes use the
  current authenticated CLIs where available.
- Absent value probe: doctor and status report UNKNOWN/unavailable when a CLI,
  auth context, linked project, env file, or database is absent.
- Default-session/default-context probe: run from the dedicated worktree, where
  ignored environment/provider link files are absent, and from the shared root
  context via explicit discovery only.
- Side-effect ordering: fixed inspection selection occurs before the database
  client receives SQL; the exact DSN remains in the child environment rather
  than argv; unit-mode database URLs are removed before pytest starts; Git
  output selects the canonical identifier before printing; env rendering
  extracts keys before any output is produced; integration target validation
  occurs before pytest or database admission; health output selects a fixed
  label rather than the configured URL.

### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/ci.md`
- `.agent/runbooks/database.md`
- `.agent/runbooks/deployment.md`
- `.agent/runbooks/discovery-ledger.md`
- `.agent/runbooks/environment.md`
- `.agent/runbooks/logs.md`
- `.agent/runbooks/testing.md`
- `AGENTS.md`
- `CLAUDE.md`
- `README.md`
- `ops`
- `plans/PR-Agent-Operations-Contract.md`
- `tests/test_agent_operations_contract.py`

## Mechanism

The YAML record stores evidence level, provider/tool, invocation,
authentication, mutation class, safe verification, and failure notes for each
capability. Focused runbooks explain prerequisites, safe procedures, and
escalation paths while linking existing deep runbooks. The stdlib-only `ops`
entrypoint discovers the repository/worktree and available tools at runtime,
normalizes common read-only checks, keeps remote mutations out of status
commands, and fails closed on database writes. Tests exercise its output,
redaction, and command-boundary behavior without requiring live providers.

## Intentional

- Provider IDs discovered only from ignored local linkage are reported by live
  inspection but are not copied into tracked files when a stable project name
  or repository command is sufficient.
- `ops` does not auto-source `.env` files, install tools, trigger deployments,
  run migrations, or offer a generic database write command.
- Existing product/subsystem runbooks remain canonical for their narrow flows;
  the new layer links to them instead of rewriting them.

## Deferred

- Provider deployment mutations remain operator-initiated; a future slice may
  add a separately consented, confirmation-gated deployment command if repeated
  operational evidence justifies it.

Parking predicate: provider-specific conveniences, optional formatting, and
additional mutating operations that are not required to answer the operator's
fresh-agent questions are parked unless current verification proves they block
the contract or create a security/data-safety defect.

Parked hardening: none.

## Verification

- `python -m py_compile ops tests/test_agent_operations_contract.py` - pass.
- PyYAML `safe_load(.agent/capabilities.yaml)` - schema version `1` parsed.
- New dotenv/unit-isolation regression nodes failed before implementation with
  undecoded dotenv text and a missing sanitized child environment.
- The four new boundary regression groups failed before implementation with
  `10 failed`, covering missing dependency, configured-URL disclosure,
  unbounded integration targets, and the nonexistent capability command.
- `./ops test focused tests/test_agent_operations_contract.py -q` - 29 passed.
- `./ops test focused tests/test_eom_render_profile.py::test_database_config_prefers_connection_string_for_asyncpg_kwargs -q` - 1 passed.
- `./ops doctor` - pass; live systemd/Brain ping, Docker, Vercel, Render CLI,
  PostgreSQL fixed inspection, GitHub auth, and environment-source discovery
  reported without secret values.
- `/usr/bin/python3 ./ops doctor` with `python-dotenv` unavailable - pass;
  database status reported `UNAVAILABLE` while the remaining snapshot completed.
- `./ops db inspect connectivity` and `./ops db migrations` - pass through the
  Atlas `DatabaseConfig`/asyncpg path.
- `./ops db inspect connectivity` - pass again after switching selected `.env`
  decoding to `python-dotenv`.
- `./ops env keys`, `./ops env systemd`, and `./ops env vercel` - pass;
  names/sources only.
- `./ops ci status 3`, `./ops logs brain`, targeted container logs, and
  `./ops logs frontend --since 1h --limit 5` - pass; the static Vercel runtime
  legitimately returned no records in that window.
- `python scripts/sync_pr_plan.py --check plans/PR-Agent-Operations-Contract.md`
  and `git diff --check` - pass after final sync.
- The original head's guarded push/local review and secret scan passed. The
  current review-fix head will rerun the same guarded push before publication;
  per operator direction, GitHub Actions owns the full unit gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 290 |
| `.agent/runbooks/ci.md` | 65 |
| `.agent/runbooks/database.md` | 85 |
| `.agent/runbooks/deployment.md` | 103 |
| `.agent/runbooks/discovery-ledger.md` | 249 |
| `.agent/runbooks/environment.md` | 109 |
| `.agent/runbooks/logs.md` | 74 |
| `.agent/runbooks/testing.md` | 101 |
| `AGENTS.md` | 13 |
| `CLAUDE.md` | 32 |
| `README.md` | 16 |
| `ops` | 827 |
| `plans/PR-Agent-Operations-Contract.md` | 363 |
| `tests/test_agent_operations_contract.py` | 468 |
| **Total** | **2795** |
