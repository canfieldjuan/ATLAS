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
  read-only execution; integration-test database-variable admission;
  `./ops env keys` secret-value suppression.
- Replaced-path behaviors: N/A - no existing runtime path is replaced.
- Guard-relevant fields: inspection name, complete PostgreSQL connection string,
  the canonical disposable-test URL set, Git origin userinfo, environment
  assignment key/value split, and subprocess output redaction.
- Caller x input shape: shell caller x fixed connectivity/migrations names;
  shell caller x arbitrary/unknown query names; test caller x each canonical
  disposable database URL independently; Git origin x credential-bearing URL;
  env files x blank/comment/export/quoted/canary-secret assignments.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: database defaults come from
  `atlas_brain/storage/config.py`; the linked Vercel project comes from ignored
  `.vercel/project.json`; neither secret value is copied into version control.
- Explicit value probe: tests pass fixed inspection names, a complete DSN with
  TLS/socket parameters, each canonical disposable-test URL, and fixture env paths;
  safe live probes use the current authenticated CLIs where available.
- Absent value probe: doctor and status report UNKNOWN/unavailable when a CLI,
  auth context, linked project, env file, or database is absent.
- Default-session/default-context probe: run from the dedicated worktree, where
  ignored environment/provider link files are absent, and from the shared root
  context via explicit discovery only.
- Side-effect ordering: fixed inspection selection occurs before the database
  client receives SQL; the exact DSN remains in the child environment rather
  than argv; Git output selects the canonical identifier before printing; env
  rendering extracts keys before any output is produced.

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

- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m py_compile ops` - pass.
- PyYAML `safe_load(.agent/capabilities.yaml)` - schema version `1` parsed.
- `./ops test focused tests/test_agent_operations_contract.py -q` - 16 passed.
- `./ops test focused tests/test_eom_render_profile.py::test_database_config_prefers_connection_string_for_asyncpg_kwargs -q` - 1 passed.
- `./ops doctor` - pass; live systemd/Brain ping, Docker, Vercel, Render CLI,
  PostgreSQL fixed inspection, GitHub auth, and environment-source discovery
  reported without secret values.
- `./ops db inspect connectivity` and `./ops db migrations` - pass through the
  Atlas `DatabaseConfig`/asyncpg path.
- `./ops env keys`, `./ops env systemd`, and `./ops env vercel` - pass;
  names/sources only.
- `./ops ci status 3`, `./ops logs brain`, targeted container logs, and
  `./ops logs frontend --since 1h --limit 5` - pass; the static Vercel runtime
  legitimately returned no records in that window.
- `python scripts/sync_pr_plan.py --check plans/PR-Agent-Operations-Contract.md`
  and `git diff --check` - pass after final sync.
- The original head's guarded push/local review and secret scan passed. The
  current review-fix head will rerun the same guarded push before publication.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 288 |
| `.agent/runbooks/ci.md` | 65 |
| `.agent/runbooks/database.md` | 85 |
| `.agent/runbooks/deployment.md` | 103 |
| `.agent/runbooks/discovery-ledger.md` | 189 |
| `.agent/runbooks/environment.md` | 109 |
| `.agent/runbooks/logs.md` | 74 |
| `.agent/runbooks/testing.md` | 85 |
| `AGENTS.md` | 13 |
| `CLAUDE.md` | 32 |
| `README.md` | 16 |
| `ops` | 780 |
| `plans/PR-Agent-Operations-Contract.md` | 255 |
| `tests/test_agent_operations_contract.py` | 258 |
| **Total** | **2352** |
