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

### Contract revision 5

- New evidence: current-head review identified a second valid Git common-dir
  shape. The live Atlas layout currently returns
  `/home/juan-canfield/Desktop/Atlas/.git`, so the existing `.git`-parent branch
  resolves shared state correctly; a linked worktree backed directly by a bare
  common directory returns that directory itself, and the current fallback would
  incorrectly select the worktree.
- Revised root cause: the `ops` shared-root resolver distinguishes only a conventional `.git`
  directory from failure and treats every other successful Git common-dir shape
  as failure, conflating a bare common directory with an unavailable lookup.
- Revised required change surface: return the parent for a conventional `.git`
  common directory, return any other successfully resolved common directory
  itself, preserve the repository-root fallback only for command failure, cover
  all three outcomes in a focused resolver regression, and record the distinction
  in the existing worktree-layout discovery entry.
- Revised non-scope: do not change worktree discovery, copy shared ignored state,
  alter Git configuration, add provider lookup behavior, or touch application,
  database, deployment, CI, schema, or product code.
- Revised verification plan: prove the bare-directory regression fails before
  implementation, then run the focused operations-contract suite, compile,
  plan/body/mechanical audits, and a live `ops` shared-root probe; GitHub retains
  the full unit gate.

#### Shared-root resolver closure declaration

- Membership is **CLOSED** by Git command outcome and path shape: command failure
  falls back to the checked-out repository root; a successful common directory
  named `.git` resolves to its parent; every other successful absolute common
  directory resolves to itself.
- The regression covers each member directly, including the live Atlas
  conventional `.git` shape and the alternate bare-directory shape.

### Contract revision 6

- New evidence: one current-head review round exposed four finite classification
  gaps. Integration admission accepts more than one destructive-test URL;
  database inspection merges unrelated worktree, shared, example, and systemd
  sources instead of selecting one application context; shared-root metadata
  describes only the conventional `.git` shape; and container status reports
  every Docker inspect failure as an absent object even when the daemon itself
  is unavailable.
- Revised root cause: the operational layer uses one-sided presence or ordered
  accumulation where safety depends on explicit cardinality, precedence, or
  provider state. Patching individual instances would leave the opposite
  members of each same finite decision class undisposed.
- Revised required change surface: require exactly one canonical integration
  database URL; select one database file context in explicit override,
  worktree, shared-root, then systemd-fallback order before applying process
  environment precedence; describe conventional and direct-bare shared roots;
  distinguish Docker CLI absence, offline mode, daemon unavailability, and
  daemon-available object state; update the canonical database/ledger claims;
  and cover every member with focused tests.
- Revised non-scope: do not add more providers, containers, database variables,
  environment formats, test modes, deployment behavior, CI changes, product
  code, or adjacent hardening. This is one frozen pass over the four named
  current-head threads; GitHub retains the full unit gate.
- Revised verification plan: run only the focused operations-contract tests,
  Python compile, YAML parse, plan sync/check, diff check, safe live doctor,
  database status/inspection, and Docker status probes. Do not run the local
  unit gate.

#### Frozen-pass closure declarations

- Integration database admission is **CLOSED** over 0, 1, or many active
  members of `TEST_DATABASE_URL_KEYS`: only 1 proceeds; 0 and 2/3 reject before
  pytest, naming keys but never values.
- Database file context is **CLOSED** by precedence: an explicit
  `ATLAS_OPS_ENV_FILES` list is one selected context; otherwise any worktree
  `.env`/`.env.local` presence selects that application context, else the
  shared-root pair, else systemd EnvironmentFiles. Tracked examples and
  `.env.tailscale` remain inventory-only. Within a selected context later
  files win, and exported process values win last.
- Shared-root metadata is **CLOSED** over the resolver's successful path shapes:
  conventional `.git` means its parent and a direct bare common directory
  means that directory itself; command failure retains the repository fallback.
- Docker container status is **CLOSED** over CLI missing, offline, daemon
  unavailable, and daemon available. Only after a successful daemon probe can
  an inspect miss mean `absent`; a successful empty inspect is `UNKNOWN`.

### Contract revision 7

- New evidence: the exact-one canonical integration URL guard still hands
  pytest the unfiltered parent environment. Current integration targets can
  select `EXTRACTED_DATABASE_URL`, `DATABASE_URL`, or Atlas application
  database settings instead of the confirmed canonical URL, and the cited
  ticket-FAQ test applies migration SQL and performs writes/deletes through
  that unconfirmed credential.
- Revised root cause: integration admission validates credential cardinality
  but does not bind subprocess authority to the credential that passed the
  confirmation gate. Validation and execution therefore operate on different
  database environments.
- Revised required change surface: construct an integration-only child
  environment that removes every inherited `DATABASE_URL`/`*_DATABASE_URL` and
  Atlas application database setting, restores the single confirmed canonical
  URL, and adapts that same value to `DATABASE_URL`,
  `EXTRACTED_DATABASE_URL`, and `ATLAS_DB_CONNECTION_STRING`; pass that child
  environment explicitly to pytest; update the capability/runbook/ledger
  claims; and add negative proof that no unconfirmed credential reaches argv or
  the child environment.
- Revised non-scope: do not add target registries, parse test source, provision
  databases, execute database-backed tests, alter pytest targets/markers,
  change application configuration, add credentials, modify CI/provider/schema/
  product code, or revisit earlier resolved review decisions.
- Revised verification plan: prove the focused credential-isolation regression
  fails before implementation, then run the focused operations-contract test
  file, Python compile, YAML/plan/diff audits, and the guarded push with the
  local full unit mirror skipped; GitHub owns the full unit gate.

#### Integration credential closure declaration

- Credential membership is **OPEN** for URL-shaped environment keys and
  libpq's uppercase `PG*` class, and **CLOSED** for Atlas application database
  settings. Every `DATABASE_URL`, `*_DATABASE_URL`, `PG*`, and
  `DATABASE_CONFIG_KEYS` member is removed from the inherited child environment.
- Exactly one canonical member of `TEST_DATABASE_URL_KEYS` is admitted. Its
  value is restored under that key and adapted to the two generic URL aliases
  and Atlas's connection-string key, so all supported current test consumers
  receive one identical confirmed credential. No database value enters argv.
- Unknown future URL-shaped keys and every non-selected canonical key take the
  safe default and remain absent. Unrelated environment keys remain inherited.

### Contract revision 8

- New operator direction: the unit gate is GitHub-only because its local mirror
  is too slow. A local pytest process survived the prior push/review path even
  though the session intended to leave the full gate to GitHub.
- Revised root cause: `scripts/local_pr_review.sh` still owns an executable
  local unit-gate mirror, and `./ops test unit` still exposes the full local
  suite as a normal operations command. The earlier convention depended on an
  ambient `GITHUB_ACTIONS=true` override instead of making the boundary
  structural.
- Revised required change surface: remove the local-review unit-gate execution
  path, make `./ops test unit` fail closed with the canonical hosted workflow,
  retain focused local tests and all cheap mechanical review checks, update the
  agent/testing/capability contract, and add focused subprocess proof that
  neither local entrypoint launches the unit checker or pytest.
- Revised non-scope: do not change the hosted unit workflow, its selection or
  baseline logic, focused-test execution, integration execution, unrelated CI,
  or any application behavior; do not diagnose or optimize the unit suite.
- Revised verification plan: exercise only focused wrapper/operations tests,
  Python/shell syntax, capability parsing, plan synchronization, and diff
  checks. Do not invoke the local unit gate or full unit suite; GitHub owns that
  result.

### Contract revision 9

- New current-head blocker: PostgreSQL's libpq interface can select a database
  through inherited `PG*` variables even after every URL-shaped and Atlas
  application credential is removed.
- Revised root cause: the credential-isolation boundary enumerates application
  and URL interfaces but leaves libpq's open environment-key class authorized,
  so confirmation and subprocess authority can still describe different
  databases.
- Revised required change surface: remove the entire inherited uppercase `PG*`
  class before restoring the one confirmed disposable DSN through the supported
  URL/application aliases; add current and novel libpq canaries; update the
  capability, database/testing runbooks, and discovery ledger.
- Revised non-scope: do not change database configuration parsing, selected env
  file error behavior, project-interpreter selection, pytest targets, hosted CI,
  or the hosted-only unit-gate boundary.
- Revised verification plan: rerun the single focused integration-environment
  matrix, the focused operations-contract file, syntax/schema/plan/diff audits,
  and the mechanical push. Do not run the local unit gate or a database-backed
  test.

### Contract revision 10

- New current-head blocker: `./ops test focused` passes the ambient environment
  directly to pytest, so any exported canonical, generic, Atlas application, or
  libpq credential can activate a database-writing test without the disposable
  confirmation and isolated child used by integration mode.
- Revised root cause: database credential isolation exists only inside the
  integration branch instead of at the common local pytest boundary. The
  recommended focused entrypoint therefore has broader database authority than
  the explicit destructive-test entrypoint.
- Revised required change surface: factor one database-credential removal
  helper over URL-shaped, uppercase `PG*`, Atlas application, and confirmation
  keys; use its credential-free child for every focused pytest invocation; let
  integration mode restore only the confirmed DSN aliases; add negative focused
  child-environment proof and align the capability/testing/database/ledger
  contract.
- Revised non-scope: do not classify test targets, parse test source, change
  pytest markers, run a database-backed test, rebase, or change Brain identity,
  container inventory, Vercel linkage, hosted CI, APIs, schemas, or providers.
- Revised verification plan: prove the focused-child regression fails before
  implementation, then run the operations-contract test file, compile/schema/
  plan/body/diff audits, and the mechanical push. Do not run the local unit gate
  or a database-backed test.

## Scope (this PR)

Ownership lane: agent-operations
Slice phase: workflow/process
Max files: 16

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
  - Integration mode admits exactly one canonical disposable database URL and
    rejects zero or many before pytest; settled by the 0/1/2/3 focused matrix.
  - Database inspection selects one application-equivalent file context before
    exported values override it; settled by explicit/worktree/shared/systemd
    canary tests and the live selected-path probe.
  - Container status proves Docker daemon access before classifying objects as
    present or absent; settled by CLI/offline/daemon/object state tests and a
    live `./ops status` invocation.
  - Integration pytest receives one confirmed disposable credential across the
    canonical, generic, and Atlas application interfaces while every other
    database credential is absent; settled by child-environment and argv
    canary assertions without executing a database-backed test.
  - Focused pytest receives no canonical, URL-shaped, Atlas application, libpq,
    or disposable-confirmation database authority while unrelated environment
    values remain available; settled by child-environment canaries without
    executing a database-backed test.
  - Local PR review and `./ops test unit` cannot launch the unit gate; both
    direct agents to `.github/workflows/unit_gate.yml`, while focused tests
    remain locally available; settled by subprocess canaries that would fail if
    the unit checker or pytest were invoked.
- Reachability proof: run the real `./ops doctor`, `./ops env keys`,
  `./ops db status`, `./ops deploy status`, `./ops ci status`, and focused test
  entrypoints and observe their redacted terminal output/exit status.
- Affected surfaces: root agent instructions, `.agent` operational metadata and
  runbooks, the root `ops` command, the local PR-review wrapper, and focused
  tests for those contracts.
- Risk areas: secret disclosure, accidental local full-suite execution, a nominally read-only database command
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
  `./ops env keys` secret-value suppression; Git common-directory shared-root
  resolution; selected database configuration context; integration database-URL
  cardinality and subprocess credential isolation; and Docker
  daemon-versus-object state classification.
- Replaced-path behaviors: N/A - no existing runtime path is replaced.
- Guard-relevant fields: inspection name, complete PostgreSQL connection string,
  the canonical disposable-test URL set, open `*_DATABASE_URL` environment-key
  class, Git origin userinfo, environment assignment key/value split, and
  subprocess output redaction; configured Brain URL; integration target path,
  node suffix, trailing pytest arguments, Git lookup result, and common-dir
  basename.
- Caller x input shape: shell caller x fixed connectivity/migrations names;
  shell caller x arbitrary/unknown query names; test caller x each canonical
  disposable database URL independently; unit caller x current/novel database
  URL keys plus an unrelated key; Git origin x credential-bearing URL; env files
  x blank/comment/export/quoted/escaped/interpolated/canary-secret assignments;
  fresh system Python x missing `python-dotenv`; configured Brain endpoint x
  credential-bearing URL; integration caller x file/node/option-only/directory/
  missing/out-of-tree/additional-positional targets; worktree x conventional
  `.git`/direct bare common directory/Git lookup failure; integration environment
  x zero/one/two/three active canonical URLs; database source x explicit/worktree/
  shared/systemd context plus process override; Docker x CLI missing/offline/
  daemon unavailable/object absent/object present/empty successful inspect;
  integration credential execution x one confirmed canonical URL plus ambient
  generic/novel/application/libpq database credentials and an unrelated
  environment key.

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
  health URL, bounded/unbounded integration targets, and confirmed/unconfirmed
  database credential canaries; safe live probes use the current authenticated
  CLIs where available.
- Absent value probe: doctor and status report UNKNOWN/unavailable when a CLI,
  auth context, linked project, env file, or database is absent.
- Default-session/default-context probe: run from the dedicated worktree, where
  ignored environment/provider link files are absent, and from the shared root
  context via explicit discovery only; the live worktree resolves its shared
  root to `/home/juan-canfield/Desktop/Atlas`.
- Side-effect ordering: fixed inspection selection occurs before the database
  client receives SQL; the exact DSN remains in the child environment rather
  than argv; local full-unit execution is unavailable; Git
  output selects the canonical identifier before printing; env rendering
  extracts keys before any output is produced; integration target validation
  and exact database-URL cardinality occur before the integration child
  environment is built, and that isolated environment is passed explicitly to
  pytest; Docker daemon access is established before object inspection; health
  output selects a fixed label rather than the configured URL.

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
- `scripts/local_pr_review.sh`
- `tests/test_agent_operations_contract.py`
- `tests/test_local_pr_review.py`

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
- Selected dotenv decoding through the project interpreter is deferred as an
  availability improvement; the current head's explicit database commands
  remain read-only and the operator directed this pass to stop at blockers.
- Fail-closed handling for unreadable selected database configuration is
  deferred as read-only inspection hardening; it does not change the current
  hosted-only unit boundary or disposable integration credential authority.
- Brain health payload identity validation is deferred as non-blocking status
  accuracy hardening; no current-head evidence shows the configured endpoint is
  serving an unrelated 2xx response.
- Home Assistant and Wyze container enrollment in the aggregate status registry
  is deferred as non-blocking inventory completeness; their tracked Compose
  files remain directly inspectable.
- Shared Vercel link identity validation is deferred as non-blocking provider
  hardening; the current authenticated link was already verified during live
  archaeology and no project mutation is exposed.

Parking predicate: provider-specific conveniences, optional formatting, and
additional mutating operations that are not required to answer the operator's
fresh-agent questions are parked unless current verification proves they block
the contract or create a security/data-safety defect.

Parked hardening: project-interpreter dotenv decoding, unreadable selected
database-configuration rejection, Brain response identity, aggregate
Home Assistant/Wyze status enrollment, and shared Vercel link identity, as
listed above.

## Verification

- `python -m py_compile ops tests/test_agent_operations_contract.py` - pass.
- PyYAML `safe_load(.agent/capabilities.yaml)` - schema version `1` parsed.
- New dotenv/unit-isolation regression nodes failed before implementation with
  undecoded dotenv text and a missing sanitized child environment.
- The four new boundary regression groups failed before implementation with
  `10 failed`, covering missing dependency, configured-URL disclosure,
  unbounded integration targets, and the nonexistent capability command.
- The bare common-directory resolver regression failed before implementation
  with `1 failed, 2 passed`; only the successful direct-bare member was wrong.
- The integration credential-isolation regression failed before implementation
  in all 3 canonical cases because pytest received no explicit child
  environment.
- Focused-child database-authority regression failed before implementation with
  `KeyError: 'env'`, proving focused mode supplied no explicit child environment;
  it then passed after the shared removal helper was wired into focused mode.
- `./ops test focused tests/test_agent_operations_contract.py -q` - 41 passed;
  the frozen pass covers integration URL counts 0/1/2/3, database selection
  across explicit/worktree/shared/systemd contexts, both shared-root path
  shapes, Docker CLI/offline/daemon/object states, and single-credential
  integration subprocess isolation across all 3 canonical URL keys.
- Focused integration child-environment matrix - 3 passed; each canonical URL
  case removes current libpq credentials and the novel `PGFUTURE_CREDENTIAL`
  class member before pytest while preserving the unrelated canary.
- Focused hosted-only unit boundary nodes across
  `tests/test_agent_operations_contract.py` and `tests/test_local_pr_review.py`
  - 9 passed; selected/empty/full/failing/missing checker states never invoke
  the local unit checker, and `./ops test unit` never invokes pytest.
- `python -m py_compile ops tests/test_agent_operations_contract.py
  tests/test_local_pr_review.py`, `bash -n scripts/local_pr_review.sh`, hosted-only
  capability parsing, and `git diff --check` - pass.
- Direct live `ops` shared-root probe -
  `/home/juan-canfield/Desktop/Atlas` from the current linked worktree.
- `./ops test focused tests/test_eom_render_profile.py::test_database_config_prefers_connection_string_for_asyncpg_kwargs -q` - 1 passed.
- `./ops doctor` - pass; live systemd/Brain ping, Docker, Vercel, Render CLI,
  PostgreSQL fixed inspection, GitHub auth, and environment-source discovery
  reported without secret values.
- `./ops status` - pass after the daemon-first container change; the reachable
  daemon reported present and absent known objects separately.
- Live database-context path probe selected only the shared-root `.env` and
  `.env.local` pair from this unprovisioned worktree; `./ops db status` and
  `./ops db inspect connectivity` then passed without printing credentials.
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
  hosted-only boundary head will rerun the guarded push before publication;
  the local bundle can no longer launch pytest, and GitHub Actions owns the
  full unit gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 302 |
| `.agent/runbooks/ci.md` | 67 |
| `.agent/runbooks/database.md` | 126 |
| `.agent/runbooks/deployment.md` | 103 |
| `.agent/runbooks/discovery-ledger.md` | 321 |
| `.agent/runbooks/environment.md` | 109 |
| `.agent/runbooks/logs.md` | 74 |
| `.agent/runbooks/testing.md` | 107 |
| `AGENTS.md` | 17 |
| `CLAUDE.md` | 32 |
| `README.md` | 16 |
| `ops` | 905 |
| `plans/PR-Agent-Operations-Contract.md` | 641 |
| `scripts/local_pr_review.sh` | 132 |
| `tests/test_agent_operations_contract.py` | 723 |
| `tests/test_local_pr_review.py` | 131 |
| **Total** | **3806** |
