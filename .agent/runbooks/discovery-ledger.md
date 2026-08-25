# Operational discovery ledger

Only reusable discoveries belong here. Current values such as runtime SHAs and
deployment IDs stay in live commands, not in this ledger.

## Git worktree identity

Problem:
A fresh agent can enter `/home/juan-canfield/Desktop/Atlas`, see repository
files, and assume it is a normal worktree.

Finding:
The shared root can be registered as bare/common state and reject `git status`.
Dedicated worktrees under the repository's worktree locations are the editable
contexts. Ignored `.env`, `.venv`, and `.vercel` state can remain at the shared
root.

Canonical method:
Run `./ops doctor`, `git worktree list`, and work from the task's dedicated
worktree. `./ops` resolves the Git common directory for read-only shared state.

Verified:
2026-08-24: `git status` failed at the shared root; a fresh worktree from
`origin/main` was clean and operational.

Failure notes:
Do not repair the shared root with reset/checkout/clean. Do not copy ignored
secrets into the worktree.

## Brain deployment

Problem:
The installed Render CLI and checked-in Dockerfile make several deployment
stories look plausible.

Finding:
The running Brain is `atlas-api.service`, a user systemd service whose
`WorkingDirectory` points to a detached runtime worktree. Its ping route is on
local port `8012`. Render's authenticated workspace had no Atlas Brain service.

Canonical method:
Use `./ops deploy status`, `./ops logs brain`, and the live systemd unit. For an
authorized cutover, follow `.agent/runbooks/deployment.md` and the applicable
product runbook.

Verified:
2026-08-24: systemd reported the unit active, the runtime worktree SHA resolved,
and `GET /api/v1/ping` returned success.

Failure notes:
Do not infer deployment from a merged PR or from provider CLI availability.
Restarting can run migrations.

## Frontend deployment and Vercel linkage

Problem:
Vercel account authentication works in every worktree, but project-scoped env
commands can still fail.

Finding:
The linked project is `atlas-churn-ui`; root `vercel.json` builds that package.
The ignored `.vercel/project.json` link exists in the shared root, not fresh
worktrees. A production-target READY deployment exists. No tracked GitHub
workflow deploys it, and the inspected deployment had no Git metadata, so its
trigger is not provable from this repository.

Canonical method:
Use `./ops deploy status`, `./ops env vercel`, and `./ops logs frontend`.

Verified:
2026-08-24: Vercel whoami/project/deployment/env/log access all succeeded when
the correct linked context or explicit project was supplied.

Failure notes:
Do not run `vercel link`, `vercel --prod`, deploy hooks, or alias changes during
discovery. An empty runtime log window is normal for a static deployment.

## PostgreSQL location and safe access

Problem:
README and CLAUDE instructions tell agents to start a Compose `postgres`
service, suggesting the database is container-owned.

Finding:
Root Compose defines no PostgreSQL service. Atlas uses native PostgreSQL 16 on
port `5433`; the current session can run fixed read-only connectivity and
migration-ledger inspections through Atlas's application configuration path.

Canonical method:
Use `./ops db status`, `./ops db inspect connectivity`, and
`./ops db migrations`. Arbitrary SQL is intentionally unavailable until Atlas
has a privilege-restricted inspection role; transaction read-only mode alone
does not prevent side effects from invoked PostgreSQL functions.

Verified:
2026-08-24: `./ops db inspect connectivity` and `./ops db migrations` succeeded
through `DatabaseConfig`/asyncpg with fixed SQL inside `READ ONLY` transactions.

Failure notes:
Do not substitute a random `5432` instance, use live `atlas` for integration
tests, or use Alembic. Application startup invokes the custom migration runner.

## Root Compose prerequisites

Problem:
Even a read-only-looking `docker compose ps` can fail before inspecting Docker.

Finding:
The root Compose file requires `ATLAS_NOCODB_DB_PASSWORD` during interpolation,
even when the intended operation is status or another service.

Canonical method:
Use `./ops status` and targeted `docker inspect` / `./ops logs container` for
discovery. Set the NocoDB credential only for an authorized Compose operation.

Verified:
2026-08-24: Docker daemon access succeeded while root `docker compose ps`
failed at NocoDB interpolation with the variable absent.

Failure notes:
Do not weaken the required-variable expression merely to make status work.

## Database migrations and test databases

Problem:
An installed `alembic` CLI and generic integration marker do not reveal Atlas's
actual schema/test workflow.

Finding:
Atlas uses a custom SQL migration runner and `schema_migrations`. The full chain
is not fresh-applicable because later files depend on out-of-band
`product_metadata`. Database workflows create disposable PostgreSQL 16 services
and pass test-specific URL variables.

Canonical method:
Inspect `atlas_brain/storage/migrations/__init__.py`, use
`.agent/runbooks/database.md`, and prefer the matching GitHub Actions workflow
unless a local disposable database is explicit.

Verified:
2026-08-24: runner code, startup call path, workflow service definitions, test
gates, and the live ledger were inspected.

Failure notes:
A skipped DB test is not proof. Never apply the full chain to a fresh target as
an exploratory action.

## Logs and CI

Problem:
Agents can chase file logs, Compose logs, or stale GitHub runs without knowing
which stream is canonical.

Finding:
Brain logs are in the user journal; container logs are in Docker; static UI
runtime/build logs are in Vercel; CI is GitHub Actions with enforcement recorded
in `ci/gates.yml`.

Canonical method:
Use `./ops logs ...`, `./ops ci status`, and `./ops ci run <id> --log-failed`.

Verified:
2026-08-24: journal metadata, Vercel log access, GitHub workflow inventory, and
current run inspection succeeded.

Failure notes:
Logs may contain sensitive data. Always match CI branch/head SHA, and treat
runnerless jobs separately from source-test failures.

## Available tooling

Problem:
Agents waste time proposing installs before checking the workstation.

Finding:
Git/GitHub, Python/project venv, Node/npm, Docker, PostgreSQL clients, Vercel,
Render, systemd/journal, curl, jq, uv, pnpm, yarn, and Alembic are present.
Legacy `docker-compose` and the Fly/Railway/major-cloud/Terraform/Kubernetes
CLIs were not found. Ollama CLI exists but its server was unavailable.

Canonical method:
Run `./ops doctor` and consult `.agent/capabilities.yaml`; install nothing as an
orientation step.

Verified:
2026-08-24: command-path/version checks plus authenticated provider probes.

Failure notes:
Installed/authenticated does not mean the tool/provider belongs to Atlas.

## Dotenv database settings and unit-test isolation

Problem:
Hand-parsing selected `.env` assignments changes quoting, inline-comment,
escape, and interpolation behavior, while a nominal unit command can inherit
database URLs left exported after integration work.

Finding:
`DatabaseConfig` uses Pydantic settings backed by `python-dotenv`. Database
inspection must decode selected values with that same parser before applying
process-environment precedence. Several unmarked PostgreSQL tests activate from
`DATABASE_URL` or `*_DATABASE_URL`, so pytest markers alone do not isolate unit
mode from database writes.

Canonical method:
Use `./ops db inspect ...` for application-equivalent dotenv decoding. Use
`./ops test unit` for database-credential-free unit execution and
`./ops test integration ...` only with an explicitly confirmed disposable
database.

Verified:
2026-08-24: focused boundary tests proved dotenv comments, escapes,
interpolation, and process overrides, plus removal of current and novel
database-URL-shaped variables from unit-mode children.

Failure notes:
Do not replace dotenv parsing with shell sourcing, and do not assume
`-m "not integration"` excludes every database-writing test. Keep database
URLs out of unit-mode subprocesses by construction.

## Fresh-agent wrapper boundaries

Problem:
A fresh checkout is told to run `./ops doctor` before installing Python
dependencies, configured health endpoints can contain credentials, and pytest
falls back to broad collection when integration arguments do not name a file.

Finding:
Orientation must degrade an unavailable database probe without weakening
explicit database commands. Health checks may consume a configured URL but must
never echo it. Integration mode needs both disposable-database confirmation and
a bounded file/node target; argument count alone is not a test boundary.

Canonical method:
Run `./ops doctor` before installation and treat an unavailable database line as
an expected partial result. Run integration tests only as
`./ops test integration tests/test_file.py[::node] [pytest-options]`; use
`--option=value` for valued options.

Verified:
2026-08-24: focused boundary tests simulated missing `python-dotenv`, a
credential-bearing configured health URL, option-only/directory/out-of-tree
integration inputs, and valid file/node targets.

Failure notes:
Do not make `doctor` import application dependencies merely to orient a new
agent. Do not print `ATLAS_OPS_BRAIN_URL`, even after a successful request. Do
not bypass the bounded-target guard with direct pytest when database credentials
are present.
