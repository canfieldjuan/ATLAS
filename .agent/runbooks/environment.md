# Environment and local runtime

Use this runbook to orient a new worktree, find configuration without exposing
it, and determine whether the Brain is already running.

## First five minutes

From a normal Atlas worktree:

```bash
./ops doctor
./ops status
./ops env keys
```

Read `.agent/capabilities.yaml` after the snapshot. It distinguishes verified,
conditional, and unavailable operations. Do not use the old README/CLAUDE
commands as evidence when they disagree with `./ops` or executable config.

## Repository and worktree layout

This checkout uses a shared Git root. The path commonly called
`/home/juan-canfield/Desktop/Atlas` can be the common/bare root even though it
contains files and ignored runtime state. `git status` can therefore fail there.
Use a dedicated path listed by `git worktree list` for edits and tests.

Ignored operational state such as `.env`, `.venv`, and `.vercel/project.json`
may exist only in the shared root. A fresh worktree does not automatically get
those files. `./ops` resolves the Git common directory so its read-only checks
can find shared state without copying it into the branch.

## Configuration sources

Atlas settings read exported variables first and `.env` / `.env.local` in the
process working directory otherwise. The live `atlas-api.service` injects
absolute `EnvironmentFiles`, so its configuration is independent of the
detached runtime worktree.

Inspect names, never values:

```bash
./ops env keys
./ops env keys --file /path/to/a/specific.env
./ops env systemd
./ops env vercel
```

`./ops env keys` parses assignment names without sourcing or evaluating the
file. `./ops env vercel` must run through the linked shared-root context; it
lists encrypted variable names and target environments, not values.

Never use `cat`, `env`, `set`, `vercel env pull`, or debug tracing merely to
learn whether a secret exists. Never copy the shared live `.env` into a
worktree. A root `.env.example` does not exist; subsystem examples live under
`atlas-ui/`, `atlas_video-processing/`, and `graphiti-wrapper/`.

## Running the Brain

The code entrypoint is `atlas_brain.main:app`. A direct development invocation
is:

```bash
python -m uvicorn atlas_brain.main:app \
  --host 127.0.0.1 --port 8001 \
  --reload --reload-dir atlas_brain \
  --reload-exclude data/postgres --reload-exclude 'data/postgres/**'
```

This is not a harmless smoke command. The FastAPI lifespan initializes the
database, runs the custom migration check, and can start configured background
services. Before starting it, use a deliberately prepared development
environment and read `.agent/runbooks/database.md`. Do not load the shared
production `.env` into an ad-hoc server.

The current production-shaped Brain is the user unit `atlas-api.service`, not a
Render service. Inspect it without restarting it:

```bash
./ops deploy status
systemctl --user show atlas-api \
  --property=LoadState,ActiveState,SubState,WorkingDirectory,EnvironmentFiles
curl --fail --silent http://127.0.0.1:8012/api/v1/ping
```

## Containers and local dependencies

Docker Engine and the Compose plugin are available. Root Compose packages the
Brain and NocoDB but expects native PostgreSQL on host port `5433`; it does not
define a `postgres` service. Graphiti/Neo4j use
`docker-compose.graphiti.yml`.

Even `docker compose ps` against the root file fails unless
`ATLAS_NOCODB_DB_PASSWORD` is present because Compose evaluates the NocoDB
required-variable expression before selecting a service. For read-only status,
use `./ops status` or targeted `docker inspect` rather than weakening that
guard.

## Failure routing

- `git status` says “must be run in a work tree”: run `git worktree list` and
  move to the task's dedicated worktree.
- A worktree lacks `.venv`: `./ops` uses the shared project venv when present;
  set `ATLAS_PYTHON` to another verified interpreter if necessary.
- Port `8012` is down: inspect `./ops deploy status` and `./ops logs brain`;
  do not restart until deployment and migration impact are understood.
- Ollama CLI exists but the server is unreachable: treat local-model features
  as degraded; do not install or reconfigure Ollama as an orientation step.
- A provider variable name exists: that proves configuration shape only, not
  authentication or live provider health. Use the provider-specific safe check.
