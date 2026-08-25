# Deployment and status

Atlas has multiple deployment surfaces. Never answer “where is Atlas deployed?”
with one provider name.

| Surface | Actual provider | Production-shaped target | Preview/staging |
|---|---|---|---|
| Brain API | local user `systemd` | `atlas-api.service`, detached runtime worktree, port `8012` | no separate backend staging service discovered |
| Churn UI | Vercel | project `atlas-churn-ui`, root `vercel.json` | Vercel preview deployments; no named staging environment discovered |
| Graph memory | Docker | `atlas-neo4j` + `atlas-graphiti-wrapper` | local containers only |
| CRM browser | Docker | `atlas_nocodb` | local container only |

Render CLI is authenticated in this machine's current workspace, but the live
service inventory contains other products and no Atlas Brain service. Do not
use Render for Atlas merely because the CLI is present.

## Read-only status

```bash
./ops deploy status
./ops status
curl --fail --silent http://127.0.0.1:8012/api/v1/ping
```

`./ops deploy status` reads the systemd unit's current working directory and
resolves that detached worktree's Git SHA. It also discovers and inspects the
latest Vercel deployment. Re-run it immediately before any deployment claim;
the runtime worktree and hosted deployment can drift independently from
`origin/main`.

For deeper, still read-only inspection:

```bash
systemctl --user show atlas-api \
  --property=LoadState,ActiveState,SubState,WorkingDirectory,EnvironmentFiles
vercel project inspect atlas-churn-ui
vercel ls atlas-churn-ui
./ops logs brain
./ops logs frontend --since 1h
```

Vercel's ignored `.vercel/project.json` link exists in the shared Git root, not
automatically in each worktree. `./ops` resolves that context. A direct
`vercel env ls` from an unlinked worktree fails even when the account is logged
in; use `./ops env vercel`.

## How Brain deployment works

The checked-in source of truth is GitHub, but the running Brain does not deploy
automatically from a merge. The established production path is a guarded local
runtime-worktree cutover:

1. Fetch and identify the exact merged commit from `origin/main`.
2. Create a fresh detached runtime worktree at that exact commit.
3. Run the slice's targeted checks in that worktree.
4. Preserve the user unit, then repoint only its `WorkingDirectory` to the new
   runtime worktree.
5. Run `systemctl --user daemon-reload` and restart `atlas-api.service`.
6. Verify the ping/health route, the runtime worktree SHA, migration status,
   and service logs.
7. Keep the prior runtime worktree until rollback risk is closed.

Steps 2-6 mutate production or local runtime state. They require an owned
deployment arc and explicit operator authority; they are intentionally not
wrapped by `./ops`. For the EOM API's exact health/auth/fence checks and
rollback order, use `docs/EOM_FUNNEL_ENABLEMENT_RUNBOOK.md`.

Starting or restarting the full app can apply pending migrations. A code
rollback does not imply a migration rollback, and destructive schema rollback
is not an orientation operation. Read `.agent/runbooks/database.md` first.

## How frontend deployment works

The root `vercel.json` installs and builds `atlas-churn-ui`, producing
`atlas-churn-ui/dist`. A live production-target deployment was verified through
the Vercel CLI. No tracked GitHub Actions workflow invokes Vercel deployment,
and the inspected live deployment did not carry Git commit metadata, so the
current trigger cannot be proven from this repository alone.

`vercel --prod` from the linked root is the provider's direct production
mutation, but it was not run during contract verification. Do not invoke it,
fire a deploy hook, or change aliases merely to prove access. Confirm the exact
project, diff, build, intended target, and operator authorization first.

Other UI directories have their own `vercel.json` or package configuration and
may belong to separate repositories/projects. The root Vercel link does not
prove those surfaces deploy with `atlas-churn-ui`.

## Rollback and failure routing

- Brain service active but ping fails: inspect `./ops logs brain`, migration
  fences, and the unit's exact working directory before restarting again.
- Runtime SHA differs from `origin/main`: report both; do not silently repoint
  production because a newer commit exists.
- Vercel project inspection works but env listing fails: use the linked shared
  root through `./ops env vercel`.
- Vercel shows `ERROR`: inspect build logs with
  `vercel inspect <deployment-url> --logs`; do not create another deployment as
  a diagnostic retry.
- Static Vercel runtime logs are empty: this can be normal. Inspect deployment
  and build status instead.
- A provider command asks to link/create/import a project: stop. Discovery must
  not provision or reconfigure infrastructure.
