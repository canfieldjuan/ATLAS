# Logs

Log access is read-only but log contents can contain customer identifiers,
request payload fragments, provider IDs, or error context. Retrieve the
smallest useful window and do not paste raw logs into issues or chat.

## Brain API

The production-shaped Brain runs as the user service `atlas-api.service`, so
journald is canonical:

```bash
./ops logs brain
./ops logs brain --follow
journalctl --user -u atlas-api --since '15 minutes ago' --no-pager
```

The default wrapper returns the last 100 lines. Use `--follow` only during an
active, bounded observation; do not leave a polling process running.

## Containers

Use Docker logs by exact container name:

```bash
./ops logs container atlas-graphiti-wrapper
./ops logs container atlas-neo4j
./ops logs container atlas_nocodb
```

Targeted `docker logs` avoids the root Compose parsing prerequisite that makes
even `docker compose ps/logs` fail when `ATLAS_NOCODB_DB_PASSWORD` is absent.
Use `./ops status` to discover which known containers exist before reading.

## Vercel frontend

```bash
./ops logs frontend --since 1h --limit 100
./ops logs frontend --since 30m --limit 100 --follow
```

The root deployment is a static Vite site, so no runtime records can be a valid
result. For build failure, use the deployment URL from `./ops deploy status`:

```bash
vercel inspect <deployment-url> --logs
```

Do not broaden a search to all account projects. The Atlas root project is
`atlas-churn-ui`.

## GitHub Actions

```bash
./ops ci status
./ops ci run <run-id> --log-failed
```

Always match the run's branch and head SHA to the code under investigation.
Workflow names recur across pushes, and a green older run is not current proof.

## Failure routing

- Journal access denied: verify the current user owns the user unit; do not use
  `sudo journalctl` as an automatic bypass.
- Container absent: inspect `docker ps` and the relevant Compose file; do not
  start it just to make logs exist.
- Vercel logs fail from a worktree: the ignored project link is shared-root
  state; use `./ops`, which supplies the project and correct working directory.
- GitHub logs show no job steps: treat runner/infrastructure allocation as a
  separate failure class before changing source.
- A log contains a secret or customer data: stop copying it, preserve only a
  redacted evidence reference, and follow `docs/INCIDENT_RESPONSE.md` if
  exposure is plausible.
