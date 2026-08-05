# EOM Funnel Enablement Runbook (#2254, arc #2275 slice S0)

Operator + DBA procedure to bring the private EOM funnel live on the **ts.net Atlas host** (full `atlas_brain.main:app`). Proven end-to-end on 2026-08-05. The Render slim `main_eom` funnel stays dormant (canonical DB = ts.net primary); this runbook does **not** cover it.

## What the boot gate requires

`atlas_brain/eom_api/funnel_store.py::require_eom_funnel_data_store` runs when `ATLAS_EOM_FUNNEL_API_ENABLED=true` and fails closed unless the DB has migrations **353–361** applied **and** a role contract: `atlas_eom_handoff_owner` owns `eom_customer_handoffs` + holds schema `CREATE`; the app login is **not** a member of it; `atlas_nocodb` is read/write on only `contacts`/`contact_interactions`/`appointments`. Migration `354` establishes that contract but aborts unless a DBA pre-provisions roles (below). "Apply migration 353" alone is **not** sufficient.

## Prerequisites (verify first)

- The `atlas-api` unit runs the **full** app: `grep ExecStart ~/.config/systemd/user/atlas-api.service` shows `uvicorn atlas_brain.main:app`. The slim `main_eom` applies only receivables migrations and can never satisfy the gate.
- The runtime is on current `origin/main` (has `funnel_store.py` + migrations 356–361). If it is behind, advance it — a **fresh-worktree cutover** is the safe form: `git -C <atlas> worktree add --detach worktrees/atlas-runtime-main origin/main`, repoint the unit's `WorkingDirectory` to it (keep a `.bak`), `systemctl --user daemon-reload`. Rollback = repoint to the old worktree + restart.
  - **Env carries automatically:** the unit's `EnvironmentFile` is an **absolute** path to the main repo's `.env` (e.g. `EnvironmentFile=-/home/<user>/Desktop/Atlas/.env`), so the fresh worktree needs **no** `.env` copy — systemd injects the process environment regardless of `WorkingDirectory`. Confirm the `EnvironmentFile` path is absolute (not worktree-relative) before cutting over.
- A superuser DB session is reachable: on this host, `psql -U postgres -d atlas -h localhost -p 5433` connects as superuser (no external DBA needed).

## Phase 1 — DBA role bootstrap (privileged psql)

```sql
-- psql -U postgres -d atlas -h localhost -p 5433
CREATE ROLE atlas_nocodb LOGIN NOINHERIT PASSWORD '<nocodb password>';   -- 354 will NOT create this
GRANT CONNECT ON DATABASE atlas TO atlas_nocodb;                         -- explicit: the gate checks has_database_privilege(...,'CONNECT'); needed where CONNECT is revoked from PUBLIC
CREATE ROLE atlas_eom_handoff_owner NOLOGIN NOINHERIT;
GRANT atlas_eom_handoff_owner TO <app-login> WITH ADMIN OPTION;          -- temporary; lets 354's ownership transfer run as the app login
```

**354 precondition gotcha:** it checks `nspowner = executor` *literally*. If `public` is owned by `pg_database_owner` (PG14+ default), 354 aborts even though the app login is the DB owner. Fix once: `ALTER SCHEMA public OWNER TO <app-login>;` (benign — the app login is already the DB owner; reversible via `OWNER TO pg_database_owner`). It also requires the executor to own every table + the two protected handoff functions (normally true).

## Phase 2 — Apply migrations (auto; keep the funnel disabled)

`systemctl --user restart atlas-api`. The full-app lifespan runs `run_migrations()` → applies 353–361 (354 now passes). Verify:

```sql
SELECT version FROM schema_migrations WHERE version BETWEEN 353 AND 361;   -- expect all nine
SELECT pg_get_userbyid(relowner) FROM pg_class WHERE relname='eom_customer_handoffs';  -- atlas_eom_handoff_owner
```
Migration failures are logged as warnings (not fatal) — **confirm 354 committed**, don't assume.

## Phase 3 — Post-commit revoke (privileged psql — the easily-missed step)

```sql
REVOKE atlas_eom_handoff_owner FROM <app-login>;   -- only the grantor can; gate's non-membership clause is false until this runs
```

## Phase 3b — Restore app DML on the handoff table (see #2286)

Migration 354's "preserve Atlas DML before ownership moves" self-grant does **not** survive the ownership transfer, so after Phase 2 the app login has **zero** privileges on `eom_customer_handoffs` (would 500 the handoff write path and `/leads`). Until #2286 fixes the migration, restore it:

```sql
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON eom_customer_handoffs TO <app-login>;
```
This does not affect the gate (which audits `atlas_nocodb` + ownership + non-membership).

## Phase 4 — Prove the gate passes as the app login (before enabling)

As the app login, confirm: not a member of `atlas_eom_handoff_owner`; handoff owned by the guard; `atlas_nocodb` has no INSERT on handoff/lifecycle nor write on `contacts.lead_stage`; `eom_onboarding_email_drafts.approved_by_employee_id` is `bigint`; and the app login can now SELECT/INSERT the handoff table (Phase 3b). All true → safe to enable.

## Phase 5 — Enable (Atlas runtime)

```bash
cd worktrees/atlas-runtime-main
python -c "from atlas_brain.eom_api.funnel_auth import generate_eom_funnel_service_token as g; import pathlib; t=g(); p=pathlib.Path.home()/'.config/atlas/eom-funnel.token'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(t.token); p.chmod(0o600); print(t.sha256)"
# add to Atlas/.env:  ATLAS_EOM_FUNNEL_API_ENABLED=true  and  ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256=<printed digest>
systemctl --user restart atlas-api
```
The raw `eomf_v1_...` token lands in `~/.config/atlas/eom-funnel.token` (0600) — never commit/paste it. Smoke (needs actor headers):
```bash
TOKEN=$(cat ~/.config/atlas/eom-funnel.token)
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOKEN" -H 'X-EOM-Actor: office' -H 'X-EOM-Actor-ID: 1' http://127.0.0.1:8012/api/v1/eom-funnel/leads   # 200
curl -s -o /dev/null -w '%{http_code}\n' -H 'Authorization: Bearer wrong' http://127.0.0.1:8012/api/v1/eom-funnel/leads   # 401
```

## Phase 6 — Tracker (Render) wiring

The Render CLI v2 cannot set env vars (no `env` command; `services update` has no `--env` flag), so use the REST API. Set on the `eom-timetracker` service:
- `ATLAS_FUNNEL_BASE_URL = https://atlas-brain.tailc7bd29.ts.net/api/v1`
- `ATLAS_FUNNEL_SERVICE_TOKEN = <raw token>`

```bash
API=https://api.render.com/v1/services/<serviceId>
curl -s -X PUT -H "Authorization: Bearer $RENDER_TOKEN" -H 'Content-Type: application/json' \
  -d '{"value":"https://atlas-brain.tailc7bd29.ts.net/api/v1"}' "$API/env-vars/ATLAS_FUNNEL_BASE_URL"
python3 -c "import json,os;print(json.dumps({'value':os.environ['FT']}))" | \
  curl -s -X PUT -H "Authorization: Bearer $RENDER_TOKEN" -H 'Content-Type: application/json' --data-binary @- "$API/env-vars/ATLAS_FUNNEL_SERVICE_TOKEN"   # FT=raw token
curl -s -X POST -H "Authorization: Bearer $RENDER_TOKEN" -H 'Content-Type: application/json' -d '{}' "$API/deploys"   # redeploy (JSON content-type required)
```
Then: admin login → portal Leads tab loads live; `/api/admin/funnel/review` returns 200 with admin creds (401 unauthenticated = wired). If the Render→ts.net call times out while a direct curl to Atlas works, check Tailscale Funnel/serve exposure.

## Rollback

`ATLAS_EOM_FUNNEL_API_ENABLED=false` + `systemctl --user restart atlas-api` — fail-closed by design; tracker degrades to prior behavior. To revert the runtime, repoint the unit `WorkingDirectory` to the previous worktree + restart.

## Failure modes

- **Slim app entrypoint** → 354/360/361 never apply; funnel can't be enabled there.
- **Skipped Phase 1** → 354 aborts (warned, not fatal); app boots but enabling later fails the boot gate.
- **Skipped Phase 3** → non-membership clause false; funnel stays fail-closed after a "successful" migration.
- **Skipped Phase 3b (#2286)** → app login has no handoff privileges; handoff write path and `/leads` 500.
