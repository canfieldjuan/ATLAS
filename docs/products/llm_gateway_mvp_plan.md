# LLM Gateway MVP — implementation plan

Status: **planning** (locked 2026-05-04). Implementation starts with PR-D1.
Sourced from session conversation 2026-05-04T20:00-21:00Z.

## Context

`extracted_llm_infrastructure` is the engine: 8-provider router, Anthropic batch wrapper, semantic cache, exact-match cache, FTL tracing, cost reconciliation, runtime budget gate. Today (2026-05-04) it became 100% standalone-operational (PR-A5a-d, PR-A6a-c). Atlas runs ~100k LLM calls/month through it.

The **product** wraps this engine in a hosted API service sold on a single-page landing page. Positioning: *cost-aware LLM API for production teams* (5-100k$/mo LLM spend, no platform team). Wedge: bundled cost-control surface (cache + batch + reconciliation + budget guards) on top of multi-provider routing — competitors do pieces, nobody does all of it integrated.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Deployment model | **Inside atlas's repo + deployment** (not separate service) | 2 weeks vs 8 weeks. Reuse auth/billing/Postgres. Extract later if scale or boundaries demand. |
| Customer auth model | **BYOK** (Bring Your Own Keys) | Customer pays providers directly. Cost savings show up on their existing Anthropic/OpenRouter bill — proves the wedge directly. No metered-billing complexity for MVP. |
| Atlas's billing model for the gateway | **Flat-tier subscription** ($X/mo per plan), no overage | MVP. Stripe metered billing deferred. |
| Atlas auth reuse | **JWT for dashboard + new API keys for production calls** | JWT (24h/30d) is wrong for production scripts. Add `api_keys` table on top of existing `saas_users`/`saas_accounts`. |
| Per-account scoping on shared tables | **Sentinel `account_id` for atlas's internal calls** (`00000000-...-0000`) | atlas's existing 100k calls/month don't know about accounts. Sentinel = "atlas internal" so existing pipelines keep their cache. New customer calls get isolated `account_id`. |

## Out of scope (explicitly)

- Metered usage billing (Stripe usage records). Add post-MVP after first paying customer.
- Separate service deployment / Docker image / pip-installable package.
- Mobile app or native clients.
- Provider keys management beyond Anthropic + OpenRouter for v1 (Together / Groq / Ollama added in v2 if requested).
- White-labeling / custom domains.
- SOC 2 / enterprise compliance certifications.
- LLM Provider passthrough for streaming embeddings — chat streaming yes, embed sync only for v1.

## Reuse from atlas (verified in 2026-05-04 audit)

- `atlas_brain/auth/dependencies.py` — `AuthUser`, `require_auth`, `require_plan`, `require_b2b_plan` (extend with `require_llm_plan`)
- `atlas_brain/auth/jwt.py` — JWT signing
- `atlas_brain/auth/passwords.py` — password hashing
- `atlas_brain/auth/rate_limit.py` — `Limiter` with plan-aware `_dynamic_limit` + per-account key. **Already extensible** — add `llm_starter / llm_growth / llm_pro` to `PLAN_RATE_LIMITS`.
- `atlas_brain/api/auth.py` — register / login / refresh / me / change-password / forgot-password / reset-password endpoints. `RegisterRequest` already accepts `product` field.
- `atlas_brain/api/billing.py` — Stripe checkout / portal / status / webhooks. `PLAN_LIMITS` dict + `PRICE_TO_PLAN` map. **Already extensible** — add `llm_*` plan price IDs in config + `PLAN_LIMITS`.
- `atlas_brain/storage/migrations/076_saas_accounts.sql` — `saas_accounts` (orgs) + `saas_users` (members) tables. `product` field already extensible via `RegisterRequest`.
- `atlas_brain/storage/migrations/127_llm_usage.sql` — usage table; PR-D3 adds `account_id` column.
- `extracted_llm_infrastructure/*` — the engine itself.

## PR sequence

### PR-D1: API key auth substrate (~2-3 days)

**Why first**: blocking for every other PR. Standalone work — no dependencies on atlas-side changes.

**Critical files (NEW)**:
- `atlas_brain/storage/migrations/312_api_keys.sql` — table:
  ```
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
  account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE
  user_id         UUID REFERENCES saas_users(id)             -- creator (audit only)
  name            VARCHAR(128) NOT NULL                       -- customer label
  key_prefix      VARCHAR(16) NOT NULL                        -- first 8 chars for lookup hint
  key_hash        VARCHAR(128) NOT NULL                       -- bcrypt or sha256+salt of full key
  scopes          TEXT[] NOT NULL DEFAULT '{llm:*}'           -- minimal scope model for v1
  last_used_at    TIMESTAMPTZ
  last_used_ip    INET
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
  revoked_at      TIMESTAMPTZ                                 -- soft-delete
  ```
  Index: `(account_id) WHERE revoked_at IS NULL`, `(key_prefix)`.
- `atlas_brain/auth/api_keys.py` — module: `generate_api_key() -> (raw, prefix, hash)`, `verify_api_key(raw) -> AuthUser | None`, `revoke_api_key(id, account_id)`.
- `atlas_brain/api/api_keys.py` — router `/api/v1/keys`:
  - `POST /` — create (returns raw key ONCE, never again)
  - `GET /` — list (no raw keys)
  - `DELETE /{id}` — revoke
- Extend `atlas_brain/auth/dependencies.py` — add `require_api_key()` that returns same `AuthUser` shape (so downstream code is auth-method-agnostic).

**Format**: `atls_live_<32-char-base32>` so customers can pattern-match in their secret-scanners.

**Tests**:
- Hash is one-way; key never persisted in raw form
- Lookup by prefix narrows hash check to 1-2 candidates
- Revoked keys reject with 401
- `require_api_key()` returns equivalent `AuthUser` shape to `require_auth`
- Cross-account scope: account A's key cannot read account B's data

**Validation**:
```
$ pytest tests/test_api_keys.py -q
$ curl -X POST localhost:8000/api/v1/keys -H "Authorization: Bearer $JWT" -d '{"name":"test"}'
# returns {"raw_key": "atls_live_xxxxx", "prefix": "atls_live", ...}
$ curl localhost:8000/api/v1/keys -H "Authorization: Bearer atls_live_xxxxx"
# returns 200 (key works for API auth)
```

### PR-D2: `llm_gateway` product + plan tier (~1 day)

**Critical files (EDIT)**:
- `atlas_brain/config.py` `SaaSAuthConfig`: add `stripe_llm_starter_price_id`, `stripe_llm_growth_price_id`, `stripe_llm_pro_price_id` fields.
- `atlas_brain/api/billing.py`: add `LLM_PLAN_LIMITS` dict (per-plan: `monthly_token_limit`, `cache_enabled`, `batch_enabled`, `byok_keys_max`); add `llm_starter / llm_growth / llm_pro` entries to `PRICE_TO_PLAN` init.
- `atlas_brain/auth/dependencies.py`: add `LLM_GATEWAY_PLAN_ORDER = ["llm_trial", "llm_starter", "llm_growth", "llm_pro"]` + `require_llm_plan()` helper mirroring `require_b2b_plan()`.
- `atlas_brain/auth/rate_limit.py` `PLAN_RATE_LIMITS`: add `llm_starter: "1000/hour"`, `llm_growth: "10000/hour"`, `llm_pro: "100000/hour"`.
- `atlas_brain/api/auth.py` `VALID_PRODUCTS`: add `"llm_gateway"`.

**Tests**: registration with `product=llm_gateway` succeeds; `require_llm_plan("llm_starter")` allows pro/growth/starter and blocks trial; `PLAN_RATE_LIMITS["llm_starter"]` returns the right rate string.

**Stripe products**: created in Stripe dashboard separately (manual ops step). Price IDs go in env vars, `_init_price_map()` picks them up.

**Out of scope here**: actual product flows. PR-D2 just adds the plumbing for plan-gating to work. The product flows arrive in PR-D4.

### PR-D3: Per-account scoping on shared Postgres tables (~2-3 days, RISKIEST)

**Why this is the riskiest**: atlas's existing pipeline writes to these tables 100k times/month. Breaking that pipeline = production incident. Sentinel account_id approach mitigates: atlas's calls all use `'00000000-0000-0000-0000-000000000000'`, customer calls use their real `account_id`, cache hits stay isolated.

**Critical files (EDIT)**:
- `atlas_brain/storage/migrations/313_llm_usage_account_scoping.sql` — `ALTER TABLE llm_usage ADD COLUMN account_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000'`. Index `(account_id, created_at DESC)`.
- `atlas_brain/storage/migrations/314_b2b_llm_exact_cache_account_scoping.sql` — same pattern on `b2b_llm_exact_cache`.
- `atlas_brain/storage/migrations/315_reasoning_semantic_cache_account_scoping.sql` — same on `reasoning_semantic_cache`.
- `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py` — add `account_id` to lookup/store function signatures with default = sentinel UUID; threaded into cache key composition.
- `extracted_llm_infrastructure/reasoning/semantic_cache.py` — same pattern; `_row_to_entry` filters on account_id.
- `extracted_llm_infrastructure/pipelines/llm.py::trace_llm_call` — accepts `account_id` (default sentinel); writes to `llm_usage`.
- `atlas_brain/services/tracing.py` (FTL trace emitter, if writes to llm_usage) — same.

**Backfill**: not needed — `DEFAULT '00000000-...-000000000000'` covers all existing rows + new atlas-internal rows.

**Tests**:
- Atlas's existing pipeline (sentinel account) cache hit ratio unchanged (regression check)
- Account A's cache hit cannot return account B's row even if cache key matches
- `llm_usage` rows from atlas's pipeline carry sentinel account_id
- `llm_usage` rows from a customer call carry the customer's `account_id`
- Cross-account isolation verified end-to-end (insert as A, query as B = empty)

**Rollback plan**: if atlas's pipeline breaks, the `DEFAULT` clause means we can drop the `WHERE account_id = $1` filters in the engine and atlas keeps working. Migration is non-destructive.

### PR-D4: `/api/v1/llm/*` FastAPI router (~3-5 days)

**Critical files (NEW)**:
- `atlas_brain/api/llm_gateway.py` — router `/api/v1/llm`:
  - `POST /chat` — sync chat completion. Body: `{provider, model, messages, max_tokens, temperature, ...}`. Threads `account_id` from `require_api_key()`. Wraps `pipelines.llm.call_llm` (or whichever public function we expose). Response includes provider's response + cache hit indicator + cost.
  - `POST /chat/stream` — SSE streaming. Same body shape.
  - `POST /batch` — Anthropic batch submission. Body: `{items: [{custom_id, messages, ...}], model}`. Returns batch_id.
  - `GET /batch/{batch_id}` — poll batch status; returns `submitted | processing | completed | failed` + results when done.
  - `POST /embed` — sync only for v1. Body: `{provider, model, input}`.
  - `GET /usage` — current period token + cost spend, by provider.
- `atlas_brain/api/llm_gateway_schemas.py` — Pydantic request/response schemas.
- BYOK key resolution: when handling each request, look up the customer's stored provider keys (from PR-D5's `byok_keys` table) and inject into the engine's per-call SDK clients.

**Reuses**:
- `Depends(require_api_key)` from PR-D1
- `Depends(require_llm_plan("llm_starter"))` from PR-D2
- Engine: `extracted_llm_infrastructure.pipelines.llm`, `extracted_llm_infrastructure.services.b2b.anthropic_batch`
- Rate limiter from `auth/rate_limit.py`

**Tests**:
- End-to-end: API key auth → engine call → response shape
- Plan gating: `llm_trial` blocked from `/batch` if `LLM_PLAN_LIMITS[trial].batch_enabled=False`
- Per-account cost tracking — request as account A increments A's `llm_usage`, not B's
- Rate limiting kicks in per plan
- `/usage` returns correct per-provider breakdown for the calling account

**Smoke test**: standalone curl against the running atlas service — chat → batch → usage round-trip.

### PR-D5: BYOK dashboard + customer key management (~3-5 days)

**Critical files (NEW)**:
- `atlas_brain/storage/migrations/316_byok_keys.sql` — table:
  ```
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
  account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE
  provider        VARCHAR(32) NOT NULL                  -- anthropic | openrouter | together | groq
  encrypted_key   BYTEA NOT NULL                        -- Fernet-encrypted
  encryption_kid  VARCHAR(64) NOT NULL                  -- key ID for rotation
  key_prefix      VARCHAR(16) NOT NULL                  -- first 8 chars (display only)
  added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
  last_used_at    TIMESTAMPTZ
  ```
- `atlas_brain/auth/encryption.py` — Fernet-based encrypt/decrypt with `ATLAS_BYOK_ENCRYPTION_KEY` env var; supports key rotation via `encryption_kid`.
- `atlas_brain/api/byok_keys.py` — router `/api/v1/byok-keys`:
  - `POST /` — add provider key (encrypts before insert)
  - `GET /` — list (returns prefix only, never raw)
  - `DELETE /{id}` — revoke
- BYOK key resolver — used by PR-D4's router to look up the customer's keys per provider.
- Customer dashboard surface — TBD between (a) extending existing `atlas-intel-ui` consumer dashboard with a `/llm-gateway` section, (b) new minimal Jinja-template HTML at `/portal/llm-gateway`. Decision deferred — both are 2-3 days.

**Tests**:
- BYOK key encrypted at rest (DB row not human-readable)
- Encryption roundtrip: encrypt → store → fetch → decrypt = original
- Key rotation: old `encryption_kid` still decrypts when KEK still in env; new keys use latest KID
- Provider key lookup correctly resolves the right key per provider per account
- API key cannot be returned in plaintext after creation (only `key_prefix` exposed)

**Open decision (defer to during PR-D5)**: dashboard surface. Lean toward (b) minimal HTML for MVP — gets the customer flow working without entangling with the existing UI codebase.

## Open decisions (small, deferred until that PR)

1. **Customer dashboard surface** (PR-D5) — extend existing UI vs new minimal Jinja route. Lean: minimal Jinja for MVP.
2. **API key format** (PR-D1) — `atls_live_<32-base32>` recommended; user could prefer `sk-atlas-...` or similar.
3. **API key scopes** (PR-D1) — single `llm:*` scope for v1, expand to `llm:chat / llm:batch / llm:embed / keys:read` later if customers ask.
4. **Encryption layer** (PR-D5) — Fernet with env-var KEK. Atlas may already have one; if so, reuse. Verified during PR-D5.
5. **Streaming embed support** (PR-D4) — out of scope for v1; chat streaming yes, embed sync only.
6. **Stripe products** — created manually in Stripe dashboard before PR-D2 ships (operational dependency).

## 18-rule production protocol

Each PR follows the same 4-phase protocol used today on PR-A5/PR-A6:

- **Phase 1 (Discovery)**: file paths + line numbers verified before each edit.
- **Phase 2 (Pre-mod validation)**: ASCII-clean Python files (`grep -nP "[^\x00-\x7F]"`); verify `Edit` anchor uniqueness; check no hard-coded values introduced.
- **Phase 3 (Implementation)**: atomic, no placeholders/TODOs/stubs in production logic; existing `Any` types stay `Any`; preserve public function signatures.
- **Phase 4 (Post-mod validation)**: targeted tests + smoke checks per PR; sync invariants where applicable; full driver script.

## Verification per PR (end-to-end)

- **PR-D1**: create key via JWT, hit `/keys` with key, verify auth works; revoke key, verify 401.
- **PR-D2**: register with `product=llm_gateway`, query `/auth/me`, verify returned plan tier; rate-limit kicks in at `llm_starter` rate.
- **PR-D3**: atlas pipeline runs through (no regression); insert as account A, query as B = isolated.
- **PR-D4**: `curl -H "Authorization: Bearer atls_live_..."` against `/api/v1/llm/chat` returns provider response; usage row written; cache hit on second identical call.
- **PR-D5**: add Anthropic key in dashboard; chat call routes through that key (verify with mock provider key); revoke key, chat call 401s.

## Risk + rollback

| Risk | Likelihood | Mitigation |
|---|---|---|
| PR-D3 breaks atlas's pipeline | Medium | Sentinel `DEFAULT` covers existing rows; migration is non-destructive; can roll back column add. |
| BYOK key leak via logs | Low | Encryption at rest; never log raw provider keys; redact in error messages. |
| API key brute force | Low | Hashed at rest; rate-limited per IP; key prefix narrows attack surface. |
| Customer's BYOK key revoked at provider | Medium | Detect 401 from provider, surface to customer dashboard, fail-fast. |
| Rate limiter scoping wrong (per-IP instead of per-key) | Low | `_key_func` already encodes account; verified in PR-D2. |

## Logging (memory + repo)

- Memory: new `~/.claude/projects/.../memory/project_llm_gateway_mvp_plan.md` with summary + link to this doc + decision log.
- Repo: this doc at `docs/products/llm_gateway_mvp_plan.md` (this file).
- Each PR description references this plan with anchors.
