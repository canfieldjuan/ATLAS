# PR-D6f: `Cache-Control: no-store` request header bypass

## Why this slice exists

Item #4 in the post-D6b LLM Gateway follow-up queue (locked
2026-05-05). Customer-supplied bypass for sensitive prompts they
don't want cached. Standard HTTP caching convention (RFC 7234).

> **Touches**: `chat` route accepts an optional `Cache-Control`
> header (FastAPI `Header(default=None)`); when set to `no-store`,
> skip both lookup and store.
> **Success criterion**: a request with `Cache-Control: no-store`
> always calls Anthropic and never writes to the cache, even when
> the gateway-cache flag is enabled.

PR-D6e (item #1) shipped the `cached: bool` field on `ChatResponse`
so customers can detect cache hits explicitly. This PR is the
counterpart: the customer's tool to **suppress** caching when they
need to.

## Scope (this PR)

Three changes to `atlas_brain/api/llm_gateway.py`:

1. New helper `_cache_control_disables_cache(cache_control)` parses
   the comma-separated directives and returns True iff `no-store` is
   present (case-insensitive, with or without a value suffix).
2. `chat()` accepts an optional `Cache-Control` request header via
   FastAPI's `Header(default=None, alias="Cache-Control")`.
3. Both the exact-cache lookup branch and the post-call store branch
   skip when `cache_disabled` is True. The lookup is gated alongside
   the existing feature-flag check; the store gets the same guard.

The cache-disabled path falls through to a normal Anthropic call,
which already returns `cached=False` (PR-D6e). No additional response
shape change.

## Header parsing

```python
def _cache_control_disables_cache(cache_control: str | None) -> bool:
    if not cache_control:
        return False
    for directive in cache_control.split(","):
        token = directive.strip().lower()
        token = token.split("=", 1)[0].strip()
        if token == "no-store":
            return True
    return False
```

Handles RFC 7234 reality:

- `Cache-Control: no-store` -> disable.
- `Cache-Control: no-cache, no-store` -> disable (both directives,
  one matches).
- `Cache-Control: NO-STORE` -> disable (case-insensitive).
- `Cache-Control: max-age=0` -> not disable (different directive).
- `Cache-Control: maybe-no-store` -> not disable (substring match
  would be wrong; we tokenize on commas + equals).
- `Cache-Control: ` (empty/whitespace) -> not disable.

## API contract change

**Additive only.** Optional request header; absent header means
"caching follows the gateway-cache flag" (existing behavior). No
existing client breaks.

## Intentional (looks wrong but is deliberate)

- **`no-store` only, not `no-cache`.** RFC 7234 distinguishes them:
  `no-store` means "don't store this anywhere"; `no-cache` means
  "always revalidate before using a cached copy." For an LLM API,
  `no-store` is the right semantic for "don't cache this prompt."
  `no-cache` is closer to "always re-run" -- a different concept that
  could be added later if customers ask.
- **No corresponding response header.** A `Cache-Control` response
  header (e.g., `max-age`) would tell the customer how long the
  cached entry is valid. The exact cache today has a fixed TTL
  managed server-side; no per-call control needed yet.
- **No analogous gate on `/chat/stream` or `/batch`.** Streaming +
  cache (queue item #3) and batch + cache (queue item #6) aren't
  wired yet; bypass headers will be added there when the cache
  layers are.
- **Helper function is module-private (underscore prefix).** No
  reason to export it; the chat route is the only consumer. If a
  future endpoint needs the same parsing, promote then.
- **Bypass flips both lookup AND store.** Bypassing only the lookup
  would still write the customer's "sensitive" prompt to the cache,
  defeating the purpose. Bypassing only the store would still read
  from a previous (possibly sensitive) cached entry. Both must skip.

## Deferred (looks missing but is on purpose)

- Items #3, #5, #6 in the post-D6b queue (`/chat/stream` cache,
  semantic cache, `/batch` cache). Each is its own slice.
- `no-cache` directive support. Different semantic; add when a
  customer asks.
- `private`, `must-revalidate`, etc. Out of scope; this slice is
  about opt-out for sensitive prompts.

## Verification

- New regression tests
  `tests/test_llm_gateway_cache_control_no_store.py`:
  - `_cache_control_disables_cache(None)` -> False
  - `_cache_control_disables_cache("no-store")` -> True
  - `_cache_control_disables_cache("no-cache, no-store")` -> True
  - `_cache_control_disables_cache("NO-STORE")` -> True
  - `_cache_control_disables_cache("max-age=0")` -> False
  - `_cache_control_disables_cache("maybe-no-store")` -> False
  - `_cache_control_disables_cache("")` -> False
  - File-text assertions: lookup branch + store branch both gate on
    `not cache_disabled`.
- `python3 -m py_compile atlas_brain/api/llm_gateway.py` -> clean.
- `bash scripts/check_ascii_python.sh` -> passed.

## Conflict check

No file overlap with any open PR.

## Diff size

- Source: ~25 LOC additions to `atlas_brain/api/llm_gateway.py`
  (helper function + header param + 2 guard updates).
- Tests: ~80 LOC, 8 unit tests + 2 file-text contract assertions.
- Plan doc: ~110 LOC.

Source-only ~25 LOC. Smallest scope that closes queue item #4 with
correct RFC-7234 directive parsing.

## After this lands

Customer-supplied cache opt-out works on sync `/chat`. Three queue
items closed (#1 PR-D6e, #2 PR-D6c, #4 this PR); three remain (#3
streaming cache, #5 semantic cache, #6 batch cache). Of those, #5
(semantic cache) is the highest-value next slice -- it's "the second
cache layer in the 5-layer pitch" and the substrate already exists.
