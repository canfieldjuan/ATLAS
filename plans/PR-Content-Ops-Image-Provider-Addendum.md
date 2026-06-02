# PR: Content Ops Image Provider -- OpenRouter Reuse Addendum

> Clarification note for the already-merged plan
> `plans/PR-Content-Ops-Image-Provider.md` (PR #1238). That plan is sound and
> its premises were verified against `main`; this addendum sharpens one
> instruction so the implementing slice does not get stuck. Plan-only PR, no
> code change.

## Why this slice exists

The merged image-provider plan tells the implementing dev to "reuse Atlas's
existing OpenRouter key/provider config ... do not hand-roll a second OpenRouter
client" (citing the #1224/#1227 anti-fork lesson). That guidance is correct in
spirit but ambiguous against the actual code, and a literal reading sends the
dev into a wall:

- `extracted_llm_infrastructure/services/llm/openrouter.py` defines
  `OpenRouterLLM` as a **chat/text** client (`BaseModelService`: `chat`,
  `chat_with_tools`, `chat_async`, `generate`). It has **no image method, no
  base64 handling, and no `modalities` plumbing** -- a trace of the class
  confirms it returns text `Message` objects only.
- So image bytes cannot come out of `OpenRouterLLM.chat()`. A dev who reads
  "reuse the client, do not fork it" literally will try to force image
  generation through the chat abstraction and stall.

Flux-via-OpenRouter itself is confirmed available (working-project precedent);
this addendum is **only** about which part of the existing OpenRouter wiring is
reusable.

## Scope (this PR)

Ownership lane: content-ops/image-provider
Slice phase: Workflow/process

Plan-only PR. It records the reuse boundary for the image-provider slice. No
code change; the merged plan doc is left intact and this note is read alongside
it.

### Files touched

- `plans/PR-Content-Ops-Image-Provider-Addendum.md`

## Mechanism

Split "reuse OpenRouter" into the two things it actually means:

1. **Reuse (yes):** the key/config resolution -- `_resolve_openrouter_api_key`
   (explicit arg -> `OPENROUTER_API_KEY` / `ATLAS_B2B_CHURN_OPENROUTER_API_KEY`
   env -> `settings.b2b_churn.openrouter_api_key`) and the base URL
   (`https://openrouter.ai/api/v1`). This is the duplication the anti-fork
   lesson is about; do not re-resolve the key a second way.
2. **New thin path (expected, not a forbidden fork):** the image request itself.
   Because `OpenRouterLLM` is chat-only, the Flux call uses a separate, small
   image-specific request/response path (its own decode-base64-to-storage
   step). Building that is **not** the kind of sibling-client drift #1224/#1227
   warned against -- it is a different transport for a capability the chat
   client does not have.

Concretely: the image provider may import and call `_resolve_openrouter_api_key`
for the key and reuse the base URL, while issuing its own image request rather
than going through `OpenRouterLLM.chat()`.

## Intentional

- Preserve the merged plan's intent (one OpenRouter key/config source, paid
  calls through existing cost tracking, free-first ordering, best-effort
  degrade-to-no-image). This note narrows the "do not hand-roll a second client"
  line so it targets config duplication, not the image transport.
- Keep the #1224/#1227 anti-fork discipline correctly scoped: it forbids a
  second weaker copy of an existing capability, not a first implementation of a
  missing one.

## Deferred

- Everything already deferred by the merged plan (templated stat/quote cards,
  per-platform sizing, brand overlays, the skill's theme system) is unchanged.
- Whether the new image path should later be generalized into the
  `extracted_llm_infrastructure` package as a typed image-provider port is a
  separate question, out of scope here.

## Parked hardening

None.

## Verification

- Plan renders the 7 sections; local PR review hook passes.
- The two claims this note rests on were verified against `main`:
  `OpenRouterLLM` exposes only chat/text methods (no image/`modalities`), and
  `_resolve_openrouter_api_key` is the reusable key resolver.

## Estimated diff size

| Area | LOC |
|---|---:|
| This PR (plan doc) | ~95 |
| **Total** | ~95 |

Plan-only. No change to the implementing-slice estimate in the merged plan
(~200-350 LOC); this note only clarifies how that slice uses OpenRouter.
