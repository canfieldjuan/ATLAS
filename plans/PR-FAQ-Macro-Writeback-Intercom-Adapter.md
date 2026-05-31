# PR-FAQ-Macro-Writeback-Intercom-Adapter

## Why this slice exists
Macro writeback now has provider-neutral preview DTOs, the double publish gate,
idempotency mappings, publish history, and Zendesk as the first adapter. This
slice adds Intercom saved replies behind the same `MacroPublishProvider` port.

## Scope (this PR)
Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add `faq_macro_writeback_intercom.py` for credentials, transport, payload mapping, idempotent create/update, and fail-closed response guards.
2. Add typed Intercom `ATLAS_*` config fields in `atlas_brain/config.py`.
3. Add no-network extracted-checks tests for the double gate, create/update, pending refusal, idempotency, credentials, malformed envelopes, and per-item failure isolation.
4. Enroll the Intercom test in the extracted pipeline runner.

### Files touched
- `plans/PR-FAQ-Macro-Writeback-Intercom-Adapter.md`
- `atlas_brain/config.py`
- `extracted_content_pipeline/faq_macro_writeback_intercom.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_ticket_faq_macro_writeback_intercom.py`

## Mechanism
The provider asks a scoped credentials provider for Intercom credentials. Missing
credentials return `intercom_credentials_missing` without mapping lookup or
transport calls. Existing mappings update `/macros/{external_id}`; missing
mappings reserve pending, create `/macros`, then complete the mapping. Pending
rows without an external id fail closed so retries do not create duplicates.
Requests include Bearer auth and an idempotency key derived from account id plus
FAQ identity. Responses must be a macro object with `type == "macro"` and `id`.

## Intentional

- Transport is mocked in CI; no live Intercom API calls are made.
- No route, scheduler, UI, live smoke, or provider-selection switch is added.
- Tokens are excluded from repr, mapping metadata, and sanitized errors.

## Deferred

- Future PR: host provider selection plus tenant Intercom credential storage.
- Future PR: live Intercom smoke after tenant credential storage exists.

Parked hardening: none.

## Verification

- Python compile check for the Intercom adapter and tests - passed.
- Focused pytest for Intercom adapter coverage - 5 passed.
- Extracted pipeline CI enrollment audit - passed, 141 matching tests enrolled.
- Local PR review bundle with the prepared PR body file - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 64 |
| Config fields | 10 |
| Intercom adapter | 161 |
| Intercom tests | 162 |
| Runner enrollment | 1 |
| **Total** | **398** |

Under the 400 LOC soft cap.
