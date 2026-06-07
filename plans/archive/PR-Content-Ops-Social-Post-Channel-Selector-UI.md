# PR-Content-Ops-Social-Post-Channel-Selector-UI

## Why this slice exists

PR #1340 added the backend/package contract for social-post channel variants:
`inputs.social_channels` / `inputs.social_post_channels` fan out one source row
into platform-specific drafts and preview cost scales for brand-voice rewrites.
PR #1341 hardened the invalid-channel preview path and mutation-pinned the core
behavior.

The remaining product gap is that a user still cannot select those channels in
the New Run UI. This slice exposes the contract at the run-configuration
surface so selecting `social_post` can send the desired platform list instead
of always using the backend default LinkedIn-only behavior.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/social-post-channel-selector-ui
Slice phase: Vertical slice

1. Add a social-post channel selector to the New Run configuration UI when the
   `social_post` output is selected.
2. Default the selector to LinkedIn only, matching the backend default and
   preserving existing runs unless the user opts into more channels.
3. Submit selected channels as `inputs.social_channels` on preview/plan/execute
   payloads without reusing campaign-owned `inputs.channels`.
4. Keep the selector state stable when outputs are toggled, but omit
   `social_channels` when `social_post` is not selected.
5. Add UI/domain tests proving default payload compatibility, multi-channel
   payload threading, and non-social output omission.

### Review Contract

- Acceptance criteria: selecting `social_post` reveals a channel selector;
  LinkedIn is selected by default; additional checked channels are included in
  `inputs.social_channels`; non-social runs do not send social channels; tests
  exercise the real request mapper/submit path used by the UI.
- Affected surfaces: atlas-intel-ui Content Ops New Run UI, request mapping,
  frontend tests, plan doc.
- Risk areas: payload contract drift, accidental collision with campaign
  `channels`, UI test enrollment, mobile/compact layout text fit.
- Reviewer rules triggered: R1 (UI behavior matches backend contract), R2
  (tests prove payload threading), R9 (frontend behavior), R12 (frontend CI
  enrollment if a new test is added).

### Files touched

- `.github/workflows/atlas_intel_ui_checks.yml`
- `atlas-intel-ui/package.json`
- `atlas-intel-ui/scripts/content-ops-social-post-channel-selector.test.mjs`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/domain/contentOps/socialPostChannels.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `plans/PR-Content-Ops-Social-Post-Channel-Selector-UI.md`

## Mechanism

The UI will keep a small, typed list of supported social channels matching the
backend normalized ids (`linkedin`, `x`, `facebook`, `instagram`, `threads`).
The New Run state stores selected social-post channels separately from any
campaign channel state. The request builder adds `inputs.social_channels` only
when `social_post` is selected and at least one social channel is checked.

The control is intentionally a compact checkbox/segmented list near the output
selection area rather than a separate settings page. The backend still enforces
valid ids and default LinkedIn behavior, so the UI does not need to duplicate
validation beyond keeping the selector non-empty.

## Intentional

- This does not add a new backend enum endpoint; the supported ids are a tiny
  product list already fixed in the backend contract.
- This does not use `inputs.channels`, which remains the email/campaign channel
  field.
- This does not add a standalone brand-voice settings page; it only exposes the
  social-post channel selector needed for the current run.

## Deferred

- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if the inline New Run panel becomes too dense.

Parked hardening: none.

## Verification

- `npm run test:content-ops-social-post-channel-selector` - 5/5 tests passed.
- `npm run lint` - passed.
- `npm run build` - passed.
- Manual enrollment grep for `test:content-ops-social-post-channel-selector` - package script and workflow run step both present.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-content-ops-social-post-channel-selector-ui-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_intel_ui_checks.yml` | 3 |
| `atlas-intel-ui/package.json` | 1 |
| `atlas-intel-ui/scripts/content-ops-social-post-channel-selector.test.mjs` | 106 |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | 9 |
| `atlas-intel-ui/src/domain/contentOps/socialPostChannels.ts` | 61 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 78 |
| `plans/PR-Content-Ops-Social-Post-Channel-Selector-UI.md` | 104 |
| **Total** | **362** |
