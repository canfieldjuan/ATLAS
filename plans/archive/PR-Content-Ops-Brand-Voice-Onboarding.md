# PR-Content-Ops-Brand-Voice-Onboarding

## Why this slice exists

PR-Content-Ops-Brand-Voice-Profile-Editor made saved brand voice profiles
editable in the New Run page, but operators still have to hand-transcribe a
customer's voice into descriptors, exemplars, POV, and reading level. The
brand-voice plan chain explicitly deferred onboarding from samples; this slice
adds the thinnest usable version: load sample copy, derive an editable draft,
and keep the human in control before save.

This keeps the product moving without adding a live scraper, LLM classifier, or
new backend contract. The current proof point is upload/paste to prefill the
existing editor; authenticated URL scrape can follow once the backend scrape
adapter and rate-limit/error contract are scoped.

This slice is expected to exceed the 400 LOC soft cap because the prior editor
helpers must be extracted into a behavioral-testable module at the same time as
the sample import UI. Splitting the helper extraction from the onboarding
controls would either keep the #1311 request-builder NIT untested or introduce
sample controls without the pure fixtures that prove their derivation behavior.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/onboarding
Slice phase: Vertical slice

1. Add a pure frontend brand-voice editor helper module that owns the editor
   request transform, save validation, and deterministic sample-to-draft
   derivation.
2. Extend the existing `ContentOpsNewRun` brand voice editor with sample copy
   onboarding: paste text, load a local text/copy sample file, and apply the
   derived fields into the editable profile form.
3. Preserve the existing create/update/archive API behavior and continue to
   save only through the tenant-scoped brand voice profile CRUD routes.
4. Add behavioral tests in the existing CI-enrolled selector script so the
   request-builder NIT from #1311 is covered and sample import heuristics are
   locked.

### Files touched

- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`
- `atlas-intel-ui/src/domain/contentOps/brandVoiceProfileEditor.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `plans/PR-Content-Ops-Brand-Voice-Onboarding.md`

## Mechanism

The new domain helper exports the page-local editor state and pure helpers:

```ts
deriveBrandVoiceProfileEditorPatch(sampleText, { fallbackName })
brandVoiceProfileEditorRequest(editor)
canSaveBrandVoiceProfileEditor(editor)
```

The derivation is intentionally deterministic and conservative. It normalizes
sample text, extracts up to three short exemplars, infers POV from pronoun
balance, marks plain reading level for short average sentences, and derives a
small descriptor set from observable markers such as concise sentences,
contractions, customer-directed language, and technical terms. It never calls a
network service, never invents claims about the business, and never saves until
the operator reviews and submits the normal profile form.

`ContentOpsNewRun` keeps local sample text/file-load state while the editor is
open. Applying a sample merges the derived patch into the current editor,
preserving manually entered name/guidance unless the field was blank.

## Intentional

- No live URL scraping in this slice. Browser-side cross-origin fetch would be
  unreliable, and a backend scrape path needs its own auth, timeout, and error
  contract.
- No LLM voice classifier. The first onboarding step should be transparent and
  predictable; human review remains the quality gate.
- No new frontend test script. The existing
  `test:content-ops-brand-voice-profile-selector` script is already enrolled
  in the intel-ui workflow, so extending it avoids another CI-enrollment risk.
- No settings page. The profile workflow stays in the New Run brand voice panel
  until the inline surface becomes too dense.

## Deferred

- `PR-Content-Ops-Brand-Voice-Scrape-Onboarding`: authenticated URL scrape or
  backend text extraction into the same sample-to-draft helper.
- `PR-Content-Ops-Brand-Voice-Presets`: descriptor-only preset library.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if inline authoring becomes too dense.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `cd atlas-intel-ui && npm run lint` -- passed.
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector`
  -- 9 passed.
- `cd atlas-intel-ui && npm run build` -- passed.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-brand-voice-onboarding.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 95 |
| `atlas-intel-ui/src/domain/contentOps/brandVoiceProfileEditor.ts` | 219 |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | 10 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 212 |
| `plans/PR-Content-Ops-Brand-Voice-Onboarding.md` | 114 |
| **Total** | **650** |
