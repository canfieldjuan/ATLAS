# PR-Content-Ops-Brand-Voice-Presets

## Why this slice exists

PR-Content-Ops-Brand-Voice-Scrape-Onboarding completed the paste/file/URL
sample path and deferred `PR-Content-Ops-Brand-Voice-Presets` as the next
brand voice convenience slice. Operators can now derive a profile from real
copy, but when a customer has no useful public copy yet, profile authoring still
starts from a blank form.

This slice adds a descriptor-only preset library to the existing inline editor.
It gives operators a deterministic starting point without creating stored
profiles automatically and without introducing an LLM prompt or backend preset
API.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/presets
Slice phase: Product polish

1. Add a small typed client-side preset catalog for common brand voice starting
   points.
2. Add a domain helper that converts a selected preset into the same
   editor-patch shape used by sample import.
3. Add an inline preset selector/button in the existing brand voice editor.
4. Preserve manually entered fields by applying presets only through the
   existing non-overwrite patch merge.
5. Extend the existing enrolled brand voice frontend test script.

### Files touched

- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`
- `atlas-intel-ui/src/domain/contentOps/brandVoiceProfileEditor.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `plans/PR-Content-Ops-Brand-Voice-Presets.md`

## Mechanism

The domain layer exports `BRAND_VOICE_PROFILE_PRESETS` and
`brandVoicePresetEditorPatch(presetId)`. Each preset supplies only deterministic
authoring fields: descriptors, preferred POV, reading level, and optional
banned terms. It intentionally does not generate exemplars.

The UI renders a preset select while an editor is open. Clicking "Apply preset"
looks up the selected preset, builds a `BrandVoiceProfileEditorPatch`, and calls
`applyBrandVoiceProfileEditorPatch(...)`, which already fills only empty fields.
The operator still reviews and saves the resulting profile through the existing
create/update route.

## Intentional

- Presets are client-side and descriptor-only. They are authoring shortcuts, not
  persisted tenant records.
- Presets do not include exemplars; real exemplars should still come from
  customer copy via paste/file/URL.
- Applying a preset does not overwrite manually entered fields. Operators can
  clear a field first if they want the preset value.
- No new frontend test script is added; this extends the already enrolled brand
  voice profile selector script.

## Deferred

- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if inline authoring becomes too dense.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `cd atlas-intel-ui && npm ci` (installed dependencies; npm reported
  pre-existing audit findings: 2 moderate, 6 high)
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector`
  (11 passed)
- `cd atlas-intel-ui && npm run lint`
- `cd atlas-intel-ui && npm run build`
- `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 37 |
| `atlas-intel-ui/src/domain/contentOps/brandVoiceProfileEditor.ts` | 61 |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | 7 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 76 |
| `plans/PR-Content-Ops-Brand-Voice-Presets.md` | 90 |
| **Total** | **271** |
