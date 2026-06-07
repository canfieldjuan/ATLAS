# PR-Content-Ops-Brand-Voice-Editor-Extraction

## Why this slice exists

`PR-Content-Ops-Brand-Voice-Settings-Page` added a dedicated
`/content-ops/brand-voice` settings surface, but intentionally left full
create/edit/archive workflows deferred back to the New Run page. That still
forces operators to leave the settings page to manage the saved profiles they
are viewing.

This slice closes that deferred item by extracting the already-shipped New Run
brand voice selector/editor into a shared frontend component and rendering that
same component on the settings page. The diff is expected to exceed the 400 LOC
soft cap because the safe implementation is a move-heavy extraction of the
existing editor markup and mutation wiring rather than a second, smaller
reimplementation. The behavior change remains narrow: settings gets the same
tenant-scoped create/edit/archive/sample-import workflow New Run already uses.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/editor-extraction
Slice phase: Product polish

1. Move the New Run brand voice selector/editor into a shared
   `BrandVoiceProfileManager` component under `atlas-intel-ui/src/components`.
2. Keep New Run wired to that shared component for run profile selection.
3. Render the same shared component on `/content-ops/brand-voice` so settings
   can create, edit, archive, refresh, and sample-import saved profiles without
   navigating to New Run.
4. Update the existing enrolled frontend tests to prove the shared component is
   the single editor surface and both pages consume it.

### Review Contract

- Acceptance criteria:
  - [ ] New Run still renders brand voice selection and threads selected
        `brandVoiceProfileId` into the run request.
  - [ ] Settings renders the shared brand voice manager and no longer routes
        create/edit back to `/content-ops/new`.
  - [ ] Create/update/delete/sample URL actions remain wired to the existing
        tenant-scoped brand voice API helpers.
  - [ ] The shared component preserves loading, stale-error, missing-selection,
        success, error, preset, and sample-import states from the existing New
        Run editor.
  - [ ] Existing enrolled brand voice frontend tests cover the shared component
        and both page integrations.
- Affected surfaces: frontend.
- Risk areas: frontend behavior, maintainability, CI enrollment.
- Reviewer rules triggered: R1, R2, R9, R10, R12.

### Files touched

- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`
- `atlas-intel-ui/scripts/content-ops-brand-voice-settings-page.test.mjs`
- `atlas-intel-ui/src/components/contentOps/BrandVoiceProfileManager.tsx`
- `atlas-intel-ui/src/pages/ContentOpsBrandVoiceSettings.tsx`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `plans/PR-Content-Ops-Brand-Voice-Editor-Extraction.md`

## Mechanism

The existing inline `BrandVoiceProfileSelector` implementation moves into a
new shared `BrandVoiceProfileManager` component. The component keeps the same
prop contract New Run already uses:

```tsx
<BrandVoiceProfileManager
  profiles={brandVoiceProfiles ?? []}
  selectedProfileId={request.brandVoiceProfileId}
  loading={brandVoiceProfilesLoading}
  refreshing={brandVoiceProfilesRefreshing}
  error={brandVoiceProfilesError}
  onRefresh={refreshBrandVoiceProfiles}
  onChange={handleBrandVoiceProfileChange}
/>
```

The shared component owns only local editor/mutation/sample-import UI state and
continues to call the existing `createContentOpsBrandVoiceProfile`,
`updateContentOpsBrandVoiceProfile`, `deleteContentOpsBrandVoiceProfile`, and
`fetchContentOpsBrandVoiceSampleUrl` API helpers. New Run keeps its existing
profile-list loader and selected-id state. Settings uses its existing
`useApiData(fetchContentOpsBrandVoiceProfiles)` loader plus a local selected id
and renders the manager above the profile inventory cards.

## Intentional

- This PR does not add URL deep-linking from a settings card into New Run. The
  point of the slice is to remove that management detour by making settings the
  management surface.
- The settings page keeps the profile inventory cards as a scan-friendly
  overview and adds the shared manager above them instead of turning every card
  into its own editor. A single editor avoids duplicate mutation controls on
  the page.
- The diff is larger than 400 LOC because it moves an existing editor rather
  than duplicating or partially reimplementing it.

## Deferred

None.

Parked hardening: none.

## Verification

- `cd atlas-intel-ui && npm ci` - installed locked UI dependencies in the new
  worktree; reported existing npm audit findings, no dependency changes made.
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector`
  - 12/12 tests passed.
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-settings-page`
  - 4/4 tests passed.
- `cd atlas-intel-ui && npm run lint` - passed.
- `cd atlas-intel-ui && npm run build` - passed.
- Manual workflow grep for both existing brand voice test scripts in
  `.github/workflows/atlas_intel_ui_checks.yml` - both scripts enrolled.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-brand-voice-editor-extraction.md`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 56 |
| `atlas-intel-ui/scripts/content-ops-brand-voice-settings-page.test.mjs` | 18 |
| `atlas-intel-ui/src/components/contentOps/BrandVoiceProfileManager.tsx` | 658 |
| `atlas-intel-ui/src/pages/ContentOpsBrandVoiceSettings.tsx` | 60 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 642 |
| `plans/PR-Content-Ops-Brand-Voice-Editor-Extraction.md` | 129 |
| **Total** | **1563** |
