# PR-Content-Ops-Brand-Voice-Profile-Editor

## Why this slice exists

PR-Content-Ops-Brand-Voice-Profile-Storage merged the tenant-scoped storage
table, CRUD API, by-id lookup, and Content Ops run selector. That makes stored
profiles usable once they exist, but an operator still cannot create, revise,
or archive profiles from the product UI. The API exists; the UI management
surface is the missing product path.

This slice closes the `PR-Content-Ops-Brand-Voice-Profile-Editor` deferred item
from the storage plan. It keeps scope to the existing New Run page so operators
can manage the saved profiles in the same place they select them for generation.

## Scope (this PR)

Ownership lane: content-ops/brand-voice-profile-editor
Slice phase: Vertical slice

1. Extend the existing Brand voice panel on `ContentOpsNewRun` with create,
   edit, cancel, save, and archive controls.
2. Reuse the already-shipped tenant CRUD API functions; do not add backend
   routes, migrations, or extracted package behavior.
3. Represent descriptors, exemplars, and banned terms as newline-delimited
   textarea fields, with POV and reading level as simple text inputs.
4. Client-side validate required name plus at least one guidance field before
   save, while leaving backend validation as the source of truth.
5. Refresh the saved-profile list after create/update/archive, select a newly
   created profile, and clear the selected profile when it is archived.
6. Update the existing CI-enrolled frontend selector script so the editor API
   calls and UI wiring remain covered by the existing intel-ui workflow.

### Files touched

- `plans/PR-Content-Ops-Brand-Voice-Profile-Editor.md`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`

## Mechanism

The panel keeps local editor state:

```ts
type BrandVoiceProfileEditorState = {
  mode: 'create' | 'edit'
  profileId: string | null
  name: string
  descriptorsText: string
  exemplarsText: string
  bannedTermsText: string
  preferredPov: string
  readingLevel: string
}
```

`Save` converts newline-delimited fields into bounded string arrays and calls
`createContentOpsBrandVoiceProfile()` or `updateContentOpsBrandVoiceProfile()`.
On create, the returned profile id becomes the selected
`brandVoiceProfileId`; on update, the current selection is preserved. Archive
uses `deleteContentOpsBrandVoiceProfile()` and clears the selected id if the
archived profile was selected.

The editor never writes `inputs.brand_voice`; it manages stored profiles only.
The run request still sends only `brand_voice_profile_id`, preserving the
storage slice's backend lookup contract.

## Intentional

- No standalone profile-management route in this slice. The immediate product
  gap is authoring a selectable profile in the run setup flow; a separate
  settings page can come later if needed.
- No sample upload/scrape onboarding. This is a manual editor over the existing
  fields; automated profile generation remains deferred.
- Metadata is not exposed in the UI. The API supports it, but showing arbitrary
  JSON would add a second editor problem unrelated to the operator-facing
  guidance fields.
- No changes to #1268 output variations or #1309 workflow docs.

## Deferred

- `PR-Content-Ops-Brand-Voice-Onboarding`: scrape/upload samples and pre-fill a
  custom profile for human edit.
- `PR-Content-Ops-Brand-Voice-Presets`: descriptor-only preset library.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if the inline New Run panel becomes too dense.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector` -- 6 passed.
- `cd atlas-intel-ui && npm run build` -- passed.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-brand-voice-profile-editor.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-Brand-Voice-Profile-Editor.md` | 110 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 353 |
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 8 |
| **Total** | **471** |

Expected final: 3 files, +459 / -12. This lands over the 400 LOC soft
cap because create/edit/archive, refresh/selection consistency, client-side
validation, and the CI-enrolled wiring test need to ship together as one usable
vertical slice.
