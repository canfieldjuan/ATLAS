# PR-Content-Ops-Brand-Voice-Settings-Page

## Why this slice exists

The brand-voice profile chain is now productized enough to use: tenant-scoped
profile storage, CRUD APIs, the New Run selector/editor, sample onboarding,
presets, LLM social-post voice, and social channel selection have all landed.
The remaining UX gap is discoverability. Brand voice management is buried in
the New Run setup page, so operators cannot quickly inspect saved profiles or
find the run-entry point without starting a generation flow.

Earlier brand-voice plans repeatedly deferred
`PR-Content-Ops-Brand-Voice-Settings-Page` as the follow-up once the inline
surface became dense. This slice adds the thinnest standalone settings surface:
a dedicated route and navigation entry that lists saved profiles, shows their
operator-facing guidance summary, and links back to New Run for create/edit/run
actions.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/settings-page
Slice phase: Product polish

1. Add a protected `/content-ops/brand-voice` route in `atlas-intel-ui`.
2. Add B2B sidebar navigation to the new brand voice settings route.
3. Implement a compact settings page that fetches tenant saved profiles through
   the existing `fetchContentOpsBrandVoiceProfiles()` API and renders loading,
   error, empty, and populated states.
4. Keep management mutations in the already-shipped New Run editor for this
   slice; the settings page links to `/content-ops/new` for new/edit/run
   actions.
5. Add a CI-enrolled frontend test proving route/nav registration, API use,
   empty/populated-state text, and workflow enrollment.

### Review Contract

- Acceptance criteria: `/content-ops/brand-voice` is registered as a protected
  route; B2B navigation exposes "Brand Voice"; the page fetches saved profiles
  with the existing tenant API; loading/error/empty/populated states are
  represented; action links go to `/content-ops/new`; the new test is enrolled
  in `.github/workflows/atlas_intel_ui_checks.yml`.
- Affected surfaces: atlas-intel-ui routing, sidebar navigation, new Content
  Ops settings page, frontend tests, plan doc.
- Risk areas: accidentally duplicating profile mutation behavior, hiding the
  New Run editor contract, frontend CI test enrollment.
- Reviewer rules triggered: R1 (settings surface matches the deferred scope),
  R2 (test evidence), R9 (frontend behavior), R12 (frontend CI enrollment).

### Files touched

- `.github/workflows/atlas_intel_ui_checks.yml`
- `atlas-intel-ui/package.json`
- `atlas-intel-ui/scripts/content-ops-brand-voice-settings-page.test.mjs`
- `atlas-intel-ui/src/App.tsx`
- `atlas-intel-ui/src/components/Sidebar.tsx`
- `atlas-intel-ui/src/pages/ContentOpsBrandVoiceSettings.tsx`
- `plans/PR-Content-Ops-Brand-Voice-Settings-Page.md`

## Mechanism

`ContentOpsBrandVoiceSettings` uses `useApiData()` with
`fetchContentOpsBrandVoiceProfiles()` and renders an unframed page layout:

```tsx
profiles.map((profile) => (
  <article>
    <h2>{profile.name}</h2>
    <p>{format profile guidance summary}</p>
    <Link to="/content-ops/new">Use in run</Link>
  </article>
))
```

The page is read/list/action-only in this slice. It does not duplicate create,
update, archive, sample import, or preset application logic from
`ContentOpsNewRun`; those flows stay in the existing New Run editor until a
separate extraction slice moves the shared editor into a reusable component.

## Intentional

- No profile create/edit/archive controls on the settings page yet. Copying the
  large New Run editor would create a second mutation surface; extracting it
  cleanly is a separate, larger refactor.
- No backend/API changes. The tenant-scoped CRUD routes already exist and the
  settings page only needs the list endpoint.
- No new design system. The page follows the existing restrained Content Ops
  dark UI and keeps cards to individual profile rows.

## Deferred

- `PR-Content-Ops-Brand-Voice-Editor-Extraction`: move the New Run brand voice
  editor into a shared component and render it on the settings page for full
  create/edit/archive workflows.

Parked hardening: none.

## Verification

- `npm run test:content-ops-brand-voice-settings-page` - 4/4 tests passed.
- `npm run lint` - passed.
- `npm run build` - passed.
- Manual enrollment grep for `test:content-ops-brand-voice-settings-page` -
  package script and workflow run step both present.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-content-ops-brand-voice-settings-page-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_intel_ui_checks.yml` | 3 |
| `atlas-intel-ui/package.json` | 1 |
| `atlas-intel-ui/scripts/content-ops-brand-voice-settings-page.test.mjs` | 52 |
| `atlas-intel-ui/src/App.tsx` | 4 |
| `atlas-intel-ui/src/components/Sidebar.tsx` | 3 |
| `atlas-intel-ui/src/pages/ContentOpsBrandVoiceSettings.tsx` | 181 |
| `plans/PR-Content-Ops-Brand-Voice-Settings-Page.md` | 117 |
| **Total** | **361** |
