# PR-Content-Ops-Brand-Voice-Scrape-Onboarding

## Why this slice exists

PR-Content-Ops-Brand-Voice-Onboarding added paste/local-file sample import, but
operators still need to manually copy text from a customer's public site before
they can prefill a profile. That PR explicitly deferred authenticated URL
scrape/backend text extraction. This slice adds the thinnest safe backend URL
path and feeds the returned text into the existing human-reviewed sample
derivation flow.

The risk in this slice is not the profile derivation; it is safe URL fetching.
The host route therefore reuses the deflection submit hardening shape:
HTTPS-only, DNS private-IP rejection before fetch, no redirects, bounded bytes,
and transport-mocked tests that prove each failure branch fires.

This PR is expected to exceed the 400 LOC soft cap because the safe URL fetch
guard, host route, UI call site, and negative SSRF/redirect tests must land
together. Splitting the UI from the guard would create an unreachable product
path; splitting the guard tests would leave the safety claim unenforced.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/scrape-onboarding
Slice phase: Vertical slice

1. Add an authenticated admin-only host route at
   `/api/v1/content-ops/brand-voice-profiles/sample-url`.
2. Fetch only public HTTPS URLs with bounded reads, no redirect following, and
   DNS resolution checks that reject private, loopback, link-local, multicast,
   and reserved targets before any request is opened.
3. Extract readable text from HTML by ignoring script/style/head content and
   falling back to plain text for non-HTML bodies.
4. Extend the Content Ops UI sample import panel with a URL field and "Fetch
   URL" button that loads returned text into the existing sample textarea.
5. Add API/UI tests with mocked transport/DNS; no live network calls in CI.

### Files touched

- `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas_brain/api/content_ops_brand_voice_profiles.py`
- `plans/PR-Content-Ops-Brand-Voice-Scrape-Onboarding.md`
- `tests/test_content_ops_brand_voice_profiles_api.py`

## Mechanism

The route accepts:

```json
{ "url": "https://example.com/about" }
```

and returns:

```json
{
  "url": "https://example.com/about",
  "title": "Example",
  "text": "Readable sample copy...",
  "source_character_count": 1234
}
```

Validation is fail-closed:

- URL must parse as absolute HTTPS with a hostname and valid port.
- Host resolution happens before fetch and every resolved IP must be public.
- The opener disables redirect following; redirect status and `HTTPError`
  redirect responses are rejected.
- Reads are capped to a fixed byte limit and oversized bodies return 413.

The UI does not derive or save on fetch. It puts returned text in the existing
sample textarea and stores the title/host as the sample name, so the already
tested `deriveBrandVoiceProfileEditorPatch(...)` path remains the single
sample-to-profile implementation.

## Intentional

- No unauthenticated public scrape endpoint. This is an account profile
  authoring helper, so it follows the existing owner/admin write guard.
- No JavaScript rendering, crawling, sitemap expansion, or form-authenticated
  fetch. The first URL path extracts one public page only.
- No LLM summarization. The same deterministic sample derivation from #1312
  remains the only prefill path.
- No external network in tests; DNS and opener boundaries are mocked.

## Deferred

- `PR-Content-Ops-Brand-Voice-Presets`: descriptor-only preset library.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if inline authoring becomes too dense.
- `PR-Content-Ops-Social-Post-LLM-Voice`: LLM social-post variant that can
  consume brand voice.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_brand_voice_profiles_api.py -q` (16 passed)
- `cd atlas-intel-ui && npm ci` (installed dependencies; npm reported
  pre-existing audit findings: 2 moderate, 6 high)
- `cd atlas-intel-ui && npm run test:content-ops-brand-voice-profile-selector`
  (10 passed)
- `python -m py_compile atlas_brain/api/content_ops_brand_voice_profiles.py tests/test_content_ops_brand_voice_profiles_api.py`
- `cd atlas-intel-ui && npm run lint`
- `cd atlas-intel-ui && npm run build`
- `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-brand-voice-profile-selector.test.mjs` | 36 |
| `atlas-intel-ui/src/api/contentOps.ts` | 22 |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | 105 |
| `atlas_brain/api/content_ops_brand_voice_profiles.py` | 413 |
| `plans/PR-Content-Ops-Brand-Voice-Scrape-Onboarding.md` | 121 |
| `tests/test_content_ops_brand_voice_profiles_api.py` | 257 |
| **Total** | **954** |
