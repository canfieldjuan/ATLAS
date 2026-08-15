# PR-Drop-Archived-Intel-Next

## Why this slice exists

`_ARCHIVED_atlas-intel-next/` contributes **46 open OSV alerts** against
`main`'s tree. `osv-full`
(`.github/workflows/security_guardrails.yml:158-159`) never evaluates a pull
request -- `if: github.event_name != 'pull_request' && github.event_name !=
'pull_request_target'` -- so this is not a PR-time check of any kind; it is a
standalone push/schedule scan of `main`. It is, by its own DO_NOT_USE note, an
"unused Next.js experiment" that "exists only as an archive"; the live
frontend is `atlas-churn-ui`.

**Root cause.** The tree was excluded from *Dependabot* but not from
*scanning*. `docs/SECURITY_GUARDRAILS.md` records that exclusion as
deliberate — it stopped update churn, and stopping churn was read as handling
the problem. It was not: `osv-full` still walks the lockfile on every scan of
`main`, so the exclusion traded routine update PRs for a standing set of
alerts on that scan. The harm is signal dilution in that alert set, not any
blocked check: 46 alerts from code the repo forbids using sit alongside
alerts from code that ships, which makes the scan's output worth less than
it should be.

### Problem-derived contract

- Root cause: an unused tree is scanned by `osv-scan`, and the mitigation
  chosen (Dependabot exclusion) does not affect scanning.
- Correct fix must touch/change: the tree itself, and the document asserting
  the exclusion policy that this supersedes.
- Must not change: any deployed manifest; the `npm_package_checks` matrix
  (which never listed this tree); the Dependabot config (no entry to remove);
  `HARDENING.md`'s dated log entries.
## Scope (this PR)

Ownership lane: security/dependency-noise
Slice phase: Workflow/process
Max files: 95

1. Delete `_ARCHIVED_atlas-intel-next/` — 91 tracked files, ~1.1 MB, last
   touched 2026-06-16 (`f253164c1`).
2. Update `docs/SECURITY_GUARDRAILS.md`, whose "Continuous Updates" section
   states the Dependabot-exclusion policy this replaces.
3. Remove the two stale repository-map entries this deletion orphans:
   `README.md`'s directory tree and `docs/product_context_pack.md`'s
   "Relevant code and product surfaces" list. Both name the bare
   `atlas-intel-next/` (no `_ARCHIVED_` prefix) as a current, relevant path;
   after this PR no directory of either name exists.
4. This plan document.

Not in this PR: the rest of ATLAS #2375 — bumping `transformers` and
`cryptography` in root `requirements.txt` (2 critical / 22 high), and deciding
whether the non-deployed manifests are fixed or scoped out. Those are dependency
changes with their own blast radius; this is a deletion of unused code.
### Review Contract

- Acceptance criteria:
  - The tree is gone from the working tree and nothing references it from code
    or CI — verified by `git grep` across the repo excluding `plans/archive/`
    and `HARDENING.md`, which are historical records.
  - No workflow globs or matrix entry loses a target: `npm_package_checks`
    lists `atlas-admin-ui`, `atlas-churn-ui`, `atlas-mobile`, `atlas-ui` only.
  - Dependabot loses nothing: the tree has no entry in `.github/dependabot.yml`.
  - `docs/SECURITY_GUARDRAILS.md` no longer asserts a policy for a path that
    does not exist.
- Reachability proof: none required — this deletes code with no importers,
  routes, or build targets. That absence IS the claim, and it is what the
  grep above establishes.
- Affected surfaces: repository contents; the OSV alert set on `main`; one
  security document.
- Risk areas: deleting something still referenced by a build or deploy;
  implying that deletion resolves the historical IndexNow-key finding, which it
  does not.
- Reviewer rules triggered: R2, R3, R5, R9, R12, R14 — R3, R9 and R12 are triggered by the CONTENT of deleted files (an AuthContext and a set of app pages), not by any behavioural change; this PR adds no code and changes no auth, config, or gate path, so for each the review question is the same: does anything still reach this code? The inbound-reference search under Verification answers it. R2 is grep evidence of absence, R5 is backward compatibility (nothing imports it), R14 is verify-against-the-codebase.

- Boundary path/seam: none. No guard, validator, route, or config predicate
  changes. The only behavioural change is which files exist.
- Replaced-path behaviors: none — the tree has no callers.
- Guard-relevant fields: none.
- Caller x input shape: not applicable; there are no callers. The enumeration
  that matters here is the inbound-reference search, recorded under
  Verification.
### Deployed-config probing

N/A — no guard or config boundary changes, and no deployed manifest is
touched. `requirements.eom.txt` (the EOM slim profile) and root
`requirements.txt` are untouched by this PR.
### Files touched

- `README.md`
- `_ARCHIVED_atlas-intel-next/.gitignore`
- `_ARCHIVED_atlas-intel-next/.gitkeep`
- `_ARCHIVED_atlas-intel-next/AGENTS.md`
- `_ARCHIVED_atlas-intel-next/CLAUDE.md`
- `_ARCHIVED_atlas-intel-next/DO_NOT_USE.md`
- `_ARCHIVED_atlas-intel-next/README.md`
- `_ARCHIVED_atlas-intel-next/app/(app)/account/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/affiliates/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/blog-review/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/briefing-review/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/campaign-review/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/challengers/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/dashboard/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/layout.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/leads/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/onboarding/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/prospects/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/reports/[id]/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/reports/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/reviews/[id]/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/reviews/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/vendor-targets/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/vendors/[name]/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(app)/vendors/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(auth)/forgot-password/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(auth)/layout.tsx`
- `_ARCHIVED_atlas-intel-next/app/(auth)/login/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(auth)/reset-password/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(auth)/signup/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/[slug]/blog-post-content.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/[slug]/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/layout.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/methodology/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/(marketing)/report/page.tsx`
- `_ARCHIVED_atlas-intel-next/app/globals.css`
- `_ARCHIVED_atlas-intel-next/app/layout.tsx`
- `_ARCHIVED_atlas-intel-next/app/robots.ts`
- `_ARCHIVED_atlas-intel-next/app/sitemap.ts`
- `_ARCHIVED_atlas-intel-next/components/ArchetypeBadge.tsx`
- `_ARCHIVED_atlas-intel-next/components/AtlasHeroScene.tsx`
- `_ARCHIVED_atlas-intel-next/components/AtlasRobotLogo.tsx`
- `_ARCHIVED_atlas-intel-next/components/AtlasRobotScene.tsx`
- `_ARCHIVED_atlas-intel-next/components/BlogChartRenderer.tsx`
- `_ARCHIVED_atlas-intel-next/components/ChurnChart.tsx`
- `_ARCHIVED_atlas-intel-next/components/DataTable.tsx`
- `_ARCHIVED_atlas-intel-next/components/ErrorBoundary.tsx`
- `_ARCHIVED_atlas-intel-next/components/Layout.tsx`
- `_ARCHIVED_atlas-intel-next/components/PipelineStatus.tsx`
- `_ARCHIVED_atlas-intel-next/components/PublicLayout.tsx`
- `_ARCHIVED_atlas-intel-next/components/SeoHead.tsx`
- `_ARCHIVED_atlas-intel-next/components/Sidebar.tsx`
- `_ARCHIVED_atlas-intel-next/components/StatCard.tsx`
- `_ARCHIVED_atlas-intel-next/components/UpgradeGate.tsx`
- `_ARCHIVED_atlas-intel-next/components/UrgencyBadge.tsx`
- `_ARCHIVED_atlas-intel-next/components/report-library/ReportEvidencePanel.tsx`
- `_ARCHIVED_atlas-intel-next/components/report-library/SubscriptionScheduler.tsx`
- `_ARCHIVED_atlas-intel-next/components/report-renderers/SpecializedReportData.tsx`
- `_ARCHIVED_atlas-intel-next/components/report-renderers/StructuredReportData.tsx`
- `_ARCHIVED_atlas-intel-next/content/blog/index.ts`
- `_ARCHIVED_atlas-intel-next/eslint.config.mjs`
- `_ARCHIVED_atlas-intel-next/lib/api/blog.ts`
- `_ARCHIVED_atlas-intel-next/lib/api/client.ts`
- `_ARCHIVED_atlas-intel-next/lib/api/config.ts`
- `_ARCHIVED_atlas-intel-next/lib/auth/AuthContext.tsx`
- `_ARCHIVED_atlas-intel-next/lib/constants.ts`
- `_ARCHIVED_atlas-intel-next/lib/hooks/useApiData.ts`
- `_ARCHIVED_atlas-intel-next/lib/hooks/usePlanGate.ts`
- `_ARCHIVED_atlas-intel-next/lib/reportConstants.ts`
- `_ARCHIVED_atlas-intel-next/lib/reportLibrary.ts`
- `_ARCHIVED_atlas-intel-next/lib/reportNormalization.ts`
- `_ARCHIVED_atlas-intel-next/lib/reportViewModels.ts`
- `_ARCHIVED_atlas-intel-next/lib/robot.ts`
- `_ARCHIVED_atlas-intel-next/lib/types.ts`
- `_ARCHIVED_atlas-intel-next/lib/types/index.ts`
- `_ARCHIVED_atlas-intel-next/lib/types/reportViewModels.ts`
- `_ARCHIVED_atlas-intel-next/next.config.ts`
- `_ARCHIVED_atlas-intel-next/package-lock.json`
- `_ARCHIVED_atlas-intel-next/package.json`
- `_ARCHIVED_atlas-intel-next/postcss.config.mjs`
- `_ARCHIVED_atlas-intel-next/public/6a1d58a9cdf74c7b9c948ac27f86ec7a.txt`
- `_ARCHIVED_atlas-intel-next/public/file.svg`
- `_ARCHIVED_atlas-intel-next/public/globe.svg`
- `_ARCHIVED_atlas-intel-next/public/next.svg`
- `_ARCHIVED_atlas-intel-next/public/og-default.png`
- `_ARCHIVED_atlas-intel-next/public/vercel.svg`
- `_ARCHIVED_atlas-intel-next/public/window.svg`
- `_ARCHIVED_atlas-intel-next/scripts/indexnow.ts`
- `_ARCHIVED_atlas-intel-next/tsconfig.json`
- `_ARCHIVED_atlas-intel-next/vercel.json`
- `docs/SECURITY_GUARDRAILS.md`
- `docs/product_context_pack.md`
- `plans/PR-Drop-Archived-Intel-Next.md`

## Mechanism

`git rm -r` on the tree. The history is untouched, so the code is recoverable
from git if the experiment is ever revived — which is the argument for deleting
rather than continuing to carry it in the working tree.

`docs/SECURITY_GUARDRAILS.md` previously said the lockfile was "intentionally
not enrolled for routine Dependabot churn". That sentence now records what was
actually wrong with it: the exclusion addressed update churn and left the
scanner untouched, so it converted a maintenance cost into a standing set of
alerts on `osv-full`'s scan of `main`. That job does not evaluate pull
requests at all (`.github/workflows/security_guardrails.yml:158-159`), so
nothing was ever blocked or gated, advisory or otherwise -- the cost is that
alerts from code the repo forbids using share the scanned set with alerts
from code that ships.
## Intentional

- **Deleted rather than scope-excluded from `osv-scan`.** Excluding it would
  suppress the symptom and keep 1.1 MB of code that its own documentation says
  must not be used. Deleting removes the reason to scan it.
- **`HARDENING.md` is left exactly as written.** Its 2026-06-16 entry "Rotate
  archived IndexNow key" is a dated log entry, and this deletion does **not**
  resolve it: the key is in git history, and removing the working-tree copy
  does not rotate it. Editing a dated entry to imply otherwise would falsify
  the record.
- **`plans/archive/PR-Security-Guardrail-CI.md` is left alone** for the same
  reason — archived plans record what was decided at the time.
## Deferred

The rest of ATLAS #2375: the two criticals and 22 highs in root
`requirements.txt` (`transformers==4.57.6`, `cryptography==49.0.0` direct;
aiohttp/setuptools transitive), and the disposition of the non-deployed
manifests.

Parking predicate: hardening is parked when it protects a caller that does not
exist yet, or an input shape this change cannot receive. Nothing qualifies —
this PR adds no code paths.

Parked hardening: none.
## Verification

- Inbound-reference search across the repo, excluding the tree itself:
  `git grep -n "_ARCHIVED_atlas-intel-next" -- . ':(exclude)_ARCHIVED_atlas-intel-next/**'`
  returns only documentation — `HARDENING.md` (dated log), this PR's
  `docs/SECURITY_GUARDRAILS.md` update, and `plans/archive/`. **No code, no
  workflow, no config.**
- That search used the prefixed name only, and missed two files naming the
  BARE `atlas-intel-next` (no prefix): `README.md`'s directory tree and
  `docs/product_context_pack.md`'s relevant-surfaces list. An independent
  review caught this; a repo-wide sweep for the bare name found no further
  live references — the remaining hits are the gitleaks baseline (a
  historical secret-scan record keyed to a specific commit,
  `20c8f7a3b`, not a live path) and the already-reviewed `HARDENING.md` /
  `plans/archive/` / `scripts/migrate_bundled_posts_to_db.py`. Both live
  references are now removed.
- `.github/workflows/npm_package_checks.yml` matrix lists `atlas-admin-ui`, `atlas-churn-ui`,
  `atlas-mobile`, `atlas-ui` — the deleted tree was never a target.
- `.github/dependabot.yml` has no entry for the path, matching what
  `docs/SECURITY_GUARDRAILS.md` claimed.
- The tree's own its own DO_NOT_USE note states it was never in production.
- Expected effect: the 46 alerts attributed to
  `_ARCHIVED_atlas-intel-next/package-lock.json` leave the open set. This is a
  **prediction to confirm after merge**, not a measured result — `osv-scan`
  reports on `main`, so it cannot be observed from the branch.
## Estimated diff size

| File | LOC |
|---|---:|
| `README.md` | 1 |
| `_ARCHIVED_atlas-intel-next/.gitignore` | 41 |
| `_ARCHIVED_atlas-intel-next/.gitkeep` | 1 |
| `_ARCHIVED_atlas-intel-next/AGENTS.md` | 5 |
| `_ARCHIVED_atlas-intel-next/CLAUDE.md` | 1 |
| `_ARCHIVED_atlas-intel-next/DO_NOT_USE.md` | 9 |
| `_ARCHIVED_atlas-intel-next/README.md` | 36 |
| `_ARCHIVED_atlas-intel-next/app/(app)/account/page.tsx` | 222 |
| `_ARCHIVED_atlas-intel-next/app/(app)/affiliates/page.tsx` | 648 |
| `_ARCHIVED_atlas-intel-next/app/(app)/blog-review/page.tsx` | 391 |
| `_ARCHIVED_atlas-intel-next/app/(app)/briefing-review/page.tsx` | 394 |
| `_ARCHIVED_atlas-intel-next/app/(app)/campaign-review/page.tsx` | 389 |
| `_ARCHIVED_atlas-intel-next/app/(app)/challengers/page.tsx` | 338 |
| `_ARCHIVED_atlas-intel-next/app/(app)/dashboard/page.tsx` | 345 |
| `_ARCHIVED_atlas-intel-next/app/(app)/layout.tsx` | 15 |
| `_ARCHIVED_atlas-intel-next/app/(app)/leads/page.tsx` | 586 |
| `_ARCHIVED_atlas-intel-next/app/(app)/onboarding/page.tsx` | 196 |
| `_ARCHIVED_atlas-intel-next/app/(app)/prospects/page.tsx` | 1270 |
| `_ARCHIVED_atlas-intel-next/app/(app)/reports/[id]/page.tsx` | 364 |
| `_ARCHIVED_atlas-intel-next/app/(app)/reports/page.tsx` | 789 |
| `_ARCHIVED_atlas-intel-next/app/(app)/reviews/[id]/page.tsx` | 412 |
| `_ARCHIVED_atlas-intel-next/app/(app)/reviews/page.tsx` | 279 |
| `_ARCHIVED_atlas-intel-next/app/(app)/vendor-targets/page.tsx` | 584 |
| `_ARCHIVED_atlas-intel-next/app/(app)/vendors/[name]/page.tsx` | 796 |
| `_ARCHIVED_atlas-intel-next/app/(app)/vendors/page.tsx` | 264 |
| `_ARCHIVED_atlas-intel-next/app/(auth)/forgot-password/page.tsx` | 112 |
| `_ARCHIVED_atlas-intel-next/app/(auth)/layout.tsx` | 16 |
| `_ARCHIVED_atlas-intel-next/app/(auth)/login/page.tsx` | 100 |
| `_ARCHIVED_atlas-intel-next/app/(auth)/reset-password/page.tsx` | 137 |
| `_ARCHIVED_atlas-intel-next/app/(auth)/signup/page.tsx` | 184 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/[slug]/blog-post-content.tsx` | 52 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/[slug]/page.tsx` | 187 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/blog/page.tsx` | 55 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/layout.tsx` | 69 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/methodology/page.tsx` | 171 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/page.tsx` | 232 |
| `_ARCHIVED_atlas-intel-next/app/(marketing)/report/page.tsx` | 1078 |
| `_ARCHIVED_atlas-intel-next/app/globals.css` | 19 |
| `_ARCHIVED_atlas-intel-next/app/layout.tsx` | 45 |
| `_ARCHIVED_atlas-intel-next/app/robots.ts` | 29 |
| `_ARCHIVED_atlas-intel-next/app/sitemap.ts` | 67 |
| `_ARCHIVED_atlas-intel-next/components/ArchetypeBadge.tsx` | 61 |
| `_ARCHIVED_atlas-intel-next/components/AtlasHeroScene.tsx` | 187 |
| `_ARCHIVED_atlas-intel-next/components/AtlasRobotLogo.tsx` | 28 |
| `_ARCHIVED_atlas-intel-next/components/AtlasRobotScene.tsx` | 58 |
| `_ARCHIVED_atlas-intel-next/components/BlogChartRenderer.tsx` | 167 |
| `_ARCHIVED_atlas-intel-next/components/ChurnChart.tsx` | 63 |
| `_ARCHIVED_atlas-intel-next/components/DataTable.tsx` | 218 |
| `_ARCHIVED_atlas-intel-next/components/ErrorBoundary.tsx` | 64 |
| `_ARCHIVED_atlas-intel-next/components/Layout.tsx` | 29 |
| `_ARCHIVED_atlas-intel-next/components/PipelineStatus.tsx` | 104 |
| `_ARCHIVED_atlas-intel-next/components/PublicLayout.tsx` | 104 |
| `_ARCHIVED_atlas-intel-next/components/SeoHead.tsx` | 103 |
| `_ARCHIVED_atlas-intel-next/components/Sidebar.tsx` | 160 |
| `_ARCHIVED_atlas-intel-next/components/StatCard.tsx` | 35 |
| `_ARCHIVED_atlas-intel-next/components/UpgradeGate.tsx` | 35 |
| `_ARCHIVED_atlas-intel-next/components/UrgencyBadge.tsx` | 26 |
| `_ARCHIVED_atlas-intel-next/components/report-library/ReportEvidencePanel.tsx` | 195 |
| `_ARCHIVED_atlas-intel-next/components/report-library/SubscriptionScheduler.tsx` | 548 |
| `_ARCHIVED_atlas-intel-next/components/report-renderers/SpecializedReportData.tsx` | 2495 |
| `_ARCHIVED_atlas-intel-next/components/report-renderers/StructuredReportData.tsx` | 525 |
| `_ARCHIVED_atlas-intel-next/content/blog/index.ts` | 58 |
| `_ARCHIVED_atlas-intel-next/eslint.config.mjs` | 18 |
| `_ARCHIVED_atlas-intel-next/lib/api/blog.ts` | 107 |
| `_ARCHIVED_atlas-intel-next/lib/api/client.ts` | 718 |
| `_ARCHIVED_atlas-intel-next/lib/api/config.ts` | 8 |
| `_ARCHIVED_atlas-intel-next/lib/auth/AuthContext.tsx` | 193 |
| `_ARCHIVED_atlas-intel-next/lib/constants.ts` | 8 |
| `_ARCHIVED_atlas-intel-next/lib/hooks/useApiData.ts` | 119 |
| `_ARCHIVED_atlas-intel-next/lib/hooks/usePlanGate.ts` | 22 |
| `_ARCHIVED_atlas-intel-next/lib/reportConstants.ts` | 15 |
| `_ARCHIVED_atlas-intel-next/lib/reportLibrary.ts` | 634 |
| `_ARCHIVED_atlas-intel-next/lib/reportNormalization.ts` | 167 |
| `_ARCHIVED_atlas-intel-next/lib/reportViewModels.ts` | 801 |
| `_ARCHIVED_atlas-intel-next/lib/robot.ts` | 593 |
| `_ARCHIVED_atlas-intel-next/lib/types.ts` | 513 |
| `_ARCHIVED_atlas-intel-next/lib/types/index.ts` | 479 |
| `_ARCHIVED_atlas-intel-next/lib/types/reportViewModels.ts` | 542 |
| `_ARCHIVED_atlas-intel-next/next.config.ts` | 17 |
| `_ARCHIVED_atlas-intel-next/package-lock.json` | 7543 |
| `_ARCHIVED_atlas-intel-next/package.json` | 33 |
| `_ARCHIVED_atlas-intel-next/postcss.config.mjs` | 7 |
| `_ARCHIVED_atlas-intel-next/public/6a1d58a9cdf74c7b9c948ac27f86ec7a.txt` | 1 |
| `_ARCHIVED_atlas-intel-next/public/file.svg` | 1 |
| `_ARCHIVED_atlas-intel-next/public/globe.svg` | 1 |
| `_ARCHIVED_atlas-intel-next/public/next.svg` | 1 |
| `_ARCHIVED_atlas-intel-next/public/og-default.png` | 0 |
| `_ARCHIVED_atlas-intel-next/public/vercel.svg` | 1 |
| `_ARCHIVED_atlas-intel-next/public/window.svg` | 1 |
| `_ARCHIVED_atlas-intel-next/scripts/indexnow.ts` | 54 |
| `_ARCHIVED_atlas-intel-next/tsconfig.json` | 34 |
| `_ARCHIVED_atlas-intel-next/vercel.json` | 3 |
| `docs/SECURITY_GUARDRAILS.md` | 15 |
| `docs/product_context_pack.md` | 1 |
| `plans/PR-Drop-Archived-Intel-Next.md` | 344 |
| **Total** | **29186** |
