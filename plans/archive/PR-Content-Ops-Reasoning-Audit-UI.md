# PR: Content Ops reasoning audit UI

## Why this slice exists

PR #418 added a compact `step.reasoning` audit object to Content Ops
execution results. The UI adapter still drops that field, so the
execution panel cannot show whether a completed step had a host
reasoning provider attached.

## Scope (this PR)

1. Add wire and domain types for the optional step reasoning audit.
2. Map the wire audit into camelCase domain fields.
3. Render a small execution-step badge when the audit is present.
4. Update one execution fixture so the TypeScript contract sees the new
   backend shape.
5. Claim this slice in the coordination table while the PR is open.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/execution-completed.json`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Reasoning-Audit-UI.md`

## Mechanism

The UI treats `step.reasoning` as execution-level readiness only. It
shows whether the service supports the seam and whether a provider was
attached. It does not expose or imply access to the full consumed
reasoning payload.

## Intentional

- No Reasoning Context Drawer.
- No backend changes.
- No full prompt payload rendering.
- No competitive-intelligence files touched.

## Deferred

- Drawer-ready consumed-context field.
- UI drawer after the backend exposes that field.

## Verification

- `npm run build` from `atlas-intel-ui`
- `git diff --check`

## Estimated diff size

- 7 files.
- About 110 inserted lines and 10 deleted lines.
- Well below the 400-line soft PR budget.
