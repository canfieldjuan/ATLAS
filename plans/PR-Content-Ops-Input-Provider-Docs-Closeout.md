# PR: Content Ops Input Provider Docs Closeout

## Why this slice exists

The support-ticket input-provider path now has the core implementation pieces:
the extracted input package, the extracted provider adapter, the Atlas host
mount, and route-level validation. The package README and STATUS still mostly
describe inline `source_material` and FAQ Markdown generation, which leaves the
new provider handoff under-documented for future sessions.

This slice updates docs only. It does not change generation, ingestion, or FAQ
article behavior.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

Slice phase: Product polish

1. Document that support-ticket source material can be packaged into Content Ops
   defaults for FAQ Markdown, landing pages, and blog planning.
2. Document the host-owned `ContentOpsInputProvider` handoff and the Atlas host
   mount at a high level.
3. Clarify that file parsing and persisted import lookup remain host/ingestion
   responsibilities, not generator responsibilities.

### Files touched

- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-Input-Provider-Docs-Closeout.md`

## Mechanism

The README gains a short support-ticket input-provider note near the existing
support-ticket example flow. STATUS gains a bullet under the component inventory
summarizing the package/provider/host handoff and the ownership boundary.

## Intentional

- No code changes. This is a documentation closeout after the implementation
  slices landed.
- No standalone FAQ article promises. That work remains owned by the FAQ
  session.
- No upload/import lookup contract. Ingestion owns loading persisted files or
  import rows before handing source material to Content Ops.

## Deferred

- Future PR owned by the ingestion lane: document persisted import lookup once
  the loader contract exists.
- Future PR owned by the FAQ session: document standalone FAQ article output
  once that contract exists.
- Parked hardening: none.

## Verification

- `git diff --check` - passed.
- `scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| README | ~25 |
| STATUS | ~15 |
| **Total** | **~105** |
