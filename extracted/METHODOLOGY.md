# Extraction Methodology

## Phase Legend

| Phase | Meaning | Product bar |
|---|---|---|
| Phase 1 | Snapshot scaffold | Files are copied from Atlas and validated for byte equality. Atlas remains the runtime owner. |
| Phase 2 | Standalone substrate | Core config, DB, auth, protocol, and service ports can run through extracted-owned adapters behind a standalone toggle. |
| Phase 3 | Product-owned runtime | Modules no longer require Atlas imports for their product path. Host integrations are explicit ports. |

## Manifest Model

Each product owns a `manifest.json`.

`mappings` entries are byte-synced from Atlas:

```json
{"source": "atlas_brain/path.py", "target": "extracted_product/path.py"}
```

`owned` entries are product-owned:

```json
{"target": "extracted_product/path.py"}
```

Owned files are skipped by byte-sync validation and sync, but still covered by
ASCII and import checks. This is the controlled handoff from snapshot scaffold
to product-owned implementation.

## Standalone Toggle Pattern

Standalone toggles must be explicit environment flags, for example:

- `EXTRACTED_LLM_INFRA_STANDALONE=1`
- `EXTRACTED_COMP_INTEL_STANDALONE=1`
- `EXTRACTED_PIPELINE_STANDALONE=1`

Default mode should preserve Atlas behavior. Standalone mode should prefer
extracted-owned substrate and fail closed where a host application must provide
an adapter.

## Dependency Rule

Cross-product dependencies should point to another extracted product when that
product owns the capability. For example, competitive intelligence should use
extracted LLM infrastructure for LLM protocols and routing substrate instead of
importing Atlas LLM internals.

Atlas can consume extracted products during the transition. Extracted products
should not add new Atlas dependencies once a surface becomes product-owned.

## Physical Move Rule

Do not move existing `extracted_*` packages into `extracted/` while active PRs
are modifying those packages. First add umbrella docs and shared tooling, then
switch wrappers to shared scripts, then move one product at a time with
compatibility imports and CI updates.
