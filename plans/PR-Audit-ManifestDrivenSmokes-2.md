# PR-Audit-ManifestDrivenSmokes-2: convert compintel-standalone smoke to manifest-driven

## Why this slice exists

Final smoke in the drift-protection track started by #427 and #428.
`scripts/smoke_extracted_competitive_intelligence_standalone.py` (209
LOC) still uses two hardcoded lists:

- A 39-entry MODULES list for the import sweep (Phase 1).
- A 26-entry `owned_files` tuple for the "still imports atlas_brain"
  string scan (Phase 4).

Both are drift hazards: when a new file enters the manifest, neither
list is updated automatically.

A diff between hardcoded and manifest reveals **real coverage gaps**:

| Direction | Count | What |
|---|---|---|
| In hardcoded list, NOT in manifest | 14 | Untracked standalone shims (config, storage/database, pipelines/llm, ...). 9 are verified by Phase 2 owner checks; 5-6 only get import-time verification here. |
| In manifest, NOT in hardcoded list | 6 | `b2b_battle_cards`, `b2b_vendor_briefing`, `autonomous/visibility`, `mcp/b2b/cross_vendor`, `mcp/b2b/displacement`, `services/b2b/product_claim` |
| In hardcoded `owned_files`, NOT in manifest owned | 1 | `vendor_briefing_delivery` (which is actually a mapping; checking mappings for atlas_brain text is conceptually wrong) |
| In manifest owned, NOT in hardcoded `owned_files` | 1 | `product_claim.py` |

The manifest-driven refactor closes both drift hazards while preserving
the substrate-verification concerns (Phases 2 and 3) intact.

## Scope (this PR)

Refactor two of the four phases in
`smoke_extracted_competitive_intelligence_standalone.py`:

- **Phase 1 (import sweep)**: replace hardcoded MODULES list with
  manifest walk (mappings + owned), plus a small
  `_EXTRA_STANDALONE_SHIMS` curated list for untracked standalone
  substrate modules that should still get import-time verification.
- **Phase 4 (owned-files atlas_brain scan)**: replace hardcoded
  `owned_files` tuple with manifest `owned` walk.

Untouched:

- **Phase 2 (owner verification)**: 12 substrate-routing assertions.
  Each entry is a specific module-attr-substrate triple, not a
  drift-prone coverage list.
- **Phase 3 (fallback probes)**: 6 module-level
  `__atlas_fallback_probe__` checks. Same shape -- substrate
  behavior, not coverage.

## The `_EXTRA_STANDALONE_SHIMS` curated list

Six standalone substrate modules are imported by the existing smoke
but are not on the manifest:

```python
_EXTRA_STANDALONE_SHIMS = (
    "extracted_competitive_intelligence.autonomous.tasks.campaign_suppression",
    "extracted_competitive_intelligence.pipelines.llm",
    "extracted_competitive_intelligence.services.b2b.pdf_renderer",
    "extracted_competitive_intelligence.services.campaign_sender",
    "extracted_competitive_intelligence.services.crm_provider",
    "extracted_competitive_intelligence.services.email_provider",
)
```

These are hand-rolled shims that satisfy imports under the standalone
toggle. They were not added to the manifest's `owned` list when they
were created (a separate methodology gap; see Deferred). To preserve
import coverage today, they're listed as a small constant. Adding them
to the manifest is the better fix; that's a separate slice.

## Why keep Phase 2 hardcoded

The `checks` list of 12 entries asserts that, e.g., `CompIntelSettings`
resolves to `extracted_competitive_intelligence._standalone.config`,
not to a fallback. Each entry encodes a substrate-routing decision,
not a coverage list -- adding a new module-attr to verify requires
the maintainer to know which substrate it should resolve to. Manifest
walks can't infer that. Leave hardcoded.

## Why keep Phase 3 hardcoded

The 6-entry tuple of namespace modules that should not have
`__atlas_fallback_probe__` is similarly non-drift-prone. It's a
fixed set of namespaces where the substrate must fail closed under
the toggle. Adding a new namespace requires knowing the fallback
contract.

## Failure classification

Tighter than #427 / #428: under the standalone toggle, ALL imports
should succeed (the toggle is meant to enable substrate-only
operation). This smoke gates on any exception, not just decoupling
ones. Matches the existing script's tight-gate behavior.

If a 3rd-party env failure surfaces under standalone (which would be
unusual since the substrate is meant to handle it), that is a real
substrate gap and should fail the gate.

## Intentional (looks wrong but is deliberate)

- **Tighter gate than #427 / #428.** Different smoke, different
  contract. Standalone-mode coverage should be hermetic.
- **`_EXTRA_STANDALONE_SHIMS` is not just merged into the manifest
  here.** Mixing a methodology change (adding entries to
  `manifest.owned`) into a refactor PR couples concerns. The 6
  shims will move to manifest in a follow-up slice.
- **Phase 4 narrows from 26 to 26 entries** (one swap:
  `vendor_briefing_delivery` (mapping) out, `product_claim` (owned)
  in). Net coverage is unchanged in count but conceptually correct.
  Mappings are byte-synced from atlas_brain and may legitimately
  contain `atlas_brain.` text -- scanning them is a category error.
- **Owner-verification (Phase 2) and fallback probes (Phase 3) stay
  hardcoded.** They're not coverage lists; they're substrate
  contracts.
- **No DRY extraction of manifest walk into shared helper.** This
  is the 4th caller, but the previous 3 share an identical Phase-1-
  only shape; this script has Phases 1, 2, 3, 4 layered. Premature
  abstraction. Wait for a 5th caller.

## Deferred (looks missing but is on purpose)

- Add the 6 `_EXTRA_STANDALONE_SHIMS` to `manifest.json:owned`.
  Methodology change; separate slice.
- Refactor Phases 2 and 3 to be data-driven (e.g., declarative
  config files). Different concern from drift hazard.
- Extract a shared manifest-walk helper (wait for 5th caller).
- Catch the broader methodology gap (50+ untracked .py files in
  `extracted_content_pipeline`).

## Verification

- `python3 scripts/smoke_extracted_competitive_intelligence_standalone.py`
  -> exits 0 with all phases green.
- New regression test asserts the smoke passes in current state and
  that `_load_modules` includes both manifest entries and the
  extra-shims constant.
- `bash scripts/check_ascii_python.sh` -> passed.
- Repeat full sweep across both packages: 0 standalone-import
  failures.

## Conflict check

- No file overlap with any open PR.
- Independent of #422, #425, #427, #428 (all merged).

## Diff size

- Script refactor: ~50-80 LOC (replace MODULES + owned_files with
  manifest walk + extras; keep Phases 2, 3 untouched)
- Regression test: ~80 LOC
- Plan doc: ~140 LOC

## After this lands

All 4 smokes we control are drift-resistant on their drift-prone
lists. The class of failures that bit #422 and #425 cannot recur in
either default-mode or standalone-mode for either package. Remaining
methodology debt:

- 6 standalone shims still untracked in compintel manifest (queued
  for follow-up).
- ~50 untracked .py files in `extracted_content_pipeline`
  (methodology question -- separate concern).
