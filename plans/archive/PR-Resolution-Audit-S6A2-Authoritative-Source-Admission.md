# PR-Resolution-Audit-S6A2-Authoritative-Source-Admission

## Why this slice exists

Issue #2060 serializes this slice after merged S6A.1 (#2061), which closes the
classifier and proves the submit route. Normal `POST /content-ops/execute` still
merges provider output as defaults: a caller sends raw `source_material`, the provider
filters private rows/comments, then `merge_content_ops_input_package` restores
the raw value over the normalized package and private prose can reach the stored
artifact. This remaining #2060 privacy risk makes Production hardening apt.

Exact-head review found two construction-boundary gaps in the new contract:
placing the field before `outputs` changes the exported positional ABI, and a
direct dataclass constructed with a bare string bypasses mapping normalization
so the merge iterates characters and loses authority. These are one contract-
normalization defect plus one compatibility defect, not merge-policy failures.
The review delta may exceed the 400-line soft target because both regressions
must ship with the privacy authority contract they harden.

### Problem-derived contract

- Root cause: `ContentOpsInputPackage` has no way to distinguish a provider
  default from a provider-owned admission result. Its merge function applies
  every caller `inputs` key after every package key, so normalization and
  privacy admission have no authority at their own boundary. The defect is the
  missing ownership contract, not the support-ticket classifier or another
  downstream scrub rule.
- Correct fix must touch/change: add an internal exact-key authority contract
  to `ContentOpsInputPackage`; make package merge preserve a present provider
  value for declared authoritative keys while retaining caller precedence for
  other inputs and request-level fields; mark only support-ticket
  `source_material` authoritative; preserve it through Atlas package rebuilds;
  normalize direct and mapping-shaped declarations identically without moving
  any existing positional parameter; and prove the real `/execute` ->
  report-store path with mixed inputs.
- Must not change: caller precedence for packages that declare no authoritative
  keys; request-level `outputs`, target mode, ingestion profile, limits, budget,
  cache, and other explicit options; review/competitive/generic source-material
  behavior; classifier vocabulary; submit route; product shape; row caps;
  adjacent processing; schemas; #2037's paused files; or unrelated lanes.
  Existing positional `provider, inputs, outputs, ...` construction must not
  change meaning.

## Scope (this PR)

Ownership lane: resolution-audit/privacy-admission
Slice phase: Production hardening

1. Add a default-empty `authoritative_input_keys` contract at the one package/
   request merge choke point without changing undeclared-key precedence.
2. Declare the support-ticket package's present `source_material` value
   authoritative, including an empty normalized list after all raw rows reject.
3. Prove generic `/execute` retains public evidence but cannot restore a nested
   private sentinel into its response, snapshot, or stored report artifact.
4. Fold the required teardown archival for this session's merged #2061 and
   #2070 plans into this branch and refresh the archive index; no other plan is
   moved.

Max files: 9

### Review Contract

- Acceptance criteria:
  - [ ] A package with no authoritative keys preserves the existing rule that
        explicit caller `inputs` override provider defaults.
  - [ ] A declared authoritative key uses the package value only when that key
        is present in package inputs; undeclared keys in the same request still
        use explicit caller values.
  - [ ] Mapping-shaped packages and `as_dict()` preserve the authority contract
        without exposing it in user-facing input-provider diagnostics.
  - [ ] A direct bare-string authority declaration normalizes to one exact key
        and blocks raw caller reinjection like the tuple form.
  - [ ] Existing third-positional-argument construction still binds `outputs`;
        the new authority field is appended after all existing fields.
  - [ ] Support-ticket packages declare only `source_material` authoritative,
        including empty filtered results and selected-source warning rebuilds.
  - [ ] All default-empty packages retain caller precedence; authority is never
        inferred from provider names.
  - [ ] Real `POST /content-ops/execute` with mixed public/private support-ticket
        comments completes, retains public evidence, and excludes the private
        sentinel from response, snapshot, and stored artifact.
  - [ ] Existing row limits, output selection, request-level overrides, and
        customer-facing artifact shape remain unchanged.
  - [ ] Only the two merged plan docs owned by this session are archived.
- Reachability proof: call the real `/content-ops/execute` route with the Atlas
  request-aware support-ticket provider, real FAQ deflection report service,
  and in-memory report-store adapter; inspect response, stored snapshot, and
  stored artifact.
- Affected surfaces: package merge, support-ticket construction, Atlas package
  reconstruction, and generic execute admission.
- Risk areas: privacy leakage, accidental precedence reversal for unrelated
  providers, empty-filtered-package fallback, serialized package compatibility.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R13, R14; boundary probe
  required for the authority guard.

### Files touched

- `atlas_brain/_content_ops_input_provider.py`
- `extracted_content_pipeline/content_ops_input_provider.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S6A2-Authoritative-Source-Admission.md`
- `plans/archive/PR-Pre-Push-Caller-Hints-Timeout.md`
- `plans/archive/PR-Resolution-Audit-S6A-Structured-Privacy-Semantics.md`
- `tests/test_atlas_content_ops_input_provider.py`
- `tests/test_extracted_content_ops_input_provider.py`

## Mechanism

`ContentOpsInputPackage` gains `authoritative_input_keys`, defaulting to empty.
Mapping normalization and `as_dict()` retain it. Merge starts from provider
inputs and applies caller values unless a key is both declared authoritative and
present in provider inputs; a declaration cannot mask a value the provider did
not produce. Request-level behavior is untouched.
The field is appended after all existing constructor fields, and frozen
dataclass `__post_init__` applies `_string_tuple` once so direct bare strings,
tuples, and mapping packages share one canonical tuple representation.
`build_support_ticket_input_package` declares exactly `source_material`. That
value is always present, including `[]` when every raw row fails admission, so
the filtered result remains authoritative in both mixed and all-private cases.
No provider-name or source-type inference is added to the generic merge. Atlas's
selected-source warning reconstruction copies the tuple; other packages keep
the empty default.

## Intentional

- Authority is explicit package data, not inferred from provider names, source
  types, or key vocabulary. This avoids silently changing unrelated providers.
- Authority is exact-key and shallow because `ContentOpsRequest.inputs` already
  treats `source_material` as one value contract; recursive partial merging
  would let rejected rows re-enter through nested bundle shapes.
- A declared key is authoritative only when present in package inputs. The
  contract protects produced normalized values without turning a declaration
  typo or omitted value into an implicit deletion.
- `authoritative_input_keys` stays out of public `input_provider` diagnostics;
  it is merge control, not buyer-visible metadata.
- The downstream generic private-field guard remains defense in depth. Widening
  its grammar would duplicate S6A.1 and would not fix the upstream restoration
  root.
- The two plan moves are required post-merge teardown owned by this session,
  folded here as AGENTS.md permits; they introduce no runtime or product change.

## Deferred

- None for #2060; after merge, verify the tracker and close the completed issue.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_content_ops_input_provider.py
  tests/test_atlas_content_ops_input_provider.py -q -k 'authoritative or
  declared_provider_input or absent_authoritative or selected_source_warning_rebuild
  or cannot_restore_provider_rejected or expands_support_ticket_source_material'`
  (`8 passed, 29 deselected`).
- Both full input-provider test files (`37 passed`); support-ticket package and
  provider callers (`254 passed`); control-surface/execution callers
  (`237 passed, 1 skipped`).
- Bash `scripts/run_extracted_pipeline_checks.sh` (`10726 passed, 21 skipped`;
  one third-party `pynvml` deprecation warning).
- Extracted manifest, forbidden-import, standalone-debt, and ASCII checks:
  passed. The managed push hook gates plan/body sync, local review, and
  whitespace before remote mutation.
- Exact-head review regressions for bare-string reinjection and positional
  output ABI (`5 passed, 7 deselected`).
- Broad input-provider, support-ticket, control-surface, and execution callers
  after the review fix (`530 passed, 1 skipped`).
- Bash `scripts/run_extracted_pipeline_checks.sh` after the review fix (`10728
  passed, 21 skipped`; one third-party `pynvml` deprecation warning).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_input_provider.py` | 1 |
| `extracted_content_pipeline/content_ops_input_provider.py` | 34 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 1 |
| `plans/INDEX.md` | 4 |
| `plans/PR-Resolution-Audit-S6A2-Authoritative-Source-Admission.md` | 184 |
| `plans/archive/PR-Pre-Push-Caller-Hints-Timeout.md` | 0 |
| `plans/archive/PR-Resolution-Audit-S6A-Structured-Privacy-Semantics.md` | 0 |
| `tests/test_atlas_content_ops_input_provider.py` | 131 |
| `tests/test_extracted_content_ops_input_provider.py` | 108 |
| **Total** | **463** |
