# PR: audit MINOR/NIT cleanup batch — execution status + immutable catalog

## Why this slice exists

The Content Ops post-merge audit listed 9 MINOR + 2 NIT findings.
Several have been addressed by intermediate PRs (input validation
by #377, quality_gates_enabled by PR-OptionA-4/5, etc.). This PR
batches three related findings still open in
``content_ops_execution.py`` and ``control_surfaces.py``:

1. **MINOR — All-steps-failed reports "partial" not "failed".**
   When every step errors, the executor still returns
   ``status="partial"`` because the logic is just "any errors ->
   partial." Operators reading the dashboard see "partial" and
   assume some steps succeeded; the truth is nothing did.
2. **MINOR — ``_failed_step`` and ``error`` dict have inconsistent
   shapes.** The ``ContentOpsStepExecution`` records use the step's
   ``output``, ``runner``, ``status``, ``error`` fields; the parallel
   ``errors`` payload at the result level uses
   ``{"output": ..., "reason": ...}``. Two different shapes for "the
   same thing went wrong." Reviewers and dashboards have to
   reconcile.
3. **NIT — ``OUTPUT_CATALOG`` and ``PRESETS`` are mutable
   module-level dicts.** Anything in the process can mutate them
   accidentally. They're effectively constants; should be
   ``Mapping`` / frozen.

## Scope (this PR)

Three small, self-contained fixes.

### Fix 1: distinguish "partial" from "failed"

```python
# Before
status = "completed" if not errors else "partial"

# After
if not errors:
    status = "completed"
elif len(errors) >= len(plan.steps):
    status = "failed"
else:
    status = "partial"
```

When every planned step errored, the result is now
``status="failed"``. ``"partial"`` is reserved for the genuine mixed
case (some succeeded, some didn't). ``"completed"`` stays for
all-success.

Edge case: ``plan.steps`` is empty (no outputs requested). The
existing executor returns ``"completed"`` because ``errors`` is also
empty. This PR doesn't change that -- ``len(errors) >= len(steps)``
where both are 0 is still 0 errors -> not the failed branch.

### Fix 2: align step-execution and result-error shapes

The ``ContentOpsStepExecution`` already carries ``output``,
``runner``, ``status``, ``error``. The result's ``errors`` tuple
uses ``{"output": ..., "reason": ...}``. Add ``runner`` to the
result-level dict and rename ``reason`` -> ``error`` for shape
consistency. Old shape stays accessible (extra ``reason`` key
preserved for backwards compat -- hosts may be parsing the field by
name).

### Fix 3: freeze OUTPUT_CATALOG and PRESETS

Switch ``dict[str, X]`` -> ``Mapping[str, X]`` at the type
annotation level and use ``types.MappingProxyType`` to enforce
immutability at runtime. Existing ``OUTPUT_CATALOG.get(...)`` /
``PRESETS.get(...)`` reads work unchanged.

## Intentional (looks wrong but is deliberate)

- **No new abstraction.** Three small fixes; nothing to abstract.
- **Backwards-compat ``reason`` key kept on the error dict.** Hosts
  parsing the JSON shape may key on ``reason``. Adding ``error`` as
  a new key alongside (with the same value) preserves the old
  contract while migrating the new field as the canonical name.
  Future cleanup can drop ``reason`` once hosts migrate.
- **``MappingProxyType`` over ``frozendict``.** Standard library;
  no new dependency. Same semantics for ``.get()`` / iteration /
  membership tests; raises ``TypeError`` on assignment.

## Deferred (still on purpose)

- ``topic`` for blog_post (still no service-side landing surface).
- ``PR-Campaign-Config-V2`` (breaking change to remove the legacy
  ``channel`` field).
- Other audit MINORs not in this batch -- separate cleanup PRs.

## Verification

- ``pytest`` on the touched test suites
- ``python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`` -> clean
- ``bash scripts/check_ascii_python.sh`` -> passed

## Sibling references

- Audit doc:
  ``docs/audits/ai_content_ops_post_merge_audit_2026-05.md``
- Input-validation MINORs handled by #377.
