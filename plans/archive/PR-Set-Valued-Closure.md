# PR-Set-Valued-Closure

## Why this slice exists

Juan identified the shared cause behind the 2026-07-26 review loops: builders
enumerated the members visible at authoring time instead of binding the set to a
source of truth or declaring the default for members not listed. The concrete
instances were Website #70, ATLAS #2216, #2222, #2223, and #2225. The current
code/doc state confirms `docs/GUARD_CLASS_CLOSURE.md` is the single source for
this bar, but its trigger is scoped to guards over open input; it does not
plainly catch copied workflow file lists, detector regex families, replaced-code
inventories, or other set-valued dependencies in decision paths.

This slice keeps the normative bar in one source because #2226 says the bar
changes by editing `docs/GUARD_CLASS_CLOSURE.md`, not by creating another
normative document. It also adds a reviewer-rule pointer, which that same source
requires for every doc that names the bar. It moves ahead of #2221-#2224 because
it attacks the round generator those PRs kept reproducing.

### Problem-derived contract

- Root cause: the guard-closure bar already rejects enumerate-instead-of-close,
  but the trigger is too narrow; authors see "guard over open input" and miss
  the same set failure in workflow mirrors, detectors, and refactor inventories.
- Correct fix must touch/change: widen `docs/GUARD_CLASS_CLOSURE.md` so
  set-valued dependencies in decision paths must declare CLOSED, DERIVED, or
  DEFAULTED before code; point the reviewer entrypoint at that widened trigger;
  and state the advisory trigger shape.
- Must not change: product/runtime behavior, current advisory-check behavior,
  required CI status, reviewer strength, review caps, or the open #2221-#2224
  branches.

## Scope (this PR)

Ownership lane: dev-workflow/set-valued-closure
Slice phase: Workflow/process

1. Widen the guard-class-closure trigger to set-valued dependencies in decision
   paths.
2. Add the mandatory Closure Declaration dispositions: CLOSED, DERIVED, and
   DEFAULTED.
3. State the advisory-first mechanical trigger shape for future detector work.
4. Add a pointer from the reviewer rules pack to the canonical set-valued
   dependency declaration, without copying the bar.

### Review Contract

- Acceptance criteria:
  - `docs/GUARD_CLASS_CLOSURE.md` still declares itself the single source for
    the closure bar.
  - Its trigger includes set-valued dependencies such as literal collections,
    regex alternations, copied lists, and behavior/caller/field inventories.
  - Builder-facing entrypoints point to the canonical trigger-B declaration, so
    the rule reaches the author before review rather than only after a bot
    finding.
  - It requires every triggering set to declare exactly one of CLOSED, DERIVED,
    or DEFAULTED before code, including docs-only PR bodies when the body is the
    admission artifact.
  - It preserves the existing three-requirement bar for trigger-A open-input
    guards and does not impose guard-only requirements on non-guard set-valued
    dependencies.
  - Implicit behavior/caller/field/input-shape inventories are bound to a named
    reviewable source surface.
  - It preserves advisory-first language for mechanical surfacing and does not
    claim the existing checker already enforces the widened trigger.
- Reachability proof: documentation/process rule only; future builders and
  reviewers read `docs/GUARD_CLASS_CLOSURE.md` through AGENTS and the reviewer
  rules pointer.
- Affected surfaces: `AGENTS.md`, `docs/GUARD_CLASS_CLOSURE.md`,
  `docs/REVIEWER_RULES.md`, and this plan artifact.
- Risk areas: overbroad trigger, accidental normative copy outside the source
  doc, implying implemented CI behavior that does not exist yet.
- Reviewer rules triggered: R1, R2, R10, R12, R13.

#### Closure declarations for this PR

- **CLOSED — closure-disposition vocabulary.** The set `{CLOSED, DERIVED,
  DEFAULTED}` is finite and repo-owned by this canonical doc. Unlisted
  dispositions are invalid because the rule requires exactly one of those three.
- **DERIVED — trigger-A inventory.** The open-input guard surface is derived from
  the pre-existing `docs/GUARD_CLASS_CLOSURE.md` trigger text and AGENTS 3k.1
  pointer; this PR preserves that source rather than replacing it with a new
  list.
- **DERIVED — trigger-B inventory.** The non-guard set-valued dependency surface
  is derived from #2226's stated pattern and its evidence ledger classes: literal
  member collections, regex/pattern families, copied repo lists, and bounded
  behavior/caller/field/input-shape inventories.
- **DEFAULTED — deferred detector coverage.** The future mechanical detector is
  explicitly advisory-first and defaults misses to review visibility rather than
  blocking, because this PR records the law before the detector has earned trust.

#### R13 same-class proof

The rule was checked against held-out set-valued dependency shapes not required
by the cited incidents. Each terminates by choosing a disposition instead of
extending a hand list:

- A release gate hand-copies environment names from deployment config -> CLOSED
  only if it cites the canonical environment enum; otherwise DERIVED from config.
- A report renderer branches on section keys -> DERIVED from the report schema,
  or CLOSED only if the schema names the finite repo-owned section set.
- A detector regex alternates over import idioms -> DERIVED from AST/schema
  inspection where possible; otherwise DEFAULTED so unrecognized idioms warn to
  the cheap side rather than claiming closure.
- A workflow mirror lists job or test names -> DERIVED from workflow YAML, not a
  copied list.
- A route or tool admission table lists allowed operations -> CLOSED only when
  the router/tool registry is the canonical source; otherwise DERIVED from that
  registry.

### Files touched

- `AGENTS.md`
- `docs/GUARD_CLASS_CLOSURE.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Set-Valued-Closure.md`

## Mechanism

The canonical doc now separates trigger A (open-input guards, still owing the
three guard requirements) from trigger B (set-valued dependencies, owing a
closure declaration unless they are also trigger-A guards). Trigger B names
examples that match the observed loops: literal collections, regex alternations,
copied repo lists, and behavior/caller/field inventories whose missing source
binding is a closure failure. The Closure Declaration section requires the author
to classify each set as CLOSED, DERIVED, or DEFAULTED before code, with the PR
body allowed as the artifact for Markdown-only docs-only changes. `AGENTS.md`
and `docs/REVIEWER_RULES.md` only point builders/reviewers to the canonical bar
so the new trigger is reachable. The advisory paragraph keeps the existing
advisory-first posture and describes the widened mechanical tell without
pretending the current Python guard checker already implements it.

## Intentional

- This PR edits `docs/REVIEWER_RULES.md` only to add a pointer. It does not copy
  the CLOSED / DERIVED / DEFAULTED bar outside the canonical source.
- This PR edits `AGENTS.md` only to add the builder-facing pointer needed for
  trigger-B reachability. It does not copy the canonical bar.
- This PR does not update `scripts/check_guard_class_closure.py`; #2226's first
  convergence step is to widen the single source. A detector expansion can
  follow from this source without another normative copy.
- The rule allows CLOSED sets when there is a real canonical repo-owned list; it
  rejects duplicated hand copies of that list.

## Deferred

- Expanding `scripts/check_guard_class_closure.py` to detect set-valued
  dependencies mechanically is deferred to a follow-up implementation slice.
- Reworking or superseding #2221-#2224 is deferred until this source-rule PR is
  reviewed.

Parked hardening: none.

## Verification

- git diff --check - OK.
- python scripts/audit_plan_doc.py plans/PR-Set-Valued-Closure.md - OK.
- python scripts/audit_plan_doc_files_touched.py plans/PR-Set-Valued-Closure.md origin/main - OK after commit.
- python scripts/audit_plan_doc_diff_size.py plans/PR-Set-Valued-Closure.md origin/main - OK after commit.
- python scripts/audit_plan_code_consistency.py plans/PR-Set-Valued-Closure.md - OK.
- python scripts/audit_review_rules_triggered.py origin/main --plan plans/PR-Set-Valued-Closure.md - OK.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | ~4 |
| `docs/GUARD_CLASS_CLOSURE.md` | ~104 |
| `docs/REVIEWER_RULES.md` | ~5 |
| `plans/PR-Set-Valued-Closure.md` | ~170 |
| **Total** | **~283** |
