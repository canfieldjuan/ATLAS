# PR-EOM-Contact-Write-Boundary-Guard

## Why this slice exists

Website issue canfieldjuan/Effingham_Office_Maids_Website#108 (Slice 0A under
umbrella #107, itself the first child of #105) asks for the enforcement floor
under the canonical EOM CRM write boundary: a CI gate that fails when a new SQL
write to `contacts` appears outside the approved provider module.

#2298 closed the operator-facing MCP create bypass. Nothing yet stops the *next*
one. A code investigation of `origin/main` found ~13 code-level contact create
paths and ~12 update paths, all currently funnelling through exactly two
`INSERT INTO contacts` statements, both in
`atlas_brain/services/crm_provider.py` (`:637` EOM inbound atomic, `:881`
generic). That narrowness is the whole opportunity: an allow-list of one module
is exact rather than aspirational, and Atlas has no ORM, so every contact write
is literal SQL a static reader can see.

The guard is worth having because a writer that bypasses the provider also
bypasses tenant stamping, provenance, normalization, and the
`eom_lead_lifecycle_events` audit ledger — the split-brain condition #105 exists
to end. `DatabaseCRMProvider.create_contact`'s own docstring (`:725-726`) already
promises a DB-level uniqueness net "migration 037 should add" that was never
built; this slice adds the enforcement that *was* buildable, at the code layer,
without pretending to be the one that was not.

### Problem-derived contract

- **Root cause:** the canonical-boundary decision is enforced only by convention.
  Every existing writer happens to route through `DatabaseCRMProvider`, but no
  mechanism detects a new module adding its own `INSERT INTO contacts`, so the
  boundary degrades silently one PR at a time.
- **Correct fix must touch/change:** add a static detector over Python string
  literals that finds contact-table writes and fails on INSERTs outside the
  allow-listed provider module; add a committed writer-inventory baseline so
  additions/moves/removals of write sites surface as a reviewable diff; add
  tests that prove the detector *fires* on planted violations, not merely that
  it runs on a clean tree; run the detector from a CI workflow whose trigger is
  broad enough to see a newly added file.
- **Must not change:** any runtime behavior. No production module, migration,
  schema, endpoint, or test of existing behavior is modified. The detector is
  read-only over the source tree and runs in CI only.

## Scope (this PR)

Ownership lane: eom-crm-canonical-boundary
Slice phase: workflow/process

1. Add `scripts/check_contact_write_boundary.py`: an AST-based detector that
   extracts Python string literals and matches SQL write statements against the
   `contacts` table. INSERT outside the allow-list is blocking; UPDATE/DELETE
   outside it is reported non-blocking while legacy writers are converged.
2. Add `tests/contact_write_boundary/baseline.json`: the committed writer
   inventory (17 production write sites today).
3. Add `tests/test_contact_write_boundary.py`: 62 tests, including planted
   violations that must fail the gate and false-positive pins that must not.
5. Record the gate in `ci/gates.yml` as `ci_blocking_not_required`; promoting it
   to a branch-required context is an operator action.
4. Wire the gate into a **dedicated, deliberately unfiltered** workflow,
   `.github/workflows/contact_write_boundary.yml`.

### Files touched

- `.github/workflows/contact_write_boundary.yml`
- `.github/workflows/contact_write_boundary_selfcheck.yml`
- `ci/gates.yml`
- `plans/PR-EOM-Contact-Write-Boundary-Guard.md`
- `scripts/audit_workflow_security_posture.py`
- `scripts/check_contact_write_boundary.py`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_contact_write_boundary.py`

### Review Contract

Acceptance criteria — each names a claim about the code or the evidence that
settles it:

1. The detector fires on a planted INSERT outside the provider and `main()`
   returns exit code 1 — `tests/test_contact_write_boundary.py::test_planted_insert_outside_provider_is_blocking`
   and `::test_planted_insert_makes_main_exit_nonzero`.
2. Baselining an INSERT does not silence it, so `--update-baseline` cannot be
   used as a one-command bypass — `::test_insert_is_never_silenced_by_the_baseline`.
3. Prose does not produce findings: the docstring "Create/update contacts in the
   Atlas CRM." in `scripts/import_calendar_contacts.py` must not be flagged —
   `::test_prose_docstring_is_not_a_write`. (This is a regression pin: the first
   draft matched bare `update contacts` and flagged exactly that line.)
4. Adjacent tables are not confused with `contacts` —
   `::test_neighbouring_tables_do_not_match` covers `contact_interactions` and
   `contacts_archive`.
5. The committed baseline is non-vacuous and matches the tree —
   `::test_baseline_inventory_is_not_vacuous` (asserts 2 INSERT + ≥10 UPDATE
   entries) and `::test_baseline_file_matches_the_tree`.
6. The approved surface is pinned at exactly two INSERT sites in one module —
   `::test_repository_has_exactly_the_known_insert_sites`.
7. Zero runtime diff: no file under `atlas_brain/` is modified by this PR.

Affected surfaces: CI only. Risk areas: false positives that would train
reviewers to bypass the gate (covered by criteria 3–4); a vacuous baseline that
detects no drift (criterion 5).

- Reviewer rules triggered: R2, R10.

R2 (test evidence / failure-branch fixtures per AGENTS.md 3h-3i) and R10
(maintainability) are the triggers `docs/REVIEWER_RULES.md` assigns to gate
predicates and evaluator scripts, which is what
`scripts/check_contact_write_boundary.py` is. R2 is satisfied by the planted-violation fixtures rather than by
clean-tree runs: criteria 1-2 prove bad input fails, criteria 3-4 prove good
input passes, so both error directions are probed. R10 is satisfied by the
allow-list living in one reviewable constant pair rather than in scattered
in-source markers.

### Boundary-change enumeration

- Boundary path/seam: CI admission for contact-table write sites; no runtime
  boundary changes.
- Replaced-path behaviors: none. No existing code path changes behavior; the
  gate is additive and read-only over the source tree.
- Guard-relevant fields: the allow-list pair (`INSERT_ALLOWED`,
  `MUTATION_ALLOWED`) and the three SQL operation patterns.
- Caller x input shape: CI job x repository tree, in four shapes - clean tree,
  tree with a planted INSERT outside the allow-list, tree with an INSERT inside
  it, and tree with prose that merely resembles SQL.

**Reachability proof:** the real entrypoint is the CI job. The detector was run
against the working tree with a temporary planted bypass module added under
`atlas_brain/services/` and returned exit code 1 naming that file; the file was
then removed and the same command returned exit code 0. Both transcripts are in
Verification below. The planted module is intentionally absent from the diff -
it exists only for the duration of the probe.

**Guard-class closure declaration**

- **Decision-driving member sets:** `INSERT_ALLOWED`, `MUTATION_ALLOWED`,
  `DYNAMIC_SCOPE`, `DRIVER_WRITE_METHODS`, `CREATE_OPERATIONS`, and the seven
  entries of `PATTERNS` (INSERT, MERGE, COPY, SELECT_INTO, TRUNCATE, UPDATE,
  DELETE) plus the derived `DYNAMIC` and `MERGE_UPDATE` classifications.
- **Input space: OPEN.** Arbitrary Python and SQL source, so coverage is
  demonstrated by planted fixtures per shape rather than enumeration.
- **Out-of-set default: REJECT for creates, REPORT for mutations.** A create
  form outside the allow-list blocks; UPDATE/DELETE/TRUNCATE surface without
  blocking while legacy writers converge; an unanalyzable file blocks; a
  missing inventory blocks.
- **Closure evidence, per shape:** every create form has a blocking fixture and
  every non-create a reporting one; both allow-list membership outcomes are
  tested; literal-vs-executable is tested in both directions for `.sql` and
  Python, for dollar bodies, and for EXECUTE position; dedup is tested for
  shared snippet, shared line, and shared key; inventory is tested for add,
  remove, unchanged, missing, and multiplicity; INSERT is proven
  non-silenceable by the baseline.
- **Reviewer rules:** R2 and R10 for the gate predicate, plus R12 (CI surface
  and trusted-base execution) and R13/R14 (class-level guard behaviour) from
  the workflow and classification changes.

**One producer per context.** The enforcement workflow registers
`pull_request_target` only. Registering the ordinary event on the same workflow
made GitHub emit two check runs named `contact-write-boundary` per update -- the
ordinary-event copy skipped by the job guard -- and Atlas's readiness consumers
pick the latest run by name and count `skipped` as green
(`scripts/pr_watcher.py:459-503`, `scripts/watch_owned_pr.sh:217-225`). The
skipped copy could therefore mask a failed enforcement run. Advisory and
post-merge jobs live in `.github/workflows/contact_write_boundary_selfcheck.yml` under distinct
names, and `::test_blocking_context_has_exactly_one_producer` pins the property.

**Bootstrap, stated plainly:** `pull_request_target` resolves the workflow from
the base branch, so on this PR -- which introduces the file -- GitHub has no
base definition and the enforcement job **cannot run in CI**. This PR therefore
does not, and cannot, demonstrate a green `contact-write-boundary` context.
Enforcement begins on the first PR opened after this merges. Two things stand in
for it here: an advisory `contact-write-boundary-selfcheck` job in a
**separate workflow file** on the ordinary `pull_request` event (PR-owned code,
deliberately not enrolled in `ci/gates.yml`), and
`tests/test_contact_write_boundary.py::test_trusted_base_invocation_ignores_a_widened_pr_allowlist`,
which runs the exact base-checker-versus-PR-tree invocation as a subprocess and
proves the trusted-base property that the CI run cannot yet show.

### Trusted-base execution

The workflow runs on `pull_request_target`, checks the **checker** out from the
base SHA, and checks the PR tree out separately as data. The first version ran
the PR's own copy of the checker against itself, which made the gate
self-defeating: a PR adding a forbidden writer could widen `INSERT_ALLOWED`,
weaken `classify()`, or rewrite the self-tests in the same diff and publish a
green context. Demonstrated:

```
base checker vs hostile PR tree -> EXIT=1, flags atlas_brain/services/evil.py
PR   checker vs the same tree   -> EXIT=0, evil.py not flagged
```

The job shape is the one audited by
`scripts/audit_workflow_security_posture.py`: an event-name `if` guard, a
SHA-pinned first-step checkout of `github.event.pull_request.base.sha`, and
explicit enrollment in `ALLOWED_PULL_REQUEST_TARGET_JOBS`. Post-merge runs are a
separate `push`-guarded job mirroring `pre_push_audit.yml`, so neither job can
execute on the wrong event and neither needs a conditional checkout ref.
Enrolling a `pull_request_target` job is a deliberate, audited decision, which
is why it is a visible one-line diff to the auditor rather than a workflow-local
setting.

The baseline is also read from base, so a PR cannot baseline away its own
finding. `ci/gates.yml` now records `trusted_base: true`, which it can only do
truthfully because of this change. No step executes PR-ref content.

### Why a new workflow instead of the existing EOM pipeline job

`.github/workflows/atlas_eom_lead_pipeline_checks.yml` is path-filtered. The violation this gate
exists to catch is a *new* module adding its own `INSERT INTO contacts`, and a
new file matches no pre-existing path filter — so wiring the check in there
would have produced a guard that is silent on precisely its motivating case.
The new workflow triggers on all pull requests. The detector is stdlib-only and
scans the tree in seconds.

## Mechanism

SQL in this repo lives inside Python string literals, so the detector parses
each file with `ast` and walks `ast.Constant` string nodes (which also covers
the literal segments of f-strings, the shape `crm_provider.py:1131` actually
uses). Comments therefore never produce findings, and a commented-out INSERT is
correctly ignored, without the detector needing its own comment tokenizer.

Each pattern requires the *syntactic shape* of the statement rather than a
keyword adjacent to the table name:

```python
"UPDATE": re.compile(
    r"\bupdate\s+(?:only\s+)?" + TABLE + r"(?:\s+(?:as\s+)?[a-z_][a-z0-9_]*)?\s+set\b", ...
)
```

Requiring the trailing `SET` is what separates a real statement from the English
sentence "Create/update contacts in the Atlas CRM." `TABLE` accepts a quoted
identifier (`"contacts"`) and an optional `public.` qualifier, so
`contact_interactions` and `contacts_archive` still do not match.

**Constant folding.** Scanning each `ast.Constant` alone misses SQL assembled
across literals, so `+` chains and f-strings are folded before matching:
`"INSERT INTO " + "contacts (...)"` is one statement at runtime and is now one
finding. Operands that cannot be folded (a variable table name) become a
sentinel `HOLE` rather than a space -- a space is indistinguishable from real
whitespace and made the rule stop firing entirely. Constants consumed by a fold
are not yielded again, so a resolved statement is never also reported as
unresolvable.

**Every row-creation form.** INSERT is not the only way a row reaches the table:
`MERGE INTO`, `COPY ... FROM`, and `SELECT ... INTO` all write rows while
skipping the provider, so all are treated as creates under the stricter
allow-list.

**`.sql` files are scanned too.** Python is not the only path to the database; a
migration or data-fix script executes SQL directly. Comments are stripped first
so migration 358's documented rollback recipe (a commented `UPDATE contacts`) is
not read as a live statement.

**Exemption is by location, never basename.** `Path(rel).name.startswith("test_")`
exempted 13 real production modules -- `scripts/test_adapter_live.py`,
`atlas_brain/test_token_tracking.py`, several `scripts/debug/test_*.py` -- any of
which could have carried a write past the gate. A file is a test only if it sits
under a `tests/` directory.

**Docstrings are excluded.** Prose about code is not a statement the database
runs, and scanning it produced the original false positive.

**Runtime targets.** `"INSERT INTO " + table` cannot be cleared by reading the
literal, so it is reported as `DYNAMIC`. This is blocking only inside
`DYNAMIC_SCOPE` (the CRM-plausible paths); the repo has 18 pre-existing
`INSERT INTO {table}` sites in unrelated importers and a generic migration
runner, and failing the build on those is how a gate earns a reputation for
noise and gets switched off.

The detector skips itself: it necessarily contains the patterns it searches for,
in its own diagnostic strings.

## Intentional

- **UPDATE/DELETE are non-blocking in this slice.** Three operator scripts
  (`backfill_business_context.py`, `import_eom_customers_live.py`,
  `sync_eom_portal_customers.py`) carry their own guarded updates by design, and
  converging them is a separate sequenced piece of work (Website#111). Blocking
  them now would either fail the build on merge or force a rushed migration of
  live import tooling. They are allow-listed and inventoried instead, so a *new*
  one is still visible in review.
- **Baseline records the full writer inventory, not just exceptions.**
  `known_writes` alone is empty today (every writer is allow-listed), which would
  make its drift test an empty-in/empty-out assertion that passes regardless of
  what the tree does. `writer_inventory` carries all 17 sites so drift is real.
- **No `warnings`-style advisory mode.** A gate that only warns is a gate that is
  ignored; the INSERT rule is either enforced or absent.
- **The allow-list is a tuple of paths, not a decorator or marker comment.** A
  marker in the source would let a bypass author self-approve inside the same
  diff being reviewed; a path list changes only in a file the reviewer reads.

## Deferred

- **Convergence of the legacy UPDATE writers** (MCP, email backfill, calendar
  imports, portal sync) — Website#111, which must first be decomposed into its
  five per-writer children. Unlocked once the canonical operator mutation
  contract exists (Website#109).
- **Promoting UPDATE/DELETE to blocking** — follows the convergence above; the
  allow-list shrinks as writers migrate.
- **Direct database-credential writes** — no application-layer gate can close
  this. Recorded as D-6 in the Slice 0 deferred-work ledger; role/least-privilege
  work is tracked on #1656.

Parking predicate: this slice parks *scope-expanding enforcement* — anything that
would require changing a live writer's behavior — by default.

Parked hardening: none.

## Verification

Commands run locally (Office PC, Python 3.12):

```
$ python -m pytest tests/test_contact_write_boundary.py -q --tb=short
16 passed in 15.88s

$ python scripts/check_contact_write_boundary.py
OK - 17 contact write(s), all inside approved modules or baselined.
(exit 0)
```

Planted-violation transcript -- the reachability proof. Copy-pasteable and run
against the real working tree, not a fixture:

```
$ cat > atlas_brain/services/_planted_bypass_probe.py <<'EOF'
async def create_contact_directly(conn, name):
    await conn.execute(
        "INSERT INTO contacts (id, full_name) VALUES (gen_random_uuid(), $1)", name
    )
EOF
$ python scripts/check_contact_write_boundary.py --baseline tests/contact_write_boundary/baseline.json
contact write-boundary check
------------------------------------------------------------
BLOCKING: INSERT INTO contacts outside the approved provider module.

  atlas_brain/services/_planted_bypass_probe.py:3
    INSERT INTO contacts (id, full_name) VALUES (gen_random_uuid(), $1)
EXIT=1

$ rm atlas_brain/services/_planted_bypass_probe.py
$ python scripts/check_contact_write_boundary.py --baseline tests/contact_write_boundary/baseline.json
OK - 43 contact write(s), all inside approved modules or baselined.
EXIT=0
```

Detector output cross-checked against the independent code investigation
recorded on Website#107: it finds exactly the 2 INSERT sites
(`crm_provider.py:637`, `:881`) and 15 UPDATE sites (9 in `crm_provider.py`, 6
across the three operator scripts) that the manual sweep found — no more, no
fewer.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/contact_write_boundary.yml` | 95 |
| `.github/workflows/contact_write_boundary_selfcheck.yml` | 84 |
| `ci/gates.yml` | 8 |
| `plans/PR-EOM-Contact-Write-Boundary-Guard.md` | 365 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| `scripts/check_contact_write_boundary.py` | 846 |
| `tests/contact_write_boundary/baseline.json` | 86 |
| `tests/test_contact_write_boundary.py` | 1112 |
| **Total** | **2597** |
