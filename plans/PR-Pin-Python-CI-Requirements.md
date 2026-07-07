# PR-Pin-Python-CI-Requirements

Ownership lane: ci-cd/enforcement-gaps

Phase 1 / slice 1 of the CI/CD enforcement gap tracker (#2035, gap G1.2 /
root cause RC3).

## Why this slice exists

The CI/CD audit (#2035) found `requirements.txt` (52 non-comment lines) and
`requirements.content_ops_ci.txt` (31 lines) carry zero `==` pins, so every
CI job resolves dependencies fresh at runtime: test outcomes are not a
function of the repo state alone, and a transitive release can flip any suite
red or green with no commit (RC3, "non-reproducible green"). This slice pins
both files so a green run is reproducible. It lands before the Phase-1
aggregate test gate (slice 2) because a regression-ratchet baseline measured
on drifting dependencies is meaningless.

Scope amendment vs the tracker: G1.2's closes-when names only
`requirements.txt`; planning found CI installs from BOTH files
(`admin_costs_checks.yml:40` vs `atlas_content_ops_input_provider_checks.yml:61`,
`atlas_content_ops_macro_writeback_checks.yml:77`), so this slice pins both.
The amendment is recorded on #2035 at close-out.

### Problem-derived contract

Root cause: dependency resolution happens at CI runtime with open specifiers.
A correct fix pins every installable line to the exact version CI has already
proven green, changes nothing else (no upgrades, no added/removed deps, no
workflow edits), and is verified by the same CI fleet that consumes the files.

## Scope (this PR)

Slice phase: workflow/process

- `requirements.txt` — every non-comment line pinned `==`; extras
  (`uvicorn[standard]`, `nemo_toolkit[asr]`) and inline comments preserved.
- `requirements.content_ops_ci.txt` — same, all 31 lines.
- `tests/test_content_ops_ci_requirements_workflows.py` — contract amendment
  found by the PR's own CI: this guard asserted the REQUIRED side as exact
  raw requirement strings (`"torch"`, `"asyncpg>=0.31.0"`, line 99), so pins
  broke all four content-ops suites. The excluded-heavy side was already
  name-based (`requirement_name`); the fix makes the required side symmetric
  (name-based, specifier-agnostic), preserving the guard's intent — needed
  packages present, heavy stack absent — under any specifier shape.
- This plan doc.

Nothing else. No workflow YAML, no other requirements files, no runtime code.

Max files: 4

### Files touched

- `plans/PR-Pin-Python-CI-Requirements.md`
- `requirements.content_ops_ci.txt`
- `requirements.txt`
- `tests/test_content_ops_ci_requirements_workflows.py`

### Review Contract

Acceptance criteria (check one-by-one):
1. Every non-comment line of both files carries `==` EXCEPT the single
   documented carve-out `nemo_toolkit[asr]` (unpinned because
   `requirements.asr.txt` supplies a direct git-ref build and
   `security_full_sweep` pip-audits both files jointly; a `==` pin conflicts
   with the direct reference).
2. No version moves vs what the harvested green runs resolved; extras and
   inline comments byte-identical; no dependency added or removed.
3. The requirements-shape guard asserts NAMES on its required side (pinned
   entries pass; a removed required package still fails; a pinned heavy
   package is still excluded).
4. No workflow YAML, no runtime code, no other requirements files change.

Reachability proof: the PR itself exercises the real entrypoints — the
path-filtered pytest fleet installs `requirements.txt` (py3.11 + py3.10) and
the four content-ops workflows install `requirements.content_ops_ci.txt` and
run the shape guard; observable result = those check runs green on this PR.
The joint pip-audit resolution over `requirements.txt` + `requirements.asr.txt`
(security_full_sweep) is restored to its pre-PR state by the nemo carve-out
(that name returns to exactly its previous unversioned form).

Affected surfaces: CI dependency installs for every Python suite; the
security full-sweep joint audit input; the content-ops requirements shape
guard.

Risk areas: a pin that resolves on py3.11 but not py3.10 (mitigated:
3.10-resolved versions win divergences; both fleets run on this PR); pinned
set drifting from `requirements.asr.txt`'s git-ref NeMo (mitigated: carve-out).

Triggered reviewer rules: R11, R12 (env/config surface — requirements files);
R2, R14 (guard/validator change — the shape-guard test edit, boundary-probed
both directions).

## Mechanism

Pin versions are harvested from the `Successfully installed ...` lines of
three recent GREEN CI runs on main — not from a local environment (local venv
is Python 3.12; CI runs 3.11 and 3.10):

- run 28893718452 (`admin_costs_checks`, py3.11, installs requirements.txt,
  main, 2026-07-07) — primary 3.11 map.
- run 28561411787 (`extracted_llm_infrastructure_checks`, py3.10, installs
  requirements.txt, main, 2026-07-02) — 3.10 map; on 3.10/3.11 divergence the
  3.10 version is taken so one pin satisfies both interpreters
  (divergences: uvicorn 0.49.0, opencv-python 4.13.0.92, langgraph 1.2.7,
  anthropic 0.115.1, boto3 1.43.39).
- run 28893718535 (`atlas_content_ops_input_provider_checks`, py3.11,
  installs requirements.content_ops_ci.txt, main, 2026-07-07) — the
  content-ops file pins from its own fleet's map.

The rewrite is mechanical (scripted): same lines, same order, same comments;
only the version specifier changes. Every pin was validated against the
original specifier with `packaging.SpecifierSet` (e.g. `fastapi==0.136.3`
inside `<0.137`, `mcp==1.28.1` inside `>=1.28.1,<2`) — the script fails loud
on any violation or any name without a harvested version.

## Intentional

- **No version moves.** Pins capture what CI already resolved and proved
  green; nothing is upgraded or downgraded relative to the harvested runs.
- **The two files may pin different versions of a shared package**
  (e.g. `uvicorn[standard]==0.49.0` in requirements.txt via the 3.10 rule vs
  `==0.50.2` in the content-ops file): each file pins what ITS consuming
  fleet proved. They are installed by disjoint jobs, never together.
- **Top-level pins only, no transitive lock** — see Deferred.
- **One documented carve-out:** `nemo_toolkit[asr]` stays unpinned.
  `requirements.asr.txt:7` supplies a direct git-ref NeMo build and
  `security_full_sweep.yml` pip-audits `requirements.txt` +
  `requirements.asr.txt` jointly; a `==` pin on nemo conflicts with the
  direct reference (Codex T3). Unpinned restores that name to its exact
  pre-PR (proven) form.
- **Commented-out optionals** (`# twilio`, `# llama-cpp-python`) stay
  commented.
- Where the 3.10 run resolved older than the 3.11 run (5 packages above), the
  older pin is taken; the PR's own 3.11 fleet run is the proof it still
  passes there.

## Deferred

- Bare unpinned `pip install pytest pytest-asyncio` lines in ~20 workflow
  files (plus `codex_wake_bridge_checks.yml:34` pyyaml,
  `atlas_deflection_migration_apply_checks.yml:60` asyncpg) — slice 2
  consolidates CI installs when the aggregate gate lands.
- `requirements.asr.txt`, `atlas_edge/`, `atlas_video-processing/`,
  `graphiti-wrapper/` requirements — separate deploy surfaces, installed by
  no PR workflow.
- `nemo_toolkit[asr]` dedup (#2040): the one line left unpinned. A `==` pin
  conflicts with `requirements.asr.txt`'s git-ref in the joint
  `security_full_sweep` pip-audit (verified: pip rejects `==` + `@ git` for
  the same name across `-r` files). nemo is installed-but-never-imported by
  any `requirements.txt`-only suite (real importers: `asr_server.py` via
  `requirements.asr.txt`, `scripts/debug/*` not in CI), so its unpinned
  version cannot flip a PR-suite outcome. Root fix = relocate/dedup, a
  dependency-set change deferred to #2040.
- Full transitive lock (constraints file / pip-compile) — hardening
  follow-up once the gate exists to validate it (#2040). Top-level pins
  narrow RC3; a transitive lock reconciled across py3.11 + py3.10 must be
  validated by the slice-2 aggregate gate before it can be trusted.
- Dependabot interaction: pinned lines change how dependabot proposes bumps
  (PRs will now move explicit pins). Acceptable; dependabot PRs get the same
  fleet validation.

## Verification

- The PR itself triggers the consuming fleet: `requirements.txt` is in the
  `paths:` filters of most pytest workflows (py3.11) and of
  `extracted_llm_infrastructure_checks`/`extracted_pipeline_checks` (py3.10);
  `extracted_content_pipeline/**`-adjacent content-ops suites install the
  content-ops file. Green fleet on both interpreters = the pins reproduce
  what CI had.
- Closes-when (from #2035, amended): every non-comment line pinned except
  the documented `nemo_toolkit` carve-out — requirements.txt 51 of 52 lines
  `==` (+ carve-out comment), requirements.content_ops_ci.txt 31 of 31. The
  G1.2 closes-when on #2035 is amended accordingly at close-out.
- Specifier-compliance already machine-checked at rewrite time (fail-loud).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Pin-Python-CI-Requirements.md` | 185 |
| `requirements.content_ops_ci.txt` | 62 |
| `requirements.txt` | 105 |
| `tests/test_content_ops_ci_requirements_workflows.py` | 22 |
| **Total** | **374** |
