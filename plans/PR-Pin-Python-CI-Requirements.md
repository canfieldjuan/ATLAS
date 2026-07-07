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

## Scope

Slice phase: workflow/process

- `requirements.txt` — every non-comment line pinned `==`; extras
  (`uvicorn[standard]`, `nemo_toolkit[asr]`) and inline comments preserved.
- `requirements.content_ops_ci.txt` — same, all 31 lines.
- This plan doc.

Nothing else. No workflow YAML, no other requirements files, no code.

Max files: 3

### Files touched

- `requirements.txt`
- `requirements.content_ops_ci.txt`
- `plans/PR-Pin-Python-CI-Requirements.md`

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
- Full transitive lock (constraints file / pip-compile) — hardening
  follow-up once the gate exists to validate it.
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
- Closes-when (from #2035): `grep -cvE '^\s*#|^\s*$' <file>` equals
  `grep -cE '==' <file>` for both files (52==52, 31==31 locally).
- Specifier-compliance already machine-checked at rewrite time (fail-loud).

## Estimated diff size

| File | ~LOC (added+deleted) |
|---|---|
| `requirements.txt` | ~104 |
| `requirements.content_ops_ci.txt` | ~62 |
| `plans/PR-Pin-Python-CI-Requirements.md` | ~130 |
| **Total** | **~296** |

Under the 400-LOC budget (the two requirements files are in-place rewrites:
52 + 31 lines each counted as an add plus a delete).
