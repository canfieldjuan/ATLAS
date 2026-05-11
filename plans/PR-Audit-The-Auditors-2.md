# PR-Audit-The-Auditors-2

## Why this slice exists

PR #486 (`PR-Audit-The-Auditors-1`) added AGENTS.md section 3f:

> Every `scripts/audit_*.py` ships with a sibling
> `tests/test_audit_<name>.py` that exercises at least three cases:
> happy path, parser-specific negative case, and pathological input
> that should be rejected.

That principle currently has zero implementations -- AGENTS.md says
"required" but no auditor under `scripts/` has a sibling test file
yet. This slice instantiates the contract for the five auditors
where Copilot reviewers actually caught a real bug during this
session, locking those exact bug-shapes in as regression tests so
they cannot recur.

The five auditors with real bugs caught:

| Auditor | Bug Copilot caught | What the test pins |
|---|---|---|
| `audit_mcp_port_assignments.py` | `ENV_VAR_LINE` regex `[A-Z_]+` rejected the "2" in `B2B_CHURN`, silently dropping the line | Fixture string with `ATLAS_MCP_B2B_CHURN_PORT=8062` must produce a claim with name "b2b_churn" |
| `audit_plan_doc.py` | Substring `"scope".lower() in "out of scope".lower()` made "Out of scope" satisfy the Scope slot | Fixture plan doc with only "## Out of scope" must report Scope as MISSING |
| `audit_extracted_manifests.py` | `startswith("/")` missed Windows drive (`C:\...`) and UNC (`\\srv\...`) absolute paths | Fixture manifest entries with Windows + UNC absolute paths must be rejected |
| `audit_claude_md_claims.py` | `HEADER_PATTERN` didn't require closing `)`, accepted malformed headers | Fixture with `### Foo MCP Server (9 tools` (no close paren) must NOT match; `(9 tools)` must match |
| `audit_mcp_tool_names_match_docs.py` | `doc_claims()` silently dropped `### <Name> MCP Server` headers whose name wasn't in `HEADER_TO_FILE` | Fixture with an unknown server header must be returned in the `unknown` list, not silently dropped |

This is the first half of the fixture-coverage work. The four
remaining pre-push-gate auditors (`audit_review_source_count`,
`audit_plan_doc_files_touched`, `audit_plan_doc_diff_size`,
`audit_plan_code_consistency`) had Copilot catches that were
plan-doc drift, not parser bugs, so they need different fixture
shapes and ship in a follow-up.

## Scope (this PR)

1. `tests/audit_helpers.py` -- shared `load_auditor(name)` helper
   that uses `importlib.util` to load `scripts/<name>.py` as a
   module. If the script does not exist on the current branch
   (because the auditor's parent PR has not merged yet), the
   helper raises `pytest.skip()` with a clear message naming the
   dependency. This is the mechanism that lets this slice land
   off main before PRs #483 / #484 / #485 / #486 have all merged.

2. `tests/test_audit_mcp_port_assignments.py` -- happy path
   (CRM port = 8056 matches), digit-in-name regression (the
   B2B_CHURN bug -> must match), missing-in-doc surfaces ports
   declared in `MCPConfig` but absent from CLAUDE.md.

3. `tests/test_audit_plan_doc.py` -- happy path (a doc with all
   7 sections in order passes), substring trap ("Out of scope"
   must NOT satisfy "Scope"), out-of-order detection.

4. `tests/test_audit_extracted_manifests.py` -- happy path
   (real manifest passes), POSIX absolute path rejected,
   Windows + UNC absolute paths rejected, `..` traversal
   rejected.

5. `tests/test_audit_claude_md_claims.py` -- happy path
   (`(9 tools)` matches), missing-close-paren must NOT match,
   `(60+ tools)` soft count reports DRIFT, missing file
   reports MISSING_FILE sentinel (not -1).

6. `tests/test_audit_mcp_tool_names_match_docs.py` -- happy
   path (every claimed tool resolves), unknown server header
   surfaces in the unknown list, extra-in-doc detection.

### Files touched

- `plans/PR-Audit-The-Auditors-2.md` (this file, new)
- `tests/audit_helpers.py` (new)
- `tests/test_audit_mcp_port_assignments.py` (new)
- `tests/test_audit_plan_doc.py` (new)
- `tests/test_audit_extracted_manifests.py` (new)
- `tests/test_audit_claude_md_claims.py` (new)
- `tests/test_audit_mcp_tool_names_match_docs.py` (new)

No existing files modified.

## Mechanism

### `tests/audit_helpers.py`

```
load_auditor(name: str) -> module
    Resolve scripts/<name>.py under the repo root.
    If it does not exist: pytest.skip(
        f"requires scripts/{name}.py to exist; "
        f"depends on an audit PR merging first"
    )
    Otherwise: load via importlib.util.spec_from_file_location,
    exec the module, cache in sys.modules, return.
```

This skip-if-missing pattern is the slice's load-bearing trick.
This PR is filed off main today; on main, none of the audit
scripts from PRs #483 / #484 / #485 / #486 exist yet. So every
test in this PR's test files will skip when CI runs on this PR's
branch (the test SUITE collects them all and reports
skip-with-reason rather than fail). As each underlying PR
merges, the corresponding tests automatically activate.

### Test shape (per auditor)

Each `tests/test_audit_<name>.py` follows the same shape:

```
import pytest
from tests.audit_helpers import load_auditor


@pytest.fixture(scope="module")
def auditor():
    return load_auditor("audit_<name>")


def test_happy_path(auditor):
    """Known-good input -> expected normalized output."""
    ...

def test_<bug_specific_negative>(auditor):
    """Regression test for <Copilot catch> on PR #<NNN>."""
    ...

def test_<pathological>(auditor):
    """Input the auditor must reject."""
    ...
```

Tests import the auditor's parsing helpers directly (e.g.
`auditor.doc_claims(...)`, `auditor.parse_files_touched(...)`)
and assert on the returned data structure. This is faster, more
focused, and more readable than subprocess-based testing, and
catches parser bugs that would slip past a smoke test.

## Intentional

- **Skip-if-missing in `load_auditor()`, not at module import.**
  pytest.skip at module top-level would be tricky to surface;
  skipping inside a fixture cleanly cascades to every test using
  that fixture, and reports the skip reason in pytest output.

- **Five auditors in this slice, not all nine.** §3f says "every
  audit script", but trying to ship nine test files plus the
  helper plus the plan in one PR would push past 600 LOC.
  Splitting at the boundary between "auditors with Copilot-caught
  parser bugs" (this PR) and "auditors with plan-doc drift
  catches" (`PR-Audit-The-Auditors-3`) keeps each PR reviewable.

- **Unit-style tests, not subprocess.** Tests import the
  parsing helpers from each auditor and exercise them with
  in-memory fixture strings. Faster than subprocess + tempdir
  setup, and tests pin the specific function shape the audit
  contract relies on.

- **Module-scoped `auditor` fixture.** The importlib load is the
  expensive bit; caching it per module keeps the suite fast and
  ensures every test in a file sees the same module instance.

- **Tests live under `tests/`, not under `scripts/`.** Matches
  the existing `pytest.ini` `testpaths = tests` convention. New
  audit-test files do not interfere with the existing 600+ test
  files because they don't import from `atlas_brain`.

- **No edits to `tests/conftest.py`.** That file is DB / async
  focused; my audit tests are pure-Python with no DB dependency.
  The audit helper lives in its own `tests/audit_helpers.py`.

## Deferred

- **`PR-Audit-The-Auditors-3`** -- test files for the remaining
  four auditors: `audit_review_source_count`,
  `audit_plan_doc_files_touched`, `audit_plan_doc_diff_size`,
  `audit_plan_code_consistency`. Same shape, ~250 LOC.

- **Test for `audit_script_hygiene.sh`.** It's bash, not Python;
  needs subprocess-based testing with fixture script trees.
  Different mechanism; deferred to its own slice.

- **CI wiring.** When pytest runs in CI, these tests will be
  collected automatically (they live under `tests/` per
  `pytest.ini`). Explicit GitHub Actions wiring is Phase 2 of
  the pre-merge gate plan.

- **Removing the skip-if-missing pattern after foundation PRs
  merge.** Once PRs #483-#486 land on main, the
  `pytest.skip()` calls in `load_auditor()` can be replaced
  with hard `pytest.fail()` so missing scripts become test
  failures. Deferred to a small follow-up cleanup PR.

## Verification

```bash
# Today, on this branch off main (no audit scripts exist):
pytest tests/test_audit_*.py -v
# -> every test skips with the message
#    "requires scripts/audit_<name>.py to exist; depends on an
#     audit PR merging first"

# After PR #483 / #484 / #485 / #486 land on main (rebase this
# branch), the same command runs the real tests:
pytest tests/test_audit_*.py -v
# -> 15+ tests run, all PASS.

# Negative-fixture sanity check (no scripts needed):
pytest tests/audit_helpers.py --collect-only
# -> module loads, no errors.
```

## Estimated diff size

| File | LOC (est) |
|---|---|
| `tests/audit_helpers.py` | ~50 |
| `tests/test_audit_mcp_port_assignments.py` | ~60 |
| `tests/test_audit_plan_doc.py` | ~55 |
| `tests/test_audit_extracted_manifests.py` | ~65 |
| `tests/test_audit_claude_md_claims.py` | ~60 |
| `tests/test_audit_mcp_tool_names_match_docs.py` | ~50 |
| `plans/PR-Audit-The-Auditors-2.md` | ~225 |
| **Total** | **~565** |

165 LOC over the 400 soft cap. Slice is indivisible: AGENTS.md
section 3f mandates fixture tests for every auditor; shipping
fewer than the five auditors with real Copilot-caught bugs
would mean the regression-locks for those exact bugs sit on a
shelf instead of in CI. Splitting into "helper + one auditor"
PRs would force five thin plan docs each ~150 LOC, net worse.
Plan doc is ~40% of the total -- same shape as the four
preceding audit PRs in the family.
