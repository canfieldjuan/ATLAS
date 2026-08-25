# Testing

Use the narrowest command that settles the changed behavior, then run the
repository's required mechanical gate before pushing.

## Python tests

The canonical runner is pytest. The shared project venv is preferred; `./ops`
selects a worktree `.venv`, then the shared-root `.venv`, then the current
interpreter. Override only with a verified executable:

```bash
ATLAS_PYTHON=/path/to/python ./ops test focused tests/test_file.py -q
```

Common commands:

```bash
# One file/node while iterating
./ops test focused tests/test_file.py -q
./ops test focused tests/test_file.py::test_name -q

# Full non-integration/e2e suite
./ops test unit

# Mandatory mechanical PR bundle; run once through the push workflow
./ops test local-review
```

The Unit Gate is branch-required and uses impacted-test selection plus the
committed baseline. The scheduled Repo-Wide Unit Backstop runs the full
non-integration/e2e suite. See `.github/workflows/unit_gate.yml`,
`.github/workflows/repo_wide_unit_backstop.yml`, and
`scripts/check_unit_gate.py`; a local green subset is not a claim that all CI
passed.

`./ops test unit` removes `DATABASE_URL` and every inherited
`*_DATABASE_URL` from the pytest child environment, along with the disposable
database confirmation flag. This keeps unmarked PostgreSQL tests in their skip
path when credentials remain exported from an earlier integration run. Use
`./ops test integration ...` for explicit database-backed proof; unrelated
environment configuration is preserved in unit mode.

## Database and integration tests

Integration tests can create, migrate, truncate, or drop objects. `./ops`
requires a focused target, an explicit test URL, and
`ATLAS_CONFIRM_DISPOSABLE_TEST_DB=1` before it will run them. Follow
`.agent/runbooks/database.md`. Do not use the live `atlas` database and do not
run concurrent suites against one test database.

## Frontend tests

Each frontend owns its package scripts. Discover them from `package.json`; do
not assume every package has a generic `test` command.

```bash
./ops test frontend atlas-ui test
./ops test frontend atlas-churn-ui test
./ops test frontend atlas-intel-ui test:content-ops-input-display
./ops test frontend portfolio-ui test:deflection-result
```

Install with the checked-in lockfile before first use in a fresh worktree:

```bash
cd <package>
npm ci
```

`atlas-admin-ui` and `atlas-mobile` currently have no generic test script;
build/lint or a feature-specific test is required instead of inventing one.

## Extracted packages

Changes under `extracted_*/` carry package gauntlets. The authoritative list is
in `AGENTS.md` and the “Per-package validation gauntlets” section of
`CLAUDE.md`. `extracted_content_pipeline` requires its validation, standalone
import/audit, ASCII, and synchronization checks before push when applicable.

## Failure routing

- Pytest cannot import an optional dependency: verify the selected interpreter
  and requirements before changing code or baselines.
- A DB test skips: check the exact test-specific URL variable; a skip is not
  integration proof.
- A frontend has no `test` script: list scripts with
  `jq -r '.scripts' <package>/package.json` and use the relevant named script.
- Hosted CI is red: use `./ops ci run <id> --log-failed` and match the run's
  head SHA before diagnosing source.
- Before push, use `scripts/push_pr.sh`; do not run a duplicate manual local
  review immediately before that wrapper.
