# PR-Audit-Install-Local-Review-Hook

## Why this slice exists

The local reviewer workflow now has a mechanical review bundle, but it
still depends on humans remembering to run it before push. This slice adds
an opt-in Git hook installer so builders can make the local review bundle
run automatically before `git push`.

## Scope (this PR)

1. Add a safe installer for `.git/hooks/pre-push`.
2. Make the installed hook run `bash scripts/local_pr_review.sh`.
3. Refuse to overwrite unmanaged hooks unless explicitly forced.
4. Repair the pre-push wrapper test fixture so the two newly wired auditors
   from the previous slice are covered.
5. Document the installer in the local review workflow.
6. Refresh the in-flight coordination row for this slice.

### Files touched

- `AGENTS.md`
- `scripts/install_local_pr_hook.sh`
- `tests/test_install_local_pr_hook.py`
- `tests/test_pre_push_audit.py`
- `plans/PR-Audit-Install-Local-Review-Hook.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`scripts/install_local_pr_hook.sh` writes a managed `.git/hooks/pre-push`
wrapper containing the marker `ATLAS_LOCAL_PR_REVIEW_HOOK`. The hook changes
to the repo root and executes `bash scripts/local_pr_review.sh`.

If an existing hook does not contain the marker, the installer exits
non-zero and leaves the hook unchanged. Passing `--force` replaces the
unmanaged hook.

The installed hook supports `ATLAS_SKIP_LOCAL_PR_REVIEW=1` as an emergency
escape hatch for local pushes when the operator intentionally wants to bypass
the local mechanical review.

## Intentional

- This installer is opt-in. It does not install itself during tests, package
  setup, or CI.
- The hook runs `local_pr_review.sh`, not only `pre_push_audit.sh`, because
  the local review bundle also runs plan/code consistency and whitespace
  checks.
- There is no uninstall command in this slice. Removing `.git/hooks/pre-push`
  is simple and keeps the installer small.

## Deferred

- Shell hygiene auditor remains deferred from the previous audit-kit work.
- Automatic invocation of the judgment reviewer remains a process step, not
  a Git hook.

## Verification

```bash
python -m pytest tests/test_install_local_pr_hook.py tests/test_pre_push_audit.py -q
bash scripts/local_pr_review.sh
python scripts/audit_plan_doc.py plans/PR-Audit-Install-Local-Review-Hook.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Install-Local-Review-Hook.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Install-Local-Review-Hook.md origin/main
bash -n scripts/install_local_pr_hook.sh
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `AGENTS.md` | 11 |
| `scripts/install_local_pr_hook.sh` | 86 |
| `tests/test_install_local_pr_hook.py` | 122 |
| `tests/test_pre_push_audit.py` | 37 |
| `plans/PR-Audit-Install-Local-Review-Hook.md` | 81 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~341** |
