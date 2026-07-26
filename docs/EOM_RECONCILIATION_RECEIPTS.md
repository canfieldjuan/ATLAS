# EOM Calendar execution receipts

Create a private directory under the operator's local state directory:

```bash
export EOM_RECEIPT_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/atlas/eom-calendar-import"
mkdir -p "$EOM_RECEIPT_DIR"
chmod 700 "$EOM_RECEIPT_DIR"
```

Supply it for the production Calendar sequence:

```bash
set -euo pipefail

ATLAS_REPO_ROOT="$(pwd -P)"
trusted_git() (
  unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
  unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR GIT_NAMESPACE
  unset GIT_PREFIX GIT_CEILING_DIRECTORIES
  unset GIT_CONFIG GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_NOSYSTEM
  unset GIT_CONFIG_COUNT
  for name in ${!GIT_CONFIG_KEY_@} ${!GIT_CONFIG_VALUE_@}; do
    unset "$name"
  done
  GIT_NO_REPLACE_OBJECTS=1 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    git -C "$ATLAS_REPO_ROOT" -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null "$@"
)

test "$(trusted_git rev-parse --show-toplevel)" = "$ATLAS_REPO_ROOT"
ATLAS_EOM_REVIEWED_SHA="$(trusted_git rev-parse --verify "HEAD^{commit}")"

run_reviewed_eom() {
  trusted_git show "$ATLAS_EOM_REVIEWED_SHA:scripts/eom_execution_receipt.py" |
    python -I - --launch-reviewed \
      --reviewed-git-sha "$ATLAS_EOM_REVIEWED_SHA" \
      scripts/import_eom_customers_live.py "$@"
}

run_reviewed_eom --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
run_reviewed_eom --receipt-dir "$EOM_RECEIPT_DIR"
run_reviewed_eom --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
```

Live writes reject a missing receipt directory. A supplied directory must
already exist, be owned by the current user, and not be group/world writable.
Each run creates a mode-0600 `.in-progress.json` artifact before the Calendar
runtime, then exclusively publishes one `.exit-N.json` receipt.

Receipted production runs resolve the reviewed commit once, load the launcher
from that exact object, and pass the same SHA into Python's isolated `-I`
startup. The outer `trusted_git` wrapper strips repository-selection and
config-injection environment variables and disables executable fsmonitor hooks.
The launcher repeats the same Git hardening, rejects Git replacement refs,
requires the checkout to resolve to the reviewed SHA, materializes every
tracked Python module from that exact Git SHA into a private read-only
snapshot, and runs the Calendar entrypoint and its repository-local imports
from that snapshot. Mutable worktree code therefore cannot enter after
preflight. Direct
`python -I scripts/import_eom_customers_live.py --receipt-dir ...` invocation
is rejected.

The launcher rejects tracked or untracked changes, tracked Python bytes or
executable modes that differ from the reviewed revision, ignored Python import
shadows, ignored package and module-file symlinks beneath the CLI import roots,
and bytecode caches for tracked source. Run from the exact clean checkout whose
resolved Git SHA belongs in the receipt.

Receipts contain only source bindings, UTC lifecycle timestamps, non-PII
counts, and changed contact UUIDs. They never contain credentials, tokens,
runtime URLs, or customer names, emails, phones, or addresses. Keep the
directory private and do not commit or attach its contents to a public issue.
