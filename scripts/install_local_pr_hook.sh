#!/usr/bin/env bash
# Install the local PR review bundle as this checkout's pre-push hook.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/install_local_pr_hook.sh [--force]

Installs .git/hooks/pre-push as a managed wrapper around:

  bash scripts/local_pr_review.sh

Options:
  --force   overwrite an existing unmanaged pre-push hook
  -h, --help
            show this help
EOF
}

force=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force)
            force=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "install_local_pr_hook.sh: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [ ! -f scripts/local_pr_review.sh ]; then
    echo "install_local_pr_hook.sh: scripts/local_pr_review.sh not found" >&2
    exit 2
fi

hook_dir="$(git rev-parse --git-path hooks)"
hook_path="$hook_dir/pre-push"
marker="ATLAS_LOCAL_PR_REVIEW_HOOK"

mkdir -p "$hook_dir"

if [ -e "$hook_path" ] && ! grep -q "$marker" "$hook_path"; then
    if [ "$force" -ne 1 ]; then
        cat >&2 <<EOF
install_local_pr_hook.sh: refusing to overwrite unmanaged hook:
  $hook_path

Re-run with --force to replace it, or merge its behavior manually.
EOF
        exit 1
    fi
fi

cat > "$hook_path" <<'EOF'
#!/usr/bin/env bash
# ATLAS_LOCAL_PR_REVIEW_HOOK
# Managed by scripts/install_local_pr_hook.sh.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [ "${ATLAS_SKIP_LOCAL_PR_REVIEW:-}" = "1" ]; then
    echo "ATLAS local PR review hook skipped (ATLAS_SKIP_LOCAL_PR_REVIEW=1)."
    exit 0
fi

if [ -t 0 ]; then
    exec bash scripts/local_pr_review.sh
fi

delete_sha="0000000000000000000000000000000000000000"
saw_ref_update=0
saw_non_delete_ref_update=0

is_object_name() {
    [[ "$1" =~ ^[0-9a-fA-F]{40}$ ]]
}

is_remote_ref_name() {
    [[ "$1" == refs/* ]] && git check-ref-format "$1" >/dev/null 2>&1
}

while IFS= read -r ref_update_line || [ -n "$ref_update_line" ]; do
    if [ -z "$ref_update_line" ]; then
        continue
    fi
    if [[ "$ref_update_line" =~ ^[[:space:]]+$ ]]; then
        saw_ref_update=1
        saw_non_delete_ref_update=1
        continue
    fi
    read -r local_ref local_sha remote_ref remote_sha extra <<< "$ref_update_line"
    saw_ref_update=1
    if [ -z "${local_sha:-}" ] || [ -z "${remote_ref:-}" ] || [ -z "${remote_sha:-}" ] || [ -n "${extra:-}" ]; then
        saw_non_delete_ref_update=1
        continue
    fi
    if [ "${local_ref:-}" = "(delete)" ] && [ "${local_sha:-}" = "$delete_sha" ] &&
        is_remote_ref_name "$remote_ref" && is_object_name "$remote_sha" &&
        [ "$remote_sha" != "$delete_sha" ]; then
        continue
    fi
    saw_non_delete_ref_update=1
done

if [ "$saw_ref_update" -eq 1 ] && [ "$saw_non_delete_ref_update" -eq 0 ]; then
    echo "ATLAS local PR review hook skipped (delete-only push)."
    exit 0
fi

exec bash scripts/local_pr_review.sh
EOF

chmod +x "$hook_path"
echo "Installed ATLAS local PR review pre-push hook at $hook_path"
