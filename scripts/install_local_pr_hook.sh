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

exec bash scripts/local_pr_review.sh
EOF

chmod +x "$hook_path"
echo "Installed ATLAS local PR review pre-push hook at $hook_path"
