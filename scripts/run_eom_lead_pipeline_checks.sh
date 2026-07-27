#!/usr/bin/env bash
# Local mirror of .github/workflows/atlas_eom_lead_pipeline_checks.yml.
#
# The test set is NOT duplicated here: this script and the workflow both read
# tests/eom_lead_pipeline_files.txt, so running this proves the same set CI
# enforces rather than a copy that drifted.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cleanup_container=""

cleanup() {
  if [[ -n "$cleanup_container" ]]; then
    docker rm -f "$cleanup_container" >/dev/null
  fi
}
trap cleanup EXIT

if [[ -z "${ATLAS_MIGRATION_TEST_DATABASE_URL:-}" ]]; then
  if ! command -v docker >/dev/null; then
    echo "ATLAS_MIGRATION_TEST_DATABASE_URL is required when docker is unavailable." >&2
    exit 1
  fi
  cleanup_container="atlas-eom-lead-pipeline-postgres-$$"
  docker run -d --rm \
    --name "$cleanup_container" \
    -e POSTGRES_USER=atlas \
    -e POSTGRES_PASSWORD=atlas \
    -e POSTGRES_DB=atlas_migration_tests \
    -p 127.0.0.1::5432 \
    postgres:16@sha256:081f1bc7bd5e143dbb6e487b710bbc27712cdcfaced4c071b8e47349aa1b4171 \
    >/dev/null
  postgres_port="$(docker port "$cleanup_container" 5432/tcp | sed 's/.*://')"
  export ATLAS_MIGRATION_TEST_DATABASE_URL="postgresql://atlas:atlas@localhost:${postgres_port}/atlas_migration_tests"
fi

if ! command -v pg_isready >/dev/null; then
  echo "pg_isready is required to prove the workflow mirror database is reachable." >&2
  exit 1
fi

for _ in {1..30}; do
  if pg_isready -d "$ATLAS_MIGRATION_TEST_DATABASE_URL" >/dev/null; then
    break
  fi
  sleep 1
done

if ! pg_isready -d "$ATLAS_MIGRATION_TEST_DATABASE_URL" >/dev/null; then
  echo "PostgreSQL service for EOM lead pipeline checks did not become ready." >&2
  exit 1
fi

# Same manifest the workflow runs; see tests/eom_lead_pipeline_files.txt.
grep -v '^[[:space:]]*#' tests/eom_lead_pipeline_files.txt \
  | grep -v '^[[:space:]]*$' \
  | xargs python -m pytest -q
