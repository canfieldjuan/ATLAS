#!/usr/bin/env bash
# Local mirror of .github/workflows/atlas_eom_lead_pipeline_checks.yml.

set -euo pipefail

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

python -m pytest \
  tests/test_crm_read_scoping.py \
  tests/test_eom_complaints_integration.py \
  tests/test_eom_contacts_api_tenant_scope.py \
  tests/test_eom_lead_pipeline_integration.py \
  tests/test_eom_mailbox_context_binding.py \
  tests/test_eom_recurring_appointments_integration.py \
  tests/test_eom_scoped_gmail_credentials.py \
  tests/test_eom_scoped_gmail_hardening.py \
  tests/test_eom_sent_email_tenant_scope.py \
  tests/test_leads_intake.py \
  tests/test_migrations_runner.py \
  -q
