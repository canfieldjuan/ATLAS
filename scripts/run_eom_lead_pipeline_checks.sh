#!/usr/bin/env bash
# Local mirror of .github/workflows/atlas_eom_lead_pipeline_checks.yml.

set -euo pipefail

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
