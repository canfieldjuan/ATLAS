#!/usr/bin/env bash
# Live integration test for the entire Intelligence Platform (Phases 0-6)
# Hits every REST endpoint against a running server with real data.
# Usage: bash scripts/test_intelligence_platform_live.sh [port]

set -euo pipefail

PORT="${1:-8000}"
BASE="http://127.0.0.1:${PORT}"
PASS=0
FAIL=0
TOTAL=0

# Sample IDs from the database
REVIEW_ID="74bb10c7-42f8-4cd4-a6c7-0282b1db5828"
SEQUENCE_ID="2531e161-73dd-4fb4-81e5-cfdd3f5948b5"
REPORT_ID="62d8c300-de78-4567-ace4-fef95a01f6c2"
SIGNAL_ID="296b7b54-8f74-454d-939e-606eb73eb950"
CORRECTION_ID="d2fd86e9-fc30-4274-b899-81c82a54dbc6"
CRM_EVENT_ID="03dab025-1ddd-4eb9-94d6-92c2bcb46384"
VENDOR="Freshdesk"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

check() {
    local label="$1"
    local method="$2"
    local url="$3"
    local expect_code="${4:-200}"
    local body="${5:-}"
    TOTAL=$((TOTAL + 1))

    local args=(-s -o /tmp/atlas_test_body -w "%{http_code}" -X "$method")
    if [ -n "$body" ]; then
        args+=(-H "Content-Type: application/json" -d "$body")
    fi
    args+=("${BASE}${url}")

    local code
    code=$(curl "${args[@]}" 2>/dev/null) || code="000"

    if [ "$code" = "$expect_code" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} [${code}] ${label}"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} [${code}] ${label} (expected ${expect_code})"
        # Show first 200 chars of body on failure
        head -c 200 /tmp/atlas_test_body 2>/dev/null | sed 's/^/       /'
        echo
    fi
}

# Check that a response body contains a substring
check_contains() {
    local label="$1"
    local method="$2"
    local url="$3"
    local needle="$4"
    local body="${5:-}"
    TOTAL=$((TOTAL + 1))

    local args=(-s -o /tmp/atlas_test_body -w "%{http_code}" -X "$method")
    if [ -n "$body" ]; then
        args+=(-H "Content-Type: application/json" -d "$body")
    fi
    args+=("${BASE}${url}")

    local code
    code=$(curl "${args[@]}" 2>/dev/null) || code="000"

    if [ "$code" = "200" ] && grep -q "$needle" /tmp/atlas_test_body 2>/dev/null; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} [${code}] ${label}"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} [${code}] ${label} (expected 200 + '${needle}')"
        head -c 200 /tmp/atlas_test_body 2>/dev/null | sed 's/^/       /'
        echo
    fi
}

echo ""
echo -e "${CYAN}===============================================${NC}"
echo -e "${CYAN} Atlas Intelligence Platform -- Live Test Suite${NC}"
echo -e "${CYAN} Server: ${BASE}${NC}"
echo -e "${CYAN}===============================================${NC}"
echo ""

# ── Sanity ──────────────────────────────────────────────────
echo -e "${YELLOW}[Sanity]${NC}"
check "Health check" GET "/api/v1/health"

# ════════════════════════════════════════════════════════════
# PHASE 0: Current-State Hardening
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 0] Source Naming / Vendor Registry / Provenance${NC}"

check "Source health" GET "/api/v1/b2b/tenant/source-health"
check "Source health telemetry" GET "/api/v1/b2b/tenant/source-health/telemetry"
check "Source health telemetry timeline" GET "/api/v1/b2b/tenant/source-health/telemetry-timeline"
check "Source capabilities" GET "/api/v1/b2b/tenant/source-capabilities"
check "Fuzzy vendor search" GET "/api/v1/b2b/tenant/fuzzy-vendor-search?q=Salesfroce"
check "Fuzzy vendor search empty" GET "/api/v1/b2b/tenant/fuzzy-vendor-search?q=" "400"
check "Fuzzy company search" GET "/api/v1/b2b/tenant/fuzzy-company-search?q=Acme"

# ════════════════════════════════════════════════════════════
# PHASE 1: Managed Intelligence Substrate
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 1] Scrape Telemetry / Parser Versioning${NC}"

check "Scrape targets list" GET "/api/v1/b2b/scrape/targets"
check "Parser version status" GET "/api/v1/b2b/tenant/parser-version-status"
check "Operational overview" GET "/api/v1/b2b/tenant/operational-overview"

# ════════════════════════════════════════════════════════════
# PHASE 2: Canonical Intelligence Model
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 2] Entity Tables / Confidence Scoring${NC}"

check "Vendor pain points" GET "/api/v1/b2b/tenant/vendor-pain-points?vendor_name=${VENDOR}"
check "Vendor pain points (min_confidence)" GET "/api/v1/b2b/tenant/vendor-pain-points?vendor_name=${VENDOR}&min_confidence=0.5"
check "Vendor use cases" GET "/api/v1/b2b/tenant/vendor-use-cases?vendor_name=${VENDOR}"
check "Vendor integrations" GET "/api/v1/b2b/tenant/vendor-integrations?vendor_name=${VENDOR}"
check "Vendor buyer profiles" GET "/api/v1/b2b/tenant/vendor-buyer-profiles?vendor_name=${VENDOR}"
check "Vendor buyer profiles (min_confidence)" GET "/api/v1/b2b/tenant/vendor-buyer-profiles?min_confidence=0.3"
check "Displacement history" GET "/api/v1/b2b/tenant/displacement-history?from_vendor=${VENDOR}&to_vendor=Zendesk"

# ════════════════════════════════════════════════════════════
# PHASE 3: Historical Memory
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 3] Snapshots / Change Events / Correlation${NC}"

check "Vendor history" GET "/api/v1/b2b/tenant/vendor-history?vendor_name=${VENDOR}"
check "Change events" GET "/api/v1/b2b/tenant/change-events"
check "Change events (type filter)" GET "/api/v1/b2b/tenant/change-events?event_type=urgency_spike"
check "Change events summary" GET "/api/v1/b2b/tenant/change-events/summary"
check "Concurrent events" GET "/api/v1/b2b/tenant/concurrent-events?days=90&min_vendors=2"
check "Vendor correlation" GET "/api/v1/b2b/tenant/vendor-correlation?vendor_a=Freshdesk&vendor_b=Zendesk&metric=churn_density"
check "Vendor correlation (bad metric)" GET "/api/v1/b2b/tenant/vendor-correlation?vendor_a=Freshdesk&vendor_b=Zendesk&metric=invalid" "400"
check "Product profile" GET "/api/v1/b2b/tenant/product-profile?vendor_name=${VENDOR}"
check "Product profile history" GET "/api/v1/b2b/tenant/product-profile-history?vendor_name=${VENDOR}"

# ════════════════════════════════════════════════════════════
# PHASE 4: Action Feedback Loop
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 4] Campaign Outcomes / CRM Events / Calibration${NC}"

# Campaign sequences
check "List sequences" GET "/api/v1/b2b/campaigns/sequences"
check "List sequences (outcome filter)" GET "/api/v1/b2b/campaigns/sequences?outcome=pending"
check "List sequences (bad outcome)" GET "/api/v1/b2b/campaigns/sequences?outcome=invalid" "400"

# Outcome recording
check "Get sequence outcome" GET "/api/v1/b2b/campaigns/sequences/${SEQUENCE_ID}/outcome"
check "Record outcome (meeting_booked)" POST "/api/v1/b2b/campaigns/sequences/${SEQUENCE_ID}/outcome" "200" '{"outcome":"meeting_booked","notes":"Live test"}'
check "Record outcome (bad value)" POST "/api/v1/b2b/campaigns/sequences/${SEQUENCE_ID}/outcome" "400" '{"outcome":"invalid_value"}'

# Signal effectiveness
check "Signal effectiveness (buying_stage)" GET "/api/v1/b2b/tenant/signal-effectiveness?group_by=buying_stage"
check "Signal effectiveness (role_type)" GET "/api/v1/b2b/tenant/signal-effectiveness?group_by=role_type"
check "Signal effectiveness (bad group)" GET "/api/v1/b2b/tenant/signal-effectiveness?group_by=invalid" "400"

# Outcome distribution
check "Outcome distribution" GET "/api/v1/b2b/tenant/outcome-distribution"
check "Outcome distribution (vendor)" GET "/api/v1/b2b/tenant/outcome-distribution?vendor_name=${VENDOR}"

# Signal-to-outcome attribution
check "Signal-to-outcome" GET "/api/v1/b2b/tenant/signal-to-outcome?group_by=buying_stage"
check "Signal-to-outcome (bad group)" GET "/api/v1/b2b/tenant/signal-to-outcome?group_by=invalid" "400"

# Calibration
check "Calibration weights" GET "/api/v1/b2b/tenant/calibration-weights"
check "Trigger calibration" POST "/api/v1/b2b/tenant/calibration/trigger" "200"

# CRM Events
check "List CRM events" GET "/api/v1/b2b/crm/events"
check "List CRM events (status filter)" GET "/api/v1/b2b/crm/events?status=pending"
check "List CRM events (bad status)" GET "/api/v1/b2b/crm/events?status=invalid" "400"
check "List CRM events (date range)" GET "/api/v1/b2b/crm/events?start_date=2026-01-01&end_date=2026-12-31"
check "CRM enrichment stats" GET "/api/v1/b2b/crm/events/enrichment-stats"

# Batch CRM ingest
check_contains "Batch CRM ingest" POST "/api/v1/b2b/crm/events/batch" "created_ids" \
    '{"events":[{"crm_provider":"hubspot","crm_event_id":"test-live-'$RANDOM'","event_type":"deal_stage_change","company_name":"Test Corp","contact_email":"live@test.com","deal_amount":1000}]}'

# HubSpot webhook format (requires auth -- expect 401 in local dev)
check "HubSpot webhook (auth-gated)" POST "/api/v1/b2b/crm/events/hubspot" "401" \
    '[{"objectType":"deal","propertyName":"dealstage","propertyValue":"closedwon","objectId":99999,"subscriptionType":"deal.propertyChange","changeSource":"CRM","eventId":'$RANDOM'}]'

# ════════════════════════════════════════════════════════════
# PHASE 5: Thin Delivery Surfaces
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 5] Webhooks / PDF / Delivery${NC}"

# Webhooks (require auth -- expect 401 in local dev)
check "List webhooks (auth-gated)" GET "/api/v1/b2b/tenant/webhooks" "401"
check "Webhook delivery summary (auth-gated)" GET "/api/v1/b2b/tenant/webhooks/delivery-summary" "401"

# PDF export
check "PDF export" GET "/api/v1/b2b/tenant/reports/${REPORT_ID}/pdf"

# Reports
check "List reports" GET "/api/v1/b2b/tenant/reports"
check "Get report" GET "/api/v1/b2b/tenant/reports/${REPORT_ID}"
check "Get report (bad UUID)" GET "/api/v1/b2b/tenant/reports/not-a-uuid" "400"

# CRM push log (requires auth -- expect 401 in local dev)
check "CRM push log (auth-gated)" GET "/api/v1/b2b/tenant/webhooks/00000000-0000-0000-0000-000000000000/crm-push-log" "401"

# ════════════════════════════════════════════════════════════
# PHASE 6: Analyst Controls
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 6] Corrections / Merge / Overrides${NC}"

# Corrections CRUD
check "List corrections" GET "/api/v1/b2b/tenant/corrections"
check "List corrections (type filter)" GET "/api/v1/b2b/tenant/corrections?correction_type=suppress"
check "List corrections (corrected_by)" GET "/api/v1/b2b/tenant/corrections?corrected_by=api"
check "List corrections (date range)" GET "/api/v1/b2b/tenant/corrections?start_date=2026-01-01&end_date=2026-12-31"
check "Correction stats" GET "/api/v1/b2b/tenant/corrections/stats"
check "Correction stats (7d)" GET "/api/v1/b2b/tenant/corrections/stats?days=7"
check "Get correction" GET "/api/v1/b2b/tenant/corrections/${CORRECTION_ID}"
check "Get correction (bad UUID)" GET "/api/v1/b2b/tenant/corrections/not-a-uuid" "400"

# Source correction impact
check "Source correction impact" GET "/api/v1/b2b/tenant/source-corrections/impact"

# Create a suppress correction for testing, then revert it
echo ""
echo -e "${YELLOW}[Phase 6] Suppress + Revert round-trip${NC}"

# Create suppress
SUPPRESS_BODY='{"entity_type":"review","entity_id":"'${REVIEW_ID}'","correction_type":"suppress","reason":"Live test suppression"}'
TOTAL=$((TOTAL + 1))
SUPPRESS_RESULT=$(curl -s -X POST "${BASE}/api/v1/b2b/tenant/corrections" \
    -H "Content-Type: application/json" \
    -d "$SUPPRESS_BODY" 2>/dev/null)
SUPPRESS_ID=$(echo "$SUPPRESS_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")

if [ -n "$SUPPRESS_ID" ] && [ "$SUPPRESS_ID" != "" ]; then
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}PASS${NC} [201] Create suppress correction (id=${SUPPRESS_ID})"

    # Verify review is suppressed
    TOTAL=$((TOTAL + 1))
    REVIEW_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/api/v1/b2b/tenant/reviews/${REVIEW_ID}" 2>/dev/null)
    if [ "$REVIEW_CODE" = "404" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} [404] Suppressed review returns 404"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} [${REVIEW_CODE}] Suppressed review should return 404"
    fi

    # Revert (body required -- RevertCorrectionBody with optional reason)
    check "Revert suppress correction" POST "/api/v1/b2b/tenant/corrections/${SUPPRESS_ID}/revert" "200" '{"reason":"Live test cleanup"}'

    # Verify review is back
    TOTAL=$((TOTAL + 1))
    REVIEW_CODE2=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/api/v1/b2b/tenant/reviews/${REVIEW_ID}" 2>/dev/null)
    if [ "$REVIEW_CODE2" = "200" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} [200] Review restored after revert"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} [${REVIEW_CODE2}] Review should return 200 after revert"
    fi
else
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL${NC} Create suppress correction failed"
    echo "       $SUPPRESS_RESULT" | head -c 300
    echo
fi

# ════════════════════════════════════════════════════════════
# PHASE 6: Field override round-trip
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Phase 6] Field Override round-trip${NC}"

OVERRIDE_BODY='{"entity_type":"review","entity_id":"'${REVIEW_ID}'","correction_type":"override_field","field_name":"vendor_name","old_value":"original","new_value":"Corrected Vendor","reason":"Live test override"}'
TOTAL=$((TOTAL + 1))
OVERRIDE_RESULT=$(curl -s -X POST "${BASE}/api/v1/b2b/tenant/corrections" \
    -H "Content-Type: application/json" \
    -d "$OVERRIDE_BODY" 2>/dev/null)
OVERRIDE_ID=$(echo "$OVERRIDE_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")

if [ -n "$OVERRIDE_ID" ] && [ "$OVERRIDE_ID" != "" ]; then
    PASS=$((PASS + 1))
    echo -e "  ${GREEN}PASS${NC} [201] Create override_field correction"

    # Check review has _overrides_applied
    check_contains "Review shows override" GET "/api/v1/b2b/tenant/reviews/${REVIEW_ID}" "_overrides_applied"

    # Revert (body required)
    check "Revert override" POST "/api/v1/b2b/tenant/corrections/${OVERRIDE_ID}/revert" "200" '{"reason":"Live test cleanup"}'
else
    FAIL=$((FAIL + 1))
    echo -e "  ${RED}FAIL${NC} Create override_field correction failed"
    echo "       $OVERRIDE_RESULT" | head -c 300
    echo
fi

# ════════════════════════════════════════════════════════════
# CROSS-CUTTING: Core B2B Dashboard Endpoints
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Cross-cutting] Core dashboard endpoints${NC}"

check "Reviews list" GET "/api/v1/b2b/tenant/reviews?limit=5"
check "Review detail" GET "/api/v1/b2b/tenant/reviews/${REVIEW_ID}"
check "Signals list" GET "/api/v1/b2b/tenant/signals"
check "High-intent companies" GET "/api/v1/b2b/tenant/high-intent"
check "Vendor detail" GET "/api/v1/b2b/tenant/vendors/${VENDOR}"

# ════════════════════════════════════════════════════════════
# VALIDATION: Bad inputs return proper errors
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Validation] Error handling${NC}"

check "Bad review UUID" GET "/api/v1/b2b/tenant/reviews/not-a-uuid" "400"
check "Nonexistent review" GET "/api/v1/b2b/tenant/reviews/00000000-0000-0000-0000-000000000000" "404"
check "Bad outcome value" POST "/api/v1/b2b/campaigns/sequences/${SEQUENCE_ID}/outcome" "400" '{"outcome":"bogus"}'
check "Bad CRM status" GET "/api/v1/b2b/crm/events?status=bogus" "400"
check "Bad correction UUID" GET "/api/v1/b2b/tenant/corrections/not-a-uuid" "400"

# ════════════════════════════════════════════════════════════
# RESTORE: Reset the test outcome we set earlier
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${YELLOW}[Cleanup] Reset test outcome${NC}"
check "Reset outcome to pending" POST "/api/v1/b2b/campaigns/sequences/${SEQUENCE_ID}/outcome" "200" '{"outcome":"pending","notes":"Reset after live test"}'

# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}===============================================${NC}"
echo -e "${CYAN} Results: ${PASS}/${TOTAL} PASS, ${FAIL} FAIL${NC}"
echo -e "${CYAN}===============================================${NC}"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
