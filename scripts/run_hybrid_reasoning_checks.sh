#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

./scripts/run_reasoning_provider_port_compat_checks.sh
./scripts/run_reasoning_provider_port_tests.sh

echo "hybrid reasoning checks complete"
