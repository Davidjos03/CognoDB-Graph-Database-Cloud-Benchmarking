#!/usr/bin/env bash
# One command, whole benchmark: start the capped comparators, wait for them to
# be healthy, prepare the dataset, benchmark every configured platform and
# publish the tables and charts.
#
#   ./scripts/run_all.sh                 # every platform in .env
#   ./scripts/run_all.sh cognodb neo4j   # a subset
#   SKIP_DOCKER=1 ./scripts/run_all.sh   # cloud targets only
#
# The containers are left running afterwards so a failed run can be
# investigated; `docker compose down -v` removes them and their data.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "error: .env not found. Copy .env.example to .env and fill in your credentials." >&2
  exit 2
fi

if [[ "${SKIP_DOCKER:-0}" != "1" ]]; then
  echo "== starting the resource-capped comparators =="
  docker compose up -d --wait
  echo "== container resource usage (should show the caps) =="
  docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
fi

echo "== configuration =="
gdbbench validate

echo "== benchmark and report =="
gdbbench run-all ${@:+--targets "$@"}

echo
echo "Done. Tables: results/tables.md — charts: results/charts/ — evidence: results/raw/"
