#!/bin/sh
set -e

################################################################################
# MedBenchJudge Agent - A2A-Compatible Launch Script
#
# This script launches the A2A-compatible MedBenchJudge agent using Google ADK
# framework with SSE transport support for agentbeats-client compatibility.
################################################################################

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${AGENT_PORT:-9008}"
DATA_PATH="${DATA_PATH:-/app/data/medagentbench/test_data_v2.json}"

# Check if dry run mode is enabled
if [ "$DRY_RUN" = "true" ] || [ "$EXPORT_OFFICIAL" = "false" ]; then
    DRY_RUN_FLAG="--dry-run"
else
    DRY_RUN_FLAG=""
fi

# Log startup
echo "=================================="
echo "Starting MedBenchJudge Agent (A2A-Compatible)"
echo "=================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Data Path: $DATA_PATH"
echo "Dry Run: ${DRY_RUN:-false}"
echo "=================================="
echo "Command: python -m scenarios.medbench.medbench_judge_a2a --host $HOST --port $PORT --data-path $DATA_PATH $DRY_RUN_FLAG"
echo "=================================="

# Execute the A2A-compatible judge
exec python -m scenarios.medbench.medbench_judge_a2a \
    --host "$HOST" \
    --port "$PORT" \
    --data-path "$DATA_PATH" \
    $DRY_RUN_FLAG
