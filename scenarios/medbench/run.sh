#!/bin/sh
set -e

################################################################################
# MedBenchJudge Agent - AgentBeats Integration Script
#
# This script serves as the integration layer between the AgentBeats controller
# and the MedBenchJudge green agent. It reads environment variables set by
# the controller and constructs the appropriate command to start the agent.
#
# Environment Variables:
#   HOST          - Host address to bind to (default: 0.0.0.0)
#   AGENT_PORT    - Port to listen on (default: 9008)
#   DATA_PATH     - Path to test_data_v2.json (required)
#   CARD_URL      - External URL for agent card (optional)
#   EXPORT_OFFICIAL - Export in official MedAgentBench format (default: false)
#   OUTPUT_DIR    - Directory for official format exports (default: outputs)
#   MODEL_NAME    - Model name for output directory (default: agentx-medical)
#
# The AgentBeats controller sets HOST and AGENT_PORT automatically.
################################################################################

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${AGENT_PORT:-9008}"
DATA_PATH="${DATA_PATH:-/app/data/medagentbench/test_data_v2.json}"
CARD_URL="${CARD_URL:-}"
EXPORT_OFFICIAL="${EXPORT_OFFICIAL:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
MODEL_NAME="${MODEL_NAME:-agentx-medical}"

# Set PYTHONPATH for agentbeats module
export PYTHONPATH="/app/tutorial/src:${PYTHONPATH:-}"

# Build the base command
CMD="python -m scenarios.medbench.medbench_judge_a2a"
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --data-path $DATA_PATH"

# Add optional arguments if provided
if [ -n "$CARD_URL" ]; then
    CMD="$CMD --card-url $CARD_URL"
fi

if [ "$EXPORT_OFFICIAL" = "true" ]; then
    CMD="$CMD --export-official"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi

# Log startup (useful for debugging)
echo "=================================="
echo "Starting MedBenchJudge Agent"
echo "=================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Data Path: $DATA_PATH"
if [ -n "$CARD_URL" ]; then
    echo "Card URL: $CARD_URL"
fi
echo "Export Official: $EXPORT_OFFICIAL"
echo "Output Dir: $OUTPUT_DIR"
echo "Model Name: $MODEL_NAME"
echo "=================================="
echo "Command: $CMD"
echo "=================================="

# Execute the command (exec replaces shell process, proper signal handling)
exec $CMD
