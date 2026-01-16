#!/bin/bash
################################################################################
# Docker Deployment Test
#
# This script simulates a docker deployment by running a test evaluation
# between the green and purple agents.
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Docker Deployment Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo ""
    echo "Please run ./setup_agents.sh first to configure your environment."
    exit 1
fi

# Verify agents are running
echo -e "${YELLOW}Checking agent status...${NC}"

GREEN_RUNNING=false
PURPLE_RUNNING=false

if docker ps --filter "name=medbench-judge" --format "{{.Status}}" | grep -q "running"; then
    GREEN_RUNNING=true
    echo -e "${GREEN}✓${NC} Green agent running on port 9008"
else
    echo -e "${YELLOW}○ Green agent not running${NC}"
fi

if docker ps --filter "name=medbench-medical" --format "{{.Status}}" | grep -q "running"; then
    PURPLE_RUNNING=true
    echo -e "${GREEN}✓${NC} Purple agent running on port 9010"
else
    echo -e "${YELLOW}○ Purple agent not running${NC}"
fi

echo ""

if ! $GREEN_RUNNING || ! $PURPLE_RUNNING; then
    echo -e "${RED}Error: One or both agents are not running!${NC}"
    echo ""
    echo "Please run: ./setup_agents.sh"
    exit 1
fi

echo -e "${GREEN}Both agents are running!${NC}"
echo ""

# Test agent communication
echo -e "${YELLOW}Testing Agent Communication...${NC}"
echo -e "${YELLOW}==================================${NC}"

echo -e "${BLUE}Test 1: Check Agent Cards${NC}"
echo ""

echo -e "Green Agent Card:"
curl -s http://localhost:9008/.well-known/agent-card.json | python3 -m json.tool | head -20
echo ""

echo -e "Purple Agent Card:"
curl -s http://localhost:9010/.p/.well-known/agent-card.json | python3 -m json.tool | head -20
echo ""

# Test message flow
echo -e "${BLUE}Test 2: Test Message Flow${NC}"
echo -e "${YELLOW}==================================${NC}"

echo -e "Sending medical query to purple agent..."
curl -s -X POST http://localhost:9010/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the treatment for type 2 diabetes?}' \
  > /tmp/purple_response.json 2>&1

echo "Purple Agent Response:"
python3 -m json.tool /tmp/purple_response.json 2>/dev/null || echo "Response parsing failed"

echo ""
echo -e "${BLUE}Test 3: Docker Deployment Verification${NC}"
echo -e "${YELLOW}======================================${NC}"

echo -e "${GREEN}Container Status:${NC}"
echo -e "  Container Name\tState\tPorts"
docker ps --filter "name=medbench" --format "table {{.Names}}\t{{.State}}\t{{.Ports}}"

echo ""
echo -e "${GREEN}Docker Images:${NC}"
echo "  Image Tag\tSize"
docker images | grep medbench

echo ""
echo -e "${GREEN}Deployment Simulation Complete!${NC}"
echo ""
echo -e "${BLUE}Local Deployment URL:${NC}"
echo "  Green Agent: http://localhost:9008"
echo "  Purple Agent: http://localhost:9010"
echo ""
echo -e "${BLUE}CI/CD Registry URLs (when deployed):${NC}"
echo "  ghcr.io/yourusername/project2:green-agent:v1.0.0"
echo "  ghcr.io/yourusername/project2:purple-agent:v1.0.0"
