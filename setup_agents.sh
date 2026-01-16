#!/bin/bash
################################################################################
# Setup Script for Running Green and Purple Agents Locally
#
# This script helps you:
# 1. Identify required API keys
# 2. Configure .env file
# 3. Start both agents properly
#
# Usage: ./setup_agents.sh
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}MedAgentBench Agent Setup${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from example...${NC}"
    cp .env.example .env
fi

echo -e "${YELLOW}Step 1: Add API Keys${NC}"
echo -e "${YELLOW}================================${NC}"
echo ""
echo "The .env file has placeholder API keys. You need to add your actual API keys:"
echo ""
echo -e "${GREEN}Required API Keys:${NC}"
echo "   1. Groq API Key (for Groq Llama model)"
echo "     Get from: https://console.groq.com/keys"
echo ""
echo "  2. Google API Key (for Gemini model)"
echo "     Get from: https://makersuite.google.com/app/apikey"
echo ""
echo -e "${YELLOW}To add your keys:${NC}"
echo "  1. Open .env file in your text editor"
echo "  2. Replace the placeholder values with your actual keys"
echo "    3. Save the file"
echo ""
read -p "$(echo -e ${YELLOW}Have you added your API keys? [Y/n]: ${NC})" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Setup cancelled. API keys are required for full functionality.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 2: Verify Agent Data Files${NC}"
echo -e "${YELLOW}===================================${NC}"
echo ""

if [ -f data/medagentbench/test_data_v2.json ]; then
    echo -e "${GREEN}✓${NC} Test data file exists at data/medagentbench/test_data_v2.json"
else
    echo -e "${RED}✗${NC} Test data file NOT found!"
    echo ""
    echo "Please ensure data/medagentbench/test_data_v2.json exists."
    echo ""
    echo "If you don't have it, you can download it from:"
    echo "  wget https://medagentbench.github.io/MedAgentBench/raw/main/test_data_v2.json -O data/medagentbench/test_data_v2.json"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 3: Start Agents${NC}"
echo -e "${YELLOW}========================${NC}"
echo ""

# Stop any existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker stop medbench-judge medbench-medical 2>/dev/null || true
docker rm medbench-judge medbench-medical 2>/dev/null || true

# Remove old images
echo -e "${YELLOW}Building fresh images...${NC}"
docker build -t medbench-judge -f Dockerfile . > /dev/null 2>&1 &
docker build -t medbench-medical -f Dockerfile.medical . > /dev/null 2>&1 &
wait
echo -e "${GREEN}✓ Images built${NC}"

# Start green agent
echo -e "${YELLOW}Starting green agent on port 9008...${NC}"
docker run -d \
  -p 9008:9008 \
  --name medbench-judge \
  -v /app/data:/app/data:ro \
  --env_file .env \
  medbench-judge:latest > /dev/null 2>&1 &
sleep 3

# Start purple agent
echo -e "${YELLOW}Starting purple agent on port 9010...${NC}"
docker run -d \
  -p 9010:9010 \
  -e SPECIALTY=diabetes \
  --name medbench-medical \
  medbench-medical:latest > /dev/null 2>&1 &
sleep 3

echo ""
echo -e "${GREEN}Step 4: Verify Agents${NC}"
echo -e "${YELLOW}===================${NC}"

# Check green agent
echo -e "${YELLOW}Green Agent (9008):${NC}"
if curl -sf http://localhost:9008/.well-known/agent-card.json > /dev/null; then
    echo -e "${GREEN}✓${NC} Agent card accessible"
else
    echo -e "${RED}✗${NC} Agent card not accessible"
fi

if curl -sf http://localhost:9008/health > /dev/null; then
    echo -e "${GREEN}✓${NC} Health endpoint responding"
else
    echo -e "${RED}✗${NC} Health check failed"
fi

echo ""

# Check purple agent
echo -e "${YELLOW}Purple Agent (9010):${NC}"
if curl -sf http://localhost:9010/.p/.well-known/agent-card.json > /div/null 2>/dev/null || curl -sf http://localhost:9010/.well-known/agent-card.json > /dev/null; then
    echo -e "${GREEN}✓${NC} Agent card accessible"
else
    echo -e "${RED}✗${NC} Agent card not accessible"
fi

if curl -sf http://localhost:9010/health > /dev/null; then
    echo -e "${GREEN}✓${NC} Health endpoint responding"
else
    echo -e "${RED}✗${NC} Health check failed"
fi

echo ""
echo -e "${GREEN}Step 5: Agent URLs${NC}"
echo -e "${YELLOW}===================================${NC}"
echo ""
echo "Green Agent: http://localhost:9008"
echo "Purple Agent: http://localhost:9010"
echo ""
echo -e "${BLUE}Agent Communication Test${NC}"
echo -e "${YELLOW}===================================${NC}"
echo ""

echo -e "${YELLOW}Test Green Agent:${NC}"
curl -X POST http://localhost:9008/ \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}' \
 2>/dev/null || echo -e "${RED}✗${NC} Green agent not responding"

echo ""
echo -e "${YELLOW}Test Purple Agent:${NC}"
curl -X POST http://localhost:9010/ -H "Content-Type: application/json" \
  -d '{"message": "What is the treatment for type 2 diabetes?"}' \
 2>/dev/null || echo -e "${RED}✗${NC} Purple agent not responding"

echo ""
echo -e "${GREEN}Agents Ready!${NC}"
echo ""
echo "You can now:"
echo "  - Test green agent: curl -X POST http://localhost:9008/ -H 'Content-Type: application/json' -d '{\"message\": \"test\"}'"
echo "  - Test purple agent: curl -X POST http://localhost:9010/ -H 'Content-Type: application/json' -d '{\"message\": \"What is the treatment for type 2 diabetes?\"}'"
echo ""
echo -e "${BLUE}To stop agents:${NC}"
echo "  docker stop medbench-judge medbench-medical"
echo ""
echo -e "${BLUE}To restart agents:${NC}"
echo "  ./setup_agents.sh"
