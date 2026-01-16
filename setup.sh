#!/bin/bash
# One-time setup for AgentX2 MedAgentBench deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================"
echo "   AgentX2 MedAgentBench Setup"
echo -e "========================================${NC}"
echo ""

# ========================================
# 1. OS Detection
# ========================================
echo -e "${BLUE}[1/7] Detecting OS...${NC}"

OS_TYPE="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
    # Check for WSL
    if grep -q Microsoft /proc/version 2>/dev/null; then
        OS_TYPE="wsl"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS_TYPE="windows"
fi

echo -e "${GREEN}✓ OS detected: $OS_TYPE${NC}"
echo ""

# ========================================
# 2. Docker Check
# ========================================
echo -e "${BLUE}[2/7] Checking Docker...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo ""
    echo "Please install Docker first:"
    echo ""
    case "$OS_TYPE" in
        "linux")
            echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
            echo "  sudo sh get-docker.sh"
            ;;
        "macos")
            echo "  Download from: https://docs.docker.com/docker-for-mac/install/"
            ;;
        "wsl"|"windows")
            echo "  Download from: https://docs.docker.com/desktop/install/windows-install/"
            ;;
        *)
            echo "  Visit: https://docs.docker.com/get-docker/"
            ;;
    esac
    echo ""
    exit 1
fi

DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
echo -e "${GREEN}✓ Docker found: $DOCKER_VERSION${NC}"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${YELLOW}[OK] Docker daemon not running. Please start Docker and retry.${NC}"
    exit 1
fi
echo ""

# ========================================
# 3. Docker Compose Check
# ========================================
echo -e "${BLUE}[3/7] Checking Docker Compose...${NC}"

DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
    COMPOSE_VERSION=$(docker compose version | awk '{print $4}' | tr -d ',')
    echo -e "${GREEN}✓ Docker Compose (plugin) found: $COMPOSE_VERSION${NC}"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
    COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | tr -d ',')
    echo -e "${GREEN}✓ Docker Compose (standalone) found: $COMPOSE_VERSION${NC}"
else
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo ""
    echo "Please install Docker Compose:"
    echo "  Linux: https://docs.docker.com/compose/install/"
    echo "  Or update Docker Desktop to latest version"
    echo ""
    exit 1
fi
echo ""

# ========================================
# 4. Create Directory Structure
# ========================================
echo -e "${BLUE}[4/7] Creating directory structure...${NC}"

mkdir -p data/medagentbench
echo -e "${GREEN}✓ Created: data/medagentbench/${NC}"

mkdir -p outputs
echo -e "${GREEN}✓ Created: outputs/${NC}"

mkdir -p logs
echo -e "${GREEN}✓ Created: logs/${NC}"
echo ""

# ========================================
# 5. Setup Test Data
# ========================================
echo -e "${BLUE}[5/7] Setting up test data...${NC}"

if [ -f "data/medagentbench/test_data_v2.json" ]; then
    echo -e "${GREEN}✓ Test data already exists: data/medagentbench/test_data_v2.json${NC}"
else
    echo "Test data not found. Downloading from MedAgentBench repository..."

    # Clone MedAgentBench repo if not exists
    if [ ! -d "MedAgentBench" ]; then
        echo "  Cloning MedAgentBench repository..."
        git clone https://github.com/stanfordmlgroup/MedAgentBench.git
    fi

    # Copy test data
    if [ -f "MedAgentBench/data/medagentbench/test_data_v2.json" ]; then
        cp MedAgentBench/data/medagentbench/test_data_v2.json data/medagentbench/
        echo -e "${GREEN}✓ Test data copied: data/medagentbench/test_data_v2.json${NC}"
    else
        echo -e "${RED}✗ Failed to find test data in MedAgentBench repository${NC}"
        echo "  Please download manually from:"
        echo "  https://github.com/stanfordmlgroup/MedAgentBench"
        exit 1
    fi
fi
echo ""

# ========================================
# 6. Create Environment File
# ========================================
echo -e "${BLUE}[6/7] Setting up environment configuration...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}✓ Environment file already exists: .env${NC}"

    # Check if GOOGLE_API_KEY is set
    if grep -q "your_google_api_key_here" .env 2>/dev/null; then
        echo -e "${YELLOW}[OK][OK]  GOOGLE_API_KEY not set in .env${NC}"
        NEEDS_API_KEY=true
    else
        echo -e "${GREEN}✓ GOOGLE_API_KEY is configured${NC}"
        NEEDS_API_KEY=false
    fi
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
    else
        echo -e "${RED}✗ .env.example not found${NC}"
        exit 1
    fi
    NEEDS_API_KEY=true
fi
echo ""

# ========================================
# 7. Verification
# ========================================
echo -e "${BLUE}[7/7] Verifying setup...${NC}"

VERIFY_PASSED=true

# Check tutorial source
if [ -d "tutorial/src/agentbeats" ]; then
    echo -e "${GREEN}✓ Tutorial source found: tutorial/src/agentbeats/${NC}"
else
    echo -e "${YELLOW}[OK][OK]  Tutorial source not found (may be needed for framework)${NC}"
fi

# Check config file
if [ -f "config/scenario.toml" ]; then
    echo -e "${GREEN}✓ Config file found: config/scenario.toml${NC}"
else
    echo -e "${RED}✗ Config file not found: config/scenario.toml${NC}"
    VERIFY_PASSED=false
fi

# Check Docker files
if [ -f "Dockerfile" ] && [ -f "Dockerfile.medical" ] && [ -f "Dockerfile.benchmark" ]; then
    echo -e "${GREEN}✓ Docker files found${NC}"
else
    echo -e "${RED}✗ Some Docker files missing${NC}"
    VERIFY_PASSED=false
fi

# Check docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    echo -e "${GREEN}✓ docker-compose.yml found${NC}"
else
    echo -e "${RED}✗ docker-compose.yml not found${NC}"
    VERIFY_PASSED=false
fi

echo ""

# ========================================
# Summary
# ========================================
echo -e "${BLUE}========================================"
echo "   Setup Summary"
echo -e "========================================${NC}"
echo ""

if [ "$VERIFY_PASSED" = true ]; then
    echo -e "${GREEN}✓ Setup verification PASSED${NC}"
else
    echo -e "${YELLOW}[OK][OK]  Setup completed with warnings${NC}"
fi
echo ""

if [ "$NEEDS_API_KEY" = true ]; then
    echo -e "${YELLOW}[OK][OK]  IMPORTANT: Configure your Google API Key${NC}"
    echo ""
    echo "1. Get your API key from: https://makersuite.google.com/app/apikey"
    echo "2. Edit .env and set:"
    echo "   GOOGLE_API_KEY=AIza..."
    echo ""
fi

echo "Next steps:"
echo ""
echo "1. ${NEEDS_API_KEY}Configure GOOGLE_API_KEY in .env:           # Edit .env file"
echo "2. Run deployment:   ./scripts/deploy.sh"
echo ""
echo "Available commands:"
echo "  ./scripts/deploy.sh          # Start all services"
echo "  ./scripts/deploy.sh --with-fhir     # Include FHIR server"
echo "  ./scripts/deploy.sh --benchmark     # Run benchmark job"
echo "  ./scripts/deploy.sh --stop          # Stop all services"
echo "  ./scripts/deploy.sh --logs          # View logs"
echo "  ./scripts/health-check.sh   # Check service health"
echo ""
