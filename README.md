# MedAgentBench A2A Benchmark

A medical AI evaluation platform that implements the A2A (Agent-to-Agent) protocol for evaluating clinical decision-making agents using specialty-specific rubrics and LLM-as-Judge methodology.

## Project Structure

```
project2/
├── src/
│   └── agentbeats/              # AgentBeats framework
│       ├── client.py            # A2A client for sending messages
│       ├── green_executor.py    # Green agent executor
│       ├── models.py            # AgentBeats models
│       └── tool_provider.py     # Tool provider for agents
├── scenarios/
│   └── medbench/                # Core benchmark package
│       ├── medbench_judge_a2a.py    # Green agent (A2A-compatible judge)
│       ├── medical_agent.py         # Purple agent (medical AI)
│       ├── medical_evaluation.py    # LLM-as-Judge evaluation engine
│       ├── medbench_models.py       # Pydantic models
│       ├── task_adapter.py          # Task loading and management
│       ├── reference_solution.py    # Reference implementations
│       ├── official_exporter.py     # Official format export
│       └── medbench_common.py       # Common utilities
├── data/medagentbench/            # MedAgentBench test data
│   └── test_data_v2.json
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── pyproject.toml                 # Python dependencies
├── run_benchmark.py               # Main orchestration script
├── scenario.toml                  # Benchmark configuration
└── README.md                      # This file
```

## Running and Testing Locally

### Install dependencies

```bash
pip install -r requirements.txt
# or: uv sync
```

### Set up environment

Create a `.env` file with your API keys:

```bash
echo "GOOGLE_API_KEY=your-google-api-key" > .env
echo "GROQ_API_KEY=your-groq-api-key" >> .env
```

### Clone MedAgentBench data (if not already present)

```bash
git clone https://github.com/stanfordmlgroup/MedAgentBench.git
cp MedAgentBench/data/medagentbench/test_data_v2.json data/medagentbench/
```

### Run the servers

```bash
# Terminal 1: Start the green agent (judge)
python -m scenarios.medbench.medbench_judge_a2a --host 0.0.0.0 --port 9008

# Terminal 2: Start the purple agent (medical AI)
python scenarios/medbench/medical_agent.py --host 0.0.0.0 --port 9010
```

### Run the benchmark

```bash
python run_benchmark.py --config scenario.toml

# Show agent logs for debugging
python run_benchmark.py --config scenario.toml --show-logs

# Start agents only (for debugging)
python run_benchmark.py --config scenario.toml --serve-only
```

### Using the A2A CLI

There is a launcher script included as `src/agentbeats/client_cli.py`.

To run the scenario, make sure both green and purple agents are running. Then run:

```bash
python -m launcher.client_cli scenario.toml output.json
```

This will use the scenario file `scenario.toml` and will save the results in `output.json`.

## Running with Docker

### Using Docker Compose (Recommended)

```bash
# Or start agents only
docker compose up green-agent purple-agent
```

## Follow the AgentBeats recommendations for the GHCR images

## Automated CI/CD Pipeline

### GitHub Actions Workflow: `.github/workflows/publish-green.yml`

The automation pipeline triggers on multiple events and builds/pushes the Docker image without manual intervention.

```yaml
# Green Agent (MedBenchJudge) - Publish to GHCR
name: Publish Green Agent

on:
  push:
    branches:
      - main                    # Auto-build on main branch changes
    tags:
      - 'green-v*'             # Build versioned releases
  pull_request:               # Test builds in PRs
  workflow_dispatch:           # Manual trigger option

jobs:
  build-and-push-green:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      packages: write          # Required for pushing to GHCR

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided by GitHub

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/yshao/medbench-judge
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}
            type=raw,value=latest,enable={{is_default_branch}}
            type=ref,event=pr

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./scenarios/medbench/Dockerfile.judge
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: linux/amd64
          cache-from: type=gha      # Cache for faster builds
          cache-to: type=gha,mode=max

      - name: Image digest
        run: |
          echo "## Green Agent Published :rocket:" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo '```bash' >> $GITHUB_STEP_SUMMARY
          echo "docker pull ghcr.io/yshao/medbench-judge:latest" >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
```

### Automation Triggers Explained

| Trigger Event | When It Runs | Image Pushed | Tags Created |
|----------------|--------------|---------------|--------------|
| **Push to main** | Every commit to main branch | ✅ Yes | `:latest` |
| **Git tag `green-v*`** | When version tag pushed | ✅ Yes | `:1.0.0`, `:1`, `:latest` |
| **Pull Request** | When PR created/updated | ❌ No | `:pr-123` |
| **Manual dispatch** | Click "Run workflow" button | ✅ Yes | `:latest` |

### Zero-Touch Deployment Flow

```
Developer Push Code
        ↓
GitHub Actions Detects Change
        ↓
Triggers "publish-green.yml" Workflow
        ↓
┌─────────────────────────────────────┐
│ 1. Checkout Repository              │
│ 2. Set up Docker Buildx              │
│ 3. Log in to GHCR (auto-token)       │
│ 4. Extract metadata (tags, labels)    │
│ 5. Build Docker image                │
│    ├─ Use cache for layers            │
│    ├─ Run tests (if configured)       │
│    └─ Scan for security issues       │
│ 6. Push to GHCR                       │
│ 7. Publish image digest              │
└─────────────────────────────────────┘
        ↓
Image Available at: ghcr.io/yshao/medbench-judge:latest
        ↓
Leaderboard Evaluations Use New Image
```

---

## End-to-End Build Process

### Local Development Build

```bash
# Navigate to project root
cd /path/to/project2

# Build the green agent image locally
docker build -f scenarios/medbench/Dockerfile.judge -t medbench-judge:local .

# Run with configuration
docker run -p 9008:9008 \
  -e GROQ_API_KEY="${GROQ_API_KEY}" \
  -e DATA_PATH="/app/data/medagentbench/test_data_v2.json" \
  medbench-judge:local
```

### Automated Production Build

```bash
# 1. Make code changes
git checkout main
vim scenarios/medbench/medbench_judge_a2a.py

# 2. Commit and push (triggers automatic build)
git add scenarios/medbench/medbench_judge_a2a.py
git commit -m "feat: improve evaluation logic"
git push origin main

# 3. GitHub Actions automatically:
#    - Builds Docker image
#    - Runs tests
#    - Pushes to ghcr.io/yshao/medbench-judge:latest
#    - Creates version tags if git tag exists

# 4. Image is now available for leaderboard evaluations
docker pull ghcr.io/yshao/medbench-judge:latest
```

### Versioned Release Build

```bash
# Create and push version tag
git tag -a green-v1.2.0 -m "Release version 1.2.0"
git push origin green-v1.2.0

# GitHub Actions automatically creates:
# - ghcr.io/yshao/medbench-judge:1.2.0
# - ghcr.io/yshao/medbench-judge:1
# - ghcr.io/yshao/medbench-judge:latest
```

---

## Runtime Configuration

### Docker Compose Integration

In leaderboard repositories, the green agent is orchestrated via `docker-compose.yml` generated from `scenario.toml`:

```yaml
services:
  green-agent:
    image: ghcr.io/yshao/medbench-judge:latest
    platform: linux/amd64
    container_name: green-agent
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DATA_PATH=/app/data/medagentbench/test_data_v2.json
      - LOG_LEVEL=INFO
      - CARD_URL=http://green-agent:9008/
      - HOST=0.0.0.0
      - AGENT_PORT=9008
    ports:
      - "9008:9008"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9008/health"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 60s
```


## Automated CI/CD Pipeline

### GitHub Actions Workflow: `.github/workflows/publish-green.yml`

The automation pipeline triggers on multiple events and builds/pushes the Docker image without manual intervention.


### Automated Production Build

```bash
# 1. Make code changes
git checkout main
vim scenarios/medbench/medbench_judge_a2a.py

# 2. Commit and push (triggers automatic build)
git add scenarios/medbench/medbench_judge_a2a.py
git commit -m "feat: improve evaluation logic"
git push origin main

# 3. GitHub Actions automatically:
#    - Builds Docker image
#    - Runs tests
#    - Pushes to ghcr.io/yshao/medbench-judge:latest
#    - Creates version tags if git tag exists

# 4. Image is now available for leaderboard evaluations
docker pull ghcr.io/yshao/medbench-judge:latest
```

## AgentBeats Evaluation

### Environment Variables Reference

#### Required for Evaluation

| Variable | Description | Example | Source |
|----------|-------------|---------|--------|
| `GROQ_API_KEY` | Groq API key for LLM evaluation | `gsk_...` | GitHub Secret |
| `DATA_PATH` | Path to test dataset | `/app/data/medagentbench/test_data_v2.json` | Built into image |

#### Optional Configuration

| Variable | Description | Default | Impact |
|----------|-------------|---------|--------|
| `GOOGLE_API_KEY` | Google API for additional services | - | Enables extra features |
| `LOG_LEVEL` | Logging verbosity | `INFO` | Debug output control |
| `CARD_URL` | External agent card URL | Auto-generated | Enables external discovery |
| `HOST` | Server binding address | `0.0.0.0` | Network accessibility |
| `AGENT_PORT` | Server port | `9008` | Port conflicts |
| `DRY_RUN` | Use mock scores | `false` | Testing vs production |

---
