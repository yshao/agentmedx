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
├── tests/                         # A2A conformance tests
│   ├── test_a2a_conformance.py
│   └── test_purple_agent.py
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
# Build and run with Docker Compose
docker-compose --profile benchmark up --build

# Or start agents only
docker-compose up green-agent purple-agent
```

### Building individual images

```bash
# Build the image
docker build --platform linux/arm64 -t medbench-green-agent .

# Run the container
docker run --env-file .env -p 9008:9008 medbench-green-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
pip install -r requirements.txt

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
pytest tests/test_a2a_conformance.py --agent-url http://localhost:9008

# Run all tests
pytest tests/
```

## Medical Evaluation Details

### Features

- **300 Clinical Tasks**: Across 10 medical categories (diabetes, cardiology, internal medicine, etc.)
- **Specialty-Specific Evaluation**:
  - **Diabetes**: 6 criteria (medication appropriateness, A1C targets, comorbidity management, lifestyle, safety, monitoring)
  - **General Medical**: 3 criteria (accuracy, completeness, medical correctness)
- **LLM-as-Judge Evaluation**: Uses Gemini 2.5-flash with structured rubrics
- **Rate Limiting**: Automatic retry logic with exponential backoff for API quotas
- **A2A Protocol**: Standard Agent-to-Agent communication for agent interactions

### Configuration

Edit `scenario.toml` to customize:

```toml
[config]
task_id = "diabetes_001"          # Which task to evaluate
medical_category = "diabetes"     # Determines evaluation rubric
```

### Available Medical Categories

| Category | Criteria | Max Score |
|----------|----------|-----------|
| `diabetes` | 6 specialty criteria | 60 |
| `cardiology` | 3 general criteria | 30 |
| `internal_medicine` | 3 general criteria | 30 |
| `general_medical` | 3 general criteria | 30 |

### Evaluation Criteria

#### Diabetes (6 criteria, 0-10 each)

1. **Medication Appropriateness**: Are medications suitable for this patient's profile?
2. **A1C Target**: Does the plan address A1C goals appropriately?
3. **Comorbidity Management**: Are comorbidities (hypertension, kidney, lipids) addressed?
4. **Lifestyle Recommendations**: Are diet and exercise guidance included?
5. **Safety**: Are there contraindications or dangerous drug interactions?
6. **Monitoring Plan**: Is there a clear follow-up and monitoring strategy?

#### General Medical (3 criteria, 0-10 each)

1. **Accuracy**: How close is the response to the expected answer?
2. **Completeness**: Does it address all aspects of the question?
3. **Medical Correctness**: Is the information clinically sound?

### Component Overview

#### Green Agent (`medbench_judge_a2a.py`)
- **Port**: 9008 (default)
- **Purpose**: Orchestrates medical benchmark evaluation
- **Responsibilities**:
  - Loads MedAgentBench tasks from JSON
  - Validates task_id and medical_category
  - Sends tasks to medical agents via A2A
  - Evaluates responses using specialty-specific rubrics
  - Returns structured evaluation results

#### Purple Agent (`medical_agent.py`)
- **Port**: 9010 (default)
- **Purpose**: Medical AI participant that receives clinical tasks
- **Features**:
  - Specialty-specific instructions (diabetes, cardiology, internal medicine, general)
  - Uses Google ADK Agent framework
  - Converts to A2A for agent communication

#### Evaluation Engine (`medical_evaluation.py`)
- **Model**: gemini-2.5-flash
- **Purpose**: LLM-as-Judge with medical rubrics
- **Features**:
  - Specialty-specific prompts for accurate evaluation
  - Rate limit handling with exponential backoff
  - JSON parsing with regex fallback
  - Extracts structured scores from responses

### Rate Limiting

The benchmark includes automatic retry logic for handling API rate limits:

- **Exponential Backoff**: 2^attempt * 5 seconds (5s, 10s, 20s)
- **Max Retries**: 3 attempts
- **Applied to**: Both participant communication and LLM evaluation calls

## Troubleshooting

### "Module not found" errors

**Problem**: Cannot import agentbeats framework

**Solution**: Ensure PYTHONPATH includes both `tutorial/src/` and `project2/`:

```bash
export PYTHONPATH=/path/to/tutorial/src:/path/to/project2:$PYTHONPATH
```

### "No such file or directory: test_data_v2.json"

**Problem**: MedAgentBench data not found

**Solution**: Clone MedAgentBench and copy data:

```bash
git clone https://github.com/stanfordmlgroup/MedAgentBench.git
cp MedAgentBench/data/medagentbench/test_data_v2.json data/medagentbench/
```

### "Invalid task_id"

**Problem**: The specified task_id doesn't exist in the loaded data

**Solution**: Check available task IDs:

```bash
python -c "
import json
with open('data/medagentbench/test_data_v2.json') as f:
    tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = tasks.get('tasks', [])
    task_ids = [t.get('id') for t in tasks if isinstance(t, dict)]
    print('Available task IDs:', task_ids[:20])  # Show first 20
"
```

### 429 RESOURCE_EXHAUSTED errors

**Problem**: API quota exceeded

**Solution**:
- Use a Google API key with available quota
- Consider switching to `gemini-1.5-flash` for higher free tier limits

### Agent not becoming ready

**Problem**: Agent doesn't respond to health checks

**Solution**:
1. Check if agent is actually running (`ps aux | grep <port>`)
2. Check agent logs for startup errors
3. Verify PYTHONPATH is set correctly
4. Check if Google API key is loaded

## Development

### Adding New Medical Categories

1. Add new category to `MedicalCategory` enum in `medbench_models.py`
2. Add corresponding evaluation method in `medical_evaluation.py`
3. Add specialty instructions to `medical_agent.py`
4. Update `_valid_categories` list in `medbench_judge_a2a.py`

### Testing

To test with a specific task:

1. Check available task IDs from `test_data_v2.json`
2. Set `task_id` and `medical_category` in `scenario.toml`
3. Run benchmark: `python run_benchmark.py --config scenario.toml`

## License

This implementation is built on:
- AgentBeats framework: Apache-2.0 License
- Google GenAI SDK: Apache 2.0 License
- Google ADK: Apache 2.0 License
- A2A SDK: Apache 2.0 License
