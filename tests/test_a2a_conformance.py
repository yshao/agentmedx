"""
A2A Protocol Conformance Tests for MedBenchJudge

Adapted from green-agent-template/tests/test_agent.py
Validates that the MedBenchJudge green agent properly implements the A2A protocol.

Run against running agent:
    pytest tests/test_a2a_conformance.py --agent-url http://localhost:9008
"""

import sys
from pathlib import Path
from typing import Any

import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


# ============================================================================
# A2A Validation Helpers (from template)
# ============================================================================

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# ============================================================================
# A2A Messaging Helpers
# ============================================================================

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# ============================================================================
# A2A Conformance Tests
# ============================================================================

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    # Send a simple greeting message
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


@pytest.mark.asyncio
async def test_green_agent_eval_request(agent):
    """Test that the green agent properly handles evaluation requests."""
    import json

    # Construct a proper EvalRequest for the MedBenchJudge
    eval_request = {
        "participants": {
            "medical_agent": "http://localhost:9010"  # Default purple agent port
        },
        "config": {
            "task_id": "test_001",
            "medical_category": "diabetes"
        }
    }

    # Send the evaluation request
    events = await send_text_message(json.dumps(eval_request), agent, streaming=False)

    # Should receive task event(s)
    assert events, "Agent should respond with at least one event for eval request"

    # Check for proper event structure
    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert not all_errors, f"Evaluation request validation failed:\n" + "\n".join(all_errors)


# ============================================================================
# Custom Tests for MedBenchJudge
# ============================================================================

def test_agent_card_has_medical_skills(agent):
    """Verify that the agent card declares medical evaluation skills."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200

    card_data = response.json()

    # Check that skills are defined
    assert 'skills' in card_data, "Agent card must have skills"
    assert len(card_data['skills']) > 0, "Agent must have at least one skill"

    # Check for medical-related keywords in skills
    skills_text = str(card_data['skills']).lower()
    medical_keywords = ['medical', 'eval', 'judge', 'assessment', 'benchmark']
    has_medical_skill = any(keyword in skills_text for keyword in medical_keywords)

    assert has_medical_skill, f"Agent skills should be medical-related. Got: {card_data['skills']}"


def test_agent_card_supports_streaming(agent):
    """Verify that the agent card declares streaming support."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200

    card_data = response.json()

    # Check capabilities include streaming
    assert 'capabilities' in card_data, "Agent card must have capabilities"
    assert 'streaming' in card_data['capabilities'], "Agent should support streaming"
    assert card_data['capabilities']['streaming'] is True, "Streaming should be enabled"


# ============================================================================
# Test Configuration (from template conftest.py)
# ============================================================================

def pytest_addoption(parser):
    """Add command-line option for agent URL."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://127.0.0.1:9008",
        help="URL of the running agent to test",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Get agent URL from command line or default."""
    agent_url = request.config.getoption("--agent-url")

    # Health check - verify agent is running
    try:
        response = httpx.get(f"{agent_url}/.well-known/agent-card.json", timeout=5)
        assert response.status_code == 200, f"Agent at {agent_url} is not responding"
    except Exception as e:
        raise AssertionError(f"Cannot connect to agent at {agent_url}: {e}")

    return agent_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
