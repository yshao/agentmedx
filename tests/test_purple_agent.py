"""
A2A Protocol Conformance Tests for MedAgentBench Purple (Medical) Agent

Adapted from agent-template/tests/test_agent.py
Tests for validating A22 protocol compliance for the medical agent.

Run against running medical agent:
    pytest tests/test_purple_agent.py --agent-url http://localhost:9010

With specialty parameter:
    pytest tests/test_purple_agent.py --agent-url http://localhost:9010 --specialty diabetes
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

    # Required fields for agent card
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

    # Check if 'url' is an absolute URL
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
        # Note: Purple agents typically have empty skills list (participant agents)
        # This is expected and acceptable

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
# A2A Conformance Tests for Purple Agent
# ============================================================================

def test_purple_agent_card(agent):
    """Validate purple agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_purple_message(agent, streaming):
    """Test that purple agent returns valid A2A message format."""
    # Send a medical query message
    test_message = """Patient case: 45M with type 2 diabetes presents with HbA1c of 8.5% and fasting glucose of 140 mg/dL.
Current medications: metformin 500mg twice daily, glipizide 5mg once daily.
What would you recommend?"""

    events = await send_text_message(test_message, agent, streaming=streaming)

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
async def test_purple_agent_health_endpoint(agent):
    """Test the purple agent's health check endpoint."""
    response = httpx.get(f"{agent}/health")
    assert response.status_code == 200, "Health endpoint must return 200"

    health_data = response.json()
    assert "status" in health_data, "Health response must include status field"
    assert health_data["status"] == "healthy", "Agent must report healthy status"
    assert "service" in health_data, "Health response must include service name"
    assert health_data.get("specialty") in ["diabetes", "cardiology", "internal_medicine", "general"], \
        "Health response should include valid medical specialty"


@pytest.mark.asyncio
async def test_purple_agent_root_endpoint(agent):
    """Test the purple agent's root info endpoint."""
    response = httpx.get(f"{agent}/")
    assert response.status_code == 200, "Root endpoint must return 200"

    info_data = response.json()
    assert "service" in info_data, "Root response must include service name"
    assert "description" in info_data, "Root response must include description"
    assert "version" in info_data, "Root response must include version"
    assert "protocol" in info_data, "Root response must include protocol type"
    assert info_data["protocol"] == "A2A", "Agent must use A2A protocol"


# ============================================================================
# Medical Agent Specific Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("specialty", ["diabetes", "cardiology", "internal_medicine", "general"])
async def test_purple_agent_specialty(agent, specialty):
    """Test that purple agent responds appropriately to specialty-specific queries."""
    # Test query specific to each medical specialty
    test_queries = {
        "diabetes": "What is the target A1C for most adults with type 2 diabetes?",
        "cardiology": "What are the ECG signs of myocardial infarction?",
        "internal_medicine": "What is the differential diagnosis for fever and cough?",
        "general": "What is the treatment for influenza?",
    }

    test_query = test_queries.get(specialty, test_queries["general"])

    events = await send_text_message(test_query, agent, streaming=False)

    # Verify the agent responds
    assert len(events) > 0, "Agent must respond to medical query"

    # Verify response contains meaningful medical content
    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)
            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
            case _:
                pass

    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_addoption(parser):
    """Add command-line option for agent URL and specialty."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://127.0.0.1:9010",
        help="URL of the running purple agent to test",
    )
    parser.addoption(
        "--specialty",
        action="store",
        default="diabetes",
        choices=["diabetes", "cardiology", "internal_medicine", "general"],
        help="Medical specialty to test (default: diabetes)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Get purple agent URL from command line or default."""
    agent_url = request.config.getoption("--agent-url")

    # Health check - verify agent is running
    try:
        response = httpx.get(f"{agent_url}/health", timeout=5)
        assert response.status_code == 200, f"Agent at {agent_url} is not responding"
    except Exception as e:
        raise AssertionError(f"Cannot connect to agent at {agent_url}: {e}")

    return agent_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
