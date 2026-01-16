"""
MedBench Common - Shared models and utilities for MedAgentBench evaluation.

This module contains shared Pydantic models and the agent card function,
following the pattern from tutorial/scenarios/debate/debate_judge_common.py
"""

from typing import Literal, Any

from pydantic import BaseModel

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Import evaluation result models and request/result types
from .medbench_models import (
    DiabetesScore,
    GeneralMedicalScore,
    MedicalEvaluationResult,
    EvalRequest,
    EvalResult,
)

# Import MedicalCategory as a type alias
from .medbench_models import MedicalCategory


# ============================================================================
# Agent Card Function
# ============================================================================

def medbench_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """
    Create the AgentCard for MedBenchJudge (green agent).

    Args:
        agent_name: Name of the agent
        card_url: URL to advertise in the agent card

    Returns:
        AgentCard configured for MedAgentBench evaluation
    """
    skill = AgentSkill(
        id='evaluate_medical_case',
        name='Evaluate medical agent responses',
        description='Orchestrate and evaluate medical case assessments against MedAgentBench benchmark data.',
        tags=['medical', 'benchmark', 'evaluation'],
        examples=["""
{
  "participants": {
    "medical_agent": "https://medical-agent.example.com:443"
  },
  "config": {
    "task_id": "diabetes_001",
    "medical_category": "diabetes"
  }
}
"""]
    )

    agent_card = AgentCard(
        name=agent_name,
        description='Green agent for MedAgentBench - orchestrates medical case evaluations and scores agent responses using specialty-specific rubrics (diabetes: 6 criteria, general: 3 criteria).',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card


# ============================================================================
# Evaluation Models for structured response parsing
# ============================================================================

class AgentEvaluationSummary(BaseModel):
    """Summary of a single agent's evaluation."""
    agent_name: str
    task_id: str
    medical_category: str
    total_score: float
    feedback: str


# ============================================================================
# Medical Category Constants
# ============================================================================

VALID_MEDICAL_CATEGORIES = [
    "diabetes",
    "cardiology",
    "internal_medicine",
    "general_medical",
]


def is_valid_medical_category(category: str) -> bool:
    """Check if a medical category is valid."""
    return category in VALID_MEDICAL_CATEGORIES
