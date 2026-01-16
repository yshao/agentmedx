"""
MedBench - Medical Agent Benchmark using A2A Protocol

This package provides a green agent for MedAgentBench evaluation,
following the AgentBeats framework pattern from the tutorial.

Components:
- MedBenchJudge: Green agent that orchestrates medical evaluations
- MedicalAgent: Purple agent (medical AI participant)
- MedicalEvaluationEngine: LLM-as-Judge with specialty rubrics
- Models: Pydantic models for tasks, scores, and results
"""

# Import data models
from .medbench_models import (
    MedAgentBenchTask,
    DiabetesScore,
    GeneralMedicalScore,
    MedicalEvaluationResult,
    EvalRequest,
    EvalResult,
)

# Import MedicalCategory as a type alias for backward compatibility
from .medbench_models import MedicalCategory

# Import evaluation engine
from .medical_evaluation import MedicalEvaluationEngine

# Import common utilities
from .medbench_common import (
    medbench_judge_agent_card,
    is_valid_medical_category,
    VALID_MEDICAL_CATEGORIES,
)

__all__ = [
    # Models
    "MedAgentBenchTask",
    "DiabetesScore",
    "GeneralMedicalScore",
    "MedicalEvaluationResult",
    "MedicalCategory",
    "EvalRequest",
    "EvalResult",
    # Evaluation
    "MedicalEvaluationEngine",
    # Utilities
    "medbench_judge_agent_card",
    "is_valid_medical_category",
    "VALID_MEDICAL_CATEGORIES",
]
