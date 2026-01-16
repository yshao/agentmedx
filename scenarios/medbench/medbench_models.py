"""
MedAgentBench Benchmark Data Models

This module contains Pydantic models for MedAgentBench tasks and evaluation results.
"""

from typing import Literal, Any
from enum import Enum
from pydantic import BaseModel, Field


# Medical specialty categories for evaluation
MedicalCategory = Literal[
    "diabetes",
    "cardiology",
    "internal_medicine",
    "general_medical"
]


class TaskFormatType(str, Enum):
    """Format type for MedAgentBench tasks."""
    CUSTOM = "custom"  # AgentX2 custom format (id, sol as array)
    OFFICIAL = "official"  # Stanford MedAgentBench format (task_id, solution as string)


class MedAgentBenchTask(BaseModel):
    """
    A single MedAgentBench task from the dataset.

    Supports both custom AgentX2 format and official Stanford MedAgentBench format.
    The adapter layer handles format detection and conversion.
    """
    task_id: str  # Primary identifier (use 'task_id' for compatibility with official format)
    instruction: str
    eval_MRN: str | None = None  # Patient MRN for FHIR lookup
    sol: list[str] = Field(default_factory=list, alias="solution")  # Expected solution(s) - array for custom, converted from string for official
    context: str | None = None  # Additional clinical context
    format_type: TaskFormatType = TaskFormatType.CUSTOM

    # For backward compatibility
    @property
    def id(self) -> str:
        """Alias for task_id for backward compatibility."""
        return self.task_id

    # For backward compatibility - access solution as string when needed
    @property
    def solution(self) -> str:
        """Return first solution as string for official format compatibility."""
        return self.sol[0] if self.sol else ""

    class Config:
        populate_by_name = True  # Allow both 'sol' and 'solution' field names
        json_encoders = {
            TaskFormatType: lambda v: v.value
        }


class DiabetesScore(BaseModel):
    """Evaluation scores for diabetes treatment plans (6 criteria, 0-10 each)."""
    medication_appropriateness: float = Field(ge=0.0, le=10.0, description="Are medications suitable for this patient's profile?")
    a1c_target: float = Field(ge=0.0, le=10.0, description="Does the plan address A1C goals appropriately?")
    comorbidity_management: float = Field(ge=0.0, le=10.0, description="Are comorbidities (hypertension, kidney, lipids) addressed?")
    lifestyle_recommendations: float = Field(ge=0.0, le=10.0, description="Are diet and exercise guidance included?")
    safety: float = Field(ge=0.0, le=10.0, description="Are there contraindications or dangerous drug interactions?")
    monitoring_plan: float = Field(ge=0.0, le=10.0, description="Is there a clear follow-up and monitoring strategy?")
    feedback: str = Field(default="", description="Clinical assessment feedback")

    # NEW: Structured improvement suggestions
    suggested_improvements: list[str] = Field(
        default_factory=list,
        description="Specific actionable improvements that would increase the score"
    )
    priority_improvements: list[str] = Field(
        default_factory=list,
        description="High-priority improvements that would most significantly increase the score"
    )

    def get_total(self) -> float:
        """Calculate total score across all 6 diabetes criteria."""
        return (
            self.medication_appropriateness +
            self.a1c_target +
            self.comorbidity_management +
            self.lifestyle_recommendations +
            self.safety +
            self.monitoring_plan
        )


class GeneralMedicalScore(BaseModel):
    """Evaluation scores for general medical responses (3 criteria, 0-10 each)."""
    accuracy: float = Field(ge=0.0, le=10.0, description="How close to expected answer?")
    completeness: float = Field(ge=0.0, le=10.0, description="Addresses all aspects?")
    medical_correctness: float = Field(ge=0.0, le=10.0, description="Clinically sound?")
    feedback: str = Field(default="", description="Clinical assessment feedback")

    # NEW: Structured improvement suggestions
    suggested_improvements: list[str] = Field(
        default_factory=list,
        description="Specific actionable improvements that would increase the score"
    )
    priority_improvements: list[str] = Field(
        default_factory=list,
        description="High-priority improvements that would most significantly increase the score"
    )

    def get_total(self) -> float:
        """Calculate total score across all 3 general medical criteria."""
        return self.accuracy + self.completeness + self.medical_correctness


class MedicalEvaluationResult(BaseModel):
    """Complete evaluation result for a MedAgentBench task."""
    task_id: str
    medical_category: str  # Use str instead of MedicalCategory Literal for Pydantic
    agent_name: str
    agent_response: str
    diabetes_score: DiabetesScore | None = None
    general_score: GeneralMedicalScore | None = None
    total_score: float
    feedback: str
    evaluated_with_model: str = "llama-3.3-70b-versatile"


# ============================================================================
# Green Agent Request/Result Models
# ============================================================================

class EvalRequest(BaseModel):
    """
    Request format for green agent assessment.

    The green agent receives this as input when an assessment starts.
    """
    participants: dict[str, str]  # role -> endpoint URL mapping
    config: dict[str, Any]  # task_id, medical_category, and other config


class EvalResult(BaseModel):
    """
    Result format for green agent output.

    The green agent produces this as the final assessment result.
    """
    winner: str  # Which agent performed best (role name)
    detail: dict[str, Any]  # Detailed results per agent


# ============================================================================
# Enhanced Results Output Models (spec_output.md)
# ============================================================================


class EvaluationDetails(BaseModel):
    """Detailed evaluation information for a single task."""
    medical_category: str
    agent_name: str
    criteria: dict[str, float]  # Individual criterion scores
    rubric_type: str  # "diabetes" or "general"
    feedback: str
    suggested_improvements: list[str] = Field(default_factory=list)
    evaluated_with_model: str = "llama-3.3-70b-versatile"
    reference_solution_provided: bool = True
    pass_threshold: float  # Score required to pass (70% of max)


class EvaluationObject(BaseModel):
    """Single evaluation result with timing information."""
    task_id: str
    model: str
    score: float
    timestamp: str  # ISO-8601 timestamp
    passed: bool
    time_seconds: float
    details: EvaluationDetails


class CategoryBreakdown(BaseModel):
    """Aggregate statistics by medical specialty."""
    count: int
    avg_score: float
    max_score: float
    min_score: float
    pass_count: int
    pass_rate: float
    max_possible_score: int


class ResultsMetadata(BaseModel):
    """Metadata for the evaluation run."""
    evaluation_model: str = "llama-3.3-70b-versatile"
    benchmark_version: str = "v2"
    rubric_versions: dict[str, str] = Field(default_factory=lambda: {"diabetes": "1.0", "general": "1.0"})
    participant_endpoint: str | None = None


class MedBenchResultsOutput(BaseModel):
    """
    Enhanced output format for MedBench judge results.

    Combines summary and evaluations into a single JSON file
    with timing metrics and pass/fail tracking.
    """
    model: str  # Name of the evaluated agent model
    total_evaluations: int  # Total number of tasks evaluated
    avg_score: float  # Average score across all tasks
    max_score: float  # Highest score achieved
    min_score: float  # Lowest score achieved
    total_time_seconds: float  # Total evaluation time in seconds
    avg_time_per_task: float  # Average time per task in seconds
    by_category: dict[str, CategoryBreakdown]  # Breakdown by medical specialty
    evaluations: list[EvaluationObject]  # Array of individual evaluation results
    metadata: ResultsMetadata  # Run metadata
    participants: dict[str, str] = Field(default_factory=dict, description="Agent role to agentbeats_id mapping (e.g., {'medical_judge': 'xxx', 'medical_agent': 'yyy'})")
