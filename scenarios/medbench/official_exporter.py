"""
MedAgentBench Official Exporter

This module exports evaluation results in the official Stanford MedAgentBench format.

Official format: outputs/[model]/[task]/overall.json

Based on the official MedAgentBench repository structure.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    # Try relative imports first (when imported as a package)
    from .medbench_models import (
        MedicalEvaluationResult,
        DiabetesScore,
        GeneralMedicalScore,
    )
    from .task_adapter import to_official_format
except ImportError:
    # Fall back to absolute imports (when run as a module)
    from medbench.medbench_models import (
        MedicalEvaluationResult,
        DiabetesScore,
        GeneralMedicalScore,
    )
    from medbench.task_adapter import to_official_format


logger = logging.getLogger("official_exporter")


# ============================================================================
# Official Format Models
# ============================================================================

class OfficialOverallResult:
    """
    Official MedAgentBench overall.json format.

    Structure based on official repo outputs/[model]/[task]/overall.json
    """

    def __init__(
        self,
        task_id: str,
        model: str,
        score: float,
        passed: bool | None = None,
        details: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ):
        self.task_id = task_id
        self.model = model
        self.score = score
        self.passed = passed
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "task_id": self.task_id,
            "model": self.model,
            "score": self.score,
            "timestamp": self.timestamp,
        }

        if self.passed is not None:
            result["passed"] = self.passed

        if self.details:
            result["details"] = self.details

        return result


# ============================================================================
# Conversion Functions
# ============================================================================

def convert_diabetes_to_official(diabetes_score: DiabetesScore) -> dict[str, Any]:
    """
    Convert DiabetesScore to official format details.

    Args:
        diabetes_score: Diabetes score from evaluation

    Returns:
        Dictionary with official format criteria
    """
    return {
        "medication_appropriateness": diabetes_score.medication_appropriateness,
        "a1c_target": diabetes_score.a1c_target,
        "comorbidity_management": diabetes_score.comorbidity_management,
        "lifestyle_recommendations": diabetes_score.lifestyle_recommendations,
        "safety": diabetes_score.safety,
        "monitoring_plan": diabetes_score.monitoring_plan,
        "feedback": diabetes_score.feedback,
    }


def convert_general_to_official(general_score: GeneralMedicalScore) -> dict[str, Any]:
    """
    Convert GeneralMedicalScore to official format details.

    Args:
        general_score: General medical score from evaluation

    Returns:
        Dictionary with official format criteria
    """
    return {
        "accuracy": general_score.accuracy,
        "completeness": general_score.completeness,
        "medical_correctness": general_score.medical_correctness,
        "feedback": general_score.feedback,
    }


def convert_evaluation_to_official(
    evaluation: MedicalEvaluationResult,
    model: str,
    reference_solution: str | None = None,
) -> OfficialOverallResult:
    """
    Convert MedicalEvaluationResult to official format.

    Args:
        evaluation: Medical evaluation result
        model: Model name used for evaluation
        reference_solution: Optional reference solution for pass/fail comparison

    Returns:
        OfficialOverallResult for export
    """
    details: dict[str, Any] = {
        "medical_category": evaluation.medical_category,
        "agent_name": evaluation.agent_name,
    }

    # Add score details based on category
    if evaluation.diabetes_score:
        details["criteria"] = convert_diabetes_to_official(evaluation.diabetes_score)
        details["rubric_type"] = "diabetes"
    elif evaluation.general_score:
        details["criteria"] = convert_general_to_official(evaluation.general_score)
        details["rubric_type"] = "general_medical"

    details["feedback"] = evaluation.feedback
    details["evaluated_with_model"] = evaluation.evaluated_with_model

    # Determine pass/fail if reference solution provided
    passed = None
    if reference_solution:
        # Simple pass/fail: check if score meets threshold
        # This is a basic implementation; official repo may have different logic
        threshold = 0.7 * (60.0 if evaluation.diabetes_score else 30.0)  # 70% of max score
        passed = evaluation.total_score >= threshold
        details["reference_solution_provided"] = True
        details["pass_threshold"] = threshold
    else:
        details["reference_solution_provided"] = False

    return OfficialOverallResult(
        task_id=evaluation.task_id,
        model=model,
        score=evaluation.total_score,
        passed=passed,
        details=details,
    )


# ============================================================================
# Export Functions
# ============================================================================

def save_overall_json(
    output_dir: str | Path,
    model: str,
    task_id: str,
    overall_result: OfficialOverallResult,
) -> Path:
    """
    Save evaluation result in official format: outputs/[model]/[task]/overall.json

    Args:
        output_dir: Base output directory (default: project2/outputs/)
        model: Model name (creates subdirectory)
        task_id: Task ID (creates subdirectory)
        overall_result: OfficialOverallResult to save

    Returns:
        Path to the saved overall.json file
    """
    output_path = Path(output_dir) / model / task_id
    output_path.mkdir(parents=True, exist_ok=True)

    overall_file = output_path / "overall.json"

    with open(overall_file, 'w') as f:
        json.dump(overall_result.to_dict(), f, indent=2)

    logger.info(f"Saved official format to: {overall_file}")
    return overall_file


def export_evaluation(
    evaluation: MedicalEvaluationResult,
    model: str,
    output_dir: str | Path = "outputs",
    reference_solution: str | None = None,
) -> Path:
    """
    Export a MedicalEvaluationResult to official format.

    Convenience function that combines conversion and saving.

    Args:
        evaluation: Medical evaluation result
        model: Model name for output directory
        output_dir: Base output directory
        reference_solution: Optional reference solution

    Returns:
        Path to the saved overall.json file
    """
    official_result = convert_evaluation_to_official(
        evaluation,
        model,
        reference_solution,
    )
    return save_overall_json(output_dir, model, evaluation.task_id, official_result)


def export_batch(
    evaluations: list[MedicalEvaluationResult],
    model: str,
    output_dir: str | Path = "outputs",
    reference_solutions: dict[str, str] | None = None,
) -> list[Path]:
    """
    Export multiple evaluations to official format.

    Args:
        evaluations: List of MedicalEvaluationResult objects
        model: Model name for output directories
        output_dir: Base output directory
        reference_solutions: Optional mapping of task_id to reference solution

    Returns:
        List of paths to saved overall.json files
    """
    paths = []
    for evaluation in evaluations:
        ref_sol = reference_solutions.get(evaluation.task_id) if reference_solutions else None
        path = export_evaluation(evaluation, model, output_dir, ref_sol)
        paths.append(path)

    logger.info(f"Exported {len(paths)} evaluation(s) to official format")
    return paths


# ============================================================================
# Summary Export
# ============================================================================

def save_summary(
    evaluations: list[MedicalEvaluationResult],
    model: str,
    output_dir: str | Path = "outputs",
) -> Path:
    """
    Save a summary of all evaluations.

    Creates outputs/[model]/summary.json with aggregated statistics.

    Args:
        evaluations: List of MedicalEvaluationResult objects
        model: Model name
        output_dir: Base output directory

    Returns:
        Path to the saved summary.json file
    """
    output_path = Path(output_dir) / model
    output_path.mkdir(parents=True, exist_ok=True)

    summary_file = output_path / "summary.json"

    # Calculate statistics
    total_score = sum(e.total_score for e in evaluations)
    avg_score = total_score / len(evaluations) if evaluations else 0

    max_score = max((e.total_score for e in evaluations), default=0)
    min_score = min((e.total_score for e in evaluations), default=0)

    # Group by medical category
    by_category: dict[str, list[float]] = {}
    for e in evaluations:
        if e.medical_category not in by_category:
            by_category[e.medical_category] = []
        by_category[e.medical_category].append(e.total_score)

    category_stats = {}
    for cat, scores in by_category.items():
        category_stats[cat] = {
            "count": len(scores),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }

    summary = {
        "model": model,
        "total_evaluations": len(evaluations),
        "avg_score": round(avg_score, 2),
        "max_score": max_score,
        "min_score": min_score,
        "by_category": category_stats,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to: {summary_file}")
    return summary_file
