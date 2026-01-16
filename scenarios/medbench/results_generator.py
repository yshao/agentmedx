"""
MedBench Results Generator

Generates enhanced results.json output format with:
- Aggregate statistics and timing metrics
- Pass/fail tracking by category
- Per-task breakdown with timing
- Leaderboard-friendly single-file structure

Compatible with spec_output.md specification.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .medbench_models import (
    MedBenchResultsOutput,
    EvaluationObject,
    EvaluationDetails,
    CategoryBreakdown,
    ResultsMetadata,
    DiabetesScore,
    GeneralMedicalScore,
    MedAgentBenchTask,
)

logger = logging.getLogger("results_generator")


# Pass thresholds (70% of max score)
DIABETES_PASS_THRESHOLD = 42.0  # 70% of 60
GENERAL_PASS_THRESHOLD = 21.0  # 70% of 30
DIABETES_MAX_SCORE = 60
GENERAL_MAX_SCORE = 30


class ResultsTracker:
    """
    Tracks evaluation results and generates enhanced results.json output.

    Maintains state across evaluations to compute aggregate statistics.
    """

    def __init__(self, model_name: str = "agentx-medical", evaluation_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the results tracker.

        Args:
            model_name: Name of the agent being evaluated
            evaluation_model: LLM model used for evaluation
        """
        self._model_name = model_name
        self._evaluation_model = evaluation_model
        self._evaluations: list[dict] = []
        self._start_time: float | None = None
        self._participant_endpoint: str | None = None
        self._participants: dict[str, str] = {}  # role -> agentbeats_id mapping

    def start_evaluation(self, participant_endpoint: str | None = None, participants: dict[str, str] | None = None):
        """Start tracking an evaluation session.

        Args:
            participant_endpoint: Optional endpoint URL of the participant being evaluated
            participants: Optional dict mapping role names to agentbeats_id (e.g., {"medical_judge": "xxx", "medical_agent": "yyy"})
        """
        self._start_time = time.time()
        self._participant_endpoint = participant_endpoint
        self._participants = participants or {}
        self._evaluations = []

    def add_evaluation(
        self,
        task: MedAgentBenchTask,
        score: float,
        agent_response: str,
        diabetes_score: DiabetesScore | None = None,
        general_score: GeneralMedicalScore | None = None,
        feedback: str = "",
        time_seconds: float = 0.0,
        agent_name: str = "unknown",
    ) -> dict:
        """
        Add an evaluation result to the tracker.

        Args:
            task: The medical task that was evaluated
            score: Total score achieved
            agent_response: The agent's response text
            diabetes_score: Diabetes-specific scores (if applicable)
            general_score: General medical scores (if applicable)
            feedback: Qualitative feedback
            time_seconds: Time taken for this evaluation
            agent_name: Name/identifier of the agent

        Returns:
            The evaluation dict that was added
        """
        # Determine rubric type and pass threshold
        is_diabetes = diabetes_score is not None
        rubric_type = "diabetes" if is_diabetes else "general"
        max_score = DIABETES_MAX_SCORE if is_diabetes else GENERAL_MAX_SCORE
        pass_threshold = DIABETES_PASS_THRESHOLD if is_diabetes else GENERAL_PASS_THRESHOLD
        passed = score >= pass_threshold

        # Extract criteria scores
        criteria: dict[str, float] = {}
        suggested_improvements: list[str] = []

        if is_diabetes and diabetes_score:
            criteria = {
                "medication_appropriateness": diabetes_score.medication_appropriateness,
                "a1c_target": diabetes_score.a1c_target,
                "comorbidity_management": diabetes_score.comorbidity_management,
                "lifestyle_recommendations": diabetes_score.lifestyle_recommendations,
                "safety": diabetes_score.safety,
                "monitoring_plan": diabetes_score.monitoring_plan,
            }
            suggested_improvements = diabetes_score.suggested_improvements
        elif general_score:
            criteria = {
                "accuracy": general_score.accuracy,
                "completeness": general_score.completeness,
                "medical_correctness": general_score.medical_correctness,
            }
            suggested_improvements = general_score.suggested_improvements

        # Create evaluation object
        evaluation = {
            "task_id": task.task_id,
            "model": self._model_name,
            "score": score,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "passed": passed,
            "time_seconds": time_seconds,
            "medical_category": task.task_id.split("_")[0] if "_" in task.task_id else "general_medical",
            "details": {
                "medical_category": task.task_id.split("_")[0] if "_" in task.task_id else "general_medical",
                "agent_name": agent_name,
                "criteria": criteria,
                "rubric_type": rubric_type,
                "feedback": feedback,
                "suggested_improvements": suggested_improvements,
                "evaluated_with_model": self._evaluation_model,
                "reference_solution_provided": bool(task.sol),
                "pass_threshold": pass_threshold,
            },
        }

        self._evaluations.append(evaluation)
        return evaluation

    def _compute_category_breakdown(self) -> dict[str, dict]:
        """Compute aggregate statistics by medical category."""
        categories: dict[str, list[dict]] = {}

        # Group evaluations by category
        for eval_result in self._evaluations:
            category = eval_result["medical_category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(eval_result)

        # Compute statistics for each category
        breakdown: dict[str, dict] = {}
        for category, evals in categories.items():
            scores = [e["score"] for e in evals]
            passed = [e for e in evals if e["passed"]]

            # Determine max score based on rubric type
            first_eval = evals[0]
            is_diabetes = first_eval["details"]["rubric_type"] == "diabetes"
            max_possible = DIABETES_MAX_SCORE if is_diabetes else GENERAL_MAX_SCORE

            breakdown[category] = {
                "count": len(evals),
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "pass_count": len(passed),
                "pass_rate": len(passed) / len(evals) if evals else 0.0,
                "max_possible_score": max_possible,
            }

        return breakdown

    def generate_results(self) -> MedBenchResultsOutput:
        """
        Generate the enhanced results.json output.

        Returns:
            MedBenchResultsOutput object ready for JSON serialization
        """
        if not self._start_time:
            raise RuntimeError("No evaluation session started. Call start_evaluation() first.")

        total_time = time.time() - self._start_time

        if not self._evaluations:
            # Return empty results if no evaluations
            return MedBenchResultsOutput(
                model=self._model_name,
                total_evaluations=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                total_time_seconds=total_time,
                avg_time_per_task=0.0,
                by_category={},
                evaluations=[],
                metadata=ResultsMetadata(
                    evaluation_model=self._evaluation_model,
                    participant_endpoint=self._participant_endpoint,
                ),
                participants=self._participants,
            )

        # Compute aggregate statistics
        scores = [e["score"] for e in self._evaluations]
        total_score = sum(scores)
        avg_score = total_score / len(scores)

        # Build evaluation objects
        evaluation_objects = []
        for eval_result in self._evaluations:
            evaluation_objects.append(
                EvaluationObject(
                    task_id=eval_result["task_id"],
                    model=eval_result["model"],
                    score=eval_result["score"],
                    timestamp=eval_result["timestamp"],
                    passed=eval_result["passed"],
                    time_seconds=eval_result["time_seconds"],
                    details=EvaluationDetails(**eval_result["details"]),
                )
            )

        # Compute category breakdown
        breakdown_dict = self._compute_category_breakdown()
        by_category = {
            category: CategoryBreakdown(**stats)
            for category, stats in breakdown_dict.items()
        }

        return MedBenchResultsOutput(
            model=self._model_name,
            total_evaluations=len(self._evaluations),
            avg_score=avg_score,
            max_score=max(scores),
            min_score=min(scores),
            total_time_seconds=total_time,
            avg_time_per_task=total_time / len(self._evaluations),
            by_category=by_category,
            evaluations=evaluation_objects,
            metadata=ResultsMetadata(
                evaluation_model=self._evaluation_model,
                participant_endpoint=self._participant_endpoint,
            ),
            participants=self._participants,
        )

    def save_results(self, output_path: str | Path = "results.json"):
        """
        Generate and save results.json to disk.

        Args:
            output_path: Path to save the results file
        """
        results = self.generate_results()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results.model_dump(mode="json"), f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return results


def create_results_from_legacy(
    legacy_results: list[dict],
    model_name: str = "agentx-medical",
    evaluation_model: str = "llama-3.3-70b-versatile",
    total_time_seconds: float = 0.0,
) -> MedBenchResultsOutput:
    """
    Create enhanced results from legacy format results.

    Useful for migrating existing results to the new format.

    Args:
        legacy_results: List of legacy evaluation result dicts
        model_name: Name of the agent
        evaluation_model: LLM model used for evaluation
        total_time_seconds: Total evaluation time (if tracking separately)

    Returns:
        MedBenchResultsOutput object
    """
    tracker = ResultsTracker(model_name=model_name, evaluation_model=evaluation_model)
    tracker.start_evaluation()

    # Simulate timing based on total time
    time_per_task = total_time_seconds / len(legacy_results) if legacy_results else 0.0

    for legacy in legacy_results:
        tracker.add_evaluation(
            task=MedAgentBenchTask(
                task_id=legacy.get("task_id", ""),
                instruction=legacy.get("instruction", ""),
                sol=legacy.get("sol", []),
            ),
            score=legacy.get("score", 0.0),
            agent_response=legacy.get("agent_response", ""),
            feedback=legacy.get("feedback", ""),
            time_seconds=time_per_task,
            agent_name=legacy.get("agent_name", model_name),
        )

    return tracker.generate_results()
