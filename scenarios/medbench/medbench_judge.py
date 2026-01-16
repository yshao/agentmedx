"""
MedBench Judge - Green Agent for MedAgentBench Evaluation

This module implements the green agent that orchestrates medical case evaluations.
It follows the pattern from tutorial/scenarios/debate/debate_judge.py.

Usage:
    python -m medbench.medbench_judge --host 127.0.0.1 --port 9008 --data-path data/medagentbench/test_data_v2.json
"""

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

# Import from agentbeats framework (tutorial)
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from dotenv import load_dotenv
from pydantic import ValidationError

# Import MedBench-specific modules
try:
    # Try relative imports first (when imported as a package)
    from .medbench_common import (
        VALID_MEDICAL_CATEGORIES,
        is_valid_medical_category,
        medbench_judge_agent_card,
    )
    from .medbench_models import (
        DiabetesScore,
        GeneralMedicalScore,
        MedAgentBenchTask,
        MedicalEvaluationResult,
    )
    from .medical_evaluation import MedicalEvaluationEngine
    from .official_exporter import (
        export_batch,
        export_evaluation,
        save_summary,
    )
    from .reference_solution import (
        get_reference_provider,
        integrate_reference_comparison,
    )
    from .task_adapter import (
        get_task_by_id,
        load_tasks,
        to_official_format,
        validate_task_compatibility,
    )
except ImportError:
    # Fall back to absolute imports (when run as a module)
    from scenarios.medbench.medbench_common import (
        VALID_MEDICAL_CATEGORIES,
        is_valid_medical_category,
        medbench_judge_agent_card,
    )
    from scenarios.medbench.medbench_models import (
        DiabetesScore,
        GeneralMedicalScore,
        MedAgentBenchTask,
        MedicalEvaluationResult,
    )
    from scenarios.medbench.medical_evaluation import MedicalEvaluationEngine
    from scenarios.medbench.official_exporter import (
        export_batch,
        export_evaluation,
        save_summary,
    )
    from scenarios.medbench.reference_solution import (
        get_reference_provider,
        integrate_reference_comparison,
    )
    from scenarios.medbench.task_adapter import (
        get_task_by_id,
        load_tasks,
        to_official_format,
        validate_task_compatibility,
    )


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medbench_judge")


# ============================================================================
# MedBenchJudge - Green Agent
# ============================================================================


class MedBenchJudge(GreenAgent):
    """
    Green agent for MedAgentBench medical case evaluation.

    This green agent:
    1. Loads medical tasks from MedAgentBench dataset
    2. Sends medical cases to participant (purple) agents
    3. Collects responses from all participants
    4. Evaluates responses using LLM-as-Judge with specialty rubrics
    5. Produces artifacts with evaluation results
    """

    def __init__(
        self,
        data_path: str,
        export_official: bool = False,
        output_dir: str = "outputs",
        model_name: str = "agentx-medical",
        webhook_enabled: bool = False,
        webhook_url: str = "",
        webhook_secret: str = "",
        webhook_dry_run: bool = False,
    ):
        """
        Initialize the MedBenchJudge.

        Args:
            data_path: Path to the MedAgentBench test_data_v2.json file
            export_official: If True, export results in official MedAgentBench format
            output_dir: Directory for official format exports
            model_name: Model name for output directory naming
            webhook_enabled: If True, send evaluation results via webhook
            webhook_url: Webhook endpoint URL
            webhook_secret: Webhook secret for authentication
            webhook_dry_run: If True, log webhook payload without sending
        """
        self._required_roles = ["medical_agent"]  # Can have multiple participants
        self._required_config_keys = ["task_id", "medical_category"]
        self._tool_provider = ToolProvider()
        self._tasks = self._load_tasks(data_path)
        self._eval_engine = MedicalEvaluationEngine()

        # Official format export settings
        self._export_official = export_official
        self._output_dir = output_dir
        self._model_name = model_name

        # Webhook configuration
        self._webhook_enabled = webhook_enabled
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret
        self._webhook_dry_run = webhook_dry_run

        # Reference solution provider (lazy load)
        self._reference_provider = None

        logger.info(f"Loaded {len(self._tasks)} tasks from {data_path}")
        if export_official:
            logger.info(f"Official format export enabled: {output_dir}/{model_name}/")
        if webhook_enabled:
            logger.info(f"Webhook notifications enabled: {webhook_url}")

    def _load_tasks(self, data_path: str) -> list[MedAgentBenchTask]:
        """
        Load medical tasks from MedAgentBench JSON file.

        Uses task_adapter to automatically detect and handle both custom and official formats.

        Args:
            data_path: Path to test_data_v2.json or official tasks file

        Returns:
            List of MedAgentBenchTask objects
        """
        # Use the task adapter which handles both formats
        return load_tasks(data_path)

    def _get_task_by_id(self, task_id: str) -> MedAgentBenchTask:
        """Get a task by its ID (supports both formats)."""
        task = get_task_by_id(self._tasks, task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        return task

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate the evaluation request.

        Args:
            request: The EvalRequest to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required roles
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing required roles: {missing_roles}"

        # Check required config keys
        missing_config = set(self._required_config_keys) - set(request.config.keys())
        if missing_config:
            return False, f"Missing config keys: {missing_config}"

        # Validate task_id exists
        task_id = request.config.get("task_id")
        if not task_id:
            return False, "task_id cannot be empty"

        try:
            self._get_task_by_id(task_id)
        except ValueError:
            return False, f"Task not found in dataset: {task_id}"

        # Validate medical_category
        category = request.config.get("medical_category")
        if not category:
            return False, "medical_category cannot be empty"

        if not is_valid_medical_category(category):
            return (
                False,
                f"Invalid medical_category: {category}. Must be one of {VALID_MEDICAL_CATEGORIES}",
            )

        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """
        Main evaluation orchestration method called by GreenExecutor.

        This method orchestrates the complete medical case evaluation process:
        1. Validates the request and retrieves the medical task
        2. Sends the case to all participant agents
        3. Collects and evaluates responses using LLM-as-Judge
        4. Produces evaluation artifacts with scores and feedback

        Args:
            req (EvalRequest): Evaluation request containing:
                - participants (dict[str, HttpUrl]): Mapping of role names to agent endpoint URLs
                  Example: {"medical_agent": "http://medbench-medical:9010"}
                - config (dict[str, Any]): Configuration dictionary:
                  - task_id (str): Medical task ID from dataset (e.g., "diabetes_001")
                  - medical_category (str): Category for rubric selection ("diabetes", "cardiology", etc.)
                  - dry_run (bool, optional): If True, uses mock scores instead of LLM evaluation

            updater (TaskUpdater): A2A TaskUpdater for sending status updates and artifacts:
                - update_status(state, message): Update task status (working/completed/failed)
                - add_artifact(parts, name): Add evaluation results as artifacts
                - complete(): Mark task as completed successfully
                - failed(error): Mark task as failed with error details

        Raises:
            ValueError: If task_id is not found in dataset
            ServerError: If evaluation fails or configuration is invalid

        Lifecycle:
            1. Validates request (via validate_request())
            2. Updates status to "working"
            3. Runs medical case evaluation
            4. Evaluates responses
            5. Creates result artifacts
            6. Calls updater.complete() or updater.failed()

        Example:
            ```python
            # Called internally by GreenExecutor when receiving A2A message
            request = EvalRequest(
                participants={"medical_agent": HttpUrl("http://agent:9010")},
                config={"task_id": "diabetes_001", "medical_category": "diabetes"}
            )
            await run_eval(request, task_updater)
            ```
        """
        logger.info(f"Starting MedBench evaluation: {req}")

        try:
            # Get the task from config
            task_id = req.config["task_id"]
            medical_category = req.config["medical_category"]
            dry_run = req.config.get("dry_run", False)

            task = self._get_task_by_id(task_id)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Starting evaluation for task: {task_id} (category: {medical_category})"
                ),
            )

            # Run the medical case with all participants
            responses = await self.run_medical_case(req.participants, task, updater)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Collected {len(responses)} response(s). Starting evaluation..."
                ),
            )

            # Evaluate all responses
            logger.info("Evaluating agent responses...")
            evaluation_results = await self.evaluate_responses(
                task, responses, medical_category, dry_run=dry_run
            )

            # Determine best agent (if multiple participants)
            best_agent = None
            if len(evaluation_results) > 1:
                best_agent = max(
                    evaluation_results.keys(),
                    key=lambda k: evaluation_results[k].total_score,
                )

            # Create result
            result = EvalResult(
                winner=best_agent
                or "medical_agent",  # Default to medical_agent if only one
                detail={
                    "task_id": task_id,
                    "medical_category": medical_category,
                    "task_instruction": task.instruction,
                    "evaluations": {
                        agent: result.model_dump()
                        for agent, result in evaluation_results.items()
                    },
                },
            )

            # Add artifacts with detailed results
            artifacts = []
            for agent, eval_result in evaluation_results.items():
                artifacts.append(Part(root=TextPart(text=f"\n=== {agent} ===\n")))
                artifacts.append(
                    Part(
                        root=TextPart(text=f"Total Score: {eval_result.total_score}\n")
                    )
                )
                artifacts.append(
                    Part(root=TextPart(text=f"Feedback: {eval_result.feedback}\n"))
                )

                # Display improvement suggestions with enhanced formatting
                # This section provides structured feedback to help improve agent responses
                if (
                    hasattr(eval_result, "diabetes_score")
                    and eval_result.diabetes_score
                ):
                    score = eval_result.diabetes_score
                    # Show positive aspects of the response
                    if (
                        hasattr(score, "whats_working_well")
                        and score.whats_working_well
                    ):
                        artifacts.append(
                            Part(root=TextPart(text=f"\n[+] What's Working Well:\n"))
                        )
                        for good_thing in score.whats_working_well:
                            artifacts.append(
                                Part(root=TextPart(text=f"  [OK] {good_thing}\n"))
                            )
                    # Show categorized improvement suggestions
                    if score.suggested_improvements:
                        artifacts.append(Part(root=TextPart(text=f"\n{'=' * 60}\n")))
                        artifacts.append(
                            Part(
                                root=TextPart(text=f"[!] SUGGESTIONS FOR IMPROVEMENT\n")
                            )
                        )
                        artifacts.append(Part(root=TextPart(text=f"{'=' * 60}\n")))
                        for imp in score.suggested_improvements:
                            # Parse category prefix and display with appropriate formatting
                            if imp.startswith("ADD:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [ADD] {imp[4:]}\n"))
                                )
                            elif imp.startswith("SPECIFY:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [SPEC] {imp[8:]}\n"))
                                )
                            elif imp.startswith("ENHANCE:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [ENH] {imp[8:]}\n"))
                                )
                            else:
                                artifacts.append(
                                    Part(root=TextPart(text=f"  -> {imp}\n"))
                                )
                    # Show highest-priority improvement for quick focus
                    if score.priority_improvements:
                        artifacts.append(Part(root=TextPart(text=f"\n{'-' * 40}\n")))
                        artifacts.append(
                            Part(
                                root=TextPart(
                                    text=f"[!] PRIORITY: {score.priority_improvements[0]}\n"
                                )
                            )
                        )
                        artifacts.append(Part(root=TextPart(text=f"{'-' * 40}\n")))

                elif (
                    hasattr(eval_result, "general_score") and eval_result.general_score
                ):
                    score = eval_result.general_score
                    # Show positive aspects of the response
                    if (
                        hasattr(score, "whats_working_well")
                        and score.whats_working_well
                    ):
                        artifacts.append(
                            Part(root=TextPart(text=f"\n[+] What's Working Well:\n"))
                        )
                        for good_thing in score.whats_working_well:
                            artifacts.append(
                                Part(root=TextPart(text=f"  [OK] {good_thing}\n"))
                            )
                    # Show categorized improvement suggestions
                    if score.suggested_improvements:
                        artifacts.append(Part(root=TextPart(text=f"\n{'=' * 60}\n")))
                        artifacts.append(
                            Part(
                                root=TextPart(text=f"[!] SUGGESTIONS FOR IMPROVEMENT\n")
                            )
                        )
                        artifacts.append(Part(root=TextPart(text=f"{'=' * 60}\n")))
                        for imp in score.suggested_improvements:
                            # Parse category prefix and display with appropriate formatting
                            if imp.startswith("ADD:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [ADD] {imp[4:]}\n"))
                                )
                            elif imp.startswith("SPECIFY:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [SPEC] {imp[8:]}\n"))
                                )
                            elif imp.startswith("ENHANCE:"):
                                artifacts.append(
                                    Part(root=TextPart(text=f"  [ENH] {imp[8:]}\n"))
                                )
                            else:
                                artifacts.append(
                                    Part(root=TextPart(text=f"  -> {imp}\n"))
                                )
                    # Show highest-priority improvement for quick focus
                    if score.priority_improvements:
                        artifacts.append(Part(root=TextPart(text=f"\n{'-' * 40}\n")))
                        artifacts.append(
                            Part(
                                root=TextPart(
                                    text=f"[!] PRIORITY: {score.priority_improvements[0]}\n"
                                )
                            )
                        )
                        artifacts.append(Part(root=TextPart(text=f"{'-' * 40}\n")))

            artifacts.append(
                Part(root=TextPart(text=f"\n{result.model_dump_json(indent=2)}"))
            )

            await updater.add_artifact(parts=artifacts, name="MedBenchEvaluationResult")

            # Export to official format if enabled
            if self._export_official:
                logger.info("Exporting to official MedAgentBench format...")
                for agent_role, eval_result in evaluation_results.items():
                    try:
                        export_path = export_evaluation(
                            eval_result,
                            self._model_name,
                            self._output_dir,
                        )
                        logger.info(f"Exported {agent_role} result to: {export_path}")
                    except Exception as e:
                        logger.warning(f"Failed to export {agent_role} result: {e}")

                # Save summary
                try:
                    summary_path = save_summary(
                        list(evaluation_results.values()),
                        self._model_name,
                        self._output_dir,
                    )
                    logger.info(f"Saved summary to: {summary_path}")
                except Exception as e:
                    logger.warning(f"Failed to save summary: {e}")

            logger.info(f"Evaluation complete. Best agent: {best_agent}")

            # Send webhook notification if enabled
            if self._webhook_enabled:
                logger.info("Sending evaluation results via webhook...")
                try:
                    from medbench.webhook_notifier import WebhookConfig, WebhookNotifier

                    webhook_config = WebhookConfig(
                        url=self._webhook_url,
                        secret=self._webhook_secret,
                        enabled=True,
                        dry_run_send=self._webhook_dry_run,
                    )

                    notifier = WebhookNotifier(webhook_config)
                    webhook_sent = await notifier.send_evaluation_complete(
                        evaluation_results
                    )

                    if webhook_sent:
                        logger.info("✅ Webhook sent successfully")
                    else:
                        logger.warning("⚠️  Webhook failed to send after retries")

                except Exception as e:
                    logger.error(f"❌ Webhook error: {e}")
                    # Don't fail evaluation if webhook fails

        finally:
            self._tool_provider.reset()

    async def run_medical_case(
        self,
        participants: dict[str, str],
        task: MedAgentBenchTask,
        updater: TaskUpdater,
    ) -> dict[str, str]:
        """
        Send the medical case to all participant agents and collect responses.

        This is similar to orchestrate_debate() but single-turn (no back-and-forth).
        Optimized to reuse conversation contexts for the same endpoint across evaluations.

        Args:
            participants: Mapping of role names to endpoint URLs
            task: The medical task to send
            updater: TaskUpdater for status updates

        Returns:
            Dictionary mapping participant role to their response
        """
        responses: dict[str, str] = {}

        # Build the prompt for the medical agent
        prompt = task.instruction
        if task.context:
            prompt = f"{task.instruction}\n\nAdditional Context:\n{task.context}"

        # Send to each participant
        for role, endpoint in participants.items():
            logger.info(f"Sending case to {role} at {endpoint}")
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Querying {role}...")
            )

            try:
                # Reuse conversation context for this endpoint if available
                context_id = self._tool_provider._context_ids.get(endpoint)

                response = await self._tool_provider.talk_to_agent(
                    prompt,
                    str(endpoint),
                    context_id=context_id,  # Reuse existing context if available
                )

                responses[role] = response
                logger.info(f"{role} responded: {response[:100]}...")

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"{role} submitted response"),
                )

            except Exception as e:
                logger.error(f"Error communicating with {role}: {e}")
                responses[role] = f"ERROR: {str(e)}"

        return responses

    async def evaluate_responses(
        self,
        task: MedAgentBenchTask,
        responses: dict[str, str],
        medical_category: str,
        dry_run: bool = False,
    ) -> dict[str, MedicalEvaluationResult]:
        """
        Evaluate all agent responses using LLM-as-Judge.

        Args:
            task: The medical task being evaluated
            responses: Mapping of agent role to their response
            medical_category: Category for evaluation rubric selection
            dry_run: If True, use mock scores instead of calling LLM API

        Returns:
            Mapping of agent role to their evaluation result
        """
        # Mock scores for dry-run mode
        if dry_run:
            logger.info("DRY-RUN MODE: Using mock scores")
            results: dict[str, MedicalEvaluationResult] = {}
            for agent_role, agent_response in responses.items():
                if agent_response.startswith("ERROR:"):
                    results[agent_role] = MedicalEvaluationResult(
                        task_id=task.id,
                        medical_category=medical_category,
                        agent_name=agent_role,
                        agent_response=agent_response,
                        diabetes_score=None,
                        general_score=None,
                        total_score=0.0,
                        feedback=f"Communication error: {agent_response}",
                    )
                elif medical_category == "diabetes":
                    # Mock excellent scores for dry-run
                    results[agent_role] = MedicalEvaluationResult(
                        task_id=task.id,
                        medical_category=medical_category,
                        agent_name=agent_role,
                        agent_response=agent_response,
                        diabetes_score=DiabetesScore(
                            medication_appropriateness=9.0,
                            a1c_target=9.0,
                            comorbidity_management=9.0,
                            lifestyle_recommendations=9.0,
                            safety=9.0,
                            monitoring_plan=9.0,
                            suggested_improvements=[
                                "[DRY-RUN] This is a mock improvement suggestion for testing",
                                "[DRY-RUN] Another mock suggestion for demonstration",
                            ],
                            priority_improvements=[
                                "[DRY-RUN] High priority item that would increase score"
                            ],
                        ),
                        general_score=None,
                        total_score=54.0,
                        feedback="[DRY-RUN MODE] Mock score - not a real evaluation. Use actual evaluation for real results.",
                    )
                else:
                    # Mock excellent scores for general medical
                    results[agent_role] = MedicalEvaluationResult(
                        task_id=task.id,
                        medical_category=medical_category,
                        agent_name=agent_role,
                        agent_response=agent_response,
                        diabetes_score=None,
                        general_score=GeneralMedicalScore(
                            accuracy=9.0,
                            completeness=9.0,
                            medical_correctness=9.0,
                            feedback="[DRY-RUN MODE] Mock score - not a real evaluation.",
                            whats_working_well=[
                                "[DRY-RUN] Accurate diagnosis and comprehensive management",
                                "[DRY-RUN] Good coverage of differential diagnosis",
                                "[DRY-RUN] Well-structured response",
                            ],
                            suggested_improvements=[
                                "[DRY-RUN] ENHANCE: Include more specific timing for interventions - Current: vague timeline, Needed: 'administer norepinephrine 5-10 mcg/min IV until MAP ≥65' instead of vague 'as needed'",
                                "[DRY-RUN] ADD: Include specific vasopressor dosing and titration - Current: 'maintain BP', Needed: 'Start norepinephrine 5-10 mcg/min IV and titrate by 2.5 mcg every 5 minutes'",
                            ],
                            priority_improvements=[
                                "[DRY-RUN] PRIORITY: Include specific timing and dosing for vasopressor therapy to increase medical_correctness score from 8.0 to 9.0"
                            ],
                        ),
                        total_score=27.0,
                        feedback="[DRY-RUN MODE] Mock score - not a real evaluation. Use actual evaluation for real results.",
                    )
            return results

        results = {}

        for agent_role, agent_response in responses.items():
            logger.info(f"Evaluating response from {agent_role}...")

            # Skip error responses
            if agent_response.startswith("ERROR:"):
                results[agent_role] = MedicalEvaluationResult(
                    task_id=task.id,
                    medical_category=medical_category,
                    agent_name=agent_role,
                    agent_response=agent_response,
                    diabetes_score=None,
                    general_score=None,
                    total_score=0.0,
                    feedback=f"Communication error: {agent_response}",
                )
                continue

            try:
                # Choose evaluation method based on medical category
                if medical_category == "diabetes":
                    diabetes_score = await self._eval_engine.evaluate_diabetes(
                        task, agent_response
                    )

                    results[agent_role] = MedicalEvaluationResult(
                        task_id=task.id,
                        medical_category=medical_category,
                        agent_name=agent_role,
                        agent_response=agent_response,
                        diabetes_score=diabetes_score,
                        general_score=None,
                        total_score=diabetes_score.get_total(),
                        feedback=diabetes_score.feedback,
                    )

                else:
                    # Use general medical evaluation for other categories
                    general_score = await self._eval_engine.evaluate_general_medical(
                        task, agent_response
                    )

                    results[agent_role] = MedicalEvaluationResult(
                        task_id=task.id,
                        medical_category=medical_category,
                        agent_name=agent_role,
                        agent_response=agent_response,
                        diabetes_score=None,
                        general_score=general_score,
                        total_score=general_score.get_total(),
                        feedback=general_score.feedback,
                    )

                logger.info(f"{agent_role} score: {results[agent_role].total_score}")

            except Exception as e:
                logger.error(f"Error evaluating {agent_role}: {e}")
                results[agent_role] = MedicalEvaluationResult(
                    task_id=task.id,
                    medical_category=medical_category,
                    agent_name=agent_role,
                    agent_response=agent_response,
                    diabetes_score=None,
                    general_score=None,
                    total_score=0.0,
                    feedback=f"Evaluation error: {str(e)}",
                )

        return results


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point for the MedBenchJudge server."""
    parser = argparse.ArgumentParser(
        description="Run the MedBenchJudge (green agent) for MedAgentBench evaluation."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9008, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to test_data_v2.json file"
    )
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url",
    )

    # Official format export options
    parser.add_argument(
        "--export-official",
        action="store_true",
        help="Export results in official MedAgentBench format (overall.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for official format (default: outputs/)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="agentx-medical",
        help="Model name for output directory (default: agentx-medical)",
    )

    # Webhook configuration options
    parser.add_argument(
        "--webhook-enabled",
        action="store_true",
        help="Enable webhook notifications to leaderboard",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        help="Webhook endpoint URL (e.g., https://agentbeats.dev/api/webhooks/evaluations)",
    )
    parser.add_argument(
        "--webhook-secret", type=str, help="Webhook secret for authentication"
    )
    parser.add_argument(
        "--webhook-dry-run-send",
        action="store_true",
        help="Log webhook payload without actually sending (for testing)",
    )

    args = parser.parse_args()

    # Setup cloudflare tunnel if requested
    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel

        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(
            args.card_url or f"http://{args.host}:{args.port}/"
        )

    async with agent_url_cm as agent_url:
        # Create the green agent with official format export and webhook settings
        agent = MedBenchJudge(
            args.data_path,
            export_official=args.export_official,
            output_dir=args.output_dir,
            model_name=args.model_name,
            webhook_enabled=args.webhook_enabled,
            webhook_url=args.webhook_url or "",
            webhook_secret=args.webhook_secret or "",
            webhook_dry_run=args.webhook_dry_run_send,
        )
        executor = GreenExecutor(agent)
        agent_card = medbench_judge_agent_card("MedBenchJudge", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        # Add health check and root info endpoints using middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.middleware.cors import CORSMiddleware
        from starlette.responses import JSONResponse

        class InfoMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # Handle GET / and GET /health before passing to A2A handler
                # Let /.well-known/* paths pass through to A2A handler for agent card discovery
                if request.method == "GET":
                    if request.url.path.startswith("/.well-known/"):
                        return await call_next(request)
                    if request.url.path == "/health":
                        return JSONResponse(
                            {
                                "status": "healthy",
                                "service": "medbench-judge",
                                "tasks_loaded": len(agent._tasks),
                                "ready": True,
                            }
                        )
                    elif request.url.path == "/":
                        return JSONResponse(
                            {
                                "service": "MedBenchJudge - Green Agent",
                                "description": "Evaluates medical agent responses using LLM-as-Judge",
                                "version": "1.0.0",
                                "protocol": "A2A",
                                "endpoints": {
                                    "POST /": "A2A protocol endpoint for agent communication",
                                    "GET /health": "Health check endpoint",
                                },
                                "tasks_loaded": len(agent._tasks),
                                "status": "running",
                            }
                        )
                return await call_next(request)

        # Build the app and add middleware
        app = server.build()

        # CORS middleware for SSE streaming (must be before InfoMiddleware)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        app.add_middleware(InfoMiddleware)

        uvicorn_config = uvicorn.Config(app, host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)

        logger.info(f"Starting MedBenchJudge on {args.host}:{args.port}")
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
