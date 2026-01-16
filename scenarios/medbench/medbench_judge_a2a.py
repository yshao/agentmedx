"""
MedBench Judge - A2A-Compatible Green Agent for MedAgentBench Evaluation

This module provides an A2A-compatible green agent for the MedBench judge that works
with the agentbeats-client using the agentbeats framework.

Usage:
    python -m scenarios.medbench.medbench_judge_a2a --host 0.0.0.0 --port 9008
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Any

import uvicorn
from dotenv import load_dotenv

# A2A server imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    TaskState,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

# CORS middleware for SSE support
from starlette.middleware.cors import CORSMiddleware

# Import agentbeats framework
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult

# Import MedBench-specific modules
try:
    from scenarios.medbench.medical_evaluation import MedicalEvaluationEngine
    from scenarios.medbench.medbench_models import (
        DiabetesScore,
        GeneralMedicalScore,
        MedAgentBenchTask,
        MedicalEvaluationResult,
    )
    from scenarios.medbench.task_adapter import get_task_by_id, load_tasks
    from scenarios.medbench.medbench_common import VALID_MEDICAL_CATEGORIES, medbench_judge_agent_card
    from scenarios.medbench.medbench_common import is_valid_medical_category
    from scenarios.medbench.results_generator import ResultsTracker, DIABETES_MAX_SCORE, GENERAL_MAX_SCORE
except ImportError:
    from .medical_evaluation import MedicalEvaluationEngine
    from .medbench_models import (
        DiabetesScore,
        GeneralMedicalScore,
        MedAgentBenchTask,
        MedicalEvaluationResult,
    )
    from .task_adapter import get_task_by_id, load_tasks
    from .medbench_common import VALID_MEDICAL_CATEGORIES, medbench_judge_agent_card
    from .medbench_common import is_valid_medical_category
    from .results_generator import ResultsTracker, DIABETES_MAX_SCORE, GENERAL_MAX_SCORE

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medbench_judge_a2a")


# ============================================================================
# MedBenchJudgeA2A - Green Agent
# ============================================================================


class MedBenchJudgeA2A(GreenAgent):
    """
    A2A-compatible green agent for MedAgentBench evaluation.

    This agent implements the GreenAgent interface for agentbeats framework
    and handles medical case evaluation requests.
    """

    def __init__(self, data_path: str = "/app/data/medagentbench/test_data_v2.json", model_name: str = "agentx-medical"):
        """
        Initialize the MedBenchJudgeA2A.

        Args:
            data_path: Path to the MedAgentBench test_data_v2.json file
            model_name: Name of the model being evaluated
        """
        self._required_roles = ["medical_agent"]
        self._required_config_keys = ["task_id", "medical_category"]
        self._tasks = load_tasks(data_path)
        self._eval_engine = MedicalEvaluationEngine()
        self._model_name = model_name
        self._results_tracker = ResultsTracker(model_name=model_name)
        logger.info(f"Loaded {len(self._tasks)} tasks from {data_path}")

    def _get_task_by_id(self, task_id: str) -> MedAgentBenchTask:
        """Get a task by its ID."""
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

        Args:
            req: The evaluation request
            updater: TaskUpdater for sending status updates

        Note:
            This method should NOT call updater.update_status(TaskState.completed).
            The GreenExecutor will call updater.complete() after this method returns.
            Use add_artifact() to return results.
        """
        import time

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation for task: {req.config.get('task_id')}")
        )

        # Start timing
        eval_start = time.time()

        # Get the medical task
        task_id = req.config.get("task_id")
        task = self._get_task_by_id(task_id)

        # Get participant URL (convert HttpUrl to string)
        participant_url_obj = req.participants.get("medical_agent")
        if not participant_url_obj:
            raise ServerError(error=InvalidParamsError(message="medical_agent participant URL not provided"))
        participant_url = str(participant_url_obj)

        # Get medical category
        medical_category = req.config.get("medical_category", "general_medical")

        # Extract agentbeats_ids from config for results tracking
        agentbeats_ids = req.config.get("agentbeats_ids", {})

        # Start results tracking with participants (agentbeats_ids)
        self._results_tracker.start_evaluation(
            participant_endpoint=participant_url,
            participants=agentbeats_ids
        )

        # For dry run mode, use a mock response
        dry_run = os.getenv("DRY_RUN", "false").lower() == "true"

        if dry_run:
            # Mock evaluation for dry run
            agent_response = f"[DRY RUN] Mock response for task {task_id}"
            diabetes_score = DiabetesScore(
                medication_appropriateness=7.0,
                a1c_target=7.0,
                comorbidity_management=7.0,
                lifestyle_recommendations=7.0,
                safety=7.0,
                monitoring_plan=7.0,
                feedback="[DRY RUN] Mock evaluation - not actually evaluated",
            ) if medical_category == "diabetes" else None
            general_score = GeneralMedicalScore(
                accuracy=7.0,
                completeness=7.0,
                medical_correctness=7.0,
                feedback="[DRY RUN] Mock evaluation - not actually evaluated",
            ) if medical_category != "diabetes" else None
            total_score = diabetes_score.get_total() if diabetes_score else general_score.get_total()
            feedback = "[DRY RUN] Mock evaluation completed"
        else:
            # TODO: In production, send task to participant and get response
            # For now, use mock response
            agent_response = f"[MOCK] Response from agent for task {task_id}"

            # Perform actual evaluation using LLM-as-Judge
            if medical_category == "diabetes":
                diabetes_score = await self._eval_engine.evaluate_diabetes(task, agent_response)
                general_score = None
                feedback = diabetes_score.feedback
            else:
                general_score = await self._eval_engine.evaluate_general_medical(task, agent_response)
                diabetes_score = None
                feedback = general_score.feedback

            total_score = diabetes_score.get_total() if diabetes_score else general_score.get_total()

        # Calculate elapsed time
        time_seconds = time.time() - eval_start

        # Determine max score and pass threshold
        is_diabetes = medical_category == "diabetes"
        max_score = DIABETES_MAX_SCORE if is_diabetes else GENERAL_MAX_SCORE
        pass_threshold = 0.7 * max_score  # 70% threshold
        passed = total_score >= pass_threshold

        # Track result using ResultsTracker
        self._results_tracker.add_evaluation(
            task=task,
            score=total_score,
            agent_response=agent_response,
            diabetes_score=diabetes_score,
            general_score=general_score,
            feedback=feedback,
            time_seconds=time_seconds,
            agent_name=self._model_name,
        )

        # Build result object
        result = {
            "task_id": task_id,
            "model": self._model_name,
            "participant": participant_url,
            "score": total_score,
            "max_score": max_score,
            "passed": passed,
            "pass_threshold": pass_threshold,
            "time_seconds": time_seconds,
            "medical_category": medical_category,
            "status": "completed",
            "message": f"Evaluation completed for task {task_id}",
        }

        # Add criteria scores if available
        if diabetes_score:
            result["criteria"] = {
                "medication_appropriateness": diabetes_score.medication_appropriateness,
                "a1c_target": diabetes_score.a1c_target,
                "comorbidity_management": diabetes_score.comorbidity_management,
                "lifestyle_recommendations": diabetes_score.lifestyle_recommendations,
                "safety": diabetes_score.safety,
                "monitoring_plan": diabetes_score.monitoring_plan,
            }
            result["suggested_improvements"] = diabetes_score.suggested_improvements
        elif general_score:
            result["criteria"] = {
                "accuracy": general_score.accuracy,
                "completeness": general_score.completeness,
                "medical_correctness": general_score.medical_correctness,
            }
            result["suggested_improvements"] = general_score.suggested_improvements

        # Add result as artifact instead of updating status to completed
        # The GreenExecutor will call updater.complete() after this method returns
        from a2a.types import Part, TextPart
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(result, indent=2)))],
            name="evaluation_result"
        )

    def start_results_tracking(self, participant_endpoint: str | None = None):
        """
        Start tracking a new evaluation session.

        Call this before running any evaluations to reset tracking.

        Args:
            participant_endpoint: Optional endpoint URL of the participant being evaluated
        """
        self._results_tracker.start_evaluation(participant_endpoint=participant_endpoint)
        logger.info("Started results tracking")

    def get_results(self):
        """
        Get the current results in enhanced format.

        Returns:
            MedBenchResultsOutput object
        """
        return self._results_tracker.generate_results()

    def save_results(self, output_path: str = "results.json"):
        """
        Save results to disk in enhanced format.

        Args:
            output_path: Path to save the results file

        Returns:
            MedBenchResultsOutput object
        """
        results = self._results_tracker.save_results(output_path)
        logger.info(f"Results saved to {output_path}")
        return results


def create_judge_agent(data_path: str = "/app/data/medagentbench/test_data_v2.json", model_name: str = "agentx-medical") -> MedBenchJudgeA2A:
    """Create and return the judge agent instance."""
    return MedBenchJudgeA2A(data_path=data_path, model_name=model_name)


def main():
    """Main entry point for running the A2A-compatible judge server."""
    parser = argparse.ArgumentParser(description="MedBench Judge A2A Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_PORT", "9008")), help="Port to bind to")
    parser.add_argument("--card-url", type=str, default=os.getenv("CARD_URL", None),
                       help="External URL to advertise in agent card (e.g., http://green-agent:9008/)")
    parser.add_argument("--data-path", default="/app/data/medagentbench/test_data_v2.json", help="Path to test data")
    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode")

    args = parser.parse_args()

    # Set dry run environment variable
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    # Determine card URL: use provided --card-url, or build from host:port
    card_url = args.card_url or f"http://{args.host}:{args.port}/"

    logger.info("==================================")
    logger.info("Starting MedBenchJudge Agent (A2A)")
    logger.info("==================================")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Card URL: {card_url}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("A2A Transport: SSE (Server-Sent Events)")
    logger.info("==================================")

    # Create the green agent
    green_agent = create_judge_agent(data_path=args.data_path)

    # Create the executor
    executor = GreenExecutor(green_agent)

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Create agent card
    agent_card = medbench_judge_agent_card(
        agent_name="MedBenchJudge",
        card_url=card_url
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the Starlette app
    app = a2a_app.build()

    # Add health check endpoint
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.middleware.base import BaseHTTPMiddleware

    class HealthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.method == "GET":
                if request.url.path == "/health":
                    return JSONResponse({
                        "status": "healthy",
                        "service": "medbench-judge-a2a",
                        "ready": True,
                        "tasks_loaded": len(green_agent._tasks),
                        "transport": "SSE"
                    })
                elif request.url.path == "/":
                    return JSONResponse({
                        "service": "MedBenchJudge (A2A-Compatible)",
                        "description": "Green agent for MedAgentBench with SSE transport support",
                        "version": "2.0.0",
                        "protocol": "A2A",
                        "transport": "SSE",
                        "endpoints": {
                            "POST /": "A2A JSON-RPC endpoint",
                            "GET /health": "Health check"
                        },
                        "status": "running"
                    })
            return await call_next(request)

    app.add_middleware(HealthMiddleware)

    # Add CORS middleware for SSE support
    # SSE connections require CORS headers for cross-origin streaming requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    logger.info("Server ready, listening for A2A requests")

    # Run the server
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

    server = uvicorn.Server(config)
    server.serve()


if __name__ == "__main__":
    # Run uvicorn directly
    import sys
    from uvicorn import run

    # Get args
    host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else "0.0.0.0"
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 9008
    card_url = sys.argv[sys.argv.index("--card-url") + 1] if "--card-url" in sys.argv else f"http://{host}:{port}/"
    data_path = sys.argv[sys.argv.index("--data-path") + 1] if "--data-path" in sys.argv else "/app/data/medagentbench/test_data_v2.json"
    dry_run = "--dry-run" in sys.argv

    # Set dry run environment variable
    if dry_run:
        os.environ["DRY_RUN"] = "true"

    # Build app
    green_agent = create_judge_agent(data_path=data_path)
    executor = GreenExecutor(green_agent)
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Create agent card
    agent_card = medbench_judge_agent_card(
        agent_name="MedBenchJudge",
        card_url=card_url
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the Starlette app
    app = a2a_app.build()

    # Add health check endpoint
    from starlette.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    class HealthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.method == "GET":
                if request.url.path == "/health":
                    return JSONResponse({
                        "status": "healthy",
                        "service": "medbench-judge-a2a",
                        "ready": True,
                        "tasks_loaded": len(green_agent._tasks),
                        "transport": "SSE"
                    })
                elif request.url.path == "/":
                    return JSONResponse({
                        "service": "MedBenchJudge (A2A-Compatible)",
                        "status": "running"
                    })
            return await call_next(request)

    app.add_middleware(HealthMiddleware)

    # Add CORS middleware for SSE support
    # SSE connections require CORS headers for cross-origin streaming requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Run server
    run(app, host=host, port=port, log_level="info")
