"""
MedAgentBench Task Adapter

This module provides compatibility between AgentX2 custom task format and
official Stanford MedAgentBench task format.

Supported Formats:
1. Custom (AgentX2): Uses 'id' and 'sol' (array)
2. Official (Stanford): Uses 'task_id' and 'solution' (string)

The adapter automatically detects the format and converts to internal MedAgentBenchTask.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

try:
    # Try relative imports first (when imported as a package)
    from .medbench_models import MedAgentBenchTask, TaskFormatType, MedicalEvaluationResult
except ImportError:
    # Fall back to absolute imports (when run as a module)
    from medbench.medbench_models import MedAgentBenchTask, TaskFormatType, MedicalEvaluationResult


logger = logging.getLogger("task_adapter")


# ============================================================================
# Format Detection
# ============================================================================

def detect_task_format(raw_task: dict[str, Any]) -> TaskFormatType:
    """
    Detect the format of a raw task dictionary.

    Args:
        raw_task: Raw task dictionary from JSON

    Returns:
        TaskFormatType.CUSTOM or TaskFormatType.OFFICIAL
    """
    # Check for official format indicators
    if "task_id" in raw_task:
        return TaskFormatType.OFFICIAL
    elif "id" in raw_task:
        return TaskFormatType.CUSTOM

    # Default to custom if can't determine
    logger.warning(f"Cannot determine format for task: {raw_task.get('id', raw_task.get('task_id', 'unknown'))}, defaulting to CUSTOM")
    return TaskFormatType.CUSTOM


def normalize_task_data(raw_task: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize task data to common format for MedAgentBenchTask.

    Args:
        raw_task: Raw task dictionary from JSON

    Returns:
        Normalized dictionary compatible with MedAgentBenchTask
    """
    format_type = detect_task_format(raw_task)
    normalized = {}

    if format_type == TaskFormatType.OFFICIAL:
        # Official format: task_id, solution (string)
        normalized["task_id"] = raw_task.get("task_id", raw_task.get("id", ""))
        normalized["instruction"] = raw_task.get("instruction", "")
        normalized["eval_MRN"] = raw_task.get("eval_MRN")
        normalized["context"] = raw_task.get("context")

        # Convert solution string to array for internal format
        solution = raw_task.get("solution", "")
        if isinstance(solution, str):
            normalized["sol"] = [solution] if solution else []
        elif isinstance(solution, list):
            normalized["sol"] = solution
        else:
            normalized["sol"] = []

        normalized["format_type"] = TaskFormatType.OFFICIAL

    else:  # CUSTOM format
        # Custom format: id, sol (array)
        # Map 'id' to 'task_id' for internal consistency
        normalized["task_id"] = raw_task.get("id", "")
        normalized["instruction"] = raw_task.get("instruction", "")
        normalized["eval_MRN"] = raw_task.get("eval_MRN")
        normalized["sol"] = raw_task.get("sol", [])
        normalized["context"] = raw_task.get("context")
        normalized["format_type"] = TaskFormatType.CUSTOM

    return normalized


# ============================================================================
# Task Loading
# ============================================================================

def load_tasks(data_path: str | Path) -> list[MedAgentBenchTask]:
    """
    Load medical tasks from MedAgentBench JSON file.

    Automatically detects and handles both custom and official formats.

    Args:
        data_path: Path to the JSON file (test_data_v2.json or official tasks)

    Returns:
        List of MedAgentBenchTask objects
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(path, 'r') as f:
        raw_tasks = json.load(f)

    tasks = []
    format_counts = {TaskFormatType.CUSTOM: 0, TaskFormatType.OFFICIAL: 0}

    for task_data in raw_tasks:
        try:
            # Detect format and normalize
            format_type = detect_task_format(task_data)
            normalized = normalize_task_data(task_data)

            # Create MedAgentBenchTask
            task = MedAgentBenchTask(**normalized)
            tasks.append(task)

            format_counts[format_type] += 1

        except ValidationError as e:
            logger.warning(f"Failed to parse task {task_data.get('id', task_data.get('task_id', 'unknown'))}: {e}")

    logger.info(f"Loaded {len(tasks)} tasks from {data_path}")
    logger.info(f"Format breakdown: {format_counts}")

    return tasks


def load_official_tasks(data_path: str | Path) -> list[MedAgentBenchTask]:
    """
    Alias for load_tasks - supports both formats automatically.

    Provided for API clarity when explicitly working with official format.
    """
    return load_tasks(data_path)


# ============================================================================
# Export Conversion
# ============================================================================

def to_official_format(task: MedAgentBenchTask) -> dict[str, Any]:
    """
    Convert MedAgentBenchTask to official Stanford MedAgentBench format.

    Args:
        task: MedAgentBenchTask object

    Returns:
        Dictionary in official format
    """
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "eval_MRN": task.eval_MRN,
        "solution": task.solution,  # First solution as string
        "context": task.context,
    }


def to_custom_format(task: MedAgentBenchTask) -> dict[str, Any]:
    """
    Convert MedAgentBenchTask to custom AgentX2 format.

    Args:
        task: MedAgentBenchTask object

    Returns:
        Dictionary in custom format
    """
    return {
        "id": task.task_id,
        "instruction": task.instruction,
        "eval_MRN": task.eval_MRN,
        "sol": task.sol,  # Full array
        "context": task.context,
    }


# ============================================================================
# Batch Conversion
# ============================================================================

def convert_batch_to_official(tasks: list[MedAgentBenchTask]) -> list[dict[str, Any]]:
    """Convert multiple tasks to official format."""
    return [to_official_format(task) for task in tasks]


def convert_batch_to_custom(tasks: list[MedAgentBenchTask]) -> list[dict[str, Any]]:
    """Convert multiple tasks to custom format."""
    return [to_custom_format(task) for task in tasks]


# ============================================================================
# Utility Functions
# ============================================================================

def get_task_by_id(tasks: list[MedAgentBenchTask], task_id: str) -> MedAgentBenchTask | None:
    """
    Get a task by its ID (supports both 'id' and 'task_id' fields).

    Args:
        tasks: List of MedAgentBenchTask objects
        task_id: Task identifier to search for

    Returns:
        MedAgentBenchTask if found, None otherwise
    """
    for task in tasks:
        if task.task_id == task_id:
            return task
    return None


def validate_task_compatibility(task: MedAgentBenchTask) -> tuple[bool, str]:
    """
    Validate that a task is compatible with official MedAgentBench.

    Args:
        task: MedAgentBenchTask to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not task.task_id:
        return False, "task_id is required"

    if not task.instruction:
        return False, "instruction is required"

    # Official format expects solution as a single string
    if len(task.sol) > 1:
        return False, f"Official format expects single solution, found {len(task.sol)} solutions"

    return True, "ok"
