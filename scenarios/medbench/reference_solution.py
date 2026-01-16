"""
MedAgentBench Reference Solution Integration

This module provides a plugin interface for integrating with Stanford Medicine's
reference solutions (refsol.py) from their Box account.

The reference solutions are used for official accuracy benchmarking in MedAgentBench.
"""

import logging
from pathlib import Path
from typing import Any, Protocol

try:
    # Try relative imports first (when imported as a package)
    from .medbench_models import MedAgentBenchTask
except ImportError:
    # Fall back to absolute imports (when run as a module)
    from medbench.medbench_models import MedAgentBenchTask


logger = logging.getLogger("reference_solution")


# ============================================================================
# Reference Solution Protocol
# ============================================================================

class ReferenceSolutionProvider(Protocol):
    """
    Protocol for reference solution providers.

    This allows different implementations:
    - Actual refsol.py from Stanford (requires credentials)
    - Mock implementation for testing
    - Local cached solutions
    """

    def get_solution(self, task_id: str) -> str | None:
        """
        Get the reference solution for a task.

        Args:
            task_id: Task identifier

        Returns:
            Reference solution string, or None if not found
        """
        ...

    def has_solution(self, task_id: str) -> bool:
        """Check if a reference solution exists for the task."""
        ...

    def compare(self, task_id: str, agent_response: str) -> dict[str, Any]:
        """
        Compare agent response against reference solution.

        Args:
            task_id: Task identifier
            agent_response: Agent's response to evaluate

        Returns:
            Dictionary with comparison results (e.g., accuracy, pass/fail)
        """
        ...


# ============================================================================
# Mock Implementation (For Testing)
# ============================================================================

class MockReferenceSolutionProvider:
    """
    Mock reference solution provider for testing and development.

    Uses the 'sol' field from the task data as the reference solution.
    """

    def __init__(self, tasks: list[MedAgentBenchTask] | None = None):
        """
        Initialize the mock provider.

        Args:
            tasks: List of tasks to extract solutions from
        """
        self._solutions: dict[str, list[str]] = {}
        if tasks:
            for task in tasks:
                self._solutions[task.task_id] = task.sol

        logger.info(f"MockReferenceSolutionProvider initialized with {len(self._solutions)} solutions")

    def get_solution(self, task_id: str) -> str | None:
        """Get the first reference solution for a task."""
        solutions = self._solutions.get(task_id, [])
        return solutions[0] if solutions else None

    def has_solution(self, task_id: str) -> bool:
        """Check if a reference solution exists."""
        return task_id in self._solutions and len(self._solutions[task_id]) > 0

    def compare(self, task_id: str, agent_response: str) -> dict[str, Any]:
        """
        Compare agent response against reference solution.

        This is a simple keyword-based comparison for demonstration.
        The actual official comparison may use more sophisticated methods.
        """
        solution = self.get_solution(task_id)
        if not solution:
            return {
                "has_reference": False,
                "accuracy": None,
                "passed": None,
                "error": "No reference solution found",
            }

        # Simple comparison: check for keyword overlap
        # This is NOT the official method, just a placeholder
        solution_words = set(solution.lower().split())
        response_words = set(agent_response.lower().split())

        if not solution_words:
            overlap = 0.0
        else:
            overlap = len(solution_words & response_words) / len(solution_words)

        # Simple threshold for pass/fail
        passed = overlap >= 0.5

        return {
            "has_reference": True,
            "accuracy": round(overlap, 3),
            "passed": passed,
            "method": "keyword_overlap",
            "reference_solution": solution,
        }


# ============================================================================
# Cached Solutions Provider
# ============================================================================

class CachedReferenceSolutionProvider:
    """
    Reference solution provider that loads from a local JSON cache file.

    Cache file format:
    {
        "task_id_1": "solution text",
        "task_id_2": "solution text"
    }
    """

    def __init__(self, cache_path: str | Path):
        """
        Initialize from a cache file.

        Args:
            cache_path: Path to JSON cache file
        """
        import json

        self._cache_path = Path(cache_path)
        self._solutions: dict[str, str] = {}

        if self._cache_path.exists():
            with open(self._cache_path, 'r') as f:
                data = json.load(f)
                self._solutions = {str(k): str(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._solutions)} cached solutions from {cache_path}")
        else:
            logger.warning(f"Cache file not found: {cache_path}")

    def get_solution(self, task_id: str) -> str | None:
        """Get the reference solution for a task."""
        return self._solutions.get(task_id)

    def has_solution(self, task_id: str) -> bool:
        """Check if a reference solution exists."""
        return task_id in self._solutions

    def compare(self, task_id: str, agent_response: str) -> dict[str, Any]:
        """
        Compare agent response against reference solution.

        Uses the mock comparison logic (keyword overlap).
        For official comparison, integrate with actual refsol.py.
        """
        solution = self.get_solution(task_id)
        if not solution:
            return {
                "has_reference": False,
                "accuracy": None,
                "passed": None,
                "error": "No reference solution found",
            }

        # Simple comparison: check for keyword overlap
        solution_words = set(solution.lower().split())
        response_words = set(agent_response.lower().split())

        if not solution_words:
            overlap = 0.0
        else:
            overlap = len(solution_words & response_words) / len(solution_words)

        passed = overlap >= 0.5

        return {
            "has_reference": True,
            "accuracy": round(overlap, 3),
            "passed": passed,
            "method": "keyword_overlap",
            "reference_solution": solution,
        }

    def add_solution(self, task_id: str, solution: str) -> None:
        """Add a solution to the cache (in memory, not persisted)."""
        self._solutions[task_id] = solution

    def save_cache(self) -> None:
        """Save the current cache to disk."""
        import json

        with open(self._cache_path, 'w') as f:
            json.dump(self._solutions, f, indent=2)
        logger.info(f"Saved {len(self._solutions)} solutions to {self._cache_path}")


# ============================================================================
# Stanford Official Integration (Placeholder)
# ============================================================================

class StanfordReferenceSolutionProvider:
    """
    Provider that integrates with official Stanford refsol.py.

    NOTE: This requires access to Stanford Medicine's Box account to download
    the official refsol.py file.

    This is a placeholder showing how integration would work.
    """

    def __init__(self, refsol_path: str | Path | None = None):
        """
        Initialize with path to official refsol.py.

        Args:
            refsol_path: Path to the official refsol.py file
        """
        self._refsol_path = Path(refsol_path) if refsol_path else None
        self._refsol_module = None

        if self._refsol_path and self._refsol_path.exists():
            try:
                # Import the refsol module
                import sys
                import importlib.util

                spec = importlib.util.spec_from_file_location("refsol", self._refsol_path)
                if spec and spec.loader:
                    self._refsol_module = importlib.util.module_from_spec(spec)
                    sys.modules["refsol"] = self._refsol_module
                    spec.loader.exec_module(self._refsol_module)
                    logger.info(f"Loaded official refsol from {refsol_path}")
                else:
                    logger.warning(f"Could not load refsol from {refsol_path}")
            except Exception as e:
                logger.warning(f"Failed to load refsol.py: {e}")
        else:
            logger.warning("refsol.py not provided, using fallback comparison")

    def get_solution(self, task_id: str) -> str | None:
        """Get the reference solution for a task."""
        if self._refsol_module and hasattr(self._refsol_module, "get_solution"):
            return self._refsol_module.get_solution(task_id)
        return None

    def has_solution(self, task_id: str) -> bool:
        """Check if a reference solution exists."""
        return self.get_solution(task_id) is not None

    def compare(self, task_id: str, agent_response: str) -> dict[str, Any]:
        """
        Compare agent response against reference solution using official method.

        If refsol.py is available, uses the official comparison.
        Otherwise, falls back to keyword overlap.
        """
        if self._refsol_module and hasattr(self._refsol_module, "compare"):
            # Use official comparison
            try:
                result = self._refsol_module.compare(task_id, agent_response)
                return {
                    "has_reference": True,
                    "method": "official_stanford",
                    **result,
                }
            except Exception as e:
                logger.warning(f"Official comparison failed: {e}")

        # Fallback to keyword overlap
        solution = self.get_solution(task_id)
        if not solution:
            return {
                "has_reference": False,
                "accuracy": None,
                "passed": None,
                "error": "No reference solution found",
            }

        solution_words = set(solution.lower().split())
        response_words = set(agent_response.lower().split())

        if not solution_words:
            overlap = 0.0
        else:
            overlap = len(solution_words & response_words) / len(solution_words)

        passed = overlap >= 0.5

        return {
            "has_reference": True,
            "accuracy": round(overlap, 3),
            "passed": passed,
            "method": "keyword_overlap_fallback",
            "reference_solution": solution,
        }


# ============================================================================
# Factory Function
# ============================================================================

def get_reference_provider(
    provider_type: str = "mock",
    tasks: list[MedAgentBenchTask] | None = None,
    cache_path: str | Path | None = None,
    refsol_path: str | Path | None = None,
) -> ReferenceSolutionProvider:
    """
    Factory function to get a reference solution provider.

    Args:
        provider_type: Type of provider ("mock", "cached", "stanford")
        tasks: List of tasks (for mock provider)
        cache_path: Path to cache file (for cached provider)
        refsol_path: Path to official refsol.py (for stanford provider)

    Returns:
        ReferenceSolutionProvider instance
    """
    if provider_type == "mock":
        return MockReferenceSolutionProvider(tasks)
    elif provider_type == "cached":
        if not cache_path:
            raise ValueError("cache_path required for cached provider")
        return CachedReferenceSolutionProvider(cache_path)
    elif provider_type == "stanford":
        return StanfordReferenceSolutionProvider(refsol_path)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# ============================================================================
# Integration Helper
# ============================================================================

def integrate_reference_comparison(
    evaluation_result: dict[str, Any],
    reference_provider: ReferenceSolutionProvider,
) -> dict[str, Any]:
    """
    Integrate reference solution comparison into evaluation result.

    Args:
        evaluation_result: Existing evaluation result dictionary
        reference_provider: Reference solution provider

    Returns:
        Enhanced evaluation result with reference comparison
    """
    task_id = evaluation_result.get("task_id")
    agent_response = evaluation_result.get("agent_response", "")

    if not task_id:
        return evaluation_result

    comparison = reference_provider.compare(task_id, agent_response)

    # Add reference comparison to result
    evaluation_result["reference_comparison"] = comparison

    # Update passed status if available
    if comparison.get("has_reference") and "passed" in comparison:
        evaluation_result["passed_reference"] = comparison["passed"]

    return evaluation_result
