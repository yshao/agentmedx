#!/usr/bin/env python3
"""
MedAgentBench Webhook Notifier

Sends evaluation results to a leaderboard webhook endpoint asynchronously
with configurable retry logic and exponential backoff.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

try:
    import aiohttp
except ImportError:
    import aiohttp
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

class WebhookConfig:
    """Webhook configuration"""
    url: str
    secret: str = ""
    enabled: bool = False
    max_retries: int = 3
    timeout: int = 5
    retry_backoff: int = 2
    include_files: bool = True
    dry_run_send: bool = False  # If true, log instead of actually sending


logger = logging.getLogger("webhook_notifier")


# ============================================================================
# WebhookNotifier Class
# ============================================================================

class WebhookNotifier:
    """
    Asynchronous webhook sender with retry logic and exponential backoff.

    This class handles sending evaluation results to a leaderboard webhook endpoint.
    """

    def __init__(self, config: WebhookConfig):
        """
        Initialize the webhook notifier.

        Args:
            config: Webhook configuration object
        """
        self.config = config
        self.session = None

        if config.enabled:
            logger.info(f"Webhook notifications enabled: {config.url}")
        else:
            logger.info("Webhook notifications disabled")

    async def _create_session(self):
        """Create aiohttp session for HTTP requests."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    def _build_payload(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build webhook payload from evaluation results.

        Args:
            evaluation_results: Dictionary with evaluation results from all participants

        Returns:
            Webhook payload as dictionary
        """
        # Get first participant's results (for now)
        participant_id = next(iter(evaluation_results))
        result = evaluation_results[participant_id]

        # Extract key information
        eval_result = result.get("eval_result", {})

        # Get detailed scores
        scores = {}
        if "diabetes_score" in result:
            diabetes_score = result["diabetes_score"]
            scores = {
                "medication_appropriateness": diabetes_score.get("medication_appropriateness", 0),
                "a1c_target": diabetes_score.get("a1c_target", 0),
                "comorbidity_management": diabetes_score.get("comorbidity_management", 0),
                "lifestyle_recommendations": diabetes_score.get("lifestyle_recommendations", 0),
                "safety": diabetes_score.get("safety", 0),
                "monitoring_plan": diabetes_score.get("monitoring_plan", 0),
            }
        elif "general_score" in result:
            general_score = result["general_score"]
            scores = {
                "accuracy": general_score.get("accuracy", 0),
                "completeness": general_score.get("completeness", 0),
                "medical_correctness": general_score.get("medical_correctness", 0),
            }

        # Get file paths
        files = {}
        if self.config.include_files:
            model_name = result.get("model_name", "agentx-medical")
            task_id = result.get("task_id", "unknown")

            # Build file paths
            overall_path = f"outputs/{model_name}/{task_id}/overall.json"
            summary_path = f"outputs/{model_name}/summary.json"

            files = {
                "overall_json": overall_path,
                "summary_json": summary_path
            }

        # Build evaluation summary
        total_score = eval_result.get("total_score", 0.0)
        max_score = 60.0 if "diabetes_score" in result else 30.0

        return {
            "event": "evaluation.completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "evaluation": {
                "task_id": result.get("task_id", ""),
                "medical_category": result.get("medical_category", ""),
                "model": result.get("model_name", ""),
                "agent_name": result.get("agent_name", ""),
                "total_score": total_score,
                "max_score": max_score,
                "passed": total_score > (max_score * 0.8)  # 80% threshold
            },
            "scores": scores,
            "feedback": eval_result.get("feedback", ""),
            "files": files,
            "metadata": {
                "evaluation_duration_ms": result.get("evaluation_duration_ms", 0),
                "dry_run": result.get("dry_run", False),
                "submission_id": f"submitter-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                "webhook_delivery_id": f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{result.get('task_id', 'unknown')}",
                "retry_count": 0
            }
        }

    async def _send_http(self, payload: Dict[str, Any]) -> bool:
        """
        Send payload to webhook endpoint via HTTP POST.

        Returns True if successful, False otherwise.
        """
        if not self.config.enabled:
            logger.info("Webhook disabled (dry_run: %s)", self.config.dry_run_send)
            return True

        if self.config.dry_run_send:
            logger.info("DRY RUN MODE: Would send webhook to %s", self.config.url)
            return True

        session = await self._create_session()

        try:
            # Prepare request
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Secret": self.config.secret if self.config.secret else ""
            }

            # Send POST request
            async with session.post(self.config.url, json=payload, headers=headers) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Webhook sent successfully: {self.config.url}")
                    return True
                else:
                    logger.warning(f"Webhook failed with HTTP {response.status}")
                    return False

        except asyncio.TimeoutError:
            logger.warning(f"Webhook timeout: {self.config.timeout}s")
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"Webhook connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Webhook unexpected error: {e}")
            return False

    async def _save_to_dead_letter_queue(self, payload: Dict[str, Any]):
        """
        Save failed webhook to dead letter queue.
        """
        try:
            dead_letter_path = Path("outputs/webhooks/failed")
            dead_letter_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            filename = f"{timestamp}_{payload.get('metadata', {}).get('submission_id', 'unknown')}.json"

            output_path = dead_letter_path / filename
            with open(output_path, 'w') as f:
                json.dump(payload, f, indent=2)

            logger.info(f"Saved failed webhook to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save to dead letter queue: {e}")

    async def send_with_retry(self, payload: Dict[str, Any]) -> bool:
        """
        Send webhook with exponential backoff retry logic.

        Returns True if successful, False if all retries exhausted.
        """
        retry_count = 0

        for attempt in range(self.config.max_retries):
            try:
                success = await self._send_http(payload)
                if success:
                    logger.info(f"Webhook delivered successfully (attempt {retry_count + 1})")
                    return True
            except Exception as e:
                retry_count += 1

                if retry_count < self.config.max_retries:
                    wait_time = self.config.retry_backoff ** retry_count
                    logger.warning(f"Webhook attempt {retry_count + 1} failed (retry in {wait_time}s): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.config.max_retries} webhook attempts failed")
                    await self._save_to_dead_letter_queue(payload)
                    return False

        return False

    async def send_evaluation_complete(self, evaluation_results: Dict[str, Any]) -> bool:
        """
        Send evaluation completed webhook.

        Args:
            evaluation_results: Dictionary mapping participant IDs to their evaluation results

        Returns:
            True if webhook sent successfully, False if all retries failed.
        """
        logger.info(f"Sending evaluation complete webhook with {len(evaluation_results)} participant(s)")

        # Check if webhook is enabled
        if not self.config.enabled:
            logger.info("Webhook disabled - skipping webhook")
            return True

        # Build and send webhook
        payload = self._build_payload(evaluation_results)

        return await self._send_with_retry(payload)


# ============================================================================
# Convenience Functions
# ============================================================================

async def send_evaluation_webhook(
    results: Dict[str, Any],
    webhook_url: str,
    secret: str = "",
    max_retries: int = 3,
    timeout: int = 5,
    retry_backoff: int = 2
) -> bool:
    """
    Helper function to send evaluation webhook.

    Args:
        results: Dictionary of evaluation results
        webhook_url: Webhook endpoint URL
        secret: Optional secret for X-Webhook-Secret header
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds
        retry_backoff: Exponential backoff multiplier

    Returns:
        True if successful, False otherwise
    """
    config = WebhookConfig(
        url=webhook_url,
        secret=secret,
        enabled=True,
        max_retries=max_retries,
        timeout=timeout,
        retry_backoff=retry_backoff
    )

    notifier = WebhookNotifier(config)
    return await notifier.send_evaluation_complete(results)


if __name__ == "__main__":
    # Test the webhook with sample payload
    import sys

    # Sample evaluation result (simulated)
    sample_result = {
        "eval_result": {
            "task_id": "diabetes_001",
            "medical_category": "disease" == "diabetes",
            "model_name": "test-agent",
            "agent_name": "medical_agent",
            "total_score": 54.0,
            "dry_run": False,
            "evaluation_duration_ms": 45000,
            "feedback": "Excellent comprehensive care plan..."
        },
        "diabetes_score": {
            "medication_appropriateness": 9.0,
            "a1c_target": 9.0,
            "comorbidity_management": 9.0,
            "lifestyle_recommendations": 9.0,
            "safety": 9.0,
            "monitoring_plan": 9.0
        },
        "model_name": "test-agent"
    }

    # Test with dry run (won't actually send)
    test_config = WebhookConfig(
        url="http://localhost:8000/webhook",
        enabled=True,
        dry_run_send=True,
        max_retries=1
    )

    notifier = Webhook(test_config)

    # Create sample evaluation results dict
    sample_results = {"medical_agent": sample_result}

    print("=" * 60)
    print("Testing Webhook Notifier")
    print("=" * 60)
    print(f"Config: {test_config}")
    print(f"Results: {list(sample_results.keys())}")
    print()

    # Run async test
    import asyncio
    result = asyncio.run(notifier.send_evaluation_complete(sample_results))

    print()
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)
    print(f"Result: {'SUCCESS' if result else 'FAILED'}")
