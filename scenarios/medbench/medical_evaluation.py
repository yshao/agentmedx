"""
Medical Evaluation Engine for MedAgentBench

Implements LLM-as-Judge evaluation with specialty-specific rubrics:
- Diabetes: 6 criteria (medication appropriateness, A1C targets, comorbidities, lifestyle, safety, monitoring)
- General Medical: 3 criteria (accuracy, completeness, medical correctness)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import ClassVar

from groq import Groq
from .medbench_models import (
    MedAgentBenchTask, DiabetesScore,
    GeneralMedicalScore, MedicalEvaluationResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_evaluation")


class EvaluationCache:
    """
    Cache for evaluation results to reduce redundant API calls.

    Caches evaluations based on task ID and response hash.
    Persists to disk for reuse across runs.
    """

    def __init__(self, cache_file: str = "data/evaluation_cache.json"):
        """
        Initialize the evaluation cache.

        Args:
            cache_file: Path to the cache file
        """
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_cache_key(self, task: MedAgentBenchTask, response: str) -> str:
        """
        Generate a cache key from task and response.

        Args:
            task: The medical task
            response: The agent's response

        Returns:
            MD5 hash of the content
        """
        content = f"{task.id}:{task.instruction}:{response[:500]}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, task: MedAgentBenchTask, response: str) -> dict | None:
        """Get cached evaluation result if available."""
        key = self._get_cache_key(task, response)
        return self.cache.get(key)

    def set(self, task: MedAgentBenchTask, response: str, result: dict):
        """Cache an evaluation result."""
        key = self._get_cache_key(task, response)
        self.cache[key] = {
            "task_id": task.id,
            "response_hash": key,
            "result": result,
            "timestamp": time.time()
        }
        self._save_cache()


class MedicalEvaluationEngine:
    """
    Medical evaluation engine with specialty-specific rubrics.

    Uses Google Gemini for LLM-as-Judge evaluation with structured rubrics
    for different medical specialties.

    Features:
    - Evaluation caching to reduce API calls
    - Rate limit handling with request spacing
    - Improved error reporting
    """

    # Class variables for rate limit handling
    _last_request_time: ClassVar[float] = 0
    _min_request_interval: ClassVar[float] = 1.0  # 1 second between requests

    def __init__(self, model: str = "llama-3.3-70b-versatile", use_cache: bool = False):
        """
        Initialize the evaluation engine.

        Args:
            model: The Groq model to use for evaluation
            use_cache: Whether to use evaluation caching
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable must be set")
        self._client = Groq(api_key=api_key)
        self._init_model = model
        self._cache = EvaluationCache() if use_cache else None
        self._use_cache = use_cache
        logger.info(f"MedicalEvaluationEngine initialized with model: {model}")

    async def _rate_limit_delay(self):
        """Add delay to respect API rate limits."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self._min_request_interval:
            delay = self._min_request_interval - time_since_last
            logger.debug(f"Rate limit delay: sleeping {delay:.2f}s")
            await asyncio.sleep(delay)
        self._last_request_time = time.time()

    async def evaluate_diabetes(
        self,
        task: MedAgentBenchTask,
        agent_response: str,
        max_retries: int = 3
    ) -> DiabetesScore:
        """
        Evaluate diabetes treatment plan using 6 specialty criteria.

        Args:
            task: The MedAgentBench task
            agent_response: The agent's response to evaluate
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            A DiabetesScore with scores for all 6 criteria
        """
        # Check cache first
        if self._use_cache:
            cached = self._cache.get(task, agent_response)
            if cached:
                logger.info(f"Using cached evaluation for task {task.id}")
                cached["result"]["feedback"] = "[FROM CACHE] " + cached["result"].get("feedback", "")
                return DiabetesScore(**cached["result"])

        rubric = """
SCORING RUBRIC (0-10):
1. medication_appropriateness: Are medications suitable for this patient's profile?
2. a1c_target: Does the plan address A1C goals appropriately?
3. comorbidity_management: Are comorbidities (hypertension, kidney function, lipids) addressed?
4. lifestyle_recommendations: Are diet and exercise guidance included?
5. safety: Are there contraindications or dangerous drug interactions?
6. monitoring_plan: Is there a clear follow-up and monitoring strategy?

**Score Guidelines:**
- 9-10: Excellent - comprehensive, evidence-based, addresses all aspects
- 7-8: Good - mostly appropriate, minor gaps
- 5-6: Adequate - basic coverage, some issues
- 3-4: Fair - significant gaps or errors
- 0-2: Poor - dangerous or incorrect recommendations
"""

        prompt = f"""You are an endocrinologist evaluating diabetes treatment plans.

TASK: {task.instruction}
EXPECTED SOLUTION: {task.sol}

AGENT RESPONSE TO EVALUATE:
{agent_response}

Evaluate on 6 diabetes-specific criteria (0-10 scale):
{rubric}

STRICT PRESCRIPTIVE REQUIREMENTS FOR IMPROVEMENT SUGGESTIONS:

1. BE PRESCRIPTIVE: Use "Prescribe", "Order", "Add", "Include", "Start", "Initiate", "Monitor"
2. BE SPECIFIC: Include drug names, dosages, routes, frequencies, target values, timelines
3. BE CLINICAL: Include expected outcomes, monitoring schedules, follow-up intervals
4. NO PASSIVE LANGUAGE: Never use "consider", "suggest", "recommend", "may", "might", "could", "should", "would"

CLINICAL IMPROVEMENT EXAMPLES:
- "Prescribe atorvastatin 20mg POQ daily, target LDL-C < 70 mg/dL, monitor liver enzymes at 6 weeks, expect 25-35% LDL-C reduction"
- "Set A1C target < 7.0% for this 65yo patient with CKD stage 3, test every 3 months"
- "Prescribe 150 min/week moderate-intensity aerobic exercise + 2x/week strength training"

CRITICAL: Return ONLY valid JSON in this exact format:
{{
    "medication_appropriateness": <float 0-10>,
    "a1c_target": <float 0-10>,
    "comorbidity_management": <float 0-10>,
    "lifestyle_recommendations": <float 0-10>,
    "safety": <float 0-10>,
    "monitoring_plan": <float 0-10>,
    "feedback": "Brief clinical assessment",
    "whats_working_well": ["Thing 1 agent did well", "Thing 2 agent did well", ...],
    "suggested_improvements": [
        "[Prescriptive suggestion with clinical specifics - drug/dose/route/frequency/target/monitoring]"
    ]
}}
"""

        for attempt in range(max_retries):
            try:
                # Rate limit handling: add delay before API call
                await self._rate_limit_delay()

                response = self._client.chat.completions.create(
                    model=self._init_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

                # Parse JSON response
                data = self._parse_json_response(response.choices[0].message.content)
                result = DiabetesScore(**data)

                # Cache the result
                if self._use_cache:
                    self._cache.set(task, agent_response, result.model_dump())

                return result

            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 5  # 5, 10, 20 seconds
                        logger.warning(f"[DEFAULT SCORE] Rate limit hit for diabetes evaluation, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"[DEFAULT SCORE] Max retries exceeded for diabetes evaluation due to rate limiting")
                        # Return default scores with CLEAR feedback
                        return DiabetesScore(
                            medication_appropriateness=5.0,
                            a1c_target=5.0,
                            comorbidity_management=5.0,
                            lifestyle_recommendations=5.0,
                            safety=5.0,
                            monitoring_plan=5.0,
                            feedback="[DEFAULT SCORE - API RATE LIMIT (429) after 3 retries - Evaluation not performed. Please check API quota.]"
                        )
                else:
                    logger.error(f"[DEFAULT SCORE] Error evaluating diabetes: {e}")
                    # Return default scores with error feedback
                    return DiabetesScore(
                        medication_appropriateness=5.0,
                        a1c_target=5.0,
                        comorbidity_management=5.0,
                        lifestyle_recommendations=5.0,
                        safety=5.0,
                        monitoring_plan=5.0,
                        feedback=f"[DEFAULT SCORE - Error during evaluation: {error_str}]"
                    )

        # Should not reach here
        return DiabetesScore(
            medication_appropriateness=5.0,
            a1c_target=5.0,
            comorbidity_management=5.0,
            lifestyle_recommendations=5.0,
            safety=5.0,
            monitoring_plan=5.0,
            feedback="[DEFAULT SCORE - Unknown error during evaluation]"
        )

    async def evaluate_general_medical(
        self,
        task: MedAgentBenchTask,
        agent_response: str,
        max_retries: int = 3
    ) -> GeneralMedicalScore:
        """
        Evaluate general medical response using 3 criteria.

        Args:
            task: The MedAgentBench task
            agent_response: The agent's response to evaluate
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            A GeneralMedicalScore with scores for all 3 criteria
        """
        # Check cache first
        if self._use_cache:
            cached = self._cache.get(task, agent_response)
            if cached:
                logger.info(f"Using cached evaluation for task {task.id}")
                cached_result = cached["result"]
                cached_result["feedback"] = "[FROM CACHE] " + cached_result.get("feedback", "")
                return GeneralMedicalScore(**cached_result)

        rubric = """
SCORING RUBRIC (0-10):
1. accuracy: How close is the response to the expected answer?
2. completeness: Does it address all aspects of the question?
3. medical_correctness: Is the information clinically accurate?

**Score Guidelines:**
- 9-10: Excellent - accurate, complete, medically sound
- 7-8: Good - mostly correct with minor omissions
- 5-6: Adequate - partially correct but has gaps
- 3-4: Fair - significant errors or misinformation
- 0-2: Poor - dangerous or incorrect medical information
"""

        prompt = f"""You are a board-certified internist evaluating medical responses.

TASK: {task.instruction}
EXPECTED SOLUTION: {task.sol}

AGENT RESPONSE TO EVALUATE:
{agent_response}

Evaluate on 3 general medical criteria (0-10 scale):
{rubric}

CRITICAL: Return ONLY valid JSON in this exact format:
{{
    "accuracy": <float 0-10>,
    "completeness": <float 0-10>,
    "medical_correctness": <float 0-10>,
    "feedback": "Brief clinical assessment"
}}
"""

        for attempt in range(max_retries):
            try:
                # Rate limit handling: add delay before API call
                await self._rate_limit_delay()

                response = self._client.chat.completions.create(
                    model=self._init_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )

                # Parse JSON response
                data = self._parse_json_response(response.choices[0].message.content)
                result = GeneralMedicalScore(**data)

                # Cache the result
                if self._use_cache:
                    self._cache.set(task, agent_response, result.model_dump())

                return result

            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 5  # 5, 10, 20 seconds
                        logger.warning(f"[DEFAULT SCORE] Rate limit hit for medical evaluation, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"[DEFAULT SCORE] Max retries exceeded for medical evaluation due to rate limiting")
                        # Return default scores with CLEAR feedback
                        return GeneralMedicalScore(
                            accuracy=5.0,
                            completeness=5.0,
                            medical_correctness=5.0,
                            feedback="[DEFAULT SCORE - API RATE LIMIT (429) after 3 retries - Evaluation not performed. Please check API quota.]"
                        )
                else:
                    logger.error(f"[DEFAULT SCORE] Error evaluating general medical: {e}")
                    # Return default scores with error feedback
                    return GeneralMedicalScore(
                        accuracy=5.0,
                        completeness=5.0,
                        medical_correctness=5.0,
                        feedback=f"[DEFAULT SCORE - Error during evaluation: {error_str}]"
                    )

        # Should not reach here
        return GeneralMedicalScore(
            accuracy=5.0,
            completeness=5.0,
            medical_correctness=5.0,
            feedback="[DEFAULT SCORE - Unknown error during evaluation]"
        )

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response from LLM with fallback handling."""
        try:
            response = response.strip()

            # Remove markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if end > start:
                    response = response[start:end].strip()
            elif response.startswith("```"):
                response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()

            # Try to find JSON object boundaries
            if not response.startswith("{"):
                start_idx = response.find("{")
                end_idx = response.rfind("}")
                if start_idx >= 0 and end_idx > start_idx:
                    response = response[start_idx:end_idx + 1]

            response = response.strip()

            # Parse JSON
            data = json.loads(response)
            return data

        except json.JSONDecodeError as je:
            # If JSON parsing fails, try to extract fields using regex
            logger.warning(f"JSON decode error: {je}, trying regex fallback")

            try:
                # Try to extract score fields using regex
                data = {}

                # Extract diabetes scores if present
                if "medication_appropriateness" in response:
                    data["medication_appropriateness"] = self._extract_float("medication_appropriateness", response)
                    data["a1c_target"] = self._extract_float("a1c_target", response)
                    data["comorbidity_management"] = self._extract_float("comorbidity_management", response)
                    data["lifestyle_recommendations"] = self._extract_float("lifestyle_recommendations", response)
                    data["safety"] = self._extract_float("safety", response)
                    data["monitoring_plan"] = self._extract_float("monitoring_plan", response)

                # Extract general medical scores if present
                if "accuracy" in response:
                    data["accuracy"] = self._extract_float("accuracy", response)
                    data["completeness"] = self._extract_float("completeness", response)
                    data["medical_correctness"] = self._extract_float("medical_correctness", response)

                # Extract feedback
                feedback_match = re.search(r'"feedback"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.DOTALL)
                if feedback_match:
                    import ast
                    feedback = feedback_match.group(1)
                    # Unescape common JSON escape sequences
                    feedback = feedback.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    data["feedback"] = feedback
                else:
                    data["feedback"] = "Could not extract feedback from LLM response"

                return data

            except Exception as fallback_error:
                logger.warning(f"Regex fallback also failed: {fallback_error}")
                # Return default scores on parse error with clear feedback
                if "medication_appropriateness" in response:
                    # Diabetes task
                    return {
                        "medication_appropriateness": 5.0,
                        "a1c_target": 5.0,
                        "comorbidity_management": 5.0,
                        "lifestyle_recommendations": 5.0,
                        "safety": 5.0,
                        "monitoring_plan": 5.0,
                        "feedback": "[DEFAULT SCORE - Could not parse LLM response as JSON. Response did not contain valid JSON.]"
                    }
                else:
                    # General medical task
                    return {
                        "accuracy": 5.0,
                        "completeness": 5.0,
                        "medical_correctness": 5.0,
                        "feedback": "[DEFAULT SCORE - Could not parse LLM response as JSON. Response did not contain valid JSON.]"
                    }

    def _extract_float(self, field_name: str, response: str) -> float:
        """Extract a float field from response text using regex."""
        pattern = r'"' + field_name + r'"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        match = re.search(pattern, response)
        if match:
            return float(match.group(1))
        return 5.0  # Default if not found
