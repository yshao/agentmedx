"""
MedAgentBench Compatibility Tests

Tests for validating format compatibility between AgentX2 custom format
and official Stanford MedAgentBench format.
"""

import json
import sys
from pathlib import Path

# Add project2 root and tutorial to path for imports
project_root = Path(__file__).parent.parent  # project2/ directory
tutorial_path = project_root.parent / "tutorial" / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(tutorial_path) not in sys.path:
    sys.path.insert(0, str(tutorial_path))

import pytest

from medbench.task_adapter import (
    load_tasks,
    detect_task_format,
    normalize_task_data,
    to_official_format,
    to_custom_format,
    validate_task_compatibility,
    TaskFormatType,
)
from medbench.medbench_models import MedAgentBenchTask, DiabetesScore, MedicalEvaluationResult
from medbench.official_exporter import (
    convert_evaluation_to_official,
    save_overall_json,
    export_evaluation,
    save_summary,
)
from medbench.reference_solution import (
    MockReferenceSolutionProvider,
    CachedReferenceSolutionProvider,
    get_reference_provider,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def official_task_data():
    """Sample official MedAgentBench task data."""
    return [
        {
            "task_id": "official_001",
            "instruction": "Test instruction for official format",
            "eval_MRN": "patient-001",
            "solution": "Reference answer here",
            "context": "Additional context",
        }
    ]


@pytest.fixture
def custom_task_data():
    """Sample custom AgentX2 task data."""
    return [
        {
            "id": "custom_001",
            "instruction": "Test instruction for custom format",
            "eval_MRN": "patient-101",
            "sol": ["Solution 1", "Solution 2"],
            "context": "Additional context",
        }
    ]


@pytest.fixture
def sample_evaluation_result():
    """Sample MedicalEvaluationResult for testing."""
    return MedicalEvaluationResult(
        task_id="test_001",
        medical_category="diabetes",
        agent_name="test_agent",
        agent_response="Test agent response here",
        diabetes_score=DiabetesScore(
            medication_appropriateness=8.0,
            a1c_target=9.0,
            comorbidity_management=7.0,
            lifestyle_recommendations=8.0,
            safety=9.0,
            monitoring_plan=8.0,
        ),
        general_score=None,
        total_score=49.0,
        feedback="Good overall performance",
    )


# ============================================================================
# Format Detection Tests
# ============================================================================

class TestFormatDetection:
    """Tests for task format detection."""

    def test_detect_official_format(self, official_task_data):
        """Test detecting official MedAgentBench format."""
        format_type = detect_task_format(official_task_data[0])
        assert format_type == TaskFormatType.OFFICIAL

    def test_detect_custom_format(self, custom_task_data):
        """Test detecting custom AgentX2 format."""
        format_type = detect_task_format(custom_task_data[0])
        assert format_type == TaskFormatType.CUSTOM

    def test_normalize_official_task(self, official_task_data):
        """Test normalizing official format to internal format."""
        normalized = normalize_task_data(official_task_data[0])

        assert normalized["task_id"] == "official_001"
        assert normalized["format_type"] == TaskFormatType.OFFICIAL
        assert normalized["sol"] == ["Reference answer here"]  # String converted to array

    def test_normalize_custom_task(self, custom_task_data):
        """Test normalizing custom format to internal format."""
        normalized = normalize_task_data(custom_task_data[0])

        assert normalized["task_id"] == "custom_001"  # 'id' mapped to 'task_id'
        assert normalized["format_type"] == TaskFormatType.CUSTOM
        assert normalized["sol"] == ["Solution 1", "Solution 2"]


# ============================================================================
# Task Loading Tests
# ============================================================================

class TestTaskLoading:
    """Tests for task loading from files."""

    def test_load_official_format_tasks(self, tmp_path):
        """Test loading tasks in official format."""
        # Create temporary file with official format
        test_file = tmp_path / "official_tasks.json"
        official_data = [
            {
                "task_id": "test_001",
                "instruction": "Test",
                "eval_MRN": "patient-001",
                "solution": "Answer",
            }
        ]
        test_file.write_text(json.dumps(official_data))

        tasks = load_tasks(str(test_file))

        assert len(tasks) == 1
        assert tasks[0].task_id == "test_001"
        assert tasks[0].format_type == TaskFormatType.OFFICIAL

    def test_load_custom_format_tasks(self, tmp_path):
        """Test loading tasks in custom format."""
        # Create temporary file with custom format
        test_file = tmp_path / "custom_tasks.json"
        custom_data = [
            {
                "id": "test_002",
                "instruction": "Test",
                "eval_MRN": "patient-002",
                "sol": ["Answer 1", "Answer 2"],
            }
        ]
        test_file.write_text(json.dumps(custom_data))

        tasks = load_tasks(str(test_file))

        assert len(tasks) == 1
        assert tasks[0].task_id == "test_002"
        assert tasks[0].format_type == TaskFormatType.CUSTOM

    def test_load_mixed_format_file(self, tmp_path):
        """Test loading a file with official format (mixed detection not needed)."""
        # Files should be consistently one format
        test_file = tmp_path / "official_tasks.json"
        official_data = [
            {
                "task_id": "test_003",
                "instruction": "Test",
                "solution": "Answer",
            }
        ]
        test_file.write_text(json.dumps(official_data))

        tasks = load_tasks(str(test_file))
        assert all(t.format_type == TaskFormatType.OFFICIAL for t in tasks)


# ============================================================================
# Format Conversion Tests
# ============================================================================

class TestFormatConversion:
    """Tests for format conversion functions."""

    def test_to_official_format(self, official_task_data):
        """Test converting MedAgentBenchTask to official format."""
        task = MedAgentBenchTask(
            task_id="test_001",
            instruction="Test instruction",
            eval_MRN="patient-001",
            sol=["Answer"],
        )

        official = to_official_format(task)

        assert official["task_id"] == "test_001"
        assert official["solution"] == "Answer"  # First solution as string
        assert "id" not in official  # 'id' not in official format

    def test_to_custom_format(self, custom_task_data):
        """Test converting MedAgentBenchTask to custom format."""
        task = MedAgentBenchTask(
            task_id="test_002",
            instruction="Test instruction",
            eval_MRN="patient-002",
            sol=["Answer 1", "Answer 2"],
        )

        custom = to_custom_format(task)

        assert custom["id"] == "test_002"  # 'task_id' mapped to 'id'
        assert custom["sol"] == ["Answer 1", "Answer 2"]  # Full array
        assert "task_id" not in custom  # 'task_id' not in custom format


# ============================================================================
# Official Export Tests
# ============================================================================

class TestOfficialExport:
    """Tests for official format export."""

    def test_convert_evaluation_to_official(self, sample_evaluation_result):
        """Test converting MedicalEvaluationResult to official format."""
        official = convert_evaluation_to_official(
            sample_evaluation_result,
            model="test-model"
        )

        assert official.task_id == "test_001"
        assert official.model == "test-model"
        assert official.score == 49.0
        assert "criteria" in official.details
        assert official.details["rubric_type"] == "diabetes"

    def test_save_overall_json(self, sample_evaluation_result, tmp_path):
        """Test saving overall.json in official format."""
        official = convert_evaluation_to_official(
            sample_evaluation_result,
            model="test-model"
        )

        output_path = save_overall_json(
            output_dir=tmp_path,
            model="test-model",
            task_id="test_001",
            overall_result=official,
        )

        # Verify file was created
        assert output_path.exists()
        assert output_path.name == "overall.json"

        # Verify directory structure
        assert "test-model" in str(output_path)
        assert "test_001" in str(output_path)

        # Verify content
        content = json.loads(output_path.read_text())
        assert content["task_id"] == "test_001"
        assert content["model"] == "test-model"
        assert content["score"] == 49.0

    def test_export_evaluation(self, sample_evaluation_result, tmp_path):
        """Test the convenience export_evaluation function."""
        output_path = export_evaluation(
            sample_evaluation_result,
            model="test-model",
            output_dir=tmp_path,
        )

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert content["task_id"] == "test_001"

    def test_save_summary(self, sample_evaluation_result, tmp_path):
        """Test saving summary.json."""
        summary_path = save_summary(
            [sample_evaluation_result],
            model="test-model",
            output_dir=tmp_path,
        )

        assert summary_path.exists()
        assert summary_path.name == "summary.json"

        content = json.loads(summary_path.read_text())
        assert content["model"] == "test-model"
        assert content["total_evaluations"] == 1
        assert content["avg_score"] == 49.0


# ============================================================================
# Reference Solution Tests
# ============================================================================

class TestReferenceSolution:
    """Tests for reference solution integration."""

    def test_mock_provider(self, sample_evaluation_result):
        """Test the mock reference solution provider."""
        tasks = [MedAgentBenchTask(
            task_id="test_001",
            instruction="Test",
            sol=["Expected solution"],
        )]

        provider = MockReferenceSolutionProvider(tasks)

        assert provider.has_solution("test_001")
        assert provider.get_solution("test_001") == "Expected solution"
        assert not provider.has_solution("nonexistent")

    def test_mock_provider_compare(self, sample_evaluation_result):
        """Test comparison using mock provider."""
        tasks = [MedAgentBenchTask(
            task_id="test_001",
            instruction="Test",
            sol=["diabetes medication metformin"],
        )]

        provider = MockReferenceSolutionProvider(tasks)

        result = provider.compare("test_001", "The patient should take metformin for diabetes")

        assert result["has_reference"] is True
        assert "accuracy" in result
        assert result["method"] == "keyword_overlap"

    def test_cached_provider(self, tmp_path):
        """Test the cached reference solution provider."""
        cache_file = tmp_path / "solutions.json"
        cache_data = {
            "task_001": "Solution for task 001",
            "task_002": "Solution for task 002",
        }
        cache_file.write_text(json.dumps(cache_data))

        provider = CachedReferenceSolutionProvider(cache_file)

        assert provider.has_solution("task_001")
        assert provider.get_solution("task_001") == "Solution for task 001"

    def test_provider_factory(self):
        """Test the provider factory function."""
        tasks = [MedAgentBenchTask(
            task_id="test_001",
            instruction="Test",
            sol=["Solution"],
        )]

        mock_provider = get_reference_provider("mock", tasks=tasks)
        assert isinstance(mock_provider, MockReferenceSolutionProvider)


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Tests for compatibility validation."""

    def test_validate_compatible_task(self):
        """Test validating a compatible task."""
        task = MedAgentBenchTask(
            task_id="test_001",
            instruction="Test instruction",
            sol=["Solution"],
        )

        is_valid, message = validate_task_compatibility(task)

        assert is_valid is True
        assert message == "ok"

    def test_validate_incompatible_task_multiple_solutions(self):
        """Test validating a task with multiple solutions (incompatible with official)."""
        task = MedAgentBenchTask(
            task_id="test_001",
            instruction="Test instruction",
            sol=["Solution 1", "Solution 2"],  # Multiple solutions
        )

        is_valid, message = validate_task_compatibility(task)

        assert is_valid is False
        assert "expects single solution" in message

    def test_validate_incompatible_task_no_id(self):
        """Test validating a task without task_id."""
        # This would fail at model validation, so we can't create it
        # Just demonstrate the validation logic exists
        pass


# ============================================================================
# End-to-End Compatibility Tests
# ============================================================================

class TestEndToEndCompatibility:
    """End-to-end tests for format compatibility."""

    def test_official_format_e2e(self, tmp_path):
        """Test complete workflow with official format."""
        # Create input file in official format
        input_file = tmp_path / "input.json"
        input_data = [{
            "task_id": "e2e_test_001",
            "instruction": "Test instruction",
            "eval_MRN": "patient-001",
            "solution": "Reference answer",
        }]
        input_file.write_text(json.dumps(input_data))

        # Load tasks
        tasks = load_tasks(str(input_file))
        assert len(tasks) == 1
        assert tasks[0].format_type == TaskFormatType.OFFICIAL

        # Create evaluation result
        evaluation = MedicalEvaluationResult(
            task_id=tasks[0].task_id,
            medical_category="general_medical",
            agent_name="test_agent",
            agent_response="Test response",
            diabetes_score=None,
            general_score=None,
            total_score=25.0,
            feedback="Test feedback",
        )

        # Export to official format
        output_path = export_evaluation(
            evaluation,
            model="e2e-test",
            output_dir=tmp_path,
        )

        # Verify output
        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert content["task_id"] == "e2e_test_001"
        assert "e2e-test" in str(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
