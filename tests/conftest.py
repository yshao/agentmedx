"""
Pytest configuration for MedAgentBench tests.
"""

def pytest_addoption(parser):
    """Add command-line option for agent URL."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://127.0.0.1:9008",
        help="URL of the running agent to test",
    )
