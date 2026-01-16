"""
Run MedAgentBench Benchmark - Main Execution Script

This script orchestrates running the MedAgentBench benchmark by:
1. Starting the green agent (medbench_judge)
2. Starting the purple agent (medical_agent)
3. Waiting for agents to be ready
4. Running the evaluation via A2A client
5.5. Collecting and displaying results

Usage:
    python run_benchmark.py --config config/scenario.toml
    python run_benchmark.py --config config/scenario.toml --show-logs
    python run_benchmark.py --config config/scenario.toml --serve-only
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add project2 root and src to path for imports
project_root = Path(__file__).parent  # project2/ directory
src_path = project_root / "src"  # agentbeats is in src/

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import tomllib
except ImportError:
    # Python < 3.11
    import tomli as tomllib
import httpx
from dotenv import load_dotenv

from a2a.client import A2ACardResolver

load_dotenv(override=True)


async def wait_for_agents(cfg: dict, timeout: int = 60) -> bool:
    """
    Wait for all agents to be healthy and responding.

    Args:
        cfg: Parsed scenario configuration
        timeout: Maximum time to wait in seconds

    Returns:
        True if all agents are ready, False otherwise
    """
    endpoints = []

    # Collect green agent endpoint
    green_ep = cfg["green_agent"].get("endpoint", "")
    if green_ep:
        endpoints.append(green_ep)

    # Collect participant endpoints
    for p in cfg.get("participants", []):
        ep = p.get("endpoint", "")
        if ep:
            endpoints.append(ep)

    if not endpoints:
        return True  # No endpoints to check

    print(f"Waiting for {len(endpoints)} agent(s) to be ready...")
    start_time = time.time()

    async def check_endpoint(endpoint: str) -> bool:
        """Check if an endpoint is responding by checking health endpoint first, then agent card."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                # First try the /health endpoint (more reliable)
                response = await client.get(f"{endpoint}/health", timeout=2)
                if response.status_code == 200:
                    return True

                # Fallback to agent card for A2A protocol
                from agentbeats.client import A2ACardResolver
                resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
                await resolver.get_agent_card()
                return True
        except Exception as e:
            # Any exception means the agent is not ready yet
            return False

    while time.time() - start_time < timeout:
        ready_count = 0
        for endpoint in endpoints:
            if await check_endpoint(endpoint):
                ready_count += 1

        if ready_count == len(endpoints):
            print(f"All {len(endpoints)} agents ready!")
            return True

        print(f"  {ready_count}/{len(endpoints)} agents ready, waiting...")
        await asyncio.sleep(2)

    print(f"Timeout: Only {ready_count}/{len(endpoints)} agents became ready after {timeout}s")
    return False


def parse_toml(scenario_path: str) -> dict:
    """
    Parse scenario TOML configuration file.

    Args:
        scenario_path: Path to scenario.toml file

    Returns:
        Parsed configuration dictionary
    """
    path = Path(scenario_path)
    if not path.exists():
        print(f"Error: Scenario file not found: {path}")
        sys.exit(1)

    data = tomllib.loads(path.read_text())

    # Extract configuration
    green_agent = data.get("green_agent", {})
    participants = data.get("participants", [])
    config = data.get("config", {})

    return {
        "green_agent": green_agent,
        "participants": participants,
        "config": config,
        "raw": data,
    }


def run_evaluation(cfg: dict, show_logs: bool = False, dry_run: bool = False):
    """
    Run the evaluation by calling the A2A client.

    Args:
        cfg: Parsed scenario configuration
        show_logs: Whether to show agent logs
    """
    # Import client modules after path setup
    from agentbeats.client import send_message
    from agentbeats.models import EvalRequest
    from a2a.types import (
        AgentCard,
        Message,
        TaskStatusUpdateEvent,
        TaskArtifactUpdateEvent,
        Artifact,
        TextPart,
        DataPart,
    )

    # Build EvalRequest
    participants = {}
    for p in cfg["participants"]:
        role = p.get("role")
        endpoint = p.get("endpoint")
        if role and endpoint:
            participants[role] = endpoint

    # Add dry_run flag to config for the green agent
    eval_config = cfg["config"].copy()
    if dry_run:
        eval_config["dry_run"] = True
        print("============================================================")
        print("DRY-RUN MODE: Using mock scores instead of LLM evaluation")
        print("============================================================\n")

    eval_req = EvalRequest(
        participants=participants,
        config=eval_config
    )

    green_url = cfg["green_agent"].get("endpoint", "")
    artifacts = []

    def parse_parts(parts) -> tuple[list, list]:
        """Parse message parts into text and data parts."""
        text_parts = []
        data_parts = []

        for part in parts:
            if isinstance(part.root, TextPart):
                try:
                    import json
                    data_item = json.loads(part.root.text)
                    data_parts.append(data_item)
                except Exception:
                    text_parts.append(part.root.text.strip())
            elif isinstance(part.root, DataPart):
                data_parts.append(part.root.data)

        return text_parts, data_parts

    def print_parts(parts, task_state: str = None):
        """Print message parts."""
        text_parts, data_parts = parse_parts(parts)

        output = []
        if task_state:
            output.append(f"[Status: {task_state}]")
        if text_parts:
            output.append("\n".join(text_parts))
        if data_parts:
            import json
            output.extend(json.dumps(item, indent=2) for item in data_parts)

        print("\n".join(output) + "\n")

    async def event_consumer(event, card: AgentCard):
        """Consume events from the A2A stream."""
        nonlocal artifacts

        match event:
            case Message() as msg:
                print_parts(msg.parts)

            case (task, TaskStatusUpdateEvent() as status_event):
                status = status_event.status
                parts = status.message.parts if status.message else []
                print_parts(parts, status.state.value)
                if status.state.value == "completed":
                    artifacts = task.artifacts
                elif status.state.value not in ["submitted", "working"]:
                    print(f"Agent returned status {status.state.value}. Exiting.")
                    sys.exit(1)

            case (task, TaskArtifactUpdateEvent() as artifact_event):
                print_parts(artifact_event.artifact.parts, "Artifact update")

            case task, None:
                status = task.status
                parts = status.message.parts if status.message else []
                print_parts(parts, task.status.state.value)
                if status.state.value == "completed":
                    artifacts = task.artifacts
                elif status.state.value not in ["submitted", "working"]:
                    print(f"Agent returned status {status.state.value}. Exiting.")
                    sys.exit(1)

            case _:
                print("Unhandled event")

    async def do_send():
        msg = eval_req.model_dump_json()
        await send_message(msg, green_url, streaming=True, consumer=event_consumer)

    asyncio.run(do_send())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run MedAgentBench benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config/scenario.toml",
        help="Path to scenario TOML file (default: config/scenario.toml)"
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show agent stdout/stderr"
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start agent servers only without running evaluation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM evaluation (use mock scores for testing)"
    )
    parser.add_argument(
        "--official-format",
        action="store_true",
        help="Enable official MedAgentBench format export (overall.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for official format (default: outputs/)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="agentx-medical",
        help="Model name for output directory (default: agentx-medical)"
    )
    args = parser.parse_args()

    # Parse configuration
    with open(args.config, 'rb') as f:
        cfg = tomllib.load(f)

    # Setup environment for subprocesses
    parent_bin = str(Path(sys.executable).parent)
    base_env = os.environ.copy()
    base_env["PATH"] = parent_bin + os.pathsep + base_env.get("PATH", "")
    base_env["PYTHONPATH"] = str(project_root) + os.pathsep + str(src_path) + os.pathsep + base_env.get("PYTHONPATH", "")

    sink = None if args.show_logs or args.serve_only else subprocess.DEVNULL

    procs = []
    try:
        # Start participant agents
        for p in cfg.get("participants", []):
            cmd = p.get("cmd")
            if cmd:
                import shlex
                cmd_args = shlex.split(cmd)
                role = p.get("role", "unknown")
                print(f"Starting {role} at {p.get('endpoint', '')}")
                procs.append(subprocess.Popen(
                    cmd_args,
                    env=base_env,
                    stdout=sink,
                    stderr=subprocess.PIPE if not args.show_logs else sink,
                    text=True,
                    start_new_session=True,
                ))

        # Start green agent
        green_cmd = cfg["green_agent"].get("cmd")
        if green_cmd:
            import shlex
            cmd_args = shlex.split(green_cmd)

            # Add official format flags if specified
            if args.official_format:
                cmd_args.extend(["--export-official"])
                cmd_args.extend(["--output-dir", args.output_dir])
                cmd_args.extend(["--model-name", args.model_name])
                print(f"Official format export enabled: {args.output_dir}/{args.model_name}/")

            print(f"Starting green agent at {cfg['green_agent'].get('endpoint', '')}")
            procs.append(subprocess.Popen(
                cmd_args,
                env=base_env,
                stdout=sink,
                stderr=subprocess.PIPE if not args.show_logs else sink,
                text=True,
                start_new_session=True,
            ))

        # Wait for agents to be ready
        if not asyncio.run(wait_for_agents(cfg)):
            print("Error: Not all agents became ready. Exiting.")
            return

        print("=" * 60)
        print("Agents started successfully!")
        print("=" * 60)

        if args.serve_only:
            print("Running in serve-only mode. Press Ctrl+C to stop.")
            try:
                while True:
                    for proc in procs:
                        if proc.poll() is not None:
                            print(f"Agent exited with code {proc.returncode}")
                            return
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        else:
            print("Starting benchmark evaluation...")
            print("=" * 60)
            run_evaluation(cfg, args.show_logs, args.dry_run)
            print("=" * 60)
            print("Benchmark evaluation complete!")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nShutting down agents...")
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        time.sleep(1)

        # Force kill if still running
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        print("Shutdown complete.")


if __name__ == "__main__":
    main()
