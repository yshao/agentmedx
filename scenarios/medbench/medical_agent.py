"""
Medical Agent - Purple Agent for MedAgentBench

This module implements the purple agent (medical AI participant) using Google ADK
and converts it to an A2A-compatible server.
"""

import argparse
import logging
import asyncio

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCard, AgentCapabilities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_agent")


def _get_specialty_instructions(specialty: str) -> str:
    """Get specialty-specific instructions for the medical agent."""
    instructions = {
        "diabetes": """You are an endocrinologist specializing in diabetes management.
Provide comprehensive treatment plans including:
- Medication recommendations with justification
- A1C target goals
- Comorbidity management (hypertension, kidney, lipids)
- Lifestyle modifications (diet, exercise)
- Safety considerations (contraindications, interactions)
- Monitoring and follow-up plans

Your response should be thorough and clinically sound.""",

        "cardiology": """You are a cardiologist specializing in cardiovascular disease.
Provide evidence-based cardiac care including:
- Diagnostic assessment
- Treatment recommendations
- Risk factor modification
- Medication management
- Follow-up planning

Your response should be evidence-based and follow current cardiology guidelines.""",

        "internal_medicine": """You are a board-certified internist.
Provide thorough medical care including:
- Accurate diagnosis
- Comprehensive treatment plans
- Evidence-based recommendations
- Clear explanations

Your response should be thorough and follow internal medicine best practices.""",

        "general": """You are a helpful medical AI assistant.
Provide clear, accurate medical information:
- Accurate diagnosis
- Comprehensive treatment plans
- Evidence-based recommendations
- Clear explanations

Your response should be helpful and medically accurate."""
    }

    return instructions.get(specialty, instructions["general"])


def main():
    parser = argparse.ArgumentParser(description="Medical AI agent for MedAgentBench")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--specialty", type=str, default="general",
                       choices=["diabetes", "cardiology", "internal_medicine", "general"])
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get specialty-specific instructions
    instruction = _get_specialty_instructions(args.specialty)

    # Create Google ADK Agent with Groq
    root_agent = Agent(
        name="medical_agent",
        model="groq/llama-3.3-70b-versatile",
        description=f"Medical AI agent - {args.specialty} specialty",
        instruction=instruction,
    )

    # Create agent card
    # Use localhost as the external URL for host-based benchmark execution
    external_url = args.card_url or 'http://localhost:9010/'
    agent_card = AgentCard(
        name="medical_agent",
        description=f"Medical AI agent - {args.specialty} specialty",
        url=external_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],  # Empty skills list for purple agent
    )

    # Convert to A2A and run
    logger.info(f"Starting medical agent ({args.specialty}) on {args.host}:{args.port}")

    a2a_app = to_a2a(root_agent, agent_card=agent_card)

    # Add health check and root info endpoints using middleware
    from starlette.responses import JSONResponse
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.types import ASGIApp

    class InfoMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Handle GET / and GET /health before passing to A2A handler
            # Let /.well-known/* paths pass through to A2A handler for agent card discovery
            if request.method == "GET":
                if request.url.path.startswith("/.well-known/"):
                    return await call_next(request)
                if request.url.path == "/health":
                    return JSONResponse({
                        "status": "healthy",
                        "service": "medbench-medical",
                        "specialty": args.specialty,
                        "ready": True
                    })
                elif request.url.path == "/":
                    return JSONResponse({
                        "service": "MedBenchMedical - Purple Agent",
                        "description": "Medical AI agent for clinical task evaluation",
                        "specialty": args.specialty,
                        "version": "1.0.0",
                        "protocol": "A2A",
                        "endpoints": {
                            "POST /": "A2A protocol endpoint for agent communication",
                            "GET /health": "Health check endpoint"
                        },
                        "status": "running"
                    })
            return await call_next(request)

    # CORS middleware for SSE streaming (must be before InfoMiddleware)
    a2a_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    a2a_app.add_middleware(InfoMiddleware)

    # Run the server
    import uvicorn
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
