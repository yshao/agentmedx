# Green Agent (MedBenchJudge) Docker Image
FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Google private packages
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy only the essential tutorial files needed
RUN mkdir -p tutorial/src/agentbeats
COPY tutorial/src/agentbeats/ tutorial/src/agentbeats/

# Copy requirements first for filtering
COPY requirements.txt .

# Filter out a2a-sdk, google-adk, and earthshaker from requirements.txt (installed via uv)
RUN grep -vE '^(a2a-sdk|google-adk|earthshaker)' requirements.txt > /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    uv pip install --system --no-cache a2a-sdk google-adk earthshaker && \
    rm /tmp/requirements.txt

# Copy scenario code
COPY scenarios/medbench/ ./scenarios/medbench/

# Copy data files
COPY data/ ./data/

# Copy A2A-compatible run script
COPY scenarios/medbench/run_a2a.sh /app/run.sh
RUN chmod +x /app/run.sh

# Set Python path
ENV PYTHONPATH=/app:/app/tutorial/src

# Expose port
EXPOSE 9008

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9008/health || exit 1

# AgentBeats integration: use run.sh as entrypoint
# The run.sh script reads environment variables (HOST, AGENT_PORT, etc.)
# and constructs the appropriate command to start the agent.
# CMD can be overridden to provide custom arguments.
ENTRYPOINT ["/app/run.sh"]
CMD []
