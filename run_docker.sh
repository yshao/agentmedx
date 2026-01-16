#!/bin/bash
# Docker run script that loads .env file

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if required API keys are set
if [ -z "$GROQ_API_KEY" ] || [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GROQ_API_KEY and GOOGLE_API_KEY must be set in .env file or environment"
    echo "Please create a .env file with:"
    echo "  GROQ_API_KEY=your_key_here"
    echo "  GOOGLE_API_KEY=your_key_here"
    exit 1
fi

# Run the container
docker run --rm \
    -e GROQ_API_KEY="$GROQ_API_KEY" \
    -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
    -v "$(pwd)/outputs:/app/outputs" \
    -p 9008:9008 \
    -p 9010:9010 \
    medbench-standalone
