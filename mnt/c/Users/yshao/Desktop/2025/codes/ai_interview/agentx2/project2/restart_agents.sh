#!/bin/bash
set -e

echo "Clearing old agents..."
docker stop medbench-judge medbench-medical 2>/dev/null || true
docker rm medbench-judge medbench-medical 2>/dev/null || true

echo "Starting green agent on port 9008..."
docker run -d -p 9008:9008 \
  --name medbench-judge \
  -v /app/data:/app/data:ro \
  --env_file .env \
  medbench-judge:latest

echo "Starting purple agent on port 9010..."
docker run -d -p 9010:9010 \
  -e SPECIALTY=diabetes \
  --name medbench-medical \
  medbench-medical:latest

echo ""
echo "Waiting for agents to start..."
sleep 10

echo "Checking agent status..."
docker ps | grep medbench

echo ""
echo "Agent URLs:"
echo "  Green Agent: http://localhost:9008"
echo "  Purple Agent: http://localhost:9010"
