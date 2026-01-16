#!/bin/bash
# Publish Docker images to GitHub Container Registry (GHCR)

set -e

REPO="ghcr.io/yshao/medapp"
VERSION=${1:-latest}

echo "=================================="
echo "Publishing to GHCR: $REPO:$VERSION"
echo "=================================="

# Build green agent image
echo "Building green agent (MedBenchJudge)..."
docker build -t ${REPO}:green-agent-$VERSION -f Dockerfile .

# Build purple agent image
echo "Building purple agent (Medical)..."
docker build -t ${REPO}:purple-agent-$VERSION -f Dockerfile.medical .

# Login to GHCR
echo "Logging in to GHCR..."
echo $GITHUB_TOKEN | docker login ghcr.io -u yshao --password-stdin

# Push images
echo "Pushing green agent..."
docker push ${REPO}:green-agent-$VERSION

echo "Pushing purple agent..."
docker push ${REPO}:purple-agent-$VERSION

echo ""
echo "=================================="
echo "✓ Images published successfully!"
echo "=================================="
echo ""
echo "Green Agent:"
echo "  docker pull ${REPO}:green-agent-$VERSION"
echo "  docker run -p 9008:9008 ${REPO}:green-agent-$VERSION"
echo ""
echo "Purple Agent:"
echo "  docker pull ${REPO}:purple-agent-$VERSION"
echo "  docker run -p 9010:9010 ${REPO}:purple-agent-$VERSION"
echo ""
