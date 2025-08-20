#!/bin/bash

# Build and push Blyan Docker images
# Usage: ./docker/build.sh [--push] [--tag TAG]

set -e

# Configuration
REGISTRY="docker.io"
NAMESPACE="mnls0115"
IMAGE_NAME="blyan-node"
DEFAULT_TAG="latest"

# Parse arguments
PUSH=false
TAG="$DEFAULT_TAG"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--push] [--tag TAG]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building Blyan Docker images..."
echo "Registry: $REGISTRY"
echo "Namespace: $NAMESPACE"
echo "Image: $IMAGE_NAME"
echo "Tag: $TAG"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Build GPU node image
echo "Building GPU node image..."
docker build \
    -f docker/Dockerfile.gpu \
    -t "$NAMESPACE/$IMAGE_NAME:$TAG" \
    -t "$NAMESPACE/$IMAGE_NAME:gpu-$TAG" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .

echo "✅ GPU image built successfully"

# Build CPU-only image (lighter weight)
echo ""
echo "Building CPU-only image..."
docker build \
    -f docker/Dockerfile.cpu \
    -t "$NAMESPACE/$IMAGE_NAME:cpu-$TAG" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    . 2>/dev/null || {
        echo "⚠️  CPU Dockerfile not found, skipping CPU build"
    }

# Tag with version if we're on a git tag
if git describe --exact-match --tags 2>/dev/null; then
    VERSION=$(git describe --exact-match --tags)
    echo ""
    echo "Tagging with version: $VERSION"
    docker tag "$NAMESPACE/$IMAGE_NAME:$TAG" "$NAMESPACE/$IMAGE_NAME:$VERSION"
    docker tag "$NAMESPACE/$IMAGE_NAME:gpu-$TAG" "$NAMESPACE/$IMAGE_NAME:gpu-$VERSION"
fi

# List built images
echo ""
echo "Built images:"
docker images | grep "$NAMESPACE/$IMAGE_NAME" | head -5

# Push if requested
if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing images to registry..."
    
    # Login check
    if ! docker info 2>/dev/null | grep -q "Username"; then
        echo "Please login to Docker Hub first:"
        echo "  docker login"
        exit 1
    fi
    
    # Push all tags
    docker push "$NAMESPACE/$IMAGE_NAME:$TAG"
    docker push "$NAMESPACE/$IMAGE_NAME:gpu-$TAG"
    
    if [ -n "$VERSION" ]; then
        docker push "$NAMESPACE/$IMAGE_NAME:$VERSION"
        docker push "$NAMESPACE/$IMAGE_NAME:gpu-$VERSION"
    fi
    
    echo "✅ Images pushed successfully"
fi

echo ""
echo "Done! To run the node:"
echo ""
echo "  # With docker-compose:"
echo "  JOIN_CODE=YOUR_CODE docker-compose up -d"
echo ""
echo "  # With docker run:"
echo "  docker run -d --name blyan-node \\"
echo "    --gpus all \\"
echo "    -p 8001:8001 \\"
echo "    -v /var/lib/blyan/data:/data \\"
echo "    -e JOIN_CODE=YOUR_CODE \\"
echo "    $NAMESPACE/$IMAGE_NAME:$TAG"