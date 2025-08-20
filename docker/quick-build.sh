#!/bin/bash

# Quick build and push script for Blyan Docker image
# This creates a lighter image without full CUDA runtime

echo "Building Blyan Node Docker image..."

# Use buildx for better performance (if available)
if docker buildx version &>/dev/null; then
    echo "Using Docker Buildx..."
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -f docker/Dockerfile.gpu \
        -t mnls0115/blyan-node:latest \
        --push \
        .
else
    echo "Building for current platform..."
    # Build the image
    docker build -f docker/Dockerfile.gpu -t mnls0115/blyan-node:latest .
    
    # Push to Docker Hub
    echo ""
    echo "Pushing to Docker Hub..."
    echo "Make sure you're logged in: docker login"
    docker push mnls0115/blyan-node:latest
fi

echo ""
echo "Done! Users can now run:"
echo "  docker pull mnls0115/blyan-node:latest"