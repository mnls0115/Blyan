#!/usr/bin/env python3
"""
Runpod GPU Server for Blyan Network
Runs on A40 (48GB) to serve 20B models with INT8 quantization
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import torch
import uvicorn
from typing import Optional
import time

# Import our model wrapper
from backend.model.arch import ModelWrapper

app = FastAPI(title="Blyan GPU Server")

# Global model instance
model_wrapper = None

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time_ms: float
    model_name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Use FastAPI lifespan events instead of deprecated on_event."""
    global model_wrapper

    # Get model configuration from environment
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    quantization = os.getenv("MODEL_QUANTIZATION", "8bit")

    print("🚀 Starting Blyan GPU Server on Runpod A40")
    print(f"📦 Loading model: {model_name}")
    print(f"🔧 Quantization: {quantization}")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ No GPU detected!")

    # Load model with appropriate settings for A40
    try:
        load_in_8bit = quantization == "8bit"
        load_in_4bit = quantization == "4bit"

        model_wrapper = ModelWrapper(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map="auto",  # Auto-distribute on available GPUs
            max_memory={0: "40GB"}  # A40 has 48GB, leave some headroom
        )

        print("✅ Model loaded successfully!")

        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"💾 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    try:
        yield
    finally:
        # No special teardown required here
        pass

# Register lifespan handler
app.router.lifespan_context = lifespan

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on the GPU."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Run inference
        generated_text = model_wrapper.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Count tokens (approximate)
        tokens_generated = len(generated_text.split()) - len(request.prompt.split())
        
        return InferenceResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            inference_time_ms=inference_time_ms,
            model_name=model_wrapper.model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/p2p/expert_inference")
async def expert_inference(request: InferenceRequest):
    """
    P2P endpoint for distributed inference.
    Compatible with Blyan's DistributedInferenceCoordinator.
    """
    # Reuse the main inference endpoint
    return await run_inference(request)

@app.get("/metrics")
async def get_metrics():
    """Get GPU metrics."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}
    
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9,
        "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else "N/A"
    }

if __name__ == "__main__":
    # Runpod typically exposes port 8000
    port = int(os.getenv("PORT", 8000))
    
    # For Runpod, bind to all interfaces
    uvicorn.run(app, host="0.0.0.0", port=port)