#!/usr/bin/env python3
"""
Runpod GPU Node for Blyan Network
Runs as a P2P node that connects to DigitalOcean main server
"""

import os
import sys
import asyncio
import aiohttp
from pathlib import Path
import torch
import time
from typing import Optional, Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Import our model wrapper
from backend.model.arch import ModelWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Blyan GPU Node (Runpod)")

# Configuration
MAIN_SERVER_URL = os.getenv("MAIN_SERVER_URL", "https://your-digitalocean-server.com")
NODE_ID = os.getenv("NODE_ID", f"runpod_a40_{os.getpid()}")
NODE_HOST = os.getenv("NODE_HOST", "0.0.0.0")  # Runpod will provide external IP
NODE_PORT = int(os.getenv("NODE_PORT", 8000))

# Global model instance
model_wrapper: Optional[ModelWrapper] = None
registration_task = None

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    expert_name: Optional[str] = None

class InferenceResponse(BaseModel):
    text: str
    tokens_generated: int
    inference_time_ms: float
    node_id: str
    model_name: str

async def register_with_main_server():
    """Register this node with the main DigitalOcean server."""
    while True:
        try:
            # Get public IP (Runpod provides this)
            public_ip = os.getenv("RUNPOD_PUBLIC_IP", NODE_HOST)
            
            registration_data = {
                "node_id": NODE_ID,
                "host": public_ip,
                "port": NODE_PORT,
                "available_experts": ["gpt-neox-20b"],  # What this node can serve
                "capabilities": {
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                    "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                    "quantization": os.getenv("MODEL_QUANTIZATION", "8bit"),
                    "max_batch_size": 4
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{MAIN_SERVER_URL}/p2p/register",
                    json=registration_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Registered with main server: {result}")
                    else:
                        logger.error(f"Registration failed: {response.status}")
            
            # Send heartbeat every 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            await asyncio.sleep(10)  # Retry after 10 seconds

@app.on_event("startup")
async def startup_event():
    """Load model and register with main server on startup."""
    global model_wrapper, registration_task
    
    # Load the model
    model_name = os.getenv("MODEL_NAME", "EleutherAI/gpt-neox-20b")
    quantization = os.getenv("MODEL_QUANTIZATION", "8bit")
    
    logger.info(f"🚀 Starting Blyan GPU Node: {NODE_ID}")
    logger.info(f"📦 Loading model: {model_name}")
    logger.info(f"🔧 Quantization: {quantization}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.warning("⚠️ No GPU detected!")
    
    # Load model
    try:
        load_in_8bit = quantization == "8bit"
        load_in_4bit = quantization == "4bit"
        
        model_wrapper = ModelWrapper(
            model_name=model_name,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            max_memory={0: "40GB"}  # A40 has 48GB
        )
        
        logger.info(f"✅ Model loaded successfully!")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"💾 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Start registration task
    registration_task = asyncio.create_task(register_with_main_server())
    logger.info(f"📡 Started registration with main server: {MAIN_SERVER_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    """Unregister from main server on shutdown."""
    global registration_task
    
    if registration_task:
        registration_task.cancel()
    
    try:
        async with aiohttp.ClientSession() as session:
            await session.delete(
                f"{MAIN_SERVER_URL}/p2p/nodes/{NODE_ID}",
                timeout=aiohttp.ClientTimeout(total=5)
            )
        logger.info(f"✅ Unregistered from main server")
    except Exception as e:
        logger.error(f"Failed to unregister: {e}")

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node_id": NODE_ID,
        "model_loaded": model_wrapper is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
        "connected_to": MAIN_SERVER_URL
    }

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on this GPU node."""
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
            node_id=NODE_ID,
            model_name=model_wrapper.model_name
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/expert/{expert_name}/infer")
async def expert_inference(expert_name: str, request: InferenceRequest):
    """
    Expert-specific inference endpoint for P2P network.
    Compatible with Blyan's distributed inference system.
    """
    # For now, we serve the full model as one "expert"
    if expert_name == "gpt-neox-20b" or expert_name in ["layer0.expert0", "full_model"]:
        return await run_inference(request)
    else:
        raise HTTPException(status_code=404, detail=f"Expert {expert_name} not available on this node")

@app.get("/metrics")
async def get_metrics():
    """Get node metrics."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available"}
    
    return {
        "node_id": NODE_ID,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9,
        "model_loaded": model_wrapper is not None,
        "uptime_seconds": time.time()
    }

if __name__ == "__main__":
    # Run the node server
    uvicorn.run(app, host=NODE_HOST, port=NODE_PORT)