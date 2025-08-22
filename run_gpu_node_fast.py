#!/usr/bin/env python3
"""
Fast GPU Node - Optimized for RunPod with proper GPU handling
Skips blockchain sync and focuses on model serving
"""
import os
import sys
import torch
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import model configuration
try:
    from config.model_profile import MODEL_ID, MODEL_NAME, COMPUTE, PRECISION
    DEFAULT_MODEL = MODEL_ID  # Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
except ImportError:
    DEFAULT_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'
    logger.warning("Model profile not found, using default Qwen3-30B")

# Configuration
PORT = int(os.environ.get('NODE_PORT', 8002))
SKIP_BLOCKCHAIN = os.environ.get('SKIP_BLOCKCHAIN', 'true').lower() == 'true'
USE_SMALL_MODEL = os.environ.get('USE_SMALL_MODEL', 'false').lower() == 'true'
MODEL_NAME = os.environ.get('MODEL_NAME', 'gpt2' if USE_SMALL_MODEL else DEFAULT_MODEL)

# FastAPI app
app = FastAPI(title="Blyan GPU Node", version="1.0.0")

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    use_moe: bool = False

class GPUNode:
    """Optimized GPU node for fast startup and inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.gpu_info = {}
        self.model_loaded = False
        self.loading_progress = 0
        self.start_time = datetime.now()
        
    def check_gpu(self) -> bool:
        """Check GPU availability with detailed info."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA not available")
                self.gpu_info = {"available": False, "error": "CUDA not available"}
                return False
            
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                logger.warning("No GPUs detected")
                self.gpu_info = {"available": False, "error": "No GPUs found"}
                return False
            
            # Get GPU details
            gpu_props = torch.cuda.get_device_properties(0)
            self.gpu_info = {
                "available": True,
                "count": num_gpus,
                "name": gpu_props.name,
                "memory_gb": gpu_props.total_memory / 1e9,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_free_gb": (gpu_props.total_memory - torch.cuda.memory_allocated()) / 1e9
            }
            
            logger.info(f"✅ GPU detected: {gpu_props.name} with {gpu_props.total_memory/1e9:.1f} GB memory")
            
            # Test memory allocation
            try:
                test_tensor = torch.zeros(1024, 1024, dtype=torch.float16, device='cuda:0')
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("✅ GPU memory allocation test passed")
            except Exception as e:
                logger.error(f"GPU memory allocation failed: {e}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            self.gpu_info = {"available": False, "error": str(e)}
            return False
    
    async def load_model_async(self):
        """Load model asynchronously with progress tracking."""
        try:
            if self.model_loaded:
                logger.info("Model already loaded")
                return True
            
            logger.info(f"Loading model: {MODEL_NAME}")
            self.loading_progress = 10
            
            # Check if we should use small model for testing
            if USE_SMALL_MODEL or not self.gpu_info.get("available"):
                logger.info("Using small GPT-2 model for testing")
                model_name = "gpt2"
            else:
                model_name = MODEL_NAME
            
            # Import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.loading_progress = 30
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimal settings
            logger.info("Loading model weights...")
            self.loading_progress = 50
            
            if self.gpu_info.get("available"):
                # GPU loading with optimizations
                # For Qwen3-30B, use auto dtype to preserve FP8 precision
                if "Qwen3-30B" in model_name or "FP8" in model_name:
                    torch_dtype = "auto"  # Preserve FP8
                    logger.info("Using FP8 precision for Qwen3-30B model")
                else:
                    torch_dtype = torch.float16  # FP16 for other models
                    
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",  # Automatically distribute across GPUs
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info(f"✅ Model loaded on GPU: {self.gpu_info['name']}")
            else:
                # CPU fallback
                logger.warning("Loading model on CPU (will be slow)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            self.loading_progress = 90
            
            # Test inference
            logger.info("Testing model inference...")
            test_input = self.tokenizer("Hello", return_tensors="pt")
            if self.gpu_info.get("available"):
                test_input = {k: v.to("cuda") for k, v in test_input.items()}
            
            with torch.no_grad():
                _ = self.model.generate(**test_input, max_new_tokens=5)
            
            self.loading_progress = 100
            self.model_loaded = True
            
            # Log memory usage
            if self.gpu_info.get("available"):
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
            
            logger.info("✅ Model loaded and ready!")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.loading_progress = -1  # Indicate failure
            return False
    
    async def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7):
        """Generate text using the loaded model."""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to GPU if available
            if self.gpu_info.get("available"):
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Create global node instance
node = GPUNode()

@app.on_event("startup")
async def startup():
    """Initialize node on startup."""
    logger.info("🚀 Starting Blyan GPU Node (Fast Mode)")
    
    # Check GPU
    gpu_available = node.check_gpu()
    
    if gpu_available:
        logger.info(f"GPU: {node.gpu_info['name']} ({node.gpu_info['memory_gb']:.1f} GB)")
    else:
        logger.warning("No GPU available - will use CPU (slow)")
    
    # Start model loading in background
    if not SKIP_BLOCKCHAIN:
        asyncio.create_task(node.load_model_async())
        logger.info("Model loading started in background...")
    else:
        logger.info("Skipping model load (SKIP_BLOCKCHAIN=true)")

@app.get("/health")
async def health():
    """Health check endpoint."""
    uptime = (datetime.now() - node.start_time).total_seconds()
    return JSONResponse({
        "status": "healthy",
        "uptime_seconds": uptime,
        "model_loaded": node.model_loaded,
        "loading_progress": node.loading_progress,
        "gpu": node.gpu_info,
        "model_name": MODEL_NAME if node.model_loaded else None
    })

@app.get("/gpu-status")
async def gpu_status():
    """Detailed GPU status."""
    if not node.gpu_info.get("available"):
        return JSONResponse({"error": "No GPU available", "info": node.gpu_info})
    
    # Update current memory usage
    torch.cuda.synchronize()
    current_info = {
        **node.gpu_info,
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9,
        "model_loaded": node.model_loaded
    }
    
    return JSONResponse(current_info)

@app.post("/load-model")
async def load_model():
    """Manually trigger model loading."""
    if node.model_loaded:
        return JSONResponse({"status": "already_loaded"})
    
    # Start loading
    asyncio.create_task(node.load_model_async())
    return JSONResponse({"status": "loading_started", "check_progress": "/health"})

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint for text generation."""
    if not node.model_loaded:
        # Try to load model if not loaded
        if node.loading_progress == 0:
            asyncio.create_task(node.load_model_async())
            return JSONResponse(
                {"error": "Model loading in progress, please try again in a few moments"},
                status_code=503
            )
        elif node.loading_progress == -1:
            return JSONResponse(
                {"error": "Model failed to load"},
                status_code=500
            )
        else:
            return JSONResponse(
                {"error": f"Model loading {node.loading_progress}% complete"},
                status_code=503
            )
    
    # Generate response
    try:
        response = await node.generate(
            request.prompt,
            request.max_new_tokens,
            request.temperature
        )
        
        # Get memory usage
        mem_used = torch.cuda.memory_allocated() / 1e9 if node.gpu_info.get("available") else 0
        
        return JSONResponse({
            "prompt": request.prompt,
            "response": response,
            "model": MODEL_NAME,
            "gpu_memory_used_gb": mem_used
        })
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/debug/moe-status")
async def moe_status():
    """MoE status endpoint for compatibility."""
    return JSONResponse({
        "moe_model_manager_initialized": False,
        "blockchain_mode": False,
        "fast_mode": True,
        "model_loaded": node.model_loaded,
        "model_name": MODEL_NAME if node.model_loaded else None,
        "gpu_available": node.gpu_info.get("available", False)
    })

if __name__ == "__main__":
    logger.info(f"Starting server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)