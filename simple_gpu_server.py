#!/usr/bin/env python3
"""
Simple GPU Server - Direct GPU inference without blockchain
"""
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Global model
model = None
tokenizer = None

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, tokenizer
    
    print("Loading model...")
    # Use GPT-2 for fast testing or Qwen3 for full model
    model_name = os.environ.get("MODEL_NAME", "gpt2")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch.cuda.is_available():
        print(f"Loading on GPU: {torch.cuda.get_device_name(0)}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
    else:
        print("Loading on CPU (will be slow)")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"✅ Model {model_name} loaded!")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Generate response using GPU."""
    if not model:
        return {"error": "Model not loaded"}
    
    # Tokenize
    inputs = tokenizer(request.prompt, return_tensors="pt")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from response
    if response.startswith(request.prompt):
        response = response[len(request.prompt):].strip()
    
    return {
        "prompt": request.prompt,
        "response": response,
        "gpu_used": torch.cuda.is_available(),
        "model": model.config.name_or_path
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)