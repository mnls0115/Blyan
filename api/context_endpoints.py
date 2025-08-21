"""
Context Management API Endpoints
Provides conversation context handling for distributed LLM inference
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from backend.context import conversation_manager, ContextStrategy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context", tags=["context"])

class StartConversationRequest(BaseModel):
    strategy: Optional[str] = "hybrid"  # full_context, kv_cache, hybrid, sliding
    max_context_length: Optional[int] = 4096
    sliding_window_size: Optional[int] = 10

class StartConversationResponse(BaseModel):
    conversation_id: str
    strategy: str
    status: str

class AddTurnRequest(BaseModel):
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None

class GetContextRequest(BaseModel):
    conversation_id: str

class GetContextResponse(BaseModel):
    messages: List[Dict[str, str]]
    strategy_used: str
    token_estimate: int
    has_kv_cache: bool

class ConversationStatsResponse(BaseModel):
    turn_count: int
    total_tokens: int
    strategy: str
    has_kv_cache: bool
    conversation_age: float

@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest):
    """Start a new conversation with specified context strategy"""
    import uuid
    
    conversation_id = str(uuid.uuid4())
    
    # Validate strategy
    try:
        strategy = ContextStrategy(request.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")
    
    # Update manager settings if provided
    if request.max_context_length:
        conversation_manager.max_context_length = request.max_context_length
    if request.sliding_window_size:
        conversation_manager.sliding_window_size = request.sliding_window_size
    
    conversation_manager.strategy = strategy
    conversation_manager.start_conversation(conversation_id)
    
    return StartConversationResponse(
        conversation_id=conversation_id,
        strategy=strategy.value,
        status="created"
    )

@router.post("/add_turn")
async def add_turn(request: AddTurnRequest):
    """Add a new turn to an existing conversation"""
    if request.role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'assistant'")
    
    turn = conversation_manager.add_turn(
        conversation_id=request.conversation_id,
        role=request.role,
        content=request.content,
        metadata=request.metadata
    )
    
    return {
        "status": "added",
        "turn_timestamp": turn.timestamp,
        "token_count": turn.token_count
    }

@router.post("/get_context", response_model=GetContextResponse)
async def get_context(request: GetContextRequest):
    """Get conversation context for model inference"""
    context = conversation_manager.get_context_for_inference(request.conversation_id)
    
    return GetContextResponse(
        messages=context['messages'],
        strategy_used=context['strategy_used'],
        token_estimate=context['token_estimate'],
        has_kv_cache=context['kv_cache'] is not None
    )

@router.get("/stats/{conversation_id}", response_model=ConversationStatsResponse)
async def get_conversation_stats(conversation_id: str):
    """Get statistics about a conversation"""
    stats = conversation_manager.get_conversation_stats(conversation_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationStatsResponse(**stats)

@router.post("/cleanup")
async def cleanup_conversations(max_age_hours: int = 24):
    """Clean up old conversations and caches"""
    cleaned = conversation_manager.cleanup_expired_conversations(max_age_hours)
    return {"status": "cleaned", "conversations_removed": cleaned}

@router.get("/strategies")
async def list_strategies():
    """List available context strategies"""
    return {
        "strategies": {
            "full_context": "Always send complete conversation history",
            "kv_cache": "Use attention key-value cache for efficiency",
            "hybrid": "Combine KV cache with recent turns",
            "sliding": "Keep only recent N turns"
        },
        "current_strategy": conversation_manager.strategy.value,
        "recommendations": {
            "development": "full_context",
            "short_conversations": "sliding",
            "production": "hybrid",
            "long_conversations": "kv_cache"
        }
    }

# Integration with main chat endpoint
class ChatContextResponse(BaseModel):
    response: str
    conversation_id: str
    context_strategy: str
    tokens_processed: int

@router.post("/chat", response_model=ChatContextResponse)
async def chat_with_context(
    message: str,
    conversation_id: Optional[str] = None,
    strategy: Optional[str] = "hybrid"
):
    """
    Chat endpoint with automatic context management
    This would integrate with your main LLM inference
    """
    # Start new conversation if needed
    if not conversation_id:
        import uuid
        conversation_id = str(uuid.uuid4())
        conversation_manager.start_conversation(conversation_id)
        conversation_manager.strategy = ContextStrategy(strategy)
    
    # Add user message
    conversation_manager.add_turn(conversation_id, "user", message)
    
    # Get context for inference
    context = conversation_manager.get_context_for_inference(conversation_id)
    
    # Integrate with actual LLM inference
    from backend.model.moe_infer import MoEModelManager
    
    try:
        # Initialize model manager if needed
        model_manager = MoEModelManager()
        
        # Prepare prompt with context
        full_prompt = ""
        for msg in context['messages']:
            if msg['role'] == 'user':
                full_prompt += f"User: {msg['content']}\n"
            else:
                full_prompt += f"Assistant: {msg['content']}\n"
        full_prompt += f"User: {message}\nAssistant: "
        
        # Generate response using actual model
        response = await model_manager.generate_text(
            prompt=full_prompt,
            max_length=2048,
            temperature=0.7,
            top_p=0.8
        )
        
        # Extract just the assistant's response
        if response.startswith("Assistant: "):
            response = response[11:]
            
    except Exception as e:
        # Fallback if model not available
        logger.error(f"Model inference failed: {e}")
        response = f"Error: Model inference unavailable. Strategy: {context['strategy_used']}"
    
    # Add assistant response
    conversation_manager.add_turn(conversation_id, "assistant", response)
    
    return ChatContextResponse(
        response=response,
        conversation_id=conversation_id,
        context_strategy=context['strategy_used'],
        tokens_processed=context['token_estimate']
    )