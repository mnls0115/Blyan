"""Enhanced HTTP routes with SSE streaming and cancellation support.

Integrates with api/streaming.py StreamingChatHandler.
"""

import asyncio
import uuid
import time
from typing import Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

# Global stream registry
stream_registry: Dict[str, Dict[str, Any]] = {}
stream_locks: Dict[str, asyncio.Lock] = {}


class StreamRequest(BaseModel):
    """SSE stream request."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    stream: bool = True
    stream_id: Optional[str] = None
    use_moe: bool = False
    model: str = "qwen_1_5_moe_a2_7b"


class CancelRequest(BaseModel):
    """Stream cancellation request."""
    stream_id: str
    reason: Optional[str] = None


def setup_streaming_routes(app: FastAPI, streaming_handler=None):
    """Setup enhanced SSE streaming routes with cancellation.
    
    Args:
        app: FastAPI application instance
        streaming_handler: Instance of StreamingChatHandler from api/streaming.py
    """
    
    @app.post("/generate")
    async def generate_endpoint(request: StreamRequest):
        """Generate text with SSE streaming support."""
        stream_id = request.stream_id or str(uuid.uuid4())
        
        # Register stream
        stream_registry[stream_id] = {
            "status": "active",
            "created_at": time.time(),
            "request": request.dict(),
            "tokens_generated": 0,
            "cancelled": False
        }
        
        # Create lock for this stream
        stream_locks[stream_id] = asyncio.Lock()
        
        if request.stream:
            async def sse_generator():
                """Generate SSE events."""
                try:
                    # Send initial event
                    yield f"event: stream_start\n"
                    yield f"data: {{\"stream_id\": \"{stream_id}\", \"model\": \"{request.model}\"}}\n\n"
                    
                    # Use existing streaming handler if available
                    if streaming_handler:
                        # Delegate to StreamingChatHandler
                        async for event in streaming_handler.stream_chat_atomic(
                            prompt=request.prompt,
                            max_tokens=request.max_tokens,
                            temperature=request.temperature,
                            use_moe=request.use_moe,
                            stream_id=stream_id
                        ):
                            # Check cancellation
                            async with stream_locks[stream_id]:
                                if stream_registry[stream_id]["cancelled"]:
                                    yield f"event: cancelled\n"
                                    yield f"data: {{\"stream_id\": \"{stream_id}\", \"reason\": \"User cancelled\"}}\n\n"
                                    break
                            
                            # Forward event
                            if event.get("type") == "token":
                                stream_registry[stream_id]["tokens_generated"] += 1
                                yield f"event: token\n"
                                yield f"data: {json.dumps(event)}\n\n"
                            elif event.get("type") == "complete":
                                yield f"event: complete\n"
                                yield f"data: {json.dumps(event)}\n\n"
                            elif event.get("type") == "error":
                                yield f"event: error\n"
                                yield f"data: {json.dumps(event)}\n\n"
                    else:
                        # Fallback mock streaming
                        tokens = ["Hello", " ", "world", "!", " ", "This", " ", "is", " ", "streaming", "."]
                        for i, token in enumerate(tokens):
                            # Check cancellation
                            async with stream_locks[stream_id]:
                                if stream_registry[stream_id]["cancelled"]:
                                    yield f"event: cancelled\n"
                                    yield f"data: {{\"stream_id\": \"{stream_id}\"}}\n\n"
                                    break
                            
                            stream_registry[stream_id]["tokens_generated"] += 1
                            
                            yield f"event: token\n"
                            yield f"data: {{\"token\": \"{token}\", \"index\": {i}, \"stream_id\": \"{stream_id}\"}}\n\n"
                            
                            await asyncio.sleep(0.1)  # Simulate generation delay
                        
                        # Send completion
                        if not stream_registry[stream_id]["cancelled"]:
                            yield f"event: complete\n"
                            yield f"data: {{\"stream_id\": \"{stream_id}\", \"tokens_generated\": {len(tokens)}}}\n\n"
                    
                except Exception as e:
                    logger.error(f"Stream {stream_id} error: {e}")
                    yield f"event: error\n"
                    yield f"data: {{\"error\": \"{str(e)}\", \"stream_id\": \"{stream_id}\"}}\n\n"
                    stream_registry[stream_id]["status"] = "error"
                    stream_registry[stream_id]["error"] = str(e)
                
                finally:
                    # Update status
                    if stream_id in stream_registry:
                        stream_registry[stream_id]["status"] = "completed"
                        stream_registry[stream_id]["completed_at"] = time.time()
                    
                    # Cleanup lock after delay
                    asyncio.create_task(_cleanup_stream(stream_id, delay=60))
            
            # Return SSE response
            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Stream-ID": stream_id,
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        else:
            # Non-streaming response
            try:
                if streaming_handler:
                    # Use handler for non-streaming
                    result = await streaming_handler.generate_non_streaming(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        use_moe=request.use_moe
                    )
                else:
                    # Mock response
                    result = {
                        "text": "Hello world! This is a non-streaming response.",
                        "tokens_generated": 8
                    }
                
                stream_registry[stream_id]["status"] = "completed"
                stream_registry[stream_id]["tokens_generated"] = result.get("tokens_generated", 0)
                
                return {
                    "stream_id": stream_id,
                    "text": result.get("text", ""),
                    "tokens_generated": result.get("tokens_generated", 0),
                    "model": request.model
                }
                
            except Exception as e:
                stream_registry[stream_id]["status"] = "error"
                raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/cancel")
    async def cancel_endpoint(
        stream_id: Optional[str] = Query(None),
        body: Optional[CancelRequest] = None
    ):
        """Cancel an active stream."""
        # Get stream_id from query or body
        sid = stream_id or (body.stream_id if body else None)
        reason = body.reason if body else "User requested"
        
        if not sid:
            raise HTTPException(status_code=400, detail="stream_id required")
        
        if sid not in stream_registry:
            raise HTTPException(status_code=404, detail=f"Stream {sid} not found")
        
        # Check if already completed
        if stream_registry[sid]["status"] == "completed":
            return {
                "status": "already_completed",
                "stream_id": sid,
                "tokens_generated": stream_registry[sid]["tokens_generated"]
            }
        
        # Set cancellation flag
        async with stream_locks[sid]:
            stream_registry[sid]["cancelled"] = True
            stream_registry[sid]["cancel_reason"] = reason
            stream_registry[sid]["status"] = "cancelled"
        
        # If using streaming handler, also cancel there
        if streaming_handler and hasattr(streaming_handler, 'cancel_stream'):
            try:
                await streaming_handler.cancel_stream(sid)
            except Exception as e:
                logger.warning(f"Failed to cancel in handler: {e}")
        
        logger.info(f"Stream {sid} cancelled: {reason}")
        
        return {
            "status": "cancelled",
            "stream_id": sid,
            "reason": reason,
            "tokens_generated": stream_registry[sid]["tokens_generated"]
        }
    
    @app.get("/streams")
    async def list_streams(
        status: Optional[str] = Query(None, description="Filter by status"),
        limit: int = Query(100, description="Maximum streams to return")
    ):
        """List active and recent streams."""
        streams = []
        
        for sid, info in list(stream_registry.items())[:limit]:
            if status and info["status"] != status:
                continue
            
            stream_info = {
                "stream_id": sid,
                "status": info["status"],
                "created_at": info["created_at"],
                "tokens_generated": info["tokens_generated"],
                "model": info["request"].get("model"),
                "cancelled": info.get("cancelled", False)
            }
            
            if "completed_at" in info:
                stream_info["duration"] = info["completed_at"] - info["created_at"]
            
            streams.append(stream_info)
        
        return {
            "streams": streams,
            "total": len(streams),
            "active": sum(1 for s in streams if s["status"] == "active")
        }
    
    @app.get("/streams/{stream_id}")
    async def get_stream_status(stream_id: str):
        """Get detailed status of a specific stream."""
        if stream_id not in stream_registry:
            raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
        
        info = stream_registry[stream_id]
        
        return {
            "stream_id": stream_id,
            "status": info["status"],
            "created_at": info["created_at"],
            "tokens_generated": info["tokens_generated"],
            "request": info["request"],
            "cancelled": info.get("cancelled", False),
            "cancel_reason": info.get("cancel_reason"),
            "error": info.get("error"),
            "completed_at": info.get("completed_at")
        }
    
    async def _cleanup_stream(stream_id: str, delay: int = 60):
        """Clean up stream data after delay."""
        await asyncio.sleep(delay)
        
        if stream_id in stream_registry:
            # Keep completed streams for a while for debugging
            if time.time() - stream_registry[stream_id]["created_at"] > 300:  # 5 minutes
                del stream_registry[stream_id]
                
        if stream_id in stream_locks:
            del stream_locks[stream_id]
    
    # Add metrics endpoint
    @app.get("/streaming/metrics")
    async def streaming_metrics():
        """Get streaming system metrics."""
        now = time.time()
        
        active_streams = [s for s in stream_registry.values() if s["status"] == "active"]
        completed_streams = [s for s in stream_registry.values() if s["status"] == "completed"]
        cancelled_streams = [s for s in stream_registry.values() if s.get("cancelled", False)]
        
        # Calculate average duration
        durations = []
        for s in completed_streams:
            if "completed_at" in s:
                durations.append(s["completed_at"] - s["created_at"])
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate p95 duration
        if durations:
            durations.sort()
            p95_idx = int(len(durations) * 0.95)
            p95_duration = durations[p95_idx]
        else:
            p95_duration = 0
        
        return {
            "stream_open": len(active_streams),
            "stream_completed_total": len(completed_streams),
            "stream_cancelled_total": len(cancelled_streams),
            "stream_duration_avg": avg_duration,
            "stream_duration_p95": p95_duration,
            "total_streams": len(stream_registry),
            "total_tokens_generated": sum(s["tokens_generated"] for s in stream_registry.values())
        }


# Import json for SSE data serialization
import json