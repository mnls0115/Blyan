#!/usr/bin/env python3
"""
Streaming Response Implementation
Real-time token streaming with cost accumulation
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional
from decimal import Decimal
from dataclasses import dataclass, asdict
import logging

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

@dataclass
class StreamToken:
    """Individual token in stream."""
    token: str
    token_id: int
    cost: Decimal
    cumulative_cost: Decimal
    timestamp: float

@dataclass
class StreamMetrics:
    """Metrics for streaming session."""
    stream_id: str
    start_time: float
    tokens_generated: int
    total_cost: Decimal
    tokens_per_second: float
    cancelled: bool = False
    error: Optional[str] = None

class TokenCostCalculator:
    """Calculate cost per token dynamically."""
    
    # Base rates per 1K tokens
    BASE_INPUT_RATE = Decimal("0.01")
    BASE_OUTPUT_RATE = Decimal("0.02")
    
    @classmethod
    def calculate_token_cost(cls, token_count: int, is_input: bool = False) -> Decimal:
        """Calculate cost for given number of tokens."""
        rate = cls.BASE_INPUT_RATE if is_input else cls.BASE_OUTPUT_RATE
        return (Decimal(token_count) / 1000) * rate
    
    @classmethod
    def calculate_incremental_cost(cls, token_position: int) -> Decimal:
        """Calculate cost for single output token."""
        # Could implement dynamic pricing based on position
        # For now, flat rate
        return cls.BASE_OUTPUT_RATE / 1000

class StreamingChatHandler:
    """
    Handles streaming chat responses with real-time cost tracking.
    """
    
    def __init__(self):
        self.active_streams: Dict[str, StreamMetrics] = {}
        self.cancellation_tokens: Dict[str, asyncio.Event] = {}
        
    async def stream_response(
        self,
        prompt: str,
        user_address: str,
        max_new_tokens: int = 100,
        use_moe: bool = True,
        quote_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming response with token-by-token cost tracking.
        
        Yields dict with:
        - token: str
        - cost_so_far: Decimal
        - tokens_generated: int
        - finished: bool
        """
        
        stream_id = f"stream_{uuid.uuid4().hex[:12]}"
        cancel_event = asyncio.Event()
        self.cancellation_tokens[stream_id] = cancel_event
        
        # Initialize metrics
        metrics = StreamMetrics(
            stream_id=stream_id,
            start_time=time.time(),
            tokens_generated=0,
            total_cost=Decimal(0),
            tokens_per_second=0
        )
        self.active_streams[stream_id] = metrics
        
        # Calculate input cost
        input_tokens = len(prompt.split())  # Simple approximation
        input_cost = TokenCostCalculator.calculate_token_cost(input_tokens, is_input=True)
        metrics.total_cost = input_cost
        
        try:
            # Import inference modules
            from backend.model.moe_infer import get_moe_manager
            from backend.p2p.distributed_inference import get_distributed_coordinator
            
            model_manager = get_moe_manager()
            distributed_coordinator = get_distributed_coordinator()
            
            # Start generation
            logger.info(f"Starting stream {stream_id} for user {user_address}")
            
            # Yield initial metadata
            yield {
                "type": "stream_start",
                "stream_id": stream_id,
                "input_cost": str(input_cost),
                "max_tokens": max_new_tokens,
                "timestamp": time.time()
            }
            
            # Generate tokens
            token_count = 0
            async for token in self._generate_tokens_async(
                prompt, 
                max_new_tokens,
                model_manager,
                distributed_coordinator,
                use_moe,
                cancel_event
            ):
                if cancel_event.is_set():
                    metrics.cancelled = True
                    break
                
                token_count += 1
                
                # Calculate incremental cost
                token_cost = TokenCostCalculator.calculate_incremental_cost(token_count)
                metrics.total_cost += token_cost
                metrics.tokens_generated = token_count
                
                # Calculate tokens per second
                elapsed = time.time() - metrics.start_time
                metrics.tokens_per_second = token_count / elapsed if elapsed > 0 else 0
                
                # Yield token with cost info
                yield {
                    "type": "token",
                    "token": token,
                    "token_number": token_count,
                    "incremental_cost": str(token_cost),
                    "cumulative_cost": str(metrics.total_cost),
                    "tokens_per_second": round(metrics.tokens_per_second, 2),
                    "timestamp": time.time()
                }
                
                # Check max tokens
                if token_count >= max_new_tokens:
                    break
            
            # Yield completion
            yield {
                "type": "stream_end",
                "stream_id": stream_id,
                "total_tokens": metrics.tokens_generated,
                "total_cost": str(metrics.total_cost),
                "duration": time.time() - metrics.start_time,
                "tokens_per_second": round(metrics.tokens_per_second, 2),
                "cancelled": metrics.cancelled,
                "timestamp": time.time()
            }
            
            # Finalize billing based on actual tokens
            await self._finalize_billing(
                user_address,
                metrics.total_cost,
                metrics.tokens_generated,
                quote_id
            )
            
        except Exception as e:
            logger.error(f"Stream {stream_id} failed: {e}")
            metrics.error = str(e)
            
            yield {
                "type": "error",
                "stream_id": stream_id,
                "error": str(e),
                "tokens_generated": metrics.tokens_generated,
                "partial_cost": str(metrics.total_cost),
                "timestamp": time.time()
            }
            
        finally:
            # Cleanup
            del self.active_streams[stream_id]
            del self.cancellation_tokens[stream_id]
    
    async def _generate_tokens_async(
        self,
        prompt: str,
        max_tokens: int,
        model_manager,
        distributed_coordinator,
        use_moe: bool,
        cancel_event: asyncio.Event
    ) -> AsyncGenerator[str, None]:
        """
        Generate tokens asynchronously.
        This is a mock implementation - replace with actual model generation.
        """
        
        # Mock token generation for demonstration
        # In production, this would interface with the actual model
        
        response_tokens = [
            "The", "answer", "to", "your", "question", "is", "quite", 
            "interesting", ".", "Let", "me", "explain", "in", "detail", ".",
            "First", ",", "we", "need", "to", "understand", "the", "context", ".",
            "Then", ",", "we", "can", "explore", "the", "implications", ".",
            "Finally", ",", "I'll", "provide", "some", "recommendations", "."
        ]
        
        for i, token in enumerate(response_tokens):
            if cancel_event.is_set():
                break
                
            # Simulate generation delay
            await asyncio.sleep(0.05)  # 50ms per token
            
            yield token + " "
            
            if i >= max_tokens - 1:
                break
    
    async def _finalize_billing(
        self,
        user_address: str,
        total_cost: Decimal,
        tokens_generated: int,
        quote_id: Optional[str]
    ):
        """
        Finalize billing based on actual token usage.
        """
        logger.info(
            f"Finalizing billing for {user_address}: "
            f"{tokens_generated} tokens, {total_cost} BLY"
        )
        
        # Import ledger
        from backend.accounting.postgres_ledger import get_postgres_ledger
        
        ledger = get_postgres_ledger()
        
        try:
            # If quote was provided, finalize it
            if quote_id:
                # Complete the quoted transaction with actual amount
                await ledger.credit_transaction(
                    transaction_id=quote_id,
                    actual_amount=total_cost
                )
            else:
                # Direct charge
                await ledger.create_quote(
                    user_address=user_address,
                    amount=total_cost,
                    quote_id=f"stream_{int(time.time())}",
                    metadata={"tokens": tokens_generated}
                )
        except Exception as e:
            logger.error(f"Billing finalization failed: {e}")
            # Could implement retry logic here
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """
        Cancel an active stream.
        """
        if stream_id in self.cancellation_tokens:
            self.cancellation_tokens[stream_id].set()
            logger.info(f"Stream {stream_id} cancelled")
            return True
        return False
    
    def get_active_streams(self) -> Dict[str, Dict]:
        """
        Get information about active streams.
        """
        return {
            stream_id: {
                "started_at": metrics.start_time,
                "tokens_generated": metrics.tokens_generated,
                "current_cost": str(metrics.total_cost),
                "tokens_per_second": metrics.tokens_per_second,
                "duration": time.time() - metrics.start_time
            }
            for stream_id, metrics in self.active_streams.items()
        }

class WebSocketStreamHandler:
    """
    WebSocket-based streaming for bidirectional communication.
    """
    
    def __init__(self):
        self.streaming_handler = StreamingChatHandler()
        self.connections: Dict[str, Any] = {}
    
    async def handle_websocket(self, websocket, user_address: str):
        """
        Handle WebSocket connection for streaming chat.
        """
        connection_id = f"ws_{uuid.uuid4().hex[:12]}"
        self.connections[connection_id] = {
            "websocket": websocket,
            "user_address": user_address,
            "active_stream": None
        }
        
        try:
            await websocket.accept()
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "connection_id": connection_id,
                "status": "connected"
            })
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_ws_message(
                        connection_id,
                        data,
                        websocket
                    )
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            
        finally:
            # Cleanup
            if connection_id in self.connections:
                # Cancel any active stream
                if self.connections[connection_id]["active_stream"]:
                    await self.streaming_handler.cancel_stream(
                        self.connections[connection_id]["active_stream"]
                    )
                del self.connections[connection_id]
    
    async def _handle_ws_message(
        self,
        connection_id: str,
        message: Dict,
        websocket
    ):
        """
        Handle incoming WebSocket message.
        """
        msg_type = message.get("type")
        
        if msg_type == "chat":
            # Start streaming response
            prompt = message.get("prompt", "")
            max_tokens = message.get("max_tokens", 100)
            
            # Get user address
            user_address = self.connections[connection_id]["user_address"]
            
            # Stream response
            stream_generator = self.streaming_handler.stream_response(
                prompt=prompt,
                user_address=user_address,
                max_new_tokens=max_tokens
            )
            
            async for chunk in stream_generator:
                await websocket.send_json(chunk)
                
                # Track active stream
                if chunk["type"] == "stream_start":
                    self.connections[connection_id]["active_stream"] = chunk["stream_id"]
                elif chunk["type"] in ["stream_end", "error"]:
                    self.connections[connection_id]["active_stream"] = None
        
        elif msg_type == "cancel":
            # Cancel active stream
            stream_id = self.connections[connection_id]["active_stream"]
            if stream_id:
                success = await self.streaming_handler.cancel_stream(stream_id)
                await websocket.send_json({
                    "type": "cancel_response",
                    "stream_id": stream_id,
                    "success": success
                })
        
        elif msg_type == "ping":
            # Heartbeat
            await websocket.send_json({
                "type": "pong",
                "timestamp": time.time()
            })

# Singleton instances
_streaming_handler = None
_websocket_handler = None

def get_streaming_handler() -> StreamingChatHandler:
    """Get or create streaming handler singleton."""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingChatHandler()
    return _streaming_handler

def get_websocket_handler() -> WebSocketStreamHandler:
    """Get or create WebSocket handler singleton."""
    global _websocket_handler
    if _websocket_handler is None:
        _websocket_handler = WebSocketStreamHandler()
    return _websocket_handler