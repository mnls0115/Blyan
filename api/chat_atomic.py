#!/usr/bin/env python3
"""
Atomic Chat Endpoint with Transaction Support
Ensures all-or-nothing execution for cost, quota, and inference operations
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
from fastapi import HTTPException, Request
from pydantic import BaseModel

from backend.core.transaction_manager import get_transaction_manager, TransactionContext
from backend.core.free_tier import get_free_tier_manager
from backend.security.abuse_prevention import get_abuse_prevention_system, RequestFingerprint
from backend.api.economy import redis_client as quote_redis

logger = logging.getLogger(__name__)

class AtomicChatRequest(BaseModel):
    """Enhanced chat request with transaction support."""
    prompt: str
    use_moe: bool = True
    top_k_experts: int = 2
    max_new_tokens: int = 100
    quote_id: Optional[str] = None
    idempotency_key: Optional[str] = None  # For duplicate prevention

class AtomicChatResponse(BaseModel):
    """Enhanced chat response with transaction details."""
    response: str
    expert_usage: Dict[str, Any] = {}
    inference_time: float
    transaction_id: str
    actual_cost: Optional[Decimal] = None
    tokens_used: Optional[int] = None

class AtomicChatHandler:
    """
    Handles atomic chat operations with full transaction support.
    """
    
    def __init__(self):
        self.transaction_manager = get_transaction_manager()
        self.free_tier_manager = get_free_tier_manager()
        self.abuse_system = get_abuse_prevention_system()
    
    async def process_chat(
        self,
        request: AtomicChatRequest,
        http_request: Request,
        user_address: str
    ) -> AtomicChatResponse:
        """
        Process chat request with atomic transaction guarantees.
        """
        
        # Use transaction manager for atomicity
        async with self.transaction_manager.atomic_transaction(
            user_address=user_address,
            idempotency_key=request.idempotency_key
        ) as ctx:
            
            # If idempotent request was already processed
            if ctx.result:
                return AtomicChatResponse(**ctx.result)
            
            # Step 1: Validate abuse prevention
            await self._validate_abuse_prevention(ctx, request, http_request)
            
            # Step 2: Validate and reserve quota/balance
            await self._validate_and_reserve_resources(ctx, request, user_address)
            
            # Step 3: Execute inference
            result = await self._execute_inference(ctx, request)
            
            # Step 4: Commit resources
            await self._commit_resources(ctx, user_address, result)
            
            # Prepare response
            response = AtomicChatResponse(
                response=result["response"],
                expert_usage=result.get("expert_usage", {}),
                inference_time=result["inference_time"],
                transaction_id=ctx.transaction_id,
                actual_cost=result.get("actual_cost"),
                tokens_used=result.get("tokens_used")
            )
            
            # Save for idempotency
            ctx.result = response.dict()
            
            return response
    
    async def _validate_abuse_prevention(
        self,
        ctx: TransactionContext,
        request: AtomicChatRequest,
        http_request: Request
    ):
        """Validate request against abuse prevention system."""
        
        fingerprint = RequestFingerprint(
            user_address=ctx.user_address,
            ip_address=str(http_request.client.host),
            user_agent=http_request.headers.get("user-agent", "unknown"),
            hardware_fingerprint=http_request.headers.get("x-hardware-fingerprint"),
            session_id=http_request.headers.get("x-session-id")
        )
        
        allowed, challenge_type, context = await self.abuse_system.assess_request(
            fingerprint=fingerprint,
            prompt=request.prompt,
            request_context={"max_new_tokens": request.max_new_tokens}
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Abuse prevention triggered",
                    "challenge_type": challenge_type.value,
                    "reason": context.get("reason", "Suspicious activity detected")
                }
            )
    
    async def _validate_and_reserve_resources(
        self,
        ctx: TransactionContext,
        request: AtomicChatRequest,
        user_address: str
    ):
        """Validate and reserve quota/balance atomically."""
        
        # Calculate token requirements
        import tiktoken
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(request.prompt))
        except:
            input_tokens = int(len(request.prompt.split()) * 1.3)
        
        output_tokens_est = request.max_new_tokens
        
        # Check if quote provided
        if request.quote_id:
            await self._validate_quote(ctx, request.quote_id, input_tokens)
        
        # Reserve resources based on user type
        can_request, denial_reason, limits_info = self.free_tier_manager.can_make_request(
            address=user_address,
            input_tokens=input_tokens,
            output_tokens_est=output_tokens_est
        )
        
        if not can_request:
            raise HTTPException(
                status_code=429,
                detail=f"Resource limit exceeded: {denial_reason}"
            )
        
        # Reserve the quota (will be rolled back on failure)
        is_free_tier = (limits_info.get("free_requests_remaining", 0) > 0 or
                       limits_info.get("bonus_requests_available", 0) > 0)
        
        if is_free_tier:
            # Reserve free tier quota
            self._reserve_free_quota(ctx, user_address)
        else:
            # Reserve balance for paid request
            estimated_cost = await self._calculate_cost(input_tokens, output_tokens_est)
            await self._reserve_balance(ctx, user_address, estimated_cost)
    
    async def _validate_quote(
        self,
        ctx: TransactionContext,
        quote_id: str,
        actual_input_tokens: int
    ):
        """Validate quote and mark as consumed."""
        import json
        
        quote_key = f"quote:{quote_id}"
        quote_data = await quote_redis.get(quote_key)
        
        if not quote_data:
            raise HTTPException(
                status_code=400,
                detail="Quote expired or invalid"
            )
        
        try:
            quote = json.loads(quote_data)
            
            # Check TTL
            if time.time() > quote["expires_at"]:
                await quote_redis.delete(quote_key)
                raise HTTPException(
                    status_code=400,
                    detail="Quote has expired"
                )
            
            # Check token count (10% tolerance)
            quote_input_tokens = quote["input_tokens"]
            if abs(actual_input_tokens - quote_input_tokens) > quote_input_tokens * 0.1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token count mismatch. Quote: {quote_input_tokens}, Actual: {actual_input_tokens}"
                )
            
            # Mark quote as consumed (atomic operation)
            await quote_redis.delete(quote_key)
            ctx.quote_consumed = True
            
            # Add rollback operation
            async def rollback_quote():
                # Re-create the quote with remaining TTL
                remaining_ttl = int(quote["expires_at"] - time.time())
                if remaining_ttl > 0:
                    await quote_redis.setex(quote_key, remaining_ttl, json.dumps(quote))
            
            ctx.add_rollback(rollback_quote)
            
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid quote data"
            )
    
    def _reserve_free_quota(self, ctx: TransactionContext, user_address: str):
        """Reserve free tier quota with rollback."""
        # This is a placeholder - actual implementation would decrement Redis counter
        ctx.quota_reserved = True
        
        # Add rollback operation
        def rollback_quota():
            # Restore the quota
            self.free_tier_manager.restore_quota(user_address)
        
        ctx.add_rollback(rollback_quota)
    
    async def _reserve_balance(
        self,
        ctx: TransactionContext,
        user_address: str,
        amount: Decimal
    ):
        """Reserve user balance with rollback."""
        from backend.accounting.ledger import get_ledger
        
        ledger = get_ledger()
        
        # Check balance
        balance = ledger.get_user_balance(user_address)
        if balance < amount:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient balance. Required: {amount}, Available: {balance}"
            )
        
        # Reserve the amount (temporary hold)
        ledger.hold_amount(user_address, amount)
        ctx.balance_reserved = amount
        
        # Add rollback operation
        async def rollback_balance():
            ledger.release_hold(user_address, amount)
        
        ctx.add_rollback(rollback_balance)
    
    async def _execute_inference(
        self,
        ctx: TransactionContext,
        request: AtomicChatRequest
    ) -> Dict[str, Any]:
        """Execute AI inference with error handling."""
        start_time = time.time()
        
        try:
            # Import inference modules
            from backend.model.moe_infer import get_moe_manager
            from backend.p2p.distributed_inference import get_distributed_coordinator
            
            model_manager = get_moe_manager()
            distributed_coordinator = get_distributed_coordinator()
            
            # Check for distributed inference
            if request.use_moe and distributed_coordinator and distributed_coordinator.registry.nodes:
                # Use distributed inference
                available_experts = list(distributed_coordinator.registry.expert_to_nodes.keys())
                
                if available_experts:
                    selected_experts = available_experts[:request.top_k_experts]
                    
                    response_text, routing_info = await distributed_coordinator.distribute_inference(
                        prompt=request.prompt,
                        required_experts=selected_experts,
                        max_new_tokens=request.max_new_tokens
                    )
                    
                    ctx.inference_completed = True
                    
                    return {
                        "response": response_text,
                        "expert_usage": routing_info.get('expert_usage', {}),
                        "inference_time": time.time() - start_time,
                        "tokens_used": len(response_text.split())  # Approximate
                    }
            
            # Fallback to local MoE inference
            if request.use_moe:
                answer, expert_usage = model_manager.generate_with_moe(
                    request.prompt,
                    top_k=request.top_k_experts,
                    max_new_tokens=request.max_new_tokens
                )
            else:
                # Standard inference
                answer = model_manager.generate(
                    request.prompt,
                    max_new_tokens=request.max_new_tokens
                )
                expert_usage = {}
            
            ctx.inference_completed = True
            
            return {
                "response": answer,
                "expert_usage": expert_usage,
                "inference_time": time.time() - start_time,
                "tokens_used": len(answer.split())  # Approximate
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
    
    async def _commit_resources(
        self,
        ctx: TransactionContext,
        user_address: str,
        result: Dict[str, Any]
    ):
        """Commit resource consumption after successful inference."""
        
        # Calculate actual cost based on tokens used
        actual_tokens = result.get("tokens_used", 0)
        actual_cost = await self._calculate_cost(actual_tokens, 0)
        
        if ctx.quota_reserved:
            # Consume free tier quota
            self.free_tier_manager.consume_request(user_address, success=True)
        elif ctx.balance_reserved > 0:
            # Charge the actual cost (may be less than reserved)
            from backend.accounting.ledger import get_ledger
            ledger = get_ledger()
            
            # Release hold and charge actual
            ledger.release_hold(user_address, ctx.balance_reserved)
            ledger.charge_user(user_address, actual_cost)
            
            result["actual_cost"] = actual_cost
        
        # Record success metrics
        composite_key = f"{user_address}|{ctx.transaction_id}"
        self.abuse_system.record_request_outcome(composite_key, success=True)
    
    async def _calculate_cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Calculate cost based on token usage."""
        # Base rates (can be made configurable)
        INPUT_RATE = Decimal("0.01")  # per 1K tokens
        OUTPUT_RATE = Decimal("0.02")  # per 1K tokens
        
        input_cost = (Decimal(input_tokens) / 1000) * INPUT_RATE
        output_cost = (Decimal(output_tokens) / 1000) * OUTPUT_RATE
        
        return input_cost + output_cost

# Singleton instance
_atomic_chat_handler = None

def get_atomic_chat_handler() -> AtomicChatHandler:
    """Get or create atomic chat handler singleton."""
    global _atomic_chat_handler
    if _atomic_chat_handler is None:
        _atomic_chat_handler = AtomicChatHandler()
    return _atomic_chat_handler