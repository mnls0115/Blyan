"""
Shared Cost Calculation Utilities
==================================
Centralized token cost calculation and verification logic.
"""

from typing import Tuple, Optional
import tiktoken
import logging

logger = logging.getLogger(__name__)

# Cost constants (in USD)
INPUT_TOKEN_COST = 0.00001  # $0.01 per 1K input tokens
OUTPUT_TOKEN_COST = 0.00002  # $0.02 per 1K output tokens


class TokenCostCalculator:
    """Unified token cost calculation."""
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            model: Model name for tokenizer selection
            
        Returns:
            Token count
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    
    @staticmethod
    def calculate_token_cost(token_count: int, is_input: bool = True) -> float:
        """
        Calculate cost for tokens.
        
        Args:
            token_count: Number of tokens
            is_input: Whether these are input tokens (vs output)
            
        Returns:
            Cost in USD
        """
        rate = INPUT_TOKEN_COST if is_input else OUTPUT_TOKEN_COST
        return token_count * rate
    
    @staticmethod
    def calculate_incremental_cost(
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[float, float, float]:
        """
        Calculate incremental costs.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        input_cost = TokenCostCalculator.calculate_token_cost(input_tokens, is_input=True)
        output_cost = TokenCostCalculator.calculate_token_cost(output_tokens, is_input=False)
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    @staticmethod
    def estimate_request_cost(prompt: str, max_tokens: int = 100) -> float:
        """
        Estimate total cost for a request.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            
        Returns:
            Estimated total cost in USD
        """
        input_tokens = TokenCostCalculator.count_tokens(prompt)
        input_cost = TokenCostCalculator.calculate_token_cost(input_tokens, is_input=True)
        max_output_cost = TokenCostCalculator.calculate_token_cost(max_tokens, is_input=False)
        
        return input_cost + max_output_cost


def verify_chat_request_cost(
    user_address: str,
    prompt: str,
    max_new_tokens: int = 100
) -> Tuple[bool, str, float]:
    """
    Verify if user can afford the request.
    
    Args:
        user_address: User identifier
        prompt: Request prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (can_afford, reason, estimated_cost)
    """
    estimated_cost = TokenCostCalculator.estimate_request_cost(prompt, max_new_tokens)
    
    # Check actual user balance/quota
    try:
        # Try to use free tier manager if available
        from backend.api.free_tier_manager import get_free_tier_manager
        manager = get_free_tier_manager()
        
        quota_info = manager.check_quota(user_address)
        if quota_info and quota_info.get('remaining_quota', 0) > 0:
            return True, "Free tier quota available", estimated_cost
        
        # Check paid balance if available
        balance = quota_info.get('balance', 0.0) if quota_info else 0.0
        if balance < estimated_cost:
            return False, f"Insufficient balance: ${balance:.4f} < ${estimated_cost:.4f}", estimated_cost
        return True, "Sufficient balance", estimated_cost
        
    except ImportError:
        # Free tier manager not available, allow request for now
        logger.warning("Free tier manager not available, bypassing cost check")
        return True, "Cost check bypassed (no quota system)", estimated_cost
    except Exception as e:
        logger.warning(f"Failed to check balance for {user_address}: {e}")
        # Allow request if balance check fails
        return True, "Balance check bypassed", estimated_cost


def finalize_request_cost(
    user_address: str,
    actual_input_tokens: int,
    actual_output_tokens: int
) -> float:
    """
    Finalize and charge actual cost after completion.
    
    Args:
        user_address: User identifier
        actual_input_tokens: Actual input token count
        actual_output_tokens: Actual output token count
        
    Returns:
        Total cost charged
    """
    input_cost, output_cost, total_cost = TokenCostCalculator.calculate_incremental_cost(
        actual_input_tokens,
        actual_output_tokens
    )
    
    # This would update actual user balance
    # For now, just log
    logger.info(f"Charging {user_address}: ${total_cost:.4f} "
                f"(input: ${input_cost:.4f}, output: ${output_cost:.4f})")
    
    return total_cost