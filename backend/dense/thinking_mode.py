"""
Thinking Mode for Distributed Inference
========================================
Implements <think>...</think> scratchpad mode for internal reasoning.
Ensures thinking tokens are never exposed to clients while maintaining
efficient multi-node execution.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, List, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Special tokens for thinking mode
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
THINK_START_ID = 151665  # Token ID for <think> in Qwen3
THINK_END_ID = 151668    # Token ID for </think> in Qwen3


class ThinkingMode(Enum):
    """Execution modes for thinking phase."""
    LOCAL_THEN_DISTRIBUTE = "local_then_distribute"  # Single node for thinking, distributed for answer
    COMPRESS_BROADCAST = "compress_broadcast"  # Summarize thinking and broadcast
    SPECULATIVE = "speculative"  # Fast node drafts, others validate


@dataclass
class ThinkingConfig:
    """Configuration for thinking mode."""
    enabled: bool = True
    mode: ThinkingMode = ThinkingMode.LOCAL_THEN_DISTRIBUTE
    max_think_tokens: int = 128  # Maximum tokens allowed in thinking phase
    kv_cache_precision: str = "int8"  # KV cache precision during thinking
    log_thinking: bool = True  # Whether to log thinking content
    encrypt_logs: bool = False  # Whether to encrypt thinking logs
    sticky_timeout_ms: int = 5000  # Timeout for sticky session
    auto_close_on_limit: bool = True  # Auto-inject </think> when limit reached


@dataclass
class ThinkingState:
    """Tracks thinking phase state during inference."""
    is_thinking: bool = False
    think_tokens_generated: int = 0
    think_start_time: Optional[float] = None
    think_content: List[str] = field(default_factory=list)
    sticky_node_id: Optional[str] = None
    exceeded_limit: bool = False


@dataclass
class ThinkingMetrics:
    """Metrics for thinking mode performance."""
    think_tokens_used: int = 0
    think_duration_ms: float = 0
    first_token_latency_ms: float = 0
    inter_node_bytes_sent: int = 0
    limit_exceeded_count: int = 0
    mode_downgrades: int = 0


class ThinkingModeHandler:
    """Handles thinking mode logic for distributed inference."""
    
    def __init__(self, config: ThinkingConfig):
        """
        Initialize thinking mode handler.
        
        Args:
            config: Thinking mode configuration
        """
        self.config = config
        self.metrics = ThinkingMetrics()
        self.active_sessions: Dict[str, ThinkingState] = {}
        logger.info(f"Thinking mode initialized: {config.mode.value}, max_tokens={config.max_think_tokens}")
    
    def create_session(self, request_id: str) -> ThinkingState:
        """Create a new thinking session."""
        state = ThinkingState()
        self.active_sessions[request_id] = state
        return state
    
    def get_session(self, request_id: str) -> Optional[ThinkingState]:
        """Get an existing thinking session."""
        return self.active_sessions.get(request_id)
    
    def should_use_sticky_assignment(self, request_id: str) -> bool:
        """Check if request should use sticky node assignment."""
        state = self.get_session(request_id)
        return state and state.is_thinking
    
    def get_sticky_node(self, request_id: str) -> Optional[str]:
        """Get the sticky node for a thinking session."""
        state = self.get_session(request_id)
        return state.sticky_node_id if state else None
    
    async def process_token_stream(
        self,
        request_id: str,
        token_stream: AsyncGenerator[Dict[str, Any], None],
        node_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process token stream to handle thinking mode.
        
        Args:
            request_id: Request identifier
            token_stream: Stream of tokens from model
            node_id: ID of the node processing this request
            
        Yields:
            Filtered tokens (excluding thinking content)
        """
        state = self.get_session(request_id)
        if not state:
            state = self.create_session(request_id)
        
        first_token_time = None
        
        async for token_data in token_stream:
            token_id = token_data.get("token_id")
            token_text = token_data.get("token", "")
            
            # Track first token latency
            if first_token_time is None:
                first_token_time = time.time()
                if state.think_start_time:
                    self.metrics.first_token_latency_ms = (first_token_time - state.think_start_time) * 1000
            
            # Check for thinking mode transitions
            if token_id == THINK_START_ID or token_text == THINK_START_TOKEN:
                # Enter thinking mode
                state.is_thinking = True
                state.think_start_time = time.time()
                state.sticky_node_id = node_id
                logger.info(f"Request {request_id} entering thinking mode on node {node_id}")
                continue  # Don't yield this token
            
            elif token_id == THINK_END_ID or token_text == THINK_END_TOKEN:
                # Exit thinking mode
                if state.is_thinking:
                    state.is_thinking = False
                    think_duration = (time.time() - state.think_start_time) * 1000
                    self.metrics.think_duration_ms = think_duration
                    self.metrics.think_tokens_used = state.think_tokens_generated
                    
                    logger.info(
                        f"Request {request_id} exiting thinking mode: "
                        f"{state.think_tokens_generated} tokens in {think_duration:.1f}ms"
                    )
                    
                    # Optionally log thinking content
                    if self.config.log_thinking:
                        self._log_thinking_content(request_id, state)
                    
                    # Clear sticky assignment
                    state.sticky_node_id = None
                continue  # Don't yield this token
            
            # Handle tokens during thinking
            if state.is_thinking:
                state.think_tokens_generated += 1
                state.think_content.append(token_text)
                
                # Check if we've exceeded the limit
                if state.think_tokens_generated >= self.config.max_think_tokens:
                    if not state.exceeded_limit:
                        state.exceeded_limit = True
                        self.metrics.limit_exceeded_count += 1
                        logger.warning(
                            f"Request {request_id} exceeded thinking limit "
                            f"({state.think_tokens_generated}/{self.config.max_think_tokens})"
                        )
                    
                    if self.config.auto_close_on_limit:
                        # Auto-inject end thinking token
                        state.is_thinking = False
                        state.sticky_node_id = None
                        logger.info(f"Auto-closing thinking mode for {request_id}")
                        # Continue to normal output
                    else:
                        continue  # Still don't yield thinking tokens
                else:
                    continue  # Don't yield thinking tokens
            
            # Yield non-thinking tokens
            yield token_data
    
    def _log_thinking_content(self, request_id: str, state: ThinkingState):
        """Log thinking content for debugging/metrics."""
        content = "".join(state.think_content)
        
        if self.config.encrypt_logs:
            # Simple obfuscation for demo (use real encryption in production)
            import base64
            content = base64.b64encode(content.encode()).decode()
            logger.debug(f"Thinking content for {request_id} (encrypted): {content[:100]}...")
        else:
            logger.debug(f"Thinking content for {request_id}: {content[:200]}...")
    
    def should_downgrade_mode(self) -> bool:
        """Check if we should downgrade to simpler thinking mode."""
        # Downgrade if inter-node traffic is too high
        if self.metrics.inter_node_bytes_sent > 100 * 1024 * 1024:  # 100MB
            logger.warning("High inter-node traffic, downgrading thinking mode")
            self.metrics.mode_downgrades += 1
            return True
        return False
    
    def get_execution_strategy(self, request_id: str) -> Dict[str, Any]:
        """Get execution strategy for current thinking state."""
        state = self.get_session(request_id)
        
        if not state or not self.config.enabled:
            return {"mode": "normal", "sticky": False}
        
        # Check if we should downgrade
        if self.should_downgrade_mode():
            self.config.mode = ThinkingMode.LOCAL_THEN_DISTRIBUTE
        
        if state.is_thinking:
            # During thinking phase
            if self.config.mode == ThinkingMode.LOCAL_THEN_DISTRIBUTE:
                return {
                    "mode": "sticky",
                    "sticky": True,
                    "node_id": state.sticky_node_id,
                    "timeout_ms": self.config.sticky_timeout_ms
                }
            elif self.config.mode == ThinkingMode.COMPRESS_BROADCAST:
                return {
                    "mode": "compress",
                    "sticky": True,
                    "node_id": state.sticky_node_id,
                    "broadcast_after": 32  # Tokens
                }
            elif self.config.mode == ThinkingMode.SPECULATIVE:
                return {
                    "mode": "speculative",
                    "draft_node": state.sticky_node_id,
                    "validators": []  # Would be populated with validator nodes
                }
        else:
            # After thinking or normal mode
            return {
                "mode": "distributed",
                "sticky": False,
                "allow_handoff": True
            }
    
    def update_inter_node_traffic(self, bytes_sent: int):
        """Update inter-node traffic metrics."""
        self.metrics.inter_node_bytes_sent += bytes_sent
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "think_tokens_used": self.metrics.think_tokens_used,
            "think_duration_ms": self.metrics.think_duration_ms,
            "first_token_latency_ms": self.metrics.first_token_latency_ms,
            "inter_node_bytes_sent": self.metrics.inter_node_bytes_sent,
            "limit_exceeded_count": self.metrics.limit_exceeded_count,
            "mode_downgrades": self.metrics.mode_downgrades,
            "active_sessions": len(self.active_sessions)
        }
    
    def cleanup_session(self, request_id: str):
        """Clean up a thinking session."""
        if request_id in self.active_sessions:
            del self.active_sessions[request_id]


class ThinkingModeCoordinator:
    """Coordinates thinking mode across distributed nodes."""
    
    def __init__(self, thinking_handler: ThinkingModeHandler):
        """
        Initialize coordinator.
        
        Args:
            thinking_handler: Thinking mode handler
        """
        self.handler = thinking_handler
        self.node_assignments: Dict[str, str] = {}  # request_id -> node_id
        logger.info("Thinking mode coordinator initialized")
    
    async def assign_node_for_request(
        self,
        request_id: str,
        available_nodes: List[str],
        request_config: Dict[str, Any]
    ) -> str:
        """
        Assign a node for processing a request.
        
        Args:
            request_id: Request identifier
            available_nodes: List of available node IDs
            request_config: Request configuration
            
        Returns:
            Selected node ID
        """
        # Check if thinking mode is enabled
        if request_config.get("thinking", False):
            # Create thinking session
            self.handler.create_session(request_id)
            
            # Get execution strategy
            strategy = self.handler.get_execution_strategy(request_id)
            
            if strategy["sticky"] and strategy.get("node_id"):
                # Use sticky assignment
                return strategy["node_id"]
        
        # Check for existing sticky assignment
        if self.handler.should_use_sticky_assignment(request_id):
            sticky_node = self.handler.get_sticky_node(request_id)
            if sticky_node and sticky_node in available_nodes:
                logger.info(f"Using sticky node {sticky_node} for {request_id}")
                return sticky_node
        
        # Default: select first available node
        selected = available_nodes[0] if available_nodes else None
        if selected:
            self.node_assignments[request_id] = selected
        
        return selected
    
    def can_handoff_request(self, request_id: str) -> bool:
        """Check if request can be handed off to another node."""
        strategy = self.handler.get_execution_strategy(request_id)
        return strategy.get("allow_handoff", True)
    
    def release_assignment(self, request_id: str):
        """Release node assignment for a request."""
        if request_id in self.node_assignments:
            del self.node_assignments[request_id]
        self.handler.cleanup_session(request_id)


# Example usage
async def demo_thinking_mode():
    """Demonstrate thinking mode functionality."""
    
    # Create configuration
    config = ThinkingConfig(
        enabled=True,
        mode=ThinkingMode.LOCAL_THEN_DISTRIBUTE,
        max_think_tokens=128,
        log_thinking=True
    )
    
    # Create handler and coordinator
    handler = ThinkingModeHandler(config)
    coordinator = ThinkingModeCoordinator(handler)
    
    # Simulate token stream with thinking
    async def mock_token_stream():
        tokens = [
            {"token_id": 1, "token": "Let"},
            {"token_id": 2, "token": " me"},
            {"token_id": THINK_START_ID, "token": THINK_START_TOKEN},
            {"token_id": 3, "token": " consider"},
            {"token_id": 4, "token": " this"},
            {"token_id": 5, "token": " carefully"},
            {"token_id": THINK_END_ID, "token": THINK_END_TOKEN},
            {"token_id": 6, "token": " The"},
            {"token_id": 7, "token": " answer"},
            {"token_id": 8, "token": " is"},
            {"token_id": 9, "token": " 42"},
        ]
        for token in tokens:
            yield token
            await asyncio.sleep(0.01)
    
    # Process stream
    request_id = "test_request"
    node_id = "node_1"
    
    print("Processing token stream with thinking mode:")
    print("-" * 50)
    
    output_tokens = []
    async for token in handler.process_token_stream(
        request_id, mock_token_stream(), node_id
    ):
        output_tokens.append(token["token"])
        print(f"Output: {token['token']}", end="", flush=True)
    
    print("\n" + "-" * 50)
    print(f"Filtered output: {''.join(output_tokens)}")
    print(f"Metrics: {handler.get_metrics()}")


if __name__ == "__main__":
    asyncio.run(demo_thinking_mode())