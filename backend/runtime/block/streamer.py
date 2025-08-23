"""
Streamer Implementation

Real-time token streaming with backpressure and cancellation support.
"""

import asyncio
import time
from typing import Optional, Callable, Awaitable, Dict, Any
from collections import deque
import logging

from .types import StreamToken
from .errors import StreamingError, SessionNotFoundError

logger = logging.getLogger(__name__)


class Streamer:
    """
    Manages real-time token streaming with backpressure control.
    """
    
    def __init__(
        self,
        max_queue_size: int = 100,
        backpressure_threshold: float = 0.8,
        batch_size: int = 5,
        flush_interval_ms: int = 50
    ):
        self.max_queue_size = max_queue_size
        self.backpressure_threshold = backpressure_threshold
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        
        # Active streams
        self.streams: Dict[str, "StreamSession"] = {}
        
        # Metrics
        self.metrics = {
            "tokens_streamed": 0,
            "streams_created": 0,
            "streams_completed": 0,
            "streams_cancelled": 0,
            "backpressure_events": 0,
            "total_latency_ms": 0
        }
    
    async def create_stream(
        self,
        session_id: str,
        callback: Callable[[StreamToken], Awaitable[None]]
    ) -> "StreamSession":
        """Create a new streaming session."""
        if session_id in self.streams:
            raise StreamingError(f"Stream already exists for session {session_id}")
        
        session = StreamSession(
            session_id=session_id,
            callback=callback,
            max_queue_size=self.max_queue_size,
            backpressure_threshold=self.backpressure_threshold,
            batch_size=self.batch_size,
            flush_interval_ms=self.flush_interval_ms,
            parent=self
        )
        
        self.streams[session_id] = session
        self.metrics["streams_created"] += 1
        
        # Start the stream worker and store the task
        session.worker_task = asyncio.create_task(session._run_worker())
        
        return session
    
    def get_stream(self, session_id: str) -> Optional["StreamSession"]:
        """Get an existing stream session."""
        return self.streams.get(session_id)
    
    async def close_stream(self, session_id: str) -> None:
        """Close a streaming session."""
        if session_id not in self.streams:
            raise SessionNotFoundError(session_id)
        
        session = self.streams[session_id]
        await session.close()
        
        del self.streams[session_id]
        
        if session.cancelled:
            self.metrics["streams_cancelled"] += 1
        else:
            self.metrics["streams_completed"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streamer metrics."""
        return {
            "active_streams": len(self.streams),
            "tokens_streamed": self.metrics["tokens_streamed"],
            "streams_created": self.metrics["streams_created"],
            "streams_completed": self.metrics["streams_completed"],
            "streams_cancelled": self.metrics["streams_cancelled"],
            "backpressure_events": self.metrics["backpressure_events"],
            "avg_latency_ms": (
                self.metrics["total_latency_ms"] / self.metrics["tokens_streamed"]
                if self.metrics["tokens_streamed"] > 0 else 0
            )
        }


class StreamSession:
    """
    Individual streaming session with queue management.
    """
    
    def __init__(
        self,
        session_id: str,
        callback: Callable[[StreamToken], Awaitable[None]],
        max_queue_size: int,
        backpressure_threshold: float,
        batch_size: int,
        flush_interval_ms: int,
        parent: Streamer
    ):
        self.session_id = session_id
        self.callback = callback
        self.max_queue_size = max_queue_size
        self.backpressure_threshold = backpressure_threshold
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        self.parent = parent
        
        # Queue for tokens
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_buffer: deque = deque(maxlen=batch_size)
        
        # State
        self.closed = False
        self.cancelled = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.tokens_sent = 0
        self.last_flush_time = time.time()
    
    async def send_token(self, token: StreamToken) -> None:
        """Send a token to the stream."""
        if self.closed:
            raise StreamingError(f"Stream {self.session_id} is closed")
        
        # Check backpressure
        queue_usage = self.queue.qsize() / self.max_queue_size
        if queue_usage > self.backpressure_threshold:
            self.parent.metrics["backpressure_events"] += 1
            # Apply backpressure - wait a bit
            await asyncio.sleep(0.01 * queue_usage)
        
        try:
            await self.queue.put(token)
        except asyncio.QueueFull:
            raise StreamingError(f"Stream queue full for session {self.session_id}")
    
    async def _run_worker(self) -> None:
        """Worker task to process tokens from queue."""
        try:
            while not self.closed:
                try:
                    # Get token with timeout for periodic flushing
                    token = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=self.flush_interval_ms / 1000
                    )
                    
                    self.batch_buffer.append(token)
                    
                    # Check if we should flush
                    should_flush = (
                        len(self.batch_buffer) >= self.batch_size or
                        (time.time() - self.last_flush_time) * 1000 >= self.flush_interval_ms
                    )
                    
                    if should_flush:
                        await self._flush_batch()
                    
                except asyncio.TimeoutError:
                    # Timeout - flush any pending tokens
                    if self.batch_buffer:
                        await self._flush_batch()
                except Exception as e:
                    logger.error(f"Error in stream worker: {e}")
                    break
            
            # Final flush
            if self.batch_buffer:
                await self._flush_batch()
                
        except Exception as e:
            logger.error(f"Fatal error in stream worker: {e}")
    
    async def _flush_batch(self) -> None:
        """Flush the batch buffer."""
        if not self.batch_buffer:
            return
        
        # Send tokens
        for token in self.batch_buffer:
            try:
                start_time = time.time()
                await self.callback(token)
                
                latency_ms = (time.time() - start_time) * 1000
                self.parent.metrics["total_latency_ms"] += latency_ms
                self.parent.metrics["tokens_streamed"] += 1
                self.tokens_sent += 1
                
            except Exception as e:
                logger.error(f"Error sending token: {e}")
        
        self.batch_buffer.clear()
        self.last_flush_time = time.time()
    
    async def cancel(self) -> None:
        """Cancel the stream."""
        self.cancelled = True
        await self.close()
    
    async def close(self) -> None:
        """Close the stream."""
        if self.closed:
            return
        
        self.closed = True
        
        # Wait for worker to finish
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get stream status."""
        return {
            "session_id": self.session_id,
            "tokens_sent": self.tokens_sent,
            "queue_size": self.queue.qsize(),
            "closed": self.closed,
            "cancelled": self.cancelled
        }