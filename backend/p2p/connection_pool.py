"""
Connection pooling for P2P distributed inference.
Reduces latency by reusing HTTP connections with keep-alive.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Any, Tuple
from urllib.parse import urlparse
import time
import ssl
import certifi

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    Manages persistent HTTP connections to P2P nodes.
    
    Features:
    - Connection reuse with keep-alive
    - Automatic cleanup of idle connections
    - Per-host connection limits
    - TLS certificate caching
    - Connection health monitoring
    """
    
    def __init__(
        self,
        max_connections_per_host: int = 10,
        keepalive_timeout: int = 30,
        total_timeout: int = 300,
        ssl_verify: bool = True,
        auth_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize connection pool.
        
        Args:
            max_connections_per_host: Maximum connections per host
            keepalive_timeout: Keep-alive timeout in seconds
            total_timeout: Total session timeout in seconds
            ssl_verify: Whether to verify SSL certificates
            auth_headers: Optional authentication headers to include in all requests
        """
        self.max_connections_per_host = max_connections_per_host
        self.keepalive_timeout = keepalive_timeout
        self.total_timeout = total_timeout
        self.ssl_verify = ssl_verify
        self.auth_headers = auth_headers or {}
        
        # Connection statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_closed': 0,
            'total_requests': 0,
            'failed_requests': 0
        }
        
        # Sessions by base URL
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._session_creation_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # SSL context for certificate verification
        self._ssl_context = self._create_ssl_context() if ssl_verify else None
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with certificate verification."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        return ssl_context
    
    def _get_base_url(self, url: str) -> str:
        """Extract base URL for connection pooling."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    async def _get_or_create_session(self, base_url: str) -> aiohttp.ClientSession:
        """Get existing session or create new one for base URL."""
        async with self._lock:
            # Check if session exists and is not closed
            if base_url in self._sessions:
                session = self._sessions[base_url]
                if not session.closed:
                    self.stats['connections_reused'] += 1
                    return session
                else:
                    # Clean up closed session
                    del self._sessions[base_url]
                    del self._session_creation_times[base_url]
            
            # Create new session with connection pooling
            connector = aiohttp.TCPConnector(
                limit_per_host=self.max_connections_per_host,
                keepalive_timeout=self.keepalive_timeout,
                force_close=False,  # Allow connection reuse
                ssl=self._ssl_context if self.ssl_verify else False
            )
            
            timeout = aiohttp.ClientTimeout(total=self.total_timeout)
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Connection': 'keep-alive',
                    'Keep-Alive': f'timeout={self.keepalive_timeout}'
                }
            )
            
            self._sessions[base_url] = session
            self._session_creation_times[base_url] = time.time()
            self.stats['connections_created'] += 1
            
            logger.info(f"Created new session pool for {base_url}")
            
            return session
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request using pooled connection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            aiohttp ClientResponse
        """
        base_url = self._get_base_url(url)
        session = await self._get_or_create_session(base_url)
        
        self.stats['total_requests'] += 1
        
        # Merge auth headers with any provided headers
        if self.auth_headers:
            headers = kwargs.get('headers', {})
            headers.update(self.auth_headers)
            kwargs['headers'] = headers
        
        try:
            response = await session.request(method, url, **kwargs)
            return response
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Request failed: {url} - {e}")
            raise
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Convenience method for GET requests."""
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Convenience method for POST requests."""
        return await self.request('POST', url, **kwargs)
    
    async def post_json(self, url: str, json_data: Any, **kwargs) -> Tuple[int, Any]:
        """
        POST JSON data and return status code and response.
        
        Args:
            url: Target URL
            json_data: Data to send as JSON
            **kwargs: Additional request arguments
            
        Returns:
            Tuple of (status_code, response_data)
        """
        try:
            async with await self.post(url, json=json_data, **kwargs) as response:
                data = await response.json()
                return response.status, data
        except aiohttp.ContentTypeError:
            # Handle non-JSON responses
            async with await self.post(url, json=json_data, **kwargs) as response:
                text = await response.text()
                return response.status, {'error': 'Non-JSON response', 'text': text}
    
    async def cleanup_idle_sessions(self, max_idle_seconds: int = 300):
        """Clean up sessions that have been idle too long."""
        async with self._lock:
            current_time = time.time()
            to_remove = []
            
            for base_url, creation_time in self._session_creation_times.items():
                if current_time - creation_time > max_idle_seconds:
                    to_remove.append(base_url)
            
            for base_url in to_remove:
                session = self._sessions.get(base_url)
                if session and not session.closed:
                    await session.close()
                    self.stats['connections_closed'] += 1
                
                del self._sessions[base_url]
                del self._session_creation_times[base_url]
                logger.info(f"Cleaned up idle session for {base_url}")
    
    async def start_cleanup_task(self, interval: int = 60):
        """Start background task to clean up idle connections."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.cleanup_idle_sessions()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def close(self):
        """Close all sessions and clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            for session in self._sessions.values():
                if not session.closed:
                    await session.close()
                    self.stats['connections_closed'] += 1
            
            self._sessions.clear()
            self._session_creation_times.clear()
        
        logger.info(f"Connection pool closed. Stats: {self.stats}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self.stats,
            'active_sessions': len(self._sessions),
            'base_urls': list(self._sessions.keys())
        }
    
    async def health_check(self, url: str, timeout: int = 5) -> bool:
        """
        Check if a URL is reachable.
        
        Args:
            url: URL to check
            timeout: Request timeout in seconds
            
        Returns:
            True if reachable, False otherwise
        """
        try:
            async with await self.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                return response.status < 500
        except Exception:
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_cleanup_task()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()