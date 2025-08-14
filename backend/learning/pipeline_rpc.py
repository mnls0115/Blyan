#!/usr/bin/env python3
"""
Pipeline RPC client/server helpers for activations/gradients with
timeouts, retries, and a simple circuit breaker.

Enhancements:
 - Environment-tunable retry/backoff/circuit-breaker parameters
 - Optional compression (gzip) for blobs
 - Optional chunked transfer for large activations/gradients
 - Basic backpressure via server buffer memory threshold
 - TLS/mTLS ready: loads client SSL context if provided

Transport: aiohttp HTTP JSON+binary stubs for now. Can be swapped for gRPC.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import ssl
import time
import gzip
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout_s: float = 10.0

    failures: int = 0
    opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.reset_timeout_s:
            # Half-open
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()


class PipelineRPCClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0, max_retries: Optional[int] = None, tls_context: Optional[ssl.SSLContext] = None):
        self.base_url = base_url.rstrip('/')
        # Allow env overrides
        self.timeout_s = float(os.getenv('BLYAN_PIPELINE_TIMEOUT_S', str(timeout_s)))
        self.max_retries = int(os.getenv('BLYAN_PIPELINE_MAX_RETRIES', str(max_retries if max_retries is not None else 2)))
        breaker_fail = int(os.getenv('BLYAN_PIPELINE_BREAKER_THRESHOLD', '5'))
        breaker_reset = float(os.getenv('BLYAN_PIPELINE_BREAKER_RESET_S', '10.0'))
        self.breaker = CircuitBreaker(failure_threshold=breaker_fail, reset_timeout_s=breaker_reset)
        self.tls_context = tls_context or build_client_ssl_context_from_env()
        self._session: Optional[aiohttp.ClientSession] = None
        # Chunk/compress parameters
        self._chunk_bytes = int(os.getenv('BLYAN_PIPELINE_CHUNK_BYTES', str(1 * 1024 * 1024)))  # 1 MiB
        self._compression = os.getenv('BLYAN_PIPELINE_COMPRESSION', 'none').lower()  # 'none' | 'gzip'
        self._backoff_base_s = float(os.getenv('BLYAN_PIPELINE_BACKOFF_BASE_S', '0.1'))

    async def __aenter__(self):
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()
            self._session = None

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.breaker.is_open():
            raise RuntimeError("circuit_open")

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                assert self._session is not None
                url = f"{self.base_url}{path}"
                async with self._session.post(url, json=payload, ssl=self.tls_context) as resp:
                    if resp.status >= 500:
                        raise RuntimeError(f"server_error:{resp.status}")
                    if resp.status >= 400:
                        # client error, don't retry
                        data = await resp.text()
                        raise RuntimeError(f"client_error:{resp.status}:{data}")
                    return await resp.json()
            except Exception as e:
                last_exc = e
                self.breaker.record_failure()
                if attempt < self.max_retries:
                    await asyncio.sleep(self._backoff_base_s * (2 ** attempt))
                else:
                    break
        raise last_exc or RuntimeError("rpc_failed")

    @staticmethod
    def _compress(blob: bytes, algo: str) -> Tuple[bytes, str]:
        algo = (algo or 'none').lower()
        if algo == 'gzip':
            return gzip.compress(blob), 'gzip'
        return blob, 'none'

    @staticmethod
    def _decompress(blob: bytes, algo: str) -> bytes:
        if algo == 'gzip':
            return gzip.decompress(blob)
        return blob

    @staticmethod
    def _encode_tensor(tensor_bytes: bytes) -> str:
        return base64.b64encode(tensor_bytes).decode('ascii')

    @staticmethod
    def _decode_tensor(b64: str) -> bytes:
        return base64.b64decode(b64.encode('ascii'))

    async def send_activations(self, stage_id: int, microbatch_id: str, tensor_bytes: bytes) -> Dict[str, Any]:
        return await self._send_blob_chunked("/pipeline/send_activations", stage_id, microbatch_id, tensor_bytes)

    async def recv_activations(self, stage_id: int, microbatch_id: str) -> bytes:
        payload = {"stage_id": stage_id, "microbatch_id": microbatch_id}
        data = await self._post_json("/pipeline/recv_activations", payload)
        return self._decode_tensor(data["activations_b64"])  # type: ignore[index]

    async def send_grads(self, stage_id: int, microbatch_id: str, tensor_bytes: bytes) -> Dict[str, Any]:
        return await self._send_blob_chunked("/pipeline/send_grads", stage_id, microbatch_id, tensor_bytes, is_grad=True)

    async def recv_grads(self, stage_id: int, microbatch_id: str) -> bytes:
        payload = {"stage_id": stage_id, "microbatch_id": microbatch_id}
        data = await self._post_json("/pipeline/recv_grads", payload)
        return self._decode_tensor(data["grads_b64"])  # type: ignore[index]

    async def _send_blob_chunked(self, path: str, stage_id: int, microbatch_id: str, blob: bytes, is_grad: bool = False) -> Dict[str, Any]:
        # Optional compression
        comp_blob, comp_algo = self._compress(blob, self._compression)
        # If small enough, send as single payload for backward compatibility
        if len(comp_blob) <= self._chunk_bytes:
            field = "grads_b64" if is_grad else "activations_b64"
            payload = {
                "stage_id": stage_id,
                "microbatch_id": microbatch_id,
                field: self._encode_tensor(comp_blob),
                "compression": comp_algo,
            }
            return await self._post_json(path, payload)

        # Chunked transfer
        total = (len(comp_blob) + self._chunk_bytes - 1) // self._chunk_bytes
        offset = 0
        for idx in range(total):
            chunk = comp_blob[offset: offset + self._chunk_bytes]
            offset += len(chunk)
            payload = {
                "stage_id": stage_id,
                "microbatch_id": microbatch_id,
                "chunk_index": idx,
                "chunks_total": total,
                "chunk_b64": self._encode_tensor(chunk),
                "compression": comp_algo,
                "is_grad": bool(is_grad),
            }
            # The server may respond with 429 if backpressured; handle with backoff
            last_exc: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._post_json(path, payload)
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < self.max_retries:
                        await asyncio.sleep(self._backoff_base_s * (2 ** attempt))
            if last_exc:
                raise last_exc
        # Finalize
        return {"status": "ok", "chunks": total}


class PipelineRPCServerBuffer:
    """
    In-memory buffer with optional chunk assembly and memory backpressure.
    Keyed by (stage_id, microbatch_id).
    """

    def __init__(self) -> None:
        self._acts: Dict[str, bytes] = {}
        self._grads: Dict[str, bytes] = {}
        self._partials: Dict[str, bytearray] = {}
        self._partials_meta: Dict[str, Tuple[int, str, bool]] = {}
        # meta: (chunks_total, compression_algo, is_grad)
        self._lock = asyncio.Lock()
        # Simple memory watermark for backpressure
        self._max_buffer_bytes = int(float(os.getenv('BLYAN_PIPELINE_MAX_BUFFER_MB', '256')) * 1024 * 1024)

    @staticmethod
    def _key(stage_id: int, microbatch_id: str) -> str:
        return f"{stage_id}:{microbatch_id}"

    def _current_mem_bytes(self) -> int:
        return sum(len(b) for b in self._acts.values()) + sum(len(b) for b in self._grads.values()) + sum(len(b) for b in self._partials.values())

    async def try_backpressure(self) -> bool:
        async with self._lock:
            return self._current_mem_bytes() < self._max_buffer_bytes

    async def put_acts(self, stage_id: int, microbatch_id: str, blob: bytes) -> None:
        async with self._lock:
            self._acts[self._key(stage_id, microbatch_id)] = blob

    async def get_acts(self, stage_id: int, microbatch_id: str) -> Optional[bytes]:
        async with self._lock:
            return self._acts.pop(self._key(stage_id, microbatch_id), None)

    async def put_grads(self, stage_id: int, microbatch_id: str, blob: bytes) -> None:
        async with self._lock:
            self._grads[self._key(stage_id, microbatch_id)] = blob

    async def get_grads(self, stage_id: int, microbatch_id: str) -> Optional[bytes]:
        async with self._lock:
            return self._grads.pop(self._key(stage_id, microbatch_id), None)

    async def append_chunk(self, stage_id: int, microbatch_id: str, idx: int, total: int, chunk: bytes, compression: str, is_grad: bool) -> bool:
        key = self._key(stage_id, microbatch_id)
        async with self._lock:
            # Backpressure
            if self._current_mem_bytes() + len(chunk) > self._max_buffer_bytes:
                return False
            buf = self._partials.get(key)
            if buf is None:
                buf = bytearray()
                self._partials[key] = buf
                self._partials_meta[key] = (total, compression, is_grad)
            buf.extend(chunk)
            # finalize if last chunk
            if idx + 1 >= total:
                total_chunks, algo, as_grad = self._partials_meta.pop(key)
                blob = bytes(self._partials.pop(key))
                # Decompress if needed
                if algo == 'gzip':
                    blob = gzip.decompress(blob)
                if as_grad:
                    self._grads[key] = blob
                else:
                    self._acts[key] = blob
        return True


def build_client_ssl_context_from_env() -> Optional[ssl.SSLContext]:
    """Build an SSL context from environment variables.

    Environment:
      - BLYAN_TLS_CERT: path to server CA or certificate bundle
      - BLYAN_TLS_CLIENT_CERT: optional client cert for mTLS
      - BLYAN_TLS_CLIENT_KEY: optional client key for mTLS
    """
    cert_path = os.getenv('BLYAN_TLS_CERT')
    if not cert_path:
        return None
    context = ssl.create_default_context(cafile=cert_path if os.path.isfile(cert_path) else None)
    # If cafile not a file, try load_verify_locations with directory
    if not os.path.isfile(cert_path):
        try:
            context.load_verify_locations(capath=cert_path)
        except Exception:
            pass
    client_cert = os.getenv('BLYAN_TLS_CLIENT_CERT')
    client_key = os.getenv('BLYAN_TLS_CLIENT_KEY')
    if client_cert and client_key:
        try:
            context.load_cert_chain(certfile=client_cert, keyfile=client_key)
        except Exception:
            pass
    return context

