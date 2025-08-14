#!/usr/bin/env python3
"""
Async gRPC client for Pipeline RPC with timeout/retry and simple circuit breaker.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
import os

import grpc

try:
    from proto import pipeline_rpc_pb2 as pb
    from proto import pipeline_rpc_pb2_grpc as pb_grpc
except Exception:
    pb = None
    pb_grpc = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 10.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures = 0
        self.opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.reset_timeout_s:
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()


class PipelineGrpcClient:
    def __init__(self, target: str, timeout_s: float = 5.0, max_retries: Optional[int] = None, tls_credentials: Optional[grpc.ChannelCredentials] = None):
        if not pb_grpc:
            raise RuntimeError("gRPC stubs not available; compile proto")
        self.target = target
        self.timeout_s = float(os.getenv('BLYAN_PIPELINE_TIMEOUT_S', str(timeout_s)))
        self.max_retries = int(os.getenv('BLYAN_PIPELINE_MAX_RETRIES', str(max_retries if max_retries is not None else 2)))
        breaker_fail = int(os.getenv('BLYAN_PIPELINE_BREAKER_THRESHOLD', '5'))
        breaker_reset = float(os.getenv('BLYAN_PIPELINE_BREAKER_RESET_S', '10.0'))
        self.breaker = CircuitBreaker(failure_threshold=breaker_fail, reset_timeout_s=breaker_reset)
        if tls_credentials:
            self.channel = grpc.aio.secure_channel(target, tls_credentials)
        else:
            self.channel = grpc.aio.insecure_channel(target)
        self.stub = pb_grpc.PipelineServiceStub(self.channel)

    async def close(self):
        await self.channel.close()

    async def send_activations(self, stage_id: int, microbatch_id: str, blob: bytes) -> bool:
        if self.breaker.is_open():
            raise RuntimeError("circuit_open")
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                await self.stub.SendActivations(pb.ActivationPacket(stage_id=stage_id, microbatch_id=microbatch_id, blob=blob), timeout=self.timeout_s)
                self.breaker.record_success()
                return True
            except Exception as e:
                self.breaker.record_failure()
                last_exc = e
                if attempt < self.max_retries:
                    base = float(os.getenv('BLYAN_PIPELINE_BACKOFF_BASE_S', '0.1'))
                    await asyncio.sleep(base * (2 ** attempt))
        raise last_exc or RuntimeError("send_activations_failed")

    async def recv_activations(self, stage_id: int, microbatch_id: str) -> bytes:
        if self.breaker.is_open():
            raise RuntimeError("circuit_open")
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.stub.RecvActivations(pb.ActivationKey(stage_id=stage_id, microbatch_id=microbatch_id), timeout=self.timeout_s)
                self.breaker.record_success()
                return bytes(resp.blob)
            except Exception as e:
                self.breaker.record_failure()
                last_exc = e
                if attempt < self.max_retries:
                    base = float(os.getenv('BLYAN_PIPELINE_BACKOFF_BASE_S', '0.1'))
                    await asyncio.sleep(base * (2 ** attempt))
        raise last_exc or RuntimeError("recv_activations_failed")

    async def send_grads(self, stage_id: int, microbatch_id: str, blob: bytes) -> bool:
        if self.breaker.is_open():
            raise RuntimeError("circuit_open")
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                await self.stub.SendGrads(pb.GradientPacket(stage_id=stage_id, microbatch_id=microbatch_id, blob=blob), timeout=self.timeout_s)
                self.breaker.record_success()
                return True
            except Exception as e:
                self.breaker.record_failure()
                last_exc = e
                if attempt < self.max_retries:
                    base = float(os.getenv('BLYAN_PIPELINE_BACKOFF_BASE_S', '0.1'))
                    await asyncio.sleep(base * (2 ** attempt))
        raise last_exc or RuntimeError("send_grads_failed")

    async def recv_grads(self, stage_id: int, microbatch_id: str) -> bytes:
        if self.breaker.is_open():
            raise RuntimeError("circuit_open")
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.stub.RecvGrads(pb.ActivationKey(stage_id=stage_id, microbatch_id=microbatch_id), timeout=self.timeout_s)
                self.breaker.record_success()
                return bytes(resp.blob)
            except Exception as e:
                self.breaker.record_failure()
                last_exc = e
                if attempt < self.max_retries:
                    base = float(os.getenv('BLYAN_PIPELINE_BACKOFF_BASE_S', '0.1'))
                    await asyncio.sleep(base * (2 ** attempt))
        raise last_exc or RuntimeError("recv_grads_failed")

