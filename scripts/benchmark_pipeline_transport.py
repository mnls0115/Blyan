#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import time

from backend.learning.pipeline_rpc import PipelineRPCClient
from backend.learning.pipeline_rpc_client_grpc import PipelineGrpcClient


async def bench_http(target: str, size_mb: int, iters: int) -> float:
    blob = b"a" * (size_mb * 1024 * 1024)
    start = time.time()
    async with PipelineRPCClient(target) as c:
        for i in range(iters):
            await c.send_activations(1, f"mb{i}", blob)
    return time.time() - start


async def bench_grpc(target: str, size_mb: int, iters: int) -> float:
    blob = b"a" * (size_mb * 1024 * 1024)
    start = time.time()
    client = PipelineGrpcClient(target)
    try:
        for i in range(iters):
            await client.send_activations(1, f"mb{i}", blob)
    finally:
        await client.close()
    return time.time() - start


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="base url (http[s]://host:port or host:port for grpc)")
    p.add_argument("--transport", choices=["http", "grpc"], default=os.getenv('BLYAN_PIPELINE_TRANSPORT', 'http'))
    p.add_argument("--size-mb", type=int, default=4)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if args.transport == "http":
        dur = loop.run_until_complete(bench_http(args.target, args.size_mb, args.iters))
    else:
        dur = loop.run_until_complete(bench_grpc(args.target, args.size_mb, args.iters))
    print(f"{args.transport} {args.size_mb}MB x{args.iters} -> {dur:.3f}s")


if __name__ == "__main__":
    main()

