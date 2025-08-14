#!/usr/bin/env python3
"""
gRPC implementation for Pipeline RPC.
Note: Assumes generated stubs from proto/pipeline_rpc.proto are available.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import grpc

try:
    from proto import pipeline_rpc_pb2 as pb
    from proto import pipeline_rpc_pb2_grpc as pb_grpc
except Exception:
    pb = None
    pb_grpc = None


class PipelineGrpcServer(pb_grpc.PipelineServiceServicer if pb_grpc else object):
    def __init__(self, buffer):
        self.buffer = buffer

    async def SendActivations(self, request, context):  # type: ignore[override]
        await self.buffer.put_acts(request.stage_id, request.microbatch_id, bytes(request.blob))
        return pb.Ack(ok=True)

    async def RecvActivations(self, request, context):  # type: ignore[override]
        blob = await self.buffer.get_acts(request.stage_id, request.microbatch_id)
        if blob is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('not_found')
            return pb.ActivationPacket()
        return pb.ActivationPacket(stage_id=request.stage_id, microbatch_id=request.microbatch_id, blob=blob)

    async def SendGrads(self, request, context):  # type: ignore[override]
        await self.buffer.put_grads(request.stage_id, request.microbatch_id, bytes(request.blob))
        return pb.Ack(ok=True)

    async def RecvGrads(self, request, context):  # type: ignore[override]
        blob = await self.buffer.get_grads(request.stage_id, request.microbatch_id)
        if blob is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('not_found')
            return pb.GradientPacket()
        return pb.GradientPacket(stage_id=request.stage_id, microbatch_id=request.microbatch_id, blob=blob)


async def serve(buffer, host: str = "0.0.0.0", port: int = 50061, tls: Optional[grpc.ssl_server_credentials] = None):
    if not pb_grpc:
        raise RuntimeError("gRPC stubs not generated; compile proto/pipeline_rpc.proto")
    server = grpc.aio.server()
    pb_grpc.add_PipelineServiceServicer_to_server(PipelineGrpcServer(buffer), server)
    addr = f"{host}:{port}"
    if tls:
        server.add_secure_port(addr, tls)
    else:
        server.add_insecure_port(addr)
    await server.start()
    await server.wait_for_termination()

