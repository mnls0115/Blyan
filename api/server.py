from __future__ import annotations

import os
from pathlib import Path
from typing import Union, List, Dict

# Third-party libraries; ignore type checker if not present in local env
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore

# Built-ins / stdlib
import base64
import io

# Third-party
# silence type check errors if missing
import torch  # type: ignore
# ecdsa for signatures
import ecdsa  # type: ignore
import hashlib

# Internal token ledger
from backend.core.ledger import Ledger
from backend.core.param_index import ParameterIndex

from backend.core.chain import Chain
from backend.model.infer import ModelManager
from backend.model.moe_infer import MoEModelManager, ExpertUsageTracker, reward_expert
from backend.p2p.distributed_inference import DistributedInferenceCoordinator, ExpertNode
from backend.core.pol import evaluate_candidate
from backend.core.pol_validator import create_pol_validator, ChainValidator

# -------------------------------------------------
# ECDSA signature verification helper
# -------------------------------------------------


def _verify_signature(pub_hex: str, message: bytes, sig_hex: str) -> bool:
    try:
        vk = ecdsa.VerifyingKey.from_string(bytes.fromhex(pub_hex), curve=ecdsa.SECP256k1)
        sig = bytes.fromhex(sig_hex)
        return vk.verify(sig, message, hashfunc=hashlib.sha256)
    except Exception:
        return False


# ---------------------------------------------------------------------
# Init chains & model manager on startup
# ---------------------------------------------------------------------
root_dir = Path(os.getenv("AIBLOCK_DATA", "./data"))
# Check for development mode (skip PoW)
skip_pow = os.getenv("SKIP_POW", "false").lower() in ("true", "1", "yes")
if skip_pow:
    print("🚧 Running in development mode - PoW disabled")

# Check for PoL mode
enable_pol = os.getenv("ENABLE_POL", "false").lower() in ("true", "1", "yes")
if enable_pol:
    print("🧠 Proof-of-Learning validation enabled")

meta_chain = Chain(root_dir, "A", skip_pow=skip_pow)
param_chain = Chain(root_dir, "B", skip_pow=skip_pow)  # parameter chain for experts
# Parameter index
param_index = ParameterIndex(root_dir / "param_index.json")

# Token ledger (simple JSON file)
ledger = Ledger(root_dir / "ledger.json")

# Usage tracking for MoE experts
usage_tracker = ExpertUsageTracker(root_dir / "usage_log.json")

model_manager: ModelManager | None = None
moe_model_manager: MoEModelManager | None = None
distributed_coordinator: DistributedInferenceCoordinator | None = None
chain_validator: ChainValidator | None = None

# Initialize PoL validator if enabled
if enable_pol:
    try:
        # Create temporary MoE manager for PoL validator initialization
        temp_moe_manager = MoEModelManager(meta_chain, param_chain, param_index, usage_tracker)
        
        chain_validator = create_pol_validator(
            model_manager=temp_moe_manager,
            enable_pol=True,
            pol_threshold=float(os.getenv("POL_THRESHOLD", "0.01"))  # 1% improvement threshold
        )
        print(f"✅ PoL validator initialized with threshold {float(os.getenv('POL_THRESHOLD', '0.01'))*100:.1f}%")
        
        # Update parameter chain to use PoL validator
        param_chain.chain_validator = chain_validator
        param_chain.enable_pol = True
        
    except Exception as e:
        print(f"⚠️  Failed to initialize PoL validator: {e}")
        enable_pol = False

app = FastAPI(title="AI-Blockchain Prototype")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    use_moe: bool = True  # Use MoE inference by default
    use_distributed: bool = False  # Use distributed inference
    top_k_experts: int = 2  # Number of experts to use per layer


class ChatResponse(BaseModel):
    response: str
    expert_usage: dict = {}  # Track which experts were used
    inference_time: float = 0.0


# ---------------------------------------------------------------------
# Mining endpoint
# ---------------------------------------------------------------------


class MineRequest(BaseModel):
    miner_address: str
    miner_pub: str  # hex compressed public key
    payload_sig: str  # hex signature of candidate_payload_b64 decoded bytes
    candidate_payload_b64: str
    candidate_loss: float
    previous_loss: Union[float, None] = None


class MineResponse(BaseModel):
    block_hash: str
    reward: float
    balance: float


# ---------------------------------------------------------------------
# Balance endpoint
# ---------------------------------------------------------------------


class BalanceResponse(BaseModel):
    address: str
    balance: float


# ============================== Transfers ==============================


class TransferRequest(BaseModel):
    sender: str
    receiver: str
    amount: float


class TransferResponse(BaseModel):
    sender_balance: float
    receiver_balance: float


@app.post("/transfer", response_model=TransferResponse)
async def transfer(req: TransferRequest):
    try:
        ledger.transfer(req.sender, req.receiver, req.amount)
        sender_bal = ledger.get_balance(req.sender)
        receiver_bal = ledger.get_balance(req.receiver)
        return TransferResponse(sender_balance=sender_bal, receiver_balance=receiver_bal)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ============================== Chain view =============================


class BlockMeta(BaseModel):
    index: int
    hash: str
    timestamp: float
    payload_size: int
    chain_id: str
    points_to: Union[str, None]
    block_type: Union[str, None] = None
    expert_name: Union[str, None] = None
    layer_id: Union[str, None] = None


class ChainBlocksResponse(BaseModel):
    blocks: List[BlockMeta]


def _resolve_chain(chain_id: str) -> Chain:
    chain_id = chain_id.upper()
    if chain_id == "A":
        return meta_chain
    if chain_id == "B":
        return param_chain
    raise HTTPException(status_code=404, detail="Unknown chain id")


@app.get("/chain/{chain_id}/blocks", response_model=ChainBlocksResponse)
async def get_chain_blocks(chain_id: str, limit: int = 10):
    chain = _resolve_chain(chain_id)
    blocks = list(chain.storage.iter_blocks())
    # sort by index descending and limit
    blocks_sorted = sorted(blocks, key=lambda b: b.header.index, reverse=True)[:limit]
    res_blocks = [
        BlockMeta(
            index=b.header.index,
            hash=b.compute_hash(),
            timestamp=b.header.timestamp,
            payload_size=b.header.payload_size,
            chain_id=b.header.chain_id,
            points_to=b.header.points_to,
            block_type=getattr(b.header, 'block_type', None),
            expert_name=getattr(b.header, 'expert_name', None),
            layer_id=getattr(b.header, 'layer_id', None),
        )
        for b in blocks_sorted
    ]
    return ChainBlocksResponse(blocks=res_blocks)


# PoL-specific endpoints

@app.get("/pol/status")
async def get_pol_status():
    """Get PoL system status and configuration."""
    return {
        "pol_enabled": enable_pol,
        "pol_threshold": float(os.getenv("POL_THRESHOLD", "0.01")),
        "skip_pow": skip_pow,
        "validator_initialized": chain_validator is not None,
        "validation_data_dir": str(root_dir / "validation_data") if enable_pol else None
    }


class PoLValidationRequest(BaseModel):
    expert_name: str
    layer_id: str
    block_hash: str


@app.post("/pol/validate")
async def validate_expert_block(request: PoLValidationRequest):
    """Manually validate an expert block using PoL."""
    if not enable_pol or not chain_validator:
        raise HTTPException(status_code=400, detail="PoL validation not enabled")
    
    try:
        # Find the block to validate
        target_block = None
        for block in param_chain.storage.iter_blocks():
            if block.compute_hash() == request.block_hash:
                target_block = block
                break
        
        if not target_block:
            raise HTTPException(status_code=404, detail="Block not found")
        
        # Run PoL validation
        is_valid, validation_details = chain_validator.validate_block(target_block)
        
        return {
            "block_hash": request.block_hash,
            "expert_name": request.expert_name,
            "is_valid": is_valid,
            "validation_details": validation_details
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


# Get single block metadata and optional payload


class BlockDetailResponse(BaseModel):
    header: dict
    payload_b64: Union[str, None] = None


@app.get("/chain/{chain_id}/block/{index}", response_model=BlockDetailResponse)
async def get_block(chain_id: str, index: int, include_payload: bool = False):
    chain = _resolve_chain(chain_id)
    blk = chain.storage.load_block(index)
    if blk is None:
        raise HTTPException(status_code=404, detail="Block not found")
    header = blk.header.__dict__
    payload_b64 = base64.b64encode(blk.payload).decode() if include_payload else None
    return BlockDetailResponse(header=header, payload_b64=payload_b64)


@app.on_event("startup")
def _startup():
    global model_manager, moe_model_manager, distributed_coordinator
    model_manager = ModelManager(meta_chain, param_chain, param_index)
    moe_model_manager = MoEModelManager(meta_chain, param_chain, param_index, usage_tracker)
    distributed_coordinator = DistributedInferenceCoordinator(usage_tracker)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    import time
    start_time = time.time()
    
    try:
        if req.use_moe and moe_model_manager is not None:
            # Use MoE inference with selective expert loading
            answer, expert_usage = moe_model_manager.selective_generate(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                top_k_experts=req.top_k_experts
            )
            
            inference_time = time.time() - start_time
            
            return ChatResponse(
                response=answer,
                expert_usage=expert_usage,
                inference_time=inference_time
            )
        
        else:
            # Fallback to standard model manager
            if model_manager is None:
                raise HTTPException(status_code=500, detail="Model manager not initialized")
            
            answer = model_manager.generate(req.prompt, max_new_tokens=req.max_new_tokens)
            inference_time = time.time() - start_time
            
            return ChatResponse(
                response=answer,
                expert_usage={},
                inference_time=inference_time
            )
            
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------ Mining ------------------------------


@app.post("/mine", response_model=MineResponse)
async def mine(req: MineRequest):
    """Submit candidate payload (weights) + validation loss.

    If PoL check passes, the payload is stored as a new block on the parameter
    chain. Returns the resulting block hash.
    """
    # 1) PoL check
    passed = evaluate_candidate(
        candidate_loss_fn=lambda: req.candidate_loss,
        previous_loss=req.previous_loss,
    )
    if not passed:
        raise HTTPException(status_code=400, detail="PoL check failed: insufficient improvement")

    # 2) Decode payload & verify sig
    try:
        payload = base64.b64decode(req.candidate_payload_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload")

    if not _verify_signature(req.miner_pub, payload, req.payload_sig):
        raise HTTPException(status_code=400, detail="Signature invalid")

    # 3) Reference latest meta block
    latest_meta = meta_chain.storage.get_latest_block()
    if latest_meta is None:
        raise HTTPException(status_code=500, detail="Meta chain empty; cannot link")

    points_to = latest_meta.compute_hash()

    # 4) Add block to parameter chain (mining includes PoW)
    try:
        new_block = param_chain.add_block(
            payload,
            points_to=points_to,
            miner_pub=req.miner_pub,
            payload_sig=req.payload_sig,
        )

        # 5) Reward miner
        reward = 10.0  # fixed reward for prototype
        ledger.credit(req.miner_address, reward)
        balance = ledger.get_balance(req.miner_address)

        return MineResponse(
            block_hash=new_block.compute_hash(),
            reward=reward,
            balance=balance,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------
# Bulk parameter upload endpoint (server-side chunking)
# ---------------------------------------------------------------------


class UploadParamsRequest(BaseModel):
    miner_address: str
    miner_pub: str
    file_b64: str
    payload_sig: str  # signature over raw file bytes
    candidate_loss: float
    previous_loss: Union[float, None] = None


class UploadParamsResponse(BaseModel):
    block_hashes: List[str]
    reward: float
    balance: float


@app.post("/upload_parameters", response_model=UploadParamsResponse)
async def upload_parameters(req: UploadParamsRequest):
    # 1. PoL check (single evaluation per entire file)
    passed = evaluate_candidate(
        candidate_loss_fn=lambda: req.candidate_loss,
        previous_loss=req.previous_loss,
    )
    if not passed:
        raise HTTPException(status_code=400, detail="PoL failed: insufficient improvement")

    # 2. Decode
    try:
        full_bytes = base64.b64decode(req.file_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 file")

    if not _verify_signature(req.miner_pub, full_bytes, req.payload_sig):
        raise HTTPException(status_code=400, detail="Signature invalid")

    # Attempt to load torch state_dict
    try:
        import backend.model.arch as arch  # local import to avoid top-level torch requirement

        state_dict = arch.bytes_to_state_dict(full_bytes)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Unable to parse state_dict: {ex}")

    if not isinstance(state_dict, dict):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid state_dict")

    latest_meta = meta_chain.storage.get_latest_block()
    if latest_meta is None:
        raise HTTPException(status_code=500, detail="Meta chain empty")
    points_to = latest_meta.compute_hash()

    block_hashes: List[str] = []
    param_count = 0
    try:
        mapping: Dict[str, int] = {}
        for name, tensor in state_dict.items():
            buf = io.BytesIO()
            torch.save({name: tensor}, buf)
            chunk = buf.getvalue()
            blk = param_chain.add_block(
                chunk,
                points_to=points_to,
                miner_pub=req.miner_pub,
                payload_sig=req.payload_sig,
            )
            block_hashes.append(blk.compute_hash())
            mapping[name] = blk.header.index
            param_count += 1
        # update index
        param_index.bulk_set(mapping)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # reward proportional to number of parameters
    reward_per_param = 5.0
    total_reward = reward_per_param * param_count
    ledger.credit(req.miner_address, total_reward)
    balance = ledger.get_balance(req.miner_address)

    return UploadParamsResponse(
        block_hashes=block_hashes,
        reward=total_reward,
        balance=balance,
    )


# ------------------------------ MoE Expert Upload ------------------------------


class MoEExpertRequest(BaseModel):
    miner_address: str
    miner_pub: str
    payload_sig: str
    expert_name: str
    layer_id: str
    block_type: str  # 'expert' or 'router'
    depends_on: List[str]
    tensor_data_b64: str
    candidate_loss: float
    previous_loss: Union[float, None] = None


class MoEExpertResponse(BaseModel):
    block_hash: str
    reward: float
    balance: float


@app.post("/upload_moe_experts", response_model=MoEExpertResponse)
async def upload_moe_expert(req: MoEExpertRequest):
    """Upload a single MoE expert or router block to the DAG chain."""
    
    # 1. Enhanced PoL check (if PoL validator is available)
    if enable_pol and chain_validator:
        print(f"🧠 Using advanced PoL validation for {req.expert_name}")
        # PoL validation will be handled by the chain during block creation
    else:
        # Fallback to simple PoL check
        passed = evaluate_candidate(
            candidate_loss_fn=lambda: req.candidate_loss,
            previous_loss=req.previous_loss,
        )
        if not passed:
            raise HTTPException(status_code=400, detail="PoL failed: insufficient improvement")
    
    # 2. Validate block type
    if req.block_type not in ['expert', 'router']:
        raise HTTPException(status_code=400, detail="block_type must be 'expert' or 'router'")
    
    # 3. Decode and verify tensor data
    try:
        tensor_bytes = base64.b64decode(req.tensor_data_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 tensor data")
    
    if not _verify_signature(req.miner_pub, tensor_bytes, req.payload_sig):
        raise HTTPException(status_code=400, detail="Signature verification failed")
    
    # 4. Validate tensor data
    try:
        import backend.model.arch as arch
        tensor_dict = arch.bytes_to_state_dict(tensor_bytes)
        if not isinstance(tensor_dict, dict):
            raise ValueError("Invalid tensor format")
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Invalid tensor data: {ex}")
    
    # 5. Validate dependencies exist
    for dep_hash in req.depends_on:
        # Check if dependency exists in meta chain or param chain
        meta_blocks = list(meta_chain.storage.iter_blocks())
        param_blocks = list(param_chain.storage.iter_blocks())
        all_hashes = {block.compute_hash() for block in meta_blocks + param_blocks}
        
        if dep_hash not in all_hashes:
            raise HTTPException(
                status_code=400, 
                detail=f"Dependency {dep_hash} not found in blockchain"
            )
    
    # 6. Create and add block to parameter chain
    try:
        new_block = param_chain.add_block(
            payload=tensor_bytes,
            points_to=req.depends_on[0] if req.depends_on else None,  # Point to first dependency
            miner_pub=req.miner_pub,
            payload_sig=req.payload_sig,
            depends_on=req.depends_on,
            block_type=req.block_type,
            expert_name=req.expert_name,
            layer_id=req.layer_id,
        )
        
        # 7. Update parameter index
        param_index.set(req.expert_name, new_block.header.index)
        
        # 8. Calculate reward based on block type and complexity
        base_reward = 15.0 if req.block_type == 'expert' else 10.0  # Experts get higher reward
        tensor_count = len(tensor_dict)
        complexity_bonus = min(tensor_count * 2.0, 20.0)  # Bonus for tensor complexity
        total_reward = base_reward + complexity_bonus
        
        ledger.credit(req.miner_address, total_reward)
        balance = ledger.get_balance(req.miner_address)
        
        return MoEExpertResponse(
            block_hash=new_block.compute_hash(),
            reward=total_reward,
            balance=balance
        )
        
    except Exception as exc:
        # Enhanced error handling for PoL validation failures
        error_msg = str(exc)
        if "PoL validation failed" in error_msg:
            raise HTTPException(status_code=422, detail=f"Expert quality insufficient: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to create block: {exc}")


# ------------------------------ Expert Analytics ------------------------------


class ExpertStatsResponse(BaseModel):
    expert_name: str
    call_count: int
    average_response_time: float
    quality_score: float
    last_used: float
    current_reward_multiplier: float


class TopExpertsResponse(BaseModel):
    experts: List[ExpertStatsResponse]


@app.get("/experts/stats/{expert_name}", response_model=ExpertStatsResponse)
async def get_expert_stats(expert_name: str):
    """Get usage statistics for a specific expert."""
    stats = usage_tracker.get_expert_stats(expert_name)
    if not stats:
        raise HTTPException(status_code=404, detail="Expert not found")
    
    reward_multiplier = reward_expert(expert_name, usage_tracker)
    
    return ExpertStatsResponse(
        expert_name=stats.expert_name,
        call_count=stats.call_count,
        average_response_time=stats.average_response_time,
        quality_score=stats.quality_score,
        last_used=stats.last_used,
        current_reward_multiplier=reward_multiplier
    )


@app.get("/experts/top", response_model=TopExpertsResponse)
async def get_top_experts(limit: int = 10):
    """Get top experts by usage."""
    top_experts = usage_tracker.get_top_experts(limit)
    
    expert_responses = []
    for stats in top_experts:
        reward_multiplier = reward_expert(stats.expert_name, usage_tracker)
        expert_responses.append(ExpertStatsResponse(
            expert_name=stats.expert_name,
            call_count=stats.call_count,
            average_response_time=stats.average_response_time,
            quality_score=stats.quality_score,
            last_used=stats.last_used,
            current_reward_multiplier=reward_multiplier
        ))
    
    return TopExpertsResponse(experts=expert_responses)


@app.post("/experts/reward/{expert_name}")
async def reward_expert_endpoint(expert_name: str, base_reward: float = 10.0):
    """Manually trigger expert reward calculation."""
    if not usage_tracker.get_expert_stats(expert_name):
        raise HTTPException(status_code=404, detail="Expert not found")
    
    reward_amount = reward_expert(expert_name, usage_tracker, base_reward)
    
    # Find the expert's miner address from the latest block
    expert_blocks = param_chain.get_expert_blocks(expert_name)
    if expert_blocks:
        latest_block = max(expert_blocks, key=lambda b: b.header.timestamp)
        if latest_block.miner_pub:
            # For demo, use miner_pub as address (in practice, derive address from pubkey)
            miner_address = latest_block.miner_pub[:16]  # Truncated for demo
            ledger.credit(miner_address, reward_amount)
            balance = ledger.get_balance(miner_address)
            
            return {
                "expert_name": expert_name,
                "reward_amount": reward_amount,
                "miner_address": miner_address,
                "new_balance": balance
            }
    
    return {
        "expert_name": expert_name,
        "reward_amount": reward_amount,
        "message": "No miner found for reward distribution"
    }


# ------------------------------ Distributed Inference ------------------------------


class RegisterNodeRequest(BaseModel):
    node_id: str
    host: str
    port: int
    available_experts: List[str]


class NodeRegistrationResponse(BaseModel):
    status: str
    message: str
    registered_experts: int


@app.post("/p2p/register", response_model=NodeRegistrationResponse)
async def register_expert_node(req: RegisterNodeRequest):
    """Register a new expert node for distributed inference."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    node = ExpertNode(
        node_id=req.node_id,
        host=req.host,
        port=req.port,
        available_experts=req.available_experts
    )
    
    distributed_coordinator.registry.register_node(node)
    
    return NodeRegistrationResponse(
        status="success",
        message=f"Node {req.node_id} registered successfully",
        registered_experts=len(req.available_experts)
    )


@app.delete("/p2p/nodes/{node_id}")
async def unregister_expert_node(node_id: str):
    """Unregister an expert node."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    distributed_coordinator.registry.unregister_node(node_id)
    
    return {"status": "success", "message": f"Node {node_id} unregistered"}


@app.get("/p2p/nodes")
async def list_expert_nodes():
    """List all registered expert nodes."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    nodes = []
    for node_id, node in distributed_coordinator.registry.nodes.items():
        nodes.append({
            "node_id": node.node_id,
            "endpoint": node.endpoint,
            "available_experts": node.available_experts,
            "load_factor": node.load_factor,
            "last_heartbeat": node.last_heartbeat
        })
    
    return {"nodes": nodes}


@app.post("/p2p/heartbeat/{node_id}")
async def node_heartbeat(node_id: str, load_factor: float = 0.0):
    """Update heartbeat and load for a node."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    distributed_coordinator.registry.heartbeat(node_id)
    distributed_coordinator.registry.update_node_load(node_id, load_factor)
    
    return {"status": "heartbeat_received", "node_id": node_id, "load_factor": load_factor}


@app.post("/chat/distributed")
async def distributed_chat(req: ChatRequest):
    """Chat using distributed inference across expert nodes."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # For demo, select some experts to use (in practice, this would be determined by routing)
        available_experts = []
        for expert_name in distributed_coordinator.registry.expert_to_nodes.keys():
            available_experts.append(expert_name)
        
        if not available_experts:
            raise HTTPException(status_code=503, detail="No expert nodes available")
        
        # Select top-k experts for this request
        selected_experts = available_experts[:req.top_k_experts]
        
        # Perform distributed inference
        response_text, expert_usage = await distributed_coordinator.distribute_inference(
            prompt=req.prompt,
            required_experts=selected_experts,
            max_new_tokens=req.max_new_tokens
        )
        
        inference_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            expert_usage=expert_usage,
            inference_time=inference_time
        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------ Ledger ------------------------------


@app.get("/balance/{address}", response_model=BalanceResponse)
async def get_balance(address: str):
    try:
        bal = ledger.get_balance(address)
        return BalanceResponse(address=address, balance=bal)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) 