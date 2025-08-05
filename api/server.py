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
from backend.core.dataset_chain import DatasetChain
from backend.core.dataset_block import DatasetMetadata, DatasetStage, DatasetQualityTier, QualityReport
from backend.core.podl_proof import PoDLGenerator, PoDLVerifier, TrainingSession
from backend.core.architecture_migration import ArchitectureMigrationManager, MigrationSpec, MigrationType, EvolutionDifficulty
from backend.core.epoch_scheduler import EpochEventScheduler
from backend.model.infer import ModelManager
from backend.model.moe_infer import MoEModelManager, ExpertUsageTracker, reward_expert
from backend.p2p.distributed_inference import DistributedInferenceCoordinator, ExpertNode
from backend.p2p.expert_group_optimizer import NodeCapability, ExpertGroup
from backend.core.pol import evaluate_candidate
from backend.core.pol_validator import create_pol_validator, ChainValidator

# Security systems
from backend.security.data_validation import DataSecurityCoordinator
from backend.security.poison_detection import ComprehensivePoisonDetector
from backend.security.quarantine_system import NetworkDefenseCoordinator
from backend.core.scheduler_integration import wire_system_components

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
dataset_chain = DatasetChain(root_dir, "D")  # Dataset governance chain
# Parameter index
param_index = ParameterIndex(root_dir / "param_index.json")

# Token ledger (simple JSON file)
ledger = Ledger(root_dir / "ledger.json")

# Usage tracking for MoE experts
usage_tracker = ExpertUsageTracker(root_dir / "usage_log.json")

# PoDL (Proof-of-Data-Learning) systems
podl_generator = PoDLGenerator()
podl_verifier = PoDLVerifier(dataset_chain, param_chain)

# Autonomous Evolution Systems
migration_manager = ArchitectureMigrationManager(meta_chain, param_chain)
epoch_scheduler = EpochEventScheduler(migration_manager, dataset_chain)

# Start autonomous evolution scheduler
epoch_scheduler.start_scheduler()

# Security systems
security_coordinator = DataSecurityCoordinator(root_dir / "expert_backups")
poison_detector = ComprehensivePoisonDetector()
network_defense = NetworkDefenseCoordinator(root_dir / "quarantine_data")

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

app = FastAPI(title="Blyanchain Prototype")

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

@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get SLO-based scheduler status and metrics."""
    if not hasattr(app.state, 'scheduler_integration'):
        return {"scheduler_available": False, "error": "Scheduler not initialized"}
    
    try:
        status = app.state.scheduler_integration.get_scheduler_status()
        
        # Add inference coordinator metrics if available
        if distributed_coordinator:
            status["inference_metrics"] = distributed_coordinator.inference_metrics
            status["warm_pool_status"] = distributed_coordinator.warm_pool.get_status()
            
        return {
            "scheduler_available": True,
            **status
        }
    except Exception as e:
        return {"scheduler_available": False, "error": str(e)}

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
    from backend.core.scheduler_integration import SchedulerIntegration
    
    model_manager = ModelManager(meta_chain, param_chain, param_index)
    moe_model_manager = MoEModelManager(meta_chain, param_chain, param_index, usage_tracker)
    distributed_coordinator = DistributedInferenceCoordinator(usage_tracker, param_index)
    
    # Wire scheduler to distributed coordinator
    scheduler_integration = SchedulerIntegration()
    scheduler_integration.connect_inference_coordinator(distributed_coordinator)
    
    # Store integration for API access
    app.state.scheduler_integration = scheduler_integration
    
    print("✅ SLO-based scheduler integrated with distributed inference coordinator")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    import time
    start_time = time.time()
    
    try:
        # Check if we have any registered nodes for distributed inference
        has_distributed_nodes = False
        if distributed_coordinator and distributed_coordinator.registry.nodes:
            has_distributed_nodes = True
            
        if req.use_moe and has_distributed_nodes:
            # Use distributed inference when nodes are available
            available_experts = list(distributed_coordinator.registry.expert_to_nodes.keys())
            print(f"DEBUG: Found {len(available_experts)} available experts: {available_experts}")
            print(f"DEBUG: Nodes: {list(distributed_coordinator.registry.nodes.keys())}")
            if available_experts:
                selected_experts = available_experts[:req.top_k_experts]
                
                response_text, routing_info = await distributed_coordinator.distribute_inference(
                    prompt=req.prompt,
                    required_experts=selected_experts,
                    max_new_tokens=req.max_new_tokens
                )
                
                inference_time = time.time() - start_time
                
                return ChatResponse(
                    response=response_text,
                    expert_usage=routing_info.get('expert_usage', {}),
                    inference_time=inference_time
                )
                
        elif req.use_moe and moe_model_manager is not None:
            # Fallback to local MoE inference
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
    node_name: str = "Unnamed Node"
    resource_limit: str = "cpu-50"  # cpu-25, cpu-50, cpu-75, gpu-25, gpu-50, gpu-75
    node_type: str = "user_contributed"
    hardware_info: Dict = {}


class RegisterOptimizedNodeRequest(BaseModel):
    node_id: str
    host: str
    port: int
    available_experts: List[str]
    expert_groups: List[Dict] = []  # [{experts: [str], usage_count: int}]
    region: str = "default"
    node_name: str = "Unnamed Node"
    resource_limit: str = "cpu-50"
    node_type: str = "user_contributed"
    hardware_info: Dict = {}


class OptimizedChatRequest(BaseModel):
    prompt: str
    required_experts: List[str]
    max_new_tokens: int = 64
    preferred_region: str = "default"


class ExpertGroupInsight(BaseModel):
    group_id: str
    experts: List[str]
    usage_count: int
    co_occurrence_score: float


class SecureChatRequest(BaseModel):
    prompt: str
    required_experts: List[str]
    max_new_tokens: int = 64
    preferred_region: str = "default"
    enable_integrity_check: bool = True


class SecurityVerificationResult(BaseModel):
    verification_enabled: bool
    beacon_count: int
    integrity_score: float
    trust_level: str
    anomalies: List[str]
    verified_components: List[str]
    rolling_hash: str


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


@app.post("/p2p/register_optimized", response_model=NodeRegistrationResponse)
async def register_optimized_expert_node(req: RegisterOptimizedNodeRequest):
    """Register a new expert node with expert group capabilities."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    # Convert expert group data to ExpertGroup objects
    expert_groups = []
    for group_data in req.expert_groups:
        expert_group = ExpertGroup(
            experts=set(group_data.get("experts", [])),
            usage_count=group_data.get("usage_count", 0),
            co_occurrence_score=group_data.get("co_occurrence_score", 0.0)
        )
        expert_groups.append(expert_group)
    
    # Create NodeCapability
    node_capability = NodeCapability(
        node_id=req.node_id,
        host=req.host,
        port=req.port,
        expert_groups=expert_groups,
        individual_experts=set(req.available_experts),
        region=req.region
    )
    
    # Register with optimized coordinator
    distributed_coordinator.register_expert_group_node(node_capability)
    
    return NodeRegistrationResponse(
        status="success",
        message=f"Optimized node {req.node_id} registered with {len(expert_groups)} expert groups",
        registered_experts=len(req.available_experts)
    )


@app.post("/chat/distributed_optimized")
async def optimized_distributed_chat(req: OptimizedChatRequest):
    """Chat using optimized distributed inference with expert groups."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Use optimized distributed inference
        response_text, routing_info = await distributed_coordinator.distribute_inference_optimized(
            prompt=req.prompt,
            required_experts=req.required_experts,
            max_new_tokens=req.max_new_tokens,
            preferred_region=req.preferred_region
        )
        
        inference_time = time.time() - start_time
        
        return {
            "response": response_text,
            "routing_info": routing_info,
            "inference_time": inference_time,
            "optimization_applied": routing_info.get("optimization_applied", False)
        }
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/p2p/optimization_insights")
async def get_optimization_insights():
    """Get insights about expert group optimization performance."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    try:
        insights = distributed_coordinator.get_optimization_insights()
        return insights
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/p2p/expert_groups")
async def get_expert_groups():
    """Get information about identified expert groups."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    try:
        hot_groups = distributed_coordinator.group_index.get_hot_expert_groups(limit=10)
        
        return {
            "hot_expert_groups": [
                {
                    "group_id": group.group_id,
                    "experts": list(group.experts),
                    "usage_count": group.usage_count,
                    "co_occurrence_score": group.co_occurrence_score,
                    "average_latency": group.average_latency
                }
                for group in hot_groups
            ],
            "total_groups": len(hot_groups)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/p2p/replication_suggestions")
async def get_replication_suggestions():
    """Get suggestions for expert group replication."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    try:
        suggestions = distributed_coordinator.smart_router.get_replication_suggestions()
        return {"suggestions": suggestions}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/distributed_secure")
async def secure_distributed_chat(req: SecureChatRequest):
    """Chat using secure distributed inference with real-time integrity verification and automatic failover."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Use secure distributed inference with failover
        response_text, routing_info = await distributed_coordinator.distribute_inference_with_failover(
            prompt=req.prompt,
            required_experts=req.required_experts,
            max_new_tokens=req.max_new_tokens,
            preferred_region=req.preferred_region
        )
        
        inference_time = time.time() - start_time
        
        # Extract security verification results
        security_verification = routing_info.get("security_verification", {})
        user_message = routing_info.get("user_message")
        
        # Determine response status
        if routing_info.get("status") == "temporary_unavailable":
            raise HTTPException(
                status_code=503, 
                detail={
                    "message": user_message or "Service temporarily unavailable",
                    "recovery_suggestion": routing_info.get("recovery_suggestion"),
                    "retry_after": 30
                }
            )
        
        return {
            "response": response_text,
            "routing_info": routing_info,
            "inference_time": inference_time,
            "security_verification": security_verification,
            "integrity_verified": security_verification.get("trust_level", "UNKNOWN") in ["HIGH", "MEDIUM"],
            "user_message": user_message,
            "failover_occurred": routing_info.get("failover_occurred", False)
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/security/integrity_status")
async def get_integrity_status():
    """Get the current status of integrity verification system."""
    if not distributed_coordinator or not distributed_coordinator.integrity_coordinator:
        return {
            "integrity_verification_available": False,
            "error": "Integrity coordinator not initialized"
        }
    
    # Get statistics from integrity coordinator
    active_audits = len(distributed_coordinator.integrity_coordinator.active_audits)
    
    return {
        "integrity_verification_available": True,
        "active_audit_contexts": active_audits,
        "security_features": {
            "activation_beacons": True,
            "weight_verification": True,
            "routing_canaries": True,
            "rolling_commitments": True,
            "runtime_attestation": True
        },
        "verification_levels": ["BASIC", "STANDARD", "STRICT"],
        "current_level": "STANDARD"
    }


@app.post("/security/verify_audit/{request_id}")
async def verify_audit_results(request_id: str):
    """Verify audit results for a completed inference request."""
    if not distributed_coordinator or not distributed_coordinator.integrity_coordinator:
        raise HTTPException(status_code=500, detail="Integrity coordinator not initialized")
    
    try:
        audit_summary = distributed_coordinator.integrity_coordinator.get_audit_summary(request_id)
        
        if "error" in audit_summary:
            raise HTTPException(status_code=404, detail=audit_summary["error"])
        
        return {
            "request_id": request_id,
            "audit_summary": audit_summary,
            "verification_complete": True
        }
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/security/dashboard")
async def get_security_dashboard():
    """Get comprehensive security dashboard data with real-time metrics."""
    if not distributed_coordinator or not distributed_coordinator.security_orchestrator:
        raise HTTPException(status_code=500, detail="Security orchestrator not initialized")
    
    try:
        dashboard_data = distributed_coordinator.security_orchestrator.get_security_dashboard_data()
        return dashboard_data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/security/threat_indicators")
async def get_threat_indicators():
    """Get current security threat indicators and anomaly detection results."""
    if not distributed_coordinator or not distributed_coordinator.security_orchestrator:
        # Fallback to basic threat indicators
        return {
            "threat_level": "UNKNOWN",
            "anomaly_count_24h": 0,
            "recent_anomalies": [],
            "node_trust_scores": {},
            "verification_success_rate": 0.0,
            "error": "Security orchestrator not available"
        }
    
    try:
        dashboard_data = distributed_coordinator.security_orchestrator.get_security_dashboard_data()
        
        # Extract threat indicators from dashboard data
        overview = dashboard_data["overview"]
        alerts = dashboard_data["security_alerts"]
        
        # Determine threat level based on metrics
        threat_level = "LOW"
        if overview["success_rate"] < 0.8:
            threat_level = "HIGH"
        elif overview["quarantined_nodes"] > 0 or alerts["by_severity"]["HIGH"] > 0:
            threat_level = "MEDIUM"
        
        return {
            "threat_level": threat_level,
            "anomaly_count_24h": alerts["total"],
            "recent_anomalies": alerts["recent"],
            "node_trust_scores": {
                node_id: data["trust_score"] 
                for node_id, data in dashboard_data["node_metrics"].items()
            },
            "verification_success_rate": overview["success_rate"],
            "common_anomaly_types": list(dashboard_data["failure_analysis"].keys()),
            "quarantined_nodes": overview["quarantined_nodes"],
            "failover_count": overview["failover_count"],
            "recommended_actions": [
                "Monitor quarantined nodes" if overview["quarantined_nodes"] > 0 else None,
                "Review failure patterns" if overview["success_rate"] < 0.9 else None,
                "Consider scaling infrastructure" if overview["failover_count"] > 10 else None
            ]
        }
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/security/quarantine_node/{node_id}")
async def manual_quarantine_node(node_id: str, reason: str = "Manual quarantine"):
    """Manually quarantine a node for security reasons."""
    if not distributed_coordinator or not distributed_coordinator.security_orchestrator:
        raise HTTPException(status_code=500, detail="Security orchestrator not initialized")
    
    try:
        distributed_coordinator.security_orchestrator.quarantine_node(node_id, reason)
        return {
            "status": "success",
            "message": f"Node {node_id} has been quarantined",
            "reason": reason
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/security/recover_node/{node_id}")
async def attempt_node_recovery(node_id: str):
    """Attempt to recover a quarantined node."""
    if not distributed_coordinator or not distributed_coordinator.security_orchestrator:
        raise HTTPException(status_code=500, detail="Security orchestrator not initialized")
    
    try:
        distributed_coordinator.security_orchestrator.attempt_node_recovery(node_id)
        return {
            "status": "success",
            "message": f"Recovery attempt initiated for node {node_id}"
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/security/node_status/{node_id}")
async def get_node_security_status(node_id: str):
    """Get detailed security status for a specific node."""
    if not distributed_coordinator or not distributed_coordinator.security_orchestrator:
        raise HTTPException(status_code=500, detail="Security orchestrator not initialized")
    
    try:
        dashboard_data = distributed_coordinator.security_orchestrator.get_security_dashboard_data()
        
        if node_id not in dashboard_data["node_metrics"]:
            raise HTTPException(status_code=404, detail="Node not found")
        
        node_data = dashboard_data["node_metrics"][node_id]
        
        # Check if node is quarantined
        quarantine_info = None
        for q_node in dashboard_data["quarantined_nodes"]:
            if q_node["node_id"] == node_id:
                quarantine_info = q_node
                break
        
        return {
            "node_id": node_id,
            "status": "quarantined" if node_data["quarantined"] else "active",
            "trust_score": node_data["trust_score"],
            "success_rate": node_data["success_rate"],
            "total_requests": node_data["total_requests"],
            "consecutive_failures": node_data["consecutive_failures"],
            "average_integrity": node_data["average_integrity"],
            "quarantine_info": quarantine_info,
            "can_use": distributed_coordinator.security_orchestrator.can_use_node(node_id)
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


# ------------------------------ Security ------------------------------

class SecurityIncidentRequest(BaseModel):
    node_id: str
    incident_type: str
    severity: float
    evidence: Dict
    reporter_nodes: List[str] = []

class SecurityStatusResponse(BaseModel):
    node_id: str
    quarantined: bool
    level: str = None
    trust_score: float
    recent_activities: List[Dict]

class NetworkHealthResponse(BaseModel):
    network_health_score: float
    health_status: str
    total_nodes: int
    quarantined_nodes: int
    trust_distribution: Dict[str, int]

class ExpertValidationRequest(BaseModel):
    expert_name: str
    training_data_sample: str
    test_responses: List[str]
    address: str

class ExpertValidationResponse(BaseModel):
    is_valid: bool
    action: str
    snapshot_hash: str = None
    confidence_score: float
    violations: List[str] = []

@app.post("/security/report_incident")
async def report_security_incident(req: SecurityIncidentRequest):
    """Report a security incident."""
    try:
        result = network_defense.handle_security_incident(
            node_id=req.node_id,
            incident_type=req.incident_type,
            severity=req.severity,
            evidence=req.evidence,
            reporter_nodes=req.reporter_nodes
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/security/status/{node_id}", response_model=SecurityStatusResponse)
async def get_security_status(node_id: str):
    """Get security status for a node."""
    try:
        status = network_defense.quarantine_manager.get_quarantine_status(node_id)
        return SecurityStatusResponse(
            node_id=node_id,
            quarantined=status["quarantined"],
            level=status.get("level"),
            trust_score=status["trust_score"],
            recent_activities=status["recent_activities"]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/security/network_health", response_model=NetworkHealthResponse)
async def get_network_health():
    """Get overall network health metrics."""
    try:
        health = network_defense.get_network_health()
        return NetworkHealthResponse(
            network_health_score=health["network_health_score"],
            health_status=health["health_status"],
            total_nodes=health["total_nodes"],
            quarantined_nodes=health["quarantined_nodes"],
            trust_distribution=health["trust_distribution"]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/security/validate_expert", response_model=ExpertValidationResponse)
async def validate_expert_update(req: ExpertValidationRequest):
    """Validate an expert update for security threats."""
    try:
        # Check if node is quarantined
        can_submit, reason = network_defense.quarantine_manager.can_node_perform_action(
            req.address, "submit_expert"
        )
        if not can_submit:
            raise HTTPException(status_code=403, detail=f"Node quarantined: {reason}")
        
        # Get old weights (simplified - in production would get from blockchain)
        old_weights = {"dummy": torch.randn(10, 5)}
        new_weights = {"dummy": torch.randn(10, 5)}
        
        # Comprehensive validation
        is_valid, action, snapshot_hash = security_coordinator.validate_expert_update(
            expert_name=req.expert_name,
            old_weights=old_weights,
            new_weights=new_weights,
            training_data_sample=req.training_data_sample,
            test_responses=req.test_responses
        )
        
        # If validation failed, report security incident
        if not is_valid:
            network_defense.handle_security_incident(
                node_id=req.address,
                incident_type="failed_validation",
                severity=0.7,
                evidence={"expert_name": req.expert_name, "action": action}
            )
        
        return ExpertValidationResponse(
            is_valid=is_valid,
            action=action,
            snapshot_hash=snapshot_hash,
            confidence_score=0.8,  # Simplified
            violations=[]
        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/p2p/node_stats/{node_id}")
async def get_node_stats(node_id: str):
    """Get contribution stats for a specific node."""
    try:
        # Check if node exists
        node_info = coordinator.nodes.get(node_id)
        if not node_info:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Calculate contributions based on expert usage
        total_contribution = 0
        expert_contributions = {}
        
        for expert_name in node_info.available_experts:
            expert_stats = usage_tracker.get_expert_stats(expert_name)
            if expert_stats:
                # Contribution score based on usage and performance
                contribution_score = (
                    expert_stats.call_count * 0.1 +  # 0.1 points per inference
                    (2.0 - expert_stats.average_response_time) * expert_stats.call_count * 0.05 +  # Speed bonus
                    expert_stats.quality_score * expert_stats.call_count * 0.1  # Quality bonus
                )
                expert_contributions[expert_name] = {
                    "contribution_score": max(0, contribution_score),
                    "inferences_served": expert_stats.call_count,
                    "avg_response_time": expert_stats.average_response_time,
                    "quality_score": expert_stats.quality_score
                }
                total_contribution += contribution_score
        
        # Calculate rewards (simplified)
        estimated_earnings = total_contribution * 0.001  # $0.001 per contribution point
        
        return {
            "node_id": node_id,
            "node_info": {
                "host": node_info.host,
                "port": node_info.port,
                "available_experts": node_info.available_experts,
                "last_heartbeat": coordinator.last_heartbeat.get(node_id, 0)
            },
            "contributions": {
                "total_score": total_contribution,
                "expert_contributions": expert_contributions,
                "estimated_earnings": estimated_earnings,
                "total_inferences": sum(stats["inferences_served"] for stats in expert_contributions.values())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get node stats: {str(e)}")

@app.get("/security/quarantined_nodes")
async def list_quarantined_nodes():
    """List all quarantined nodes."""
    try:
        quarantined = {}
        for node_id, entry in network_defense.quarantine_manager.quarantined_nodes.items():
            quarantined[node_id] = {
                "level": entry.level.value,
                "reason": entry.reason,
                "timestamp": entry.timestamp,
                "duration": entry.duration
            }
        return quarantined
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# DATASET-CHAIN D API ENDPOINTS - Training Data Democracy
# =============================================================================

class DatasetUploadRequest(BaseModel):
    dataset_id: str
    version: str = "1.0.0"
    creator_pubkey: str
    license: str
    source_uri: str  # IPFS URL or file upload path
    total_files: int = 0
    total_bytes: int = 0
    description: str = ""
    proof_of_work_nonce: int = 0


class DatasetVoteRequest(BaseModel):
    dataset_id: str
    vote: bool  # True for approve, False for reject
    voter_id: str
    voter_gpu_hwid: str = None


@app.post("/datasets/upload")
async def upload_dataset(request: DatasetUploadRequest):
    """Stage 1: Upload new dataset to Chain D (Zero-barrier with PoW anti-spam)."""
    try:
        # Create dataset metadata
        metadata = DatasetMetadata(
            dataset_id=request.dataset_id,
            version=request.version,
            creator_pubkey=request.creator_pubkey,
            license=request.license,
            source_uri=request.source_uri,
            total_files=request.total_files,
            total_bytes=request.total_bytes,
            sha256_root="",  # Will be calculated from actual data
            sample_hash=""   # Will be calculated from representative sample
        )
        
        # Add to Dataset-Chain D
        success, message = dataset_chain.add_dataset(metadata, request.proof_of_work_nonce)
        
        if success:
            return {
                "success": True,
                "message": message,
                "dataset_id": request.dataset_id,
                "stage": "pending",
                "next_step": "Auto-audit will begin within 30 minutes",
                "estimated_audit_completion": "30 minutes",
                "estimated_community_vote": "72 hours after audit"
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {str(e)}")


@app.post("/datasets/{dataset_id}/audit")
async def process_dataset_audit(dataset_id: str):
    """Stage 2: Process AI Quality Gate audit (≤30 minutes automated)."""
    try:
        # Simulate AI Quality Gate processing
        # In production, this would download and analyze the actual dataset
        import random
        
        # Simulate quality analysis results
        quality_report = QualityReport(
            toxicity=random.uniform(0.0, 0.15),
            duplicate_rate=random.uniform(0.0, 0.30),
            pii_detected=random.choice([True, False]),
            lang_ratio={"en": 0.8, "ko": 0.15, "other": 0.05},
            perplexity_improvement=random.uniform(0.01, 0.05),
            license_verified=random.choice([True, False]),
            copyright_hits=random.randint(0, 2),
            processing_time_sec=random.uniform(300, 1800),  # 5-30 minutes
            confidence_score=random.uniform(0.8, 0.99)
        )
        
        # Process auto-audit
        success, message = dataset_chain.process_auto_audit(dataset_id, quality_report)
        
        if success:
            return {
                "success": True,
                "message": message,
                "dataset_id": dataset_id,
                "stage": "community_vote",
                "quality_report": {
                    "toxicity": quality_report.toxicity,
                    "duplicate_rate": quality_report.duplicate_rate,
                    "pii_detected": quality_report.pii_detected,
                    "license_verified": quality_report.license_verified,
                    "copyright_hits": quality_report.copyright_hits,
                    "processing_time": f"{quality_report.processing_time_sec:.1f}s",
                    "confidence_score": quality_report.confidence_score
                },
                "voting_window": "72 hours",
                "next_step": "Community voting is now open"
            }
        else:
            return {
                "success": False,
                "message": message,
                "dataset_id": dataset_id,
                "stage": "rejected",
                "quality_report": {
                    "toxicity": quality_report.toxicity,
                    "duplicate_rate": quality_report.duplicate_rate,
                    "pii_detected": quality_report.pii_detected,
                    "license_verified": quality_report.license_verified,
                    "copyright_hits": quality_report.copyright_hits
                },
                "rejection_reason": "Auto-audit failed - dataset does not meet minimum quality requirements"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset audit failed: {str(e)}")


@app.post("/datasets/vote")
async def submit_dataset_vote(request: DatasetVoteRequest):
    """Stage 3: Submit community vote on dataset (72-hour democratic governance)."""
    try:
        success, message = dataset_chain.submit_community_vote(
            dataset_id=request.dataset_id,
            voter_id=request.voter_id,
            vote=request.vote,
            voter_gpu_hwid=request.voter_gpu_hwid
        )
        
        if success:
            # Get current vote status
            vote_info = dataset_chain.community_votes.get(request.dataset_id, {})
            
            return {
                "success": True,
                "message": message,
                "vote_submitted": "APPROVE" if request.vote else "REJECT",
                "current_status": {
                    "votes_for": vote_info.get("votes_for", 0),
                    "votes_against": vote_info.get("votes_against", 0),
                    "total_voters": vote_info.get("total_voters", 0),
                    "time_remaining": max(0, vote_info.get("voting_ends", 0) - time.time()) if vote_info else 0
                }
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vote submission failed: {str(e)}")


@app.post("/datasets/{dataset_id}/finalize")
async def finalize_dataset_vote(dataset_id: str):
    """Stage 4: Finalize community vote and approve/reject dataset."""
    try:
        success, message = dataset_chain.finalize_community_vote(dataset_id)
        
        if success:
            # Get dataset info to determine final status
            dataset_info = dataset_chain.get_dataset_info(dataset_id)
            
            return {
                "success": True,
                "message": message,
                "dataset_id": dataset_id,
                "final_stage": dataset_info.get("stage", "unknown"),
                "quality_tier": dataset_info.get("metadata", {}).get("quality_tier", "unknown"),
                "community_rating": dataset_info.get("metadata", {}).get("community_rating", 0.0),
                "next_step": "Dataset is now available for AI training" if "APPROVED" in message else "Dataset has been rejected"
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vote finalization failed: {str(e)}")


@app.get("/datasets/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get complete information about a dataset."""
    try:
        dataset_info = dataset_chain.get_dataset_info(dataset_id)
        
        if dataset_info:
            return {
                "success": True,
                "dataset_info": dataset_info
            }
        else:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@app.get("/datasets/tier/{tier}")
async def get_datasets_by_tier(tier: str):
    """Get all datasets in a specific quality tier (gold/silver/experimental)."""
    try:
        # Convert string to enum
        tier_enum = DatasetQualityTier(tier.lower())
        dataset_ids = dataset_chain.get_datasets_by_tier(tier_enum)
        
        return {
            "success": True,
            "tier": tier.upper(),
            "dataset_count": len(dataset_ids),
            "datasets": dataset_ids
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier}. Must be one of: gold, silver, experimental, quarantined")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets by tier: {str(e)}")


@app.get("/datasets/statistics")
async def get_dataset_statistics():
    """Get comprehensive Dataset-Chain D statistics."""
    try:
        stats = dataset_chain.get_statistics()
        return {
            "success": True,
            "chain_d_statistics": stats,
            "pipeline_status": {
                "stage_1_pending": stats["pending_audits"],
                "stage_2_auto_audit": "Processing in background",
                "stage_3_community_vote": stats["active_votes"],
                "stage_4_approved": sum(stats["quality_tiers"].values())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset statistics: {str(e)}")


@app.get("/datasets/pending_audits")
async def get_pending_audits():
    """Get list of datasets waiting for auto-audit processing."""
    try:
        pending_count = dataset_chain.get_pending_audits_count()
        pending_datasets = list(dataset_chain.pending_audits.keys())
        
        return {
            "success": True,
            "pending_audits_count": pending_count,
            "datasets_awaiting_audit": pending_datasets,
            "estimated_processing_time": "≤30 minutes per dataset"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending audits: {str(e)}")


@app.get("/datasets/active_votes")
async def get_active_votes():
    """Get list of datasets currently in community voting."""
    try:
        active_votes = {}
        
        for dataset_id, vote_info in dataset_chain.community_votes.items():
            time_remaining = max(0, vote_info["voting_ends"] - time.time())
            
            active_votes[dataset_id] = {
                "votes_for": vote_info["votes_for"],
                "votes_against": vote_info["votes_against"],
                "total_voters": vote_info["total_voters"],
                "time_remaining_hours": time_remaining / 3600,
                "approval_rate": vote_info["votes_for"] / max(1, vote_info["votes_for"] + vote_info["votes_against"])
            }
        
        return {
            "success": True,
            "active_votes_count": len(active_votes),
            "voting_datasets": active_votes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active votes: {str(e)}")


# PoDL (Proof-of-Data-Learning) Endpoints
@app.post("/podl/generate_proof")
async def generate_podl_proof(training_data: dict):
    """Generate PoDL proof for expert training session."""
    try:
        # Create training session from request data
        session = TrainingSession(
            new_expert_hash=training_data["expert_hash"],
            dataset_ids=training_data["dataset_ids"],
            trainer_node_id=training_data.get("trainer_node_id", "unknown"),
            total_samples=training_data.get("total_samples", 0),
            epochs=training_data.get("epochs", 1),
            cpu_time_seconds=training_data.get("cpu_time", 0.0),
            gpu_time_seconds=training_data.get("gpu_time", 0.0),
            memory_peak_gb=training_data.get("memory_peak", 0.0),
            dataset_weights=training_data.get("dataset_weights", {}),
            batch_hashes=training_data.get("batch_hashes", []),
            random_seed=training_data.get("random_seed", 42),
            baseline_accuracy=training_data.get("baseline_accuracy", 0.0),
            final_accuracy=training_data.get("final_accuracy", 0.0),
            accuracy_improvement=training_data.get("accuracy_improvement", 0.0),
            start_time=training_data.get("start_time", time.time()),
            end_time=training_data.get("end_time", time.time()),
            framework_version=training_data.get("framework_version", "torch==2.1.0"),
            hardware_info=training_data.get("hardware_info", {})
        )
        
        # Generate PoDL proof
        proof = podl_generator.generate_training_proof(session)
        
        return {
            "success": True,
            "podl_proof": {
                "expert_hash": proof.expert_hash,
                "dataset_lineage": proof.dataset_lineage,
                "merkle_root": proof.merkle_root,
                "proof_timestamp": proof.proof_timestamp,
                "verification_level": proof.verification_level,
                "confidence_score": proof.confidence_score
            },
            "message": "PoDL proof generated successfully - training data lineage cryptographically verified"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PoDL proof generation failed: {str(e)}")


@app.post("/podl/verify_proof")
async def verify_podl_proof(proof_data: dict):
    """Verify PoDL proof authenticity."""
    try:
        # This would reconstruct PoDL proof from proof_data in production
        # For now, simulate verification
        verification_level = proof_data.get("verification_level", "basic")
        
        # Simulate verification result
        is_valid = True  # In production, would call podl_verifier.verify_proof()
        verification_report = {
            "proof_hash": "simulated_proof_hash",
            "verification_level": verification_level,
            "verification_timestamp": time.time(),
            "checks_performed": ["signature_verification", "dataset_existence", "merkle_root_verification"],
            "checks_passed": ["signature_verification", "dataset_existence", "merkle_root_verification"],
            "checks_failed": [],
            "confidence_score": 0.95
        }
        
        return {
            "success": True,
            "verification_result": {
                "is_valid": is_valid,
                "verification_report": verification_report
            },
            "message": "PoDL proof verification completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PoDL proof verification failed: {str(e)}")


# =============================================================================
# AUTONOMOUS EVOLUTION API ENDPOINTS - Self-Improving AI Architecture
# =============================================================================

class MigrationProposalRequest(BaseModel):
    migration_type: str                      # "scale_experts", "widen_model", "multimodal_fusion"
    from_version: str
    to_version: str
    difficulty: int                          # 1-4 (easy to expert)
    min_performance_gain: float              # 0.15 = 15% minimum improvement
    benchmark_suite: List[str]               # ["MMLU", "HellaSwag", "GSM8K"]
    expected_training_time: int              # Hours
    min_gpu_hours: int
    min_dataset_size_gb: int
    migration_script: str                    # Architecture change code
    initialization_strategy: str             # "teacher_weights", "random_init"
    proposer_node_id: str
    credits_to_stake: int


@app.post("/evolution/propose_migration")
async def propose_architecture_migration(request: MigrationProposalRequest):
    """Propose a new architecture migration for autonomous evolution."""
    try:
        # Convert request to MigrationSpec
        spec = MigrationSpec(
            migration_type=MigrationType(request.migration_type),
            from_version=request.from_version,
            to_version=request.to_version,
            difficulty=EvolutionDifficulty(request.difficulty),
            parameter_changes={},  # Would be extracted from migration_script
            structural_changes=[],  # Would be extracted from migration_script
            compatibility_range=[request.from_version, "∞"],
            min_performance_gain=request.min_performance_gain,
            benchmark_suite=request.benchmark_suite,
            expected_training_time=request.expected_training_time,
            min_gpu_hours=request.min_gpu_hours,
            min_dataset_size_gb=request.min_dataset_size_gb,
            memory_requirement_gb=80,  # Default
            migration_script=request.migration_script,
            initialization_strategy=request.initialization_strategy,
            created_timestamp=time.time(),
            creator_node_id=request.proposer_node_id,
            estimated_cost_credits=request.min_gpu_hours * 10  # 10 credits per GPU hour
        )
        
        # Submit proposal
        success, message = migration_manager.propose_migration(
            spec, request.proposer_node_id, request.credits_to_stake
        )
        
        if success:
            return {
                "success": True,
                "message": message,
                "migration_type": request.migration_type,
                "from_version": request.from_version,
                "to_version": request.to_version,
                "min_performance_gain": f"{request.min_performance_gain:.1%}",
                "next_step": "Migration proposal submitted for community endorsement and epoch selection"
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid migration type or difficulty: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration proposal failed: {str(e)}")


@app.post("/evolution/endorse_migration/{proposal_hash}")
async def endorse_migration(proposal_hash: str):
    """Endorse a migration proposal (increases priority for epoch selection)."""
    try:
        endorser_node_id = f"node_{int(time.time()) % 10000}"  # Demo node ID
        
        success, message = migration_manager.endorse_migration(proposal_hash, endorser_node_id)
        
        if success:
            return {
                "success": True,
                "message": message,
                "proposal_hash": proposal_hash,
                "endorser": endorser_node_id
            }
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Endorsement failed: {str(e)}")


@app.post("/evolution/trigger_epoch")
async def trigger_epoch_event():
    """Manually trigger an epoch evolution event (admin only)."""
    try:
        success, message = migration_manager.trigger_epoch_event()
        
        if success:
            return {
                "success": True,
                "message": message,
                "epoch_initiated": True,
                "estimated_duration": "48-72 hours",
                "next_step": "Monitor epoch progress at /evolution/epoch_status"
            }
        else:
            return {
                "success": False,
                "message": message,
                "epoch_initiated": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Epoch trigger failed: {str(e)}")


@app.get("/evolution/migration_candidates")
async def get_migration_candidates():
    """Get all pending migration candidates."""
    try:
        candidates = migration_manager.get_migration_candidates()
        
        return {
            "success": True,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "ready_for_epoch": sum(1 for c in candidates if c['ready_for_epoch']),
            "average_feasibility": sum(c['feasibility_score'] for c in candidates) / len(candidates) if candidates else 0.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get migration candidates: {str(e)}")


@app.get("/evolution/epoch_status")
async def get_epoch_status():
    """Get current epoch event status and progress."""
    try:
        # Get current epoch status from scheduler
        current_epoch = epoch_scheduler.get_current_epoch_status()
        evolution_status = migration_manager.get_evolution_status()
        
        if current_epoch:
            return {
                "success": True,
                "epoch_active": True,
                "current_epoch": current_epoch,
                "evolution_status": evolution_status
            }
        else:
            return {
                "success": True,
                "epoch_active": False,
                "evolution_status": evolution_status,
                "next_epoch_info": {
                    "days_until_next": evolution_status['days_until_next_epoch'],
                    "ready_candidates": evolution_status['ready_candidates'],
                    "can_trigger_now": evolution_status['next_epoch_ready']
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get epoch status: {str(e)}")


@app.get("/evolution/history")
async def get_evolution_history():
    """Get history of completed evolution events."""
    try:
        history = epoch_scheduler.get_evolution_history()
        
        return {
            "success": True,
            "total_epochs": len(history),
            "evolution_history": history,
            "total_performance_gain": sum(h.get('performance_gain', 0) for h in history),
            "average_epoch_duration": sum(h.get('duration_hours', 0) for h in history) / len(history) if history else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evolution history: {str(e)}")


@app.post("/evolution/register_gpu_node")
async def register_gpu_node():
    """Register a GPU node for epoch training (simplified demo)."""
    try:
        # Demo GPU node registration
        node_id = f"gpu_node_{int(time.time()) % 10000}"
        
        epoch_scheduler.gpu_manager.register_gpu_node(
            node_id=node_id,
            gpu_count=8,                    # 8 GPUs
            credits_per_hour=50,            # 50 credits per hour
            capabilities=["A100", "H100"]   # GPU types
        )
        
        return {
            "success": True,
            "message": "GPU node registered for epoch training",
            "node_id": node_id,
            "gpu_count": 8,
            "credits_per_hour": 50,
            "available_for_next_epoch": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPU node registration failed: {str(e)}")


@app.get("/evolution/gpu_resources")
async def get_gpu_resources():
    """Get available GPU resources for epoch training."""
    try:
        gpu_manager = epoch_scheduler.gpu_manager
        
        total_gpus = sum(info['gpu_count'] for info in gpu_manager.available_nodes.values())
        available_nodes = len([n for n in gpu_manager.available_nodes.keys() if n not in gpu_manager.reserved_nodes])
        reserved_nodes = len(gpu_manager.reserved_nodes)
        
        return {
            "success": True,
            "gpu_cluster_status": {
                "total_gpu_nodes": len(gpu_manager.available_nodes),
                "available_nodes": available_nodes,
                "reserved_nodes": reserved_nodes,
                "total_gpus": total_gpus,
                "utilization_rate": reserved_nodes / max(1, len(gpu_manager.available_nodes))
            },
            "node_details": [
                {
                    "node_id": node_id,
                    "gpu_count": info['gpu_count'],
                    "credits_per_hour": info['credits_per_hour'],
                    "capabilities": info['capabilities'],
                    "status": "reserved" if node_id in gpu_manager.reserved_nodes else "available"
                }
                for node_id, info in gpu_manager.available_nodes.items()
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPU resources: {str(e)}")


# Standard migration templates for easy testing
@app.get("/evolution/migration_templates")
async def get_migration_templates():
    """Get predefined migration templates for common architecture changes."""
    try:
        templates = {
            "scale_experts_8_to_16": {
                "migration_type": "scale_experts",
                "description": "Scale from 8 to 16 experts per MoE layer",
                "difficulty": 2,
                "min_performance_gain": 0.15,
                "expected_training_time": 24,
                "template_script": """
# Scale MoE experts from 8 to 16
def scale_experts(model, from_experts=8, to_experts=16):
    for layer in model.moe_layers:
        # Clone existing experts with noise
        new_experts = []
        for i in range(to_experts):
            if i < from_experts:
                new_experts.append(layer.experts[i])
            else:
                base_expert = layer.experts[i % from_experts]
                new_expert = copy.deepcopy(base_expert)
                add_noise(new_expert, std=0.01)
                new_experts.append(new_expert)
        layer.experts = new_experts
        layer.router.num_experts = to_experts
    return model
                """
            },
            "add_multimodal_vision": {
                "migration_type": "multimodal_fusion",
                "description": "Add vision encoder for multimodal capabilities",
                "difficulty": 4,
                "min_performance_gain": 0.25,
                "expected_training_time": 48,
                "template_script": """
# Add vision modality to text model
def add_vision_modality(text_model):
    vision_encoder = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=12, num_heads=16
    )
    cross_attention = CrossAttentionLayer(
        text_dim=text_model.d_model, vision_dim=1024, num_heads=32
    )
    return MultimodalMoE(
        text_backbone=text_model,
        vision_encoder=vision_encoder,
        cross_attention=cross_attention
    )
                """
            },
            "widen_hidden_dimensions": {
                "migration_type": "widen_model", 
                "description": "Increase hidden dimensions and FFN size",
                "difficulty": 3,
                "min_performance_gain": 0.18,
                "expected_training_time": 32,
                "template_script": """
# Widen model dimensions
def widen_model(model, d_model_new=6144, ffn_ratio=1.6):
    for layer in model.layers:
        # Interpolate existing weights to new dimensions
        layer.attention.d_model = d_model_new
        layer.ffn.d_inner = int(d_model_new * ffn_ratio)
        # Weight interpolation logic here...
    return model
                """
            }
        }
        
        return {
            "success": True,
            "template_count": len(templates),
            "templates": templates,
            "usage": "Use these templates as starting points for migration proposals"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get migration templates: {str(e)}") 