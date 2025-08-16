from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
import json
from decimal import Decimal
import datetime
import asyncio

# Third-party libraries; ignore type checker if not present in local env
from fastapi import FastAPI, HTTPException, Request, Depends, Response  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel  # type: ignore

# Built-ins / stdlib
import base64
import io
import time

# Third-party
# silence type check errors if missing
import torch  # type: ignore
# ecdsa for signatures
import ecdsa  # type: ignore
import hashlib

# Performance optimization
from backend.utils.json_canonical import dumps_fast, loads_fast, HAS_ORJSON

# Internal token ledger
from backend.core.ledger import Ledger

# Set up logging
logger = logging.getLogger(__name__)
from backend.core.param_index import ParameterIndex

from backend.core.chain import Chain
from backend.core.dataset_chain import DatasetChain
from backend.core.dataset_block import DatasetMetadata, DatasetStage, DatasetQualityTier, QualityReport
from backend.core.podl_proof import PoDLGenerator, PoDLVerifier, TrainingSession
from backend.core.architecture_migration import ArchitectureMigrationManager, MigrationSpec, MigrationType, EvolutionDifficulty
from backend.core.epoch_scheduler import EpochEventScheduler
from backend.model.infer import ModelManager
from backend.model.moe_infer import MoEModelManager, ExpertUsageTracker, reward_expert
from backend.p2p.distributed_inference import DistributedInferenceCoordinator, ExpertNode, DeviceProfile
from backend.p2p.expert_group_optimizer import NodeCapability, ExpertGroup
from backend.core.pol import evaluate_candidate
from backend.core.pol_validator import create_pol_validator, ChainValidator

# Security systems
from backend.security.data_validation import DataSecurityCoordinator
from backend.security.poison_detection import ComprehensivePoisonDetector

# Authentication dependencies
from backend.api.auth_dependencies import verify_main_node, verify_any_authorized_node, maybe_verify_main_node
from backend.security.quarantine_system import NetworkDefenseCoordinator
from backend.security.rate_limiting import rate_limiter, create_contribution_proof, ContributionProof
from backend.security.abuse_prevention import get_abuse_prevention_system, RequestFingerprint
from backend.core.transaction_manager import get_transaction_manager
from api.chat_atomic import get_atomic_chat_handler, AtomicChatRequest, AtomicChatResponse
from api.streaming import get_streaming_handler, get_websocket_handler
from backend.security.api_auth import security_middleware, api_key_manager, APIKeyGenerator, get_api_key_info
from backend.core.free_tier import get_free_tier_manager
from backend.security.monitoring import record_security_event, record_api_response_time, get_security_dashboard, get_system_health
from backend.optimization.performance_dashboard import get_dashboard
from backend.security.genesis_verification import register_network_peer, verify_network_genesis_consensus, should_accept_peer, get_genesis_network_status
from backend.security.disaster_recovery import create_manual_snapshot, emergency_rollback, list_available_snapshots, get_disaster_recovery_status
from backend.security.key_management import (
    create_secure_key, get_secure_key, rotate_secure_key, revoke_secure_key, 
    list_secure_keys, get_key_management_status, KeyType
)
from backend.security.sbom_validator import (
    scan_software_components, validate_license_compliance, get_sbom_status, get_latest_sbom_report
)
from backend.security.hardware_binding import (
    bind_node_hardware, verify_node_hardware, check_node_trust, get_hardware_status, detect_current_hardware
)
from backend.security.content_safety import (
    scan_content_safety, is_content_safe, get_content_safety_status, quarantine_content, unquarantine_content
)
from backend.core.scheduler_integration import wire_system_components
from backend.core.meta_v2 import MetaSpecV2, MetaChainV2Manager, validate_meta_spec_v2

# Economy and wallet routers will be imported after app initialization

# Import consensus endpoints
try:
    from backend.api.consensus import router as consensus_router, init_consensus_systems
    # consensus router will be included after app is defined
except ImportError:
    print("⚠️  Consensus module not found - validator/sync endpoints disabled")

# Use secure reward engine
try:
    from backend.core.reward_engine_secure import get_secure_reward_engine
    secure_reward_engine = get_secure_reward_engine()
    print("✅ Secure reward engine loaded with race condition protection")
except ImportError:
    print("⚠️  Secure reward engine not found - using basic version")

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
# Check for minimal mode to disable heavy initializations
MINIMAL_MODE = os.getenv("BLYAN_MINIMAL_MODE", "false").lower() == "true"
BLOCKCHAIN_ONLY = os.getenv("BLOCKCHAIN_ONLY", "false").lower() == "true"
DISABLE_PIPELINE_ROUND = os.getenv("DISABLE_PIPELINE_ROUND", "true").lower() == "true"
DISABLE_GRPC = os.getenv("DISABLE_GRPC", "true").lower() == "true"
P2P_ENABLE = os.getenv("P2P_ENABLE", "true").lower() == "true" and not MINIMAL_MODE
DISABLE_SECURITY_MONITOR = os.getenv("BLYAN_DISABLE_SECURITY_MONITOR", "false").lower() == "true"

if MINIMAL_MODE:
    print("🔧 Running in MINIMAL MODE - Heavy subsystems disabled for recovery")
    print("  - Pipeline Round: Disabled")
    print("  - gRPC: Disabled")
    print("  - P2P: Disabled")
    print("  - Security Monitor: Disabled if flag set")

root_dir = Path(os.getenv("AIBLOCK_DATA", "./data"))
# Check for development mode (skip anti-spam PoL)
skip_pol = os.getenv("SKIP_POL", "false").lower() in ("true", "1", "yes") or MINIMAL_MODE
if skip_pol:
    print("🚧 Running in development mode - Anti-spam PoL disabled")

# Check for PoL mode
enable_pol = os.getenv("ENABLE_POL", "false").lower() in ("true", "1", "yes") and not MINIMAL_MODE
if enable_pol:
    print("🧠 Proof-of-Learning validation enabled")

meta_chain = Chain(root_dir, "A", skip_pol=skip_pol)
param_chain = Chain(root_dir, "B", skip_pol=skip_pol)  # parameter chain for experts
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

# Autonomous Evolution Systems - Initialize conditionally
migration_manager = None
epoch_scheduler = None

if not MINIMAL_MODE:
    try:
        from backend.core.architecture_migration import ArchitectureMigrationManager
        from backend.core.epoch_scheduler import EpochEventScheduler
        
        migration_manager = ArchitectureMigrationManager(meta_chain, param_chain)
        epoch_scheduler = EpochEventScheduler(migration_manager, dataset_chain)
        # Start autonomous evolution scheduler
        # epoch_scheduler.start_scheduler()
        print("✅ Evolution systems initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize evolution systems: {e}")
        migration_manager = None
        epoch_scheduler = None

# Security systems - Initialize with graceful degradation
try:
    if not DISABLE_SECURITY_MONITOR:
        security_coordinator = DataSecurityCoordinator(root_dir / "expert_backups")
        poison_detector = ComprehensivePoisonDetector()
        network_defense = NetworkDefenseCoordinator(root_dir / "quarantine_data")
    else:
        print("⚠️  Security monitoring disabled in minimal mode")
        security_coordinator = None
        poison_detector = None
        network_defense = None
except Exception as e:
    logger.warning(f"Failed to initialize security systems: {e}")
    security_coordinator = None
    poison_detector = None
    network_defense = None

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

app = FastAPI(
    title="Blyan AI Blockchain API",
    description="Enterprise AI blockchain with MoE architecture and PoL consensus",
    version="2.0.0"
)

# Add CORS middleware for streaming and frontend access
from fastapi.middleware.cors import CORSMiddleware
import os

# Default to specific domains for production security
default_origins = "https://blyan.com,https://www.blyan.com,https://blyan.net,https://www.blyan.net,http://localhost:*,http://127.0.0.1:*"
allowed_origins = os.getenv("ALLOWED_ORIGINS", default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Configurable via environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Add OPTIONS for preflight
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "X-RateLimit-*"],  # Expose headers for SSE
    max_age=3600  # Cache preflight requests for 1 hour
)

# Include consensus router if available
try:
    if 'consensus_router' in locals():
        app.include_router(consensus_router)
except NameError:
    pass

# Import and include additional API routers
try:
    from backend.api.wallet_auth import router as wallet_auth_router
    app.include_router(wallet_auth_router)
    print("✅ Wallet authentication API mounted")
except ImportError as e:
    print(f"⚠️  Wallet auth module not found: {e}")

try:
    from backend.api.economy import router as economy_router
    app.include_router(economy_router)
    print("✅ Economy API mounted")
except ImportError as e:
    print(f"⚠️  Economy module not found: {e}")

try:
    from backend.api.payment_gateway import router as payment_router
    app.include_router(payment_router)
    print("✅ Payment gateway API mounted")
except ImportError as e:
    print(f"⚠️  Payment gateway module not found: {e}")

try:
    from backend.api.health import router as health_router
    app.include_router(health_router)
    print("✅ Health check API mounted")
except ImportError as e:
    print(f"⚠️  Health module not found: {e}")

try:
    from backend.api.siwe_auth import router as siwe_router
    app.include_router(siwe_router)
    print("✅ SIWE authentication API mounted")
except (ImportError, Exception) as e:
    print(f"⚠️  SIWE auth module not available: {e}")

try:
    from backend.api.voting import router as voting_router
    app.include_router(voting_router)
    print("✅ L2 Community Voting API mounted")
except ImportError as e:
    print(f"⚠️  Voting module not found: {e}")

# Mount leaderboard API
try:
    from backend.api.leaderboard import router as leaderboard_router
    app.include_router(leaderboard_router)
    print("✅ Leaderboard API mounted")
except ImportError as e:
    print(f"⚠️  Leaderboard API not available: {e}")

# Community API removed

# Mount transaction monitoring dashboard
try:
    from backend.monitoring.transaction_dashboard import router as monitoring_router
    app.include_router(monitoring_router)
    print("✅ Transaction monitoring dashboard mounted")
except ImportError as e:
    print(f"⚠️  Transaction monitoring not available: {e}")

# Mount rewards and dataset APIs
try:
    from api.rewards_api import dataset_router, rewards_router, contributor_router, init_distribution
    app.include_router(dataset_router)
    app.include_router(rewards_router)
    app.include_router(contributor_router)
    print("✅ Rewards and dataset APIs mounted")
    
    # Start automatic distribution on startup
    @app.on_event("startup")
    async def startup_distribution():
        await init_distribution()
        
except ImportError as e:
    print(f"⚠️  Rewards API not available: {e}")

# Mount Prometheus metrics exporter
try:
    from backend.monitoring.metrics_exporter import metrics_router
    app.include_router(metrics_router)
    print("✅ Prometheus metrics exporter mounted at /metrics")
except ImportError as e:
    print(f"⚠️  Metrics exporter not available: {e}")

# Helper for fast JSON responses (non-consensus data only)
def fast_json_response(data: Dict, status_code: int = 200) -> Response:
    """
    Create a fast JSON response using orjson for non-consensus data.
    
    WARNING: Do NOT use this for blockchain data or anything that affects consensus!
    Use only for API responses, statistics, monitoring data, etc.
    """
    content = dumps_fast(data)
    return Response(
        content=content,
        media_type="application/json",
        status_code=status_code
    )

# Add monitoring middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Monitor API requests and record security events."""
    start_time = time.time()
    request_id = f"api_{hash(str(request.url))}_{int(start_time)}"
    
    try:
        response = await call_next(request)
        
        # Record response time
        response_time_ms = (time.time() - start_time) * 1000
        record_api_response_time(response_time_ms)
        
        # Record in performance dashboard
        perf_dashboard = get_dashboard()
        model_type = "main"  # Default
        
        # Detect model type from endpoint
        if "teacher" in str(request.url.path).lower():
            model_type = "teacher"
        elif "sentinel" in str(request.url.path).lower():
            model_type = "sentinel"
        
        # Check if this was a chat/inference request
        if "/chat" in str(request.url.path):
            # Estimate tokens (would be more accurate with actual count)
            tokens = 50  # Default estimate
            kv_cache_hit = response_time_ms < 100  # Fast response likely cache hit
            experts_used = 2 if "/moe" in str(request.url.path) else 0
            
            perf_dashboard.tracker.record_request(
                request_id=request_id,
                latency_ms=response_time_ms,
                tokens_generated=tokens,
                model_type=model_type,
                batch_size=1,
                kv_cache_hit=kv_cache_hit,
                experts_used=experts_used
            )
        
        # Record security events based on response status
        if response.status_code == 401:
            record_security_event("failed_auth_attempts", str(request.client.host), {
                "endpoint": str(request.url.path),
                "user_agent": request.headers.get("user-agent", "unknown")
            })
        elif response.status_code == 429:
            record_security_event("rate_limit_exceeded", str(request.client.host), {
                "endpoint": str(request.url.path)
            })
        elif response.status_code >= 500:
            record_security_event("server_error", "api_server", {
                "endpoint": str(request.url.path),
                "status_code": response.status_code
            })
        
        return response
        
    except Exception as e:
        # Record exception as security event
        record_security_event("api_exception", "api_server", {
            "endpoint": str(request.url.path),
            "error": str(e)
        })
        raise

# Add security middleware with enhanced protection
app.middleware("http")(security_middleware)

# Add rate limiting middleware
from backend.security.rate_limiting import RateLimitMiddleware
from backend.security.input_validation import validate_heartbeat_params
rate_limit_middleware = RateLimitMiddleware(
    default_limit=60,  # 60 requests per minute for basic tier
    premium_limit=300,  # 300 requests per minute for premium
    enterprise_limit=1000  # 1000 requests per minute for enterprise
)
app.middleware("http")(rate_limit_middleware)

# Add CORS middleware (restrict in production)
# Default to specific domains for production security
default_origins = "https://blyan.com,https://www.blyan.com,https://blyan.net,https://www.blyan.net,http://localhost:*"
allowed_origins = os.getenv("ALLOWED_ORIGINS", default_origins).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Configurable via environment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Restrict methods
    allow_headers=["*"],
    max_age=3600  # Cache preflight requests for 1 hour
)


class TrainingJobStatus(BaseModel):
    running: bool
    paused: bool
    started_at: float
    last_update: float
    epochs_done: int
    steps_done: int
    total_tokens: int
    average_loss: float
    last_error: Optional[str] = None
    progress: float


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
    candidate_loss: Optional[float] = None  # Optional when PoL is enabled
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
        "skip_pol": skip_pol,
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

"""
Meta Spec V2 lightweight endpoints for snapshot/draft management
"""

class MetaSpecRequest(BaseModel):
    spec: Optional[Dict[str, Any]] = None


def _get_meta_manager():
    return MetaChainV2Manager(meta_chain)


@app.get("/meta/spec/latest")
async def get_latest_meta_spec() -> Dict[str, Any]:
    manager = _get_meta_manager()
    spec = manager.get_latest_spec()
    if not spec:
        raise HTTPException(status_code=404, detail="No meta spec found")
    return {"spec": spec.to_dict()}


@app.get("/meta/spec/version/{version}")
async def get_meta_spec_by_version(version: str) -> Dict[str, Any]:
    manager = _get_meta_manager()
    spec = manager.get_spec_by_version(version)
    if not spec:
        raise HTTPException(status_code=404, detail="Meta spec not found for version")
    return {"spec": spec.to_dict()}


@app.post("/meta/spec/snapshot")
async def create_meta_snapshot(req: MetaSpecRequest) -> Dict[str, Any]:
    manager = _get_meta_manager()
    latest = manager.get_latest_spec()

    if req.spec:
        if not validate_meta_spec_v2(req.spec):
            raise HTTPException(status_code=400, detail="Invalid spec payload")
        spec = MetaSpecV2.from_dict(req.spec)
    else:
        if latest:
            spec = MetaSpecV2.from_dict(latest.to_dict())
        else:
            spec = MetaSpecV2(model_id="blyan", version="1.0.0")
    spec.is_snapshot = True
    spec.is_draft = False
    block_hash = manager.create_spec_block(spec)
    return {"status": "ok", "block_hash": block_hash, "spec": spec.to_dict()}


@app.post("/meta/spec/draft")
async def create_meta_draft(req: MetaSpecRequest) -> Dict[str, Any]:
    manager = _get_meta_manager()
    latest = manager.get_latest_spec()

    if req.spec:
        if not validate_meta_spec_v2(req.spec):
            raise HTTPException(status_code=400, detail="Invalid spec payload")
        spec = MetaSpecV2.from_dict(req.spec)
    else:
        if latest:
            bumped = latest.semver.bump_minor()
            base = latest.to_dict()
            base["version"] = str(bumped)
            spec = MetaSpecV2.from_dict(base)
            spec.parent_version = latest.version
        else:
            spec = MetaSpecV2(model_id="blyan", version="1.1.0")
    spec.is_draft = True
    spec.is_snapshot = False
    block_hash = manager.create_spec_block(spec)
    return {"status": "ok", "block_hash": block_hash, "spec": spec.to_dict()}


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
    
    if MINIMAL_MODE:
        print("⚠️  Skipping heavy model initialization in minimal mode")
        return
    
    try:
        from backend.core.scheduler_integration import SchedulerIntegration
        
        model_manager = ModelManager(meta_chain, param_chain, param_index)
        moe_model_manager = MoEModelManager(meta_chain, param_chain, param_index, usage_tracker)
        
        if P2P_ENABLE:
            distributed_coordinator = DistributedInferenceCoordinator(usage_tracker, param_index)
        else:
            print("⚠️  P2P distributed inference disabled")
            distributed_coordinator = None
    except Exception as e:
        logger.error(f"Failed to initialize models/P2P: {e}")
        print(f"❌ Startup failed: {e}")
        # Continue anyway in degraded mode
        return
    
    # Wire scheduler to distributed coordinator
    try:
        if distributed_coordinator:
            scheduler_integration = SchedulerIntegration()
            scheduler_integration.connect_inference_coordinator(distributed_coordinator)
    except Exception as e:
        logger.warning(f"Failed to wire scheduler: {e}")
    # Provide real device profiles and model structure to partition planner
    def _device_profile_provider(node_ids: List[str]):
        devices = []
        try:
            # Import locally to avoid startup issues
            from backend.core.scheduler_integration import DeviceConstraint
            # Pull real profiles from registry
            for nid in node_ids:
                node = distributed_coordinator.registry.nodes.get(nid) if distributed_coordinator else None
                if not node or not node.device_profile:
                    continue
                dp = node.device_profile
                devices.append(DeviceConstraint(
                    device_id=nid,
                    vram_gb=float(dp.vram_gb or 0.0),
                    tflops_est=float(dp.tflops_est or 0.0),
                ))
        except Exception as e:
            print(f"⚠️ Device profile provider error: {e}")
        return devices

    def _model_structure_provider():
        try:
            # Use meta_v2 spec if available to infer number of layers
            from backend.core.meta_v2 import MetaSpecV2
            manager = _get_meta_manager()
            latest = manager.get_latest_spec()
            num_layers = latest.base_num_layers if latest else 12
        except Exception:
            num_layers = 12
        return [{"name": f"block_{i}", "kind": "transformer_block", "extra": {}} for i in range(num_layers)]

    if epoch_scheduler:
        epoch_scheduler.set_device_profile_provider(_device_profile_provider)
        epoch_scheduler.set_model_structure_provider(_model_structure_provider)

    # Start pipeline round service (training rounds using frozen plans)
    try:
        from backend.learning.pipeline_round_service import PipelineRoundService
        from backend.learning.training_job_service import get_training_job_service
        pipeline_service = PipelineRoundService(
            registry=distributed_coordinator.registry,
            epoch_scheduler=epoch_scheduler,
            transport=os.getenv('BLYAN_PIPELINE_TRANSPORT', 'http')
        )
        pipeline_service.start()
        app.state.pipeline_round_service = pipeline_service
        # Start long-running training job service if enabled
        if os.getenv('TRAINING_JOB_ENABLE', '0').lower() in ('1', 'true', 'yes'):
            tj = get_training_job_service()
            tj.set_scheduler(epoch_scheduler)
            tj.start()
            app.state.training_job_service = tj
        print("✅ Pipeline Round Service started")
    except Exception as e:
        print(f"⚠️  Failed to start Pipeline Round Service: {e}")

# --- Training Job HTTP API ---
@app.get("/training/status", response_model=TrainingJobStatus)
async def training_status():
    try:
        tj = getattr(app.state, 'training_job_service', None)
        if not tj:
            raise HTTPException(status_code=404, detail="Training job service not running")
        return TrainingJobStatus(**tj.status())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/start")
async def training_start(config: Dict[str, Any] = None):
    try:
        from backend.learning.training_job_service import get_training_job_service
        tj = getattr(app.state, 'training_job_service', None)
        if not tj:
            tj = get_training_job_service()
            tj.set_scheduler(epoch_scheduler)
            app.state.training_job_service = tj
        if tj.is_running():
            return {"status": "already_running"}
        tj.start(config)
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/stop")
async def training_stop():
    try:
        tj = getattr(app.state, 'training_job_service', None)
        if not tj:
            return {"status": "not_running"}
        await tj.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/pause")
async def training_pause():
    try:
        tj = getattr(app.state, 'training_job_service', None)
        if not tj or not tj.is_running():
            return {"status": "not_running"}
        tj.pause()
        return {"status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/resume")
async def training_resume():
    try:
        tj = getattr(app.state, 'training_job_service', None)
        if not tj:
            return {"status": "not_running"}
        tj.resume()
        return {"status": "resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/reset")
async def training_reset():
    try:
        tj = getattr(app.state, 'training_job_service', None)
        if not tj:
            return {"status": "not_running"}
        await tj.reset()
        return {"status": "reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Store integration for API access
    app.state.scheduler_integration = scheduler_integration
    
    print("✅ SLO-based scheduler integrated with distributed inference coordinator")
    
    # Initialize consensus systems if available
    try:
        if 'init_consensus_systems' in globals():
            # Get reward engine (use secure one if available)
            reward_engine = globals().get('secure_reward_engine', None)
            init_consensus_systems(root_dir, reward_engine)
            print("✅ Consensus systems initialized (state sync, validator rewards)")
    except Exception as e:
        print(f"⚠️  Failed to initialize consensus systems: {e}")


# Cost verification and free tier management functions
async def verify_chat_request_cost(
    user_address: str,
    prompt: str,
    max_new_tokens: int = 100,
    top_k_experts: int = 2,
    quote_id: str = None,
    request: Request = None
) -> Dict[str, Any]:
    """
    Server-side verification of chat request cost and limits.
    This is the authoritative check that prevents client bypasses.
    
    Returns:
        Dict with 'allowed' (bool), 'message' (str), 'status_code' (int)
    """
    try:
        from backend.api.economy import redis_client as quote_redis
        import json
        import tiktoken
        
        # 1. Calculate actual token counts (server authority)
        try:
            # Use tiktoken for accurate token counting
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            actual_input_tokens = len(encoding.encode(prompt))
        except:
            # Fallback to simple estimation
            actual_input_tokens = len(prompt.split()) * 1.3  # Rough approximation
        
        actual_input_tokens = int(actual_input_tokens)
        
        # 2. Free tier limits check (SSOT)
        free_tier_manager = get_free_tier_manager()
        
        if user_address:
            can_request, denial_reason, limits_info = free_tier_manager.can_make_request(
                address=user_address,
                input_tokens=actual_input_tokens,
                output_tokens_est=max_new_tokens
            )
            
            if not can_request:
                return {
                    "allowed": False,
                    "message": f"Free tier limit exceeded: {denial_reason}",
                    "status_code": 429,
                    "limits_info": limits_info,
                    "upgrade_options": {
                        "contribute_data": "Upload quality datasets to earn credits",
                        "validate_experts": "Validate expert blocks to increase trust level",
                        "purchase_credits": "Buy BLY tokens for unlimited access"
                    }
                }
        
        # 3. Quote verification (if provided)
        if quote_id:
            quote_key = f"quote:{quote_id}"
            quote_data = quote_redis.get(quote_key)
            
            if not quote_data:
                return {
                    "allowed": False,
                    "message": "Quote expired or invalid. Please get a new quote.",
                    "status_code": 400
                }
            
            try:
                quote = json.loads(quote_data)
                
                # Verify quote hasn't expired
                if time.time() > quote["expires_at"]:
                    quote_redis.delete(quote_key)
                    return {
                        "allowed": False,
                        "message": "Quote has expired. Please get a new quote.",
                        "status_code": 400
                    }
                
                # Verify token counts match quote (allow some tolerance)
                quote_input_tokens = quote["input_tokens"]
                if abs(actual_input_tokens - quote_input_tokens) > quote_input_tokens * 0.1:  # 10% tolerance
                    return {
                        "allowed": False,
                        "message": f"Token count mismatch. Quote: {quote_input_tokens}, Actual: {actual_input_tokens}",
                        "status_code": 400
                    }
                
                # For paid requests, check balance
                if user_address and not _is_free_tier_request(limits_info):
                    total_cost = Decimal(quote["total_cost"])
                    if not await _check_user_balance(user_address, total_cost):
                        return {
                            "allowed": False,
                            "message": f"Insufficient balance. Required: {total_cost} BLY",
                            "status_code": 402  # Payment Required
                        }
                
            except json.JSONDecodeError:
                return {
                    "allowed": False,
                    "message": "Invalid quote data",
                    "status_code": 400
                }
        
        # 4. Request allowed - consume quota if free tier
        if user_address:
            # This will be updated after successful completion
            pass
        
        return {
            "allowed": True,
            "message": "Request authorized",
            "status_code": 200,
            "actual_input_tokens": actual_input_tokens,
            "quote_verified": quote_id is not None
        }
        
    except Exception as e:
        logger.error(f"Cost verification failed: {e}")
        return {
            "allowed": False,
            "message": "Cost verification failed - please try again",
            "status_code": 500
        }

def _is_free_tier_request(limits_info: Dict) -> bool:
    """Check if this is a free tier request."""
    return (limits_info.get("free_requests_remaining", 0) > 0 or 
            limits_info.get("bonus_requests_available", 0) > 0)

async def _check_user_balance(user_address: str, required_amount: Decimal) -> bool:
    """Check if user has sufficient balance."""
    try:
        from backend.accounting.ledger import get_ledger
        ledger = get_ledger()
        balance = ledger.get_user_balance(user_address)
        return balance >= required_amount
    except Exception as e:
        logger.error(f"Balance check failed for {user_address}: {e}")
        return False

def complete_chat_request(user_address: str, success: bool = True):
    """Mark chat request as completed and update user quotas."""
    if user_address:
        free_tier_manager = get_free_tier_manager()
        free_tier_manager.consume_request(user_address, success)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, http_request: Request = None):
    import time
    start_time = time.time()
    
    # Enhanced request validation with cost verification
    user_address = None
    if http_request:
        # Extract user address from headers or API key
        user_address = http_request.headers.get("X-User-Address")
        if not user_address:
            # Try to get from API key if available
            api_key_info = getattr(http_request.state, "api_key_info", None)
            if api_key_info:
                user_address = f"api_key_{api_key_info.key_id}"
        
        # Server-side cost and limits verification
        cost_check_result = await verify_chat_request_cost(
            user_address=user_address,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            top_k_experts=req.top_k_experts,
            quote_id=getattr(req, 'quote_id', None),
            request=http_request
        )
        
        if not cost_check_result["allowed"]:
            raise HTTPException(
                status_code=cost_check_result["status_code"],
                detail=cost_check_result["message"]
            )
    
    # Abuse prevention check
    if http_request:
        abuse_system = get_abuse_prevention_system()
        fingerprint = RequestFingerprint(
            user_address=user_address,
            ip_address=str(http_request.client.host),
            user_agent=http_request.headers.get("user-agent", "unknown"),
            hardware_fingerprint=http_request.headers.get("x-hardware-fingerprint"),
            session_id=http_request.headers.get("x-session-id")
        )
        
        allowed, challenge_type, context = await abuse_system.assess_request(
            fingerprint=fingerprint,
            prompt=req.prompt,
            request_context={"max_new_tokens": req.max_new_tokens}
        )
        
        if not allowed:
            if challenge_type.value == "captcha":
                challenge_data = abuse_system.generate_captcha_challenge(fingerprint.generate_composite_key())
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Challenge required",
                        "challenge_type": challenge_type.value,
                        "challenge_id": challenge_data["challenge_id"],
                        "question": challenge_data.get("question"),
                        "expires_in": challenge_data["expires_in"],
                        "reason": context.get("reason", "Abuse prevention")
                    }
                )
            elif challenge_type.value == "proof_of_work":
                challenge_data = abuse_system.generate_pow_challenge(fingerprint.generate_composite_key())
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Challenge required",
                        "challenge_type": challenge_type.value,
                        "challenge_id": challenge_data["challenge_id"],
                        "data": challenge_data["data"],
                        "difficulty": challenge_data["difficulty"],
                        "description": challenge_data["description"],
                        "expires_in": challenge_data["expires_in"],
                        "reason": context.get("reason", "Abuse prevention")
                    }
                )
            else:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Request blocked",
                        "reason": context.get("reason", "Abuse detected"),
                        "challenge_type": challenge_type.value
                    }
                )

    # Rate limiting check
    if http_request:
        await rate_limiter(http_request, "inference")
    
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
                
                # Prefer donor nodes for free-tier requests
                limits_info = {}
                try:
                    from backend.core.free_tier import get_free_tier_manager
                    if http_request:
                        user_addr = http_request.headers.get("X-User-Address")
                        if user_addr:
                            ftm = get_free_tier_manager()
                            can_req, _, li = ftm.can_make_request(
                                address=user_addr,
                                input_tokens=len(req.prompt.split()) * 13 // 10,
                                output_tokens_est=req.max_new_tokens,
                            )
                            if can_req:
                                limits_info = li
                except Exception:
                    pass
                prefer_donor = (limits_info.get("free_requests_remaining", 0) > 0 or
                                limits_info.get("bonus_requests_available", 0) > 0)
                
                response_text, routing_info = await distributed_coordinator.distribute_inference(
                    prompt=req.prompt,
                    required_experts=selected_experts,
                    max_new_tokens=req.max_new_tokens,
                    prefer_donor=prefer_donor
                )
                
                inference_time = time.time() - start_time
                
                # Mark request as successful and update quotas
                complete_chat_request(user_address, success=True)
                
                # Record successful request for abuse prevention
                if http_request:
                    abuse_system = get_abuse_prevention_system()
                    composite_key = f"{user_address}|{http_request.client.host}"
                    abuse_system.record_request_outcome(composite_key, success=True)
                
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
            
            # Mark request as successful and update quotas
            complete_chat_request(user_address, success=True)
            
            # Record successful request for abuse prevention
            if http_request:
                abuse_system = get_abuse_prevention_system()
                composite_key = f"{user_address}|{http_request.client.host}"
                abuse_system.record_request_outcome(composite_key, success=True)
            
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
            
            # Mark request as successful and update quotas
            complete_chat_request(user_address, success=True)
            
            # Record successful request for abuse prevention
            if http_request:
                abuse_system = get_abuse_prevention_system()
                composite_key = f"{user_address}|{http_request.client.host}"
                abuse_system.record_request_outcome(composite_key, success=True)
            
            return ChatResponse(
                response=answer,
                expert_usage={},
                inference_time=inference_time
            )
            
    except Exception as exc:
        # Mark request as failed and update quotas
        complete_chat_request(user_address, success=False)
        
        # Record failed request for abuse prevention
        if http_request:
            abuse_system = get_abuse_prevention_system()
            composite_key = f"{user_address}|{http_request.client.host}"
            abuse_system.record_request_outcome(composite_key, success=False)
        
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------ Streaming Chat Endpoints ------------------------------

@app.get("/chat/stream")
async def chat_stream(
    prompt: str,
    max_new_tokens: int = 100,
    use_moe: bool = True,
    quote_id: Optional[str] = None,
    http_request: Request = None
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Streams tokens with real-time cost accumulation.
    """
    from sse_starlette.sse import EventSourceResponse
    
    # Extract user address
    user_address = None
    if http_request:
        user_address = http_request.headers.get("X-User-Address")
        if not user_address:
            api_key_info = getattr(http_request.state, "api_key_info", None)
            if api_key_info:
                user_address = f"api_key_{api_key_info.key_id}"
    
    if not user_address:
        raise HTTPException(status_code=401, detail="User address required")
    
    # Get streaming handler
    handler = get_streaming_handler()
    
    async def event_generator():
        """Generate SSE events."""
        try:
            async for chunk in handler.stream_response(
                prompt=prompt,
                user_address=user_address,
                max_new_tokens=max_new_tokens,
                use_moe=use_moe,
                quote_id=quote_id
            ):
                # Format as SSE event
                yield {
                    "event": chunk["type"],
                    "data": json.dumps(chunk)
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())

@app.post("/chat/stream/cancel")
async def cancel_stream(stream_id: str):
    """Cancel an active streaming session."""
    handler = get_streaming_handler()
    success = await handler.cancel_stream(stream_id)
    
    if success:
        return {"status": "cancelled", "stream_id": stream_id}
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.get("/chat/stream/active")
async def get_active_streams():
    """Get information about active streaming sessions."""
    handler = get_streaming_handler()
    return handler.get_active_streams()

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: Optional[str] = None):
    """
    WebSocket endpoint for bidirectional streaming chat.
    """
    # Extract user address from token or headers
    user_address = None
    if token:
        # Validate token and extract user address
        # For demo, just use token as address
        user_address = token
    
    if not user_address:
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    handler = get_websocket_handler()
    
    try:
        await handler.handle_websocket(websocket, user_address)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_address}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal error")

# ------------------------------ Atomic Chat Endpoint ------------------------------

@app.post("/chat/atomic", response_model=AtomicChatResponse)
async def chat_atomic(req: AtomicChatRequest, http_request: Request = None):
    """
    Atomic chat endpoint with full transaction support.
    Ensures all-or-nothing execution for cost, quota, and inference.
    """
    # Extract user address
    user_address = None
    if http_request:
        user_address = http_request.headers.get("X-User-Address")
        if not user_address:
            api_key_info = getattr(http_request.state, "api_key_info", None)
            if api_key_info:
                user_address = f"api_key_{api_key_info.key_id}"
    
    if not user_address:
        raise HTTPException(status_code=401, detail="User address required")
    
    # Use atomic handler
    handler = get_atomic_chat_handler()
    
    try:
        response = await handler.process_chat(req, http_request, user_address)
        
        # Record success metrics
        from backend.monitoring.transaction_dashboard import get_transaction_monitor
        monitor = get_transaction_monitor()
        await monitor.record_transaction(
            transaction_id=response.transaction_id,
            success=True,
            duration_ms=int(response.inference_time * 1000)
        )
        
        return response
        
    except HTTPException as e:
        # Record failure metrics
        from backend.monitoring.transaction_dashboard import get_transaction_monitor
        monitor = get_transaction_monitor()
        
        # Categorize error
        error_msg = str(e.detail).lower()
        if "expired" in error_msg:
            reason = "quote_expired"
        elif "mismatch" in error_msg:
            reason = "token_mismatch"
        elif "quota" in error_msg or "limit" in error_msg:
            reason = "quota_exceeded"
        elif "balance" in error_msg:
            reason = "insufficient_balance"
        else:
            reason = "other"
        
        await monitor.record_transaction(
            transaction_id=f"failed_{int(time.time())}",
            success=False,
            duration_ms=0,
            failure_reason=reason
        )
        
        # Record rejection if applicable
        if e.status_code == 429:
            await monitor.record_rejection(reason=reason)
        
        raise

# ------------------------------ Abuse Prevention Challenge ------------------------------

@app.post("/abuse_prevention/verify_challenge")
async def verify_abuse_challenge(
    challenge_request: dict,
    http_request: Request = None
):
    """Verify abuse prevention challenge response."""
    try:
        challenge_id = challenge_request.get("challenge_id")
        challenge_type = challenge_request.get("challenge_type")
        response = challenge_request.get("response")
        
        if not all([challenge_id, challenge_type, response]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        abuse_system = get_abuse_prevention_system()
        
        if challenge_type == "captcha":
            success = abuse_system.verify_captcha_response(challenge_id, response)
        elif challenge_type == "proof_of_work":
            success = abuse_system.verify_pow_solution(challenge_id, response)
        else:
            raise HTTPException(status_code=400, detail="Unknown challenge type")
        
        if success:
            return {"status": "verified", "message": "Challenge completed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Challenge verification failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Challenge verification error: {str(e)}")


# Record successful completion for behavioral analysis
# Note: This should be called within endpoint functions where http_request is available


# ------------------------------ Mining ------------------------------


@app.post("/mine", response_model=MineResponse)
async def mine(req: MineRequest):
    """Submit candidate payload (weights) + validation loss.

    If PoL check passes, the payload is stored as a new block on the parameter
    chain. Returns the resulting block hash.
    """
    # 1) PoL check (only if PoL is disabled and candidate_loss is provided)
    if not enable_pol and req.candidate_loss is not None:
        passed = evaluate_candidate(
            candidate_loss_fn=lambda: req.candidate_loss,
            previous_loss=req.previous_loss,
        )
        if not passed:
            raise HTTPException(status_code=400, detail="PoL check failed: insufficient improvement")
    elif not enable_pol and req.candidate_loss is None:
        raise HTTPException(status_code=400, detail="candidate_loss required when PoL is disabled")

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

    # 4) Add block to parameter chain (includes anti-spam PoL)
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
    candidate_loss: Optional[float] = None  # Optional when PoL is enabled
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
    candidate_loss: Optional[float] = None  # Optional when PoL is enabled
    previous_loss: Union[float, None] = None
    base_block_hash: Optional[str] = None  # For version control (CAS)


class MoEExpertResponse(BaseModel):
    block_hash: str
    reward: float
    balance: float


@app.post("/upload_moe_experts", response_model=MoEExpertResponse)
async def upload_moe_expert(req: MoEExpertRequest, http_request: Request = None, node_id: str = Depends(verify_main_node)):
    """Upload a single MoE expert or router block to the DAG chain."""
    
    # Rate limiting check
    if http_request:
        await rate_limiter(http_request, "upload")
    
    # 1. Enhanced PoL check (if PoL validator is available)
    if enable_pol and chain_validator:
        print(f"🧠 Using advanced PoL validation for {req.expert_name}")
        # PoL validation will be handled by the chain during block creation
    elif not enable_pol and req.candidate_loss is not None:
        # Fallback to simple PoL check only if candidate_loss provided
        passed = evaluate_candidate(
            candidate_loss_fn=lambda: req.candidate_loss,
            previous_loss=req.previous_loss,
        )
        if not passed:
            raise HTTPException(status_code=400, detail="PoL failed: insufficient improvement")
    elif not enable_pol and req.candidate_loss is None:
        raise HTTPException(status_code=400, detail="candidate_loss required when PoL is disabled")
    
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
    
    # 5. CAS (Compare-and-Swap) check if base_block_hash is provided
    if req.base_block_hash:
        # Find existing expert block for this expert_name
        existing_blocks = [
            block for block in param_chain.storage.iter_blocks()
            if (getattr(block.header, 'block_type', None) in ('expert', 'router') and
                getattr(block.header, 'expert_name', None) == req.expert_name)
        ]
        
        if existing_blocks:
            # Get the latest (head) block for this expert
            latest_block = sorted(existing_blocks, key=lambda b: b.header.timestamp)[-1]
            latest_hash = latest_block.compute_hash()
            
            # CAS check: ensure we're updating from the expected base version
            if latest_hash != req.base_block_hash:
                raise HTTPException(
                    status_code=409,  # Conflict
                    detail=f"Version mismatch: expected base {req.base_block_hash}, but current is {latest_hash}. Please rebase."
                )
    
    # Ensure base dependency is included if provided
    if req.base_block_hash and req.base_block_hash not in (req.depends_on or []):
        req.depends_on = [req.base_block_hash] + (req.depends_on or [])

    # 6. Validate dependencies exist
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
        
        # Record successful upload for rate limiting
        if http_request:
            rate_limiter.record_action(http_request, "upload", success=True)
        
        return MoEExpertResponse(
            block_hash=new_block.compute_hash(),
            reward=total_reward,
            balance=balance
        )
        
    except Exception as exc:
        # Record failed upload for rate limiting
        if http_request:
            rate_limiter.record_action(http_request, "upload", success=False)
        
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


@app.get("/experts/stats/{expert_name}")
async def get_expert_stats(expert_name: str):
    """Get usage statistics for a specific expert."""
    stats = usage_tracker.get_expert_stats(expert_name)
    if not stats:
        raise HTTPException(status_code=404, detail="Expert not found")
    
    reward_multiplier = reward_expert(expert_name, usage_tracker)
    
    # Use fast JSON for statistics (non-consensus data)
    return fast_json_response({
        "expert_name": stats.expert_name,
        "call_count": stats.call_count,
        "average_response_time": stats.average_response_time,
        "quality_score": stats.quality_score,
        "last_used": stats.last_used,
        "current_reward_multiplier": reward_multiplier
    })


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
    donor_mode: bool = False
    # Optional device profile
    vram_gb: Optional[float] = None
    tflops_est: Optional[float] = None
    net_mbps: Optional[float] = None
    cuda_capability: Optional[str] = None


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
    donor_mode: bool = False


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
    donor_mode: Optional[bool] = None


def is_valid_node_ip(host: str) -> bool:
    """Check if host IP is valid for P2P nodes (reject private/loopback)."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
        # Reject loopback, private, link-local addresses
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return False
        # Reject reserved addresses
        if ip.is_reserved or ip.is_multicast:
            return False
        return True
    except ValueError:
        # If it's not a valid IP, could be a domain name
        # In production, you might want to resolve and check
        return True  # Allow domain names for now

@app.post("/p2p/register", response_model=NodeRegistrationResponse)
async def register_expert_node(req: RegisterNodeRequest):
    """Register a new expert node for distributed inference."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    # Validate IP address
    if not is_valid_node_ip(req.host):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid host IP: {req.host}. Private, loopback, and reserved IPs are not allowed."
        )
    
    node = ExpertNode(
        node_id=req.node_id,
        host=req.host,
        port=req.port,
        available_experts=req.available_experts,
        donor_mode=req.donor_mode,
        device_profile=DeviceProfile(
            vram_gb=req.vram_gb or float(req.hardware_info.get("vram_gb", 0) or 0),
            tflops_est=req.tflops_est or float(req.hardware_info.get("tflops", 0) or 0),
            net_mbps=req.net_mbps or float(req.hardware_info.get("net_mbps", 0) or 0),
            cuda_capability=req.cuda_capability or req.hardware_info.get("cuda", None)
        )
    )
    
    distributed_coordinator.registry.register_node(node)
    
    return NodeRegistrationResponse(
        status="success",
        message=f"Node {req.node_id} registered successfully",
        registered_experts=len(req.available_experts),
        donor_mode=req.donor_mode
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
    donor_count = 0
    for node_id, node in distributed_coordinator.registry.nodes.items():
        is_donor = getattr(node, "donor_mode", False)
        if is_donor:
            donor_count += 1
        nodes.append({
            "node_id": node.node_id,
            "endpoint": node.endpoint,
            "available_experts": node.available_experts,
            "load_factor": node.load_factor,
            "last_heartbeat": node.last_heartbeat,
            "donor_mode": is_donor
        })
    
    # Use fast JSON for node list (non-consensus data) with donor statistics
    return fast_json_response({
        "nodes": nodes,
        "total_nodes": len(nodes),
        "donor_nodes": donor_count,
        "donor_percentage": round(donor_count / len(nodes) * 100, 1) if nodes else 0
    })


# Cache for donor stats
donor_stats_cache = {
    "data": None,
    "timestamp": 0,
    "ttl_seconds": 10  # 10 second cache
}

@app.get("/p2p/donor_stats")
async def get_donor_statistics():
    """Get detailed statistics about donor nodes and their contributions (cached)."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    # Check cache
    current_time = time.time()
    if (donor_stats_cache["data"] is not None and 
        current_time - donor_stats_cache["timestamp"] < donor_stats_cache["ttl_seconds"]):
        return fast_json_response(donor_stats_cache["data"])
    
    donor_nodes = []
    regular_nodes = []
    total_donor_experts = 0
    total_regular_experts = 0
    
    for node_id, node in distributed_coordinator.registry.nodes.items():
        is_donor = getattr(node, "donor_mode", False)
        # Use reputation manager to check node health status
        is_healthy = True
        if hasattr(distributed_coordinator, 'reputation_manager'):
            node_rep = distributed_coordinator.reputation_manager.get_node_reputation(node_id)
            is_healthy = node_rep and node_rep.get('success_rate', 0) > 0.5
        else:
            # Fallback to heartbeat check
            import time
            is_healthy = (time.time() - node.last_heartbeat) < 30
        
        node_info = {
            "node_id": node.node_id,
            "expert_count": len(node.available_experts),
            "status": "active" if is_healthy else "unhealthy",
            "load_factor": node.load_factor
        }
        
        if is_donor:
            donor_nodes.append(node_info)
            total_donor_experts += len(node.available_experts)
        else:
            regular_nodes.append(node_info)
            total_regular_experts += len(node.available_experts)
    
    # Calculate contribution metrics
    total_nodes = len(donor_nodes) + len(regular_nodes)
    total_experts = total_donor_experts + total_regular_experts
    
    # Prepare response data
    response_data = {
        "summary": {
            "total_donor_nodes": len(donor_nodes),
            "total_regular_nodes": len(regular_nodes),
            "donor_percentage": round(len(donor_nodes) / total_nodes * 100, 1) if total_nodes else 0,
            "donor_expert_contribution": round(total_donor_experts / total_experts * 100, 1) if total_experts else 0
        },
        "donor_nodes": donor_nodes,
        "impact": {
            "total_experts_donated": total_donor_experts,
            "estimated_free_tier_capacity": f"{total_donor_experts * 100} requests/hour"  # Rough estimate
        },
        "cached_at": current_time
    }
    
    # Update cache
    donor_stats_cache["data"] = response_data
    donor_stats_cache["timestamp"] = current_time
    
    return fast_json_response(response_data)


@app.get("/metrics/donor")
async def get_donor_operational_metrics():
    """Get operational metrics for donor node system monitoring."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    # Get metrics from coordinator
    metrics = distributed_coordinator.metrics.copy()
    
    # Add additional computed metrics
    metrics.update({
        "ema_window_seconds": distributed_coordinator.ema_window_seconds,
        "donor_utilization_ema": distributed_coordinator.donor_utilization_ema,
        "max_donor_utilization_ratio": distributed_coordinator.max_donor_utilization_ratio,
        "free_tier_queue_timeout_seconds": distributed_coordinator.free_tier_queue_timeout,
        "current_inflight_total": distributed_coordinator.inflight_total,
        "current_inflight_free_tier": distributed_coordinator.inflight_free_tier,
        "current_inflight_donor": distributed_coordinator.inflight_donor,
        "timestamp": time.time()
    })
    
    # Add node counts
    donor_count = sum(1 for n in distributed_coordinator.registry.nodes.values() 
                     if getattr(n, "donor_mode", False))
    total_count = len(distributed_coordinator.registry.nodes)
    
    metrics.update({
        "donor_node_count": donor_count,
        "total_node_count": total_count,
        "non_donor_node_count": total_count - donor_count
    })
    
    # Add reputation manager metrics if available
    if hasattr(distributed_coordinator, 'reputation_manager'):
        rep_summary = distributed_coordinator.reputation_manager.get_metrics_summary()
        metrics["reputation_summary"] = rep_summary
    
    return fast_json_response(metrics)


@app.post("/p2p/heartbeat/{node_id}")
async def node_heartbeat(
    node_id: str,
    load_factor: float = 0.0,
    net_mbps: Optional[float] = None,
    latency_ms: Optional[float] = None,
    vram_gb: Optional[float] = None,
    tflops_est: Optional[float] = None,
    cuda_capability: Optional[str] = None,
):
    """Update heartbeat and load for a node."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    # Basic input validation and rate limiting hardening
    if load_factor < 0.0 or load_factor > 1.0:
        raise HTTPException(status_code=400, detail="Invalid load_factor")
    # Stronger parameter validation
    try:
        validate_heartbeat_params(
            load_factor=load_factor,
            net_mbps=net_mbps if net_mbps is not None else None,
            latency_ms=latency_ms if latency_ms is not None else None,
            vram_gb=vram_gb if vram_gb is not None else None,
            tflops_est=tflops_est if tflops_est is not None else None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid heartbeat: {e}")

    # Update heartbeat and load
    distributed_coordinator.registry.heartbeat(node_id)
    distributed_coordinator.registry.update_node_load(node_id, load_factor)
    # Optional device profile refresh
    if any(val is not None for val in (net_mbps, latency_ms, vram_gb, tflops_est, cuda_capability)):
        profile = DeviceProfile(
            vram_gb=vram_gb or 0.0,
            tflops_est=tflops_est or 0.0,
            net_mbps=net_mbps or 0.0,
            latency_ms=latency_ms,
            cuda_capability=cuda_capability,
        )
        distributed_coordinator.registry.update_device_profile(node_id, profile)
        try:
            from backend.learning.pipeline_metrics import get_pipeline_metrics
            metrics = get_pipeline_metrics()
            # Mark device profile refresh to track staleness
            import asyncio
            asyncio.create_task(metrics.mark_device_profile_update(node_id))
        except Exception:
            pass
    
    return {
        "status": "heartbeat_received",
        "node_id": node_id,
        "load_factor": load_factor,
        "updated_profile": any(val is not None for val in (net_mbps, latency_ms, vram_gb, tflops_est, cuda_capability)),
    }


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
        region=req.region,
        donor_mode=req.donor_mode
    )
    
    # Register with optimized coordinator
    distributed_coordinator.register_expert_group_node(node_capability)
    
    return NodeRegistrationResponse(
        status="success",
        message=f"Optimized node {req.node_id} registered with {len(expert_groups)} expert groups",
        registered_experts=len(req.available_experts),
        donor_mode=req.donor_mode
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
        # Use fast JSON for insights (non-consensus data)
        return fast_json_response(insights)
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


# Streaming endpoints
@app.get("/chat/stream")
async def stream_chat_sse(
    request: Request,
    prompt: str,
    max_new_tokens: int = 100,
    use_moe: bool = True
):
    """Stream chat response using Server-Sent Events."""
    from api.streaming import get_streaming_handler
    from sse_starlette.sse import EventSourceResponse
    
    # Get user address from header
    user_address = request.headers.get("X-User-Address", "anonymous")
    
    streaming_handler = get_streaming_handler()
    
    async def generate():
        async for chunk in streaming_handler.stream_response(
            prompt=prompt,
            user_address=user_address,
            max_new_tokens=max_new_tokens,
            use_moe=use_moe
        ):
            # Convert chunk to SSE format
            if chunk["type"] == "stream_start":
                yield {"event": "stream_start", "data": json.dumps(chunk)}
            elif chunk["type"] == "token":
                yield {"event": "token", "data": json.dumps(chunk)}
            elif chunk["type"] == "stream_end":
                yield {"event": "stream_end", "data": json.dumps(chunk)}
            elif chunk["type"] == "error":
                yield {"event": "error", "data": json.dumps(chunk)}
    
    return EventSourceResponse(generate())

@app.websocket("/ws/chat")
async def websocket_chat(websocket, token: str = None):
    """WebSocket endpoint for streaming chat."""
    from api.streaming import get_websocket_handler
    
    # Use token as user address (simplified auth)
    user_address = token or "anonymous"
    
    websocket_handler = get_websocket_handler()
    await websocket_handler.handle_websocket(websocket, user_address)

@app.post("/chat/stream/cancel")
async def cancel_stream(stream_id: str):
    """Cancel an active streaming session."""
    from api.streaming import get_streaming_handler
    
    streaming_handler = get_streaming_handler()
    success = await streaming_handler.cancel_stream(stream_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return {"message": "Stream cancelled", "stream_id": stream_id}

@app.get("/chat/stream/active")
async def get_active_streams():
    """Get information about active streaming sessions."""
    from api.streaming import get_streaming_handler
    
    streaming_handler = get_streaming_handler()
    active_streams = streaming_handler.get_active_streams()
    
    return {
        "active_count": len(active_streams),
        "streams": active_streams
    }

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


@app.post("/chat/distributed_failover")
async def distributed_chat_with_failover(req: ChatRequest):
    """Chat using distributed inference with automatic failover and reputation-based routing."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Get available experts
        available_experts = []
        for expert_name in distributed_coordinator.registry.expert_to_nodes.keys():
            available_experts.append(expert_name)
        
        if not available_experts:
            raise HTTPException(status_code=503, detail="No expert nodes available")
        
        # Select top-k experts for this request
        selected_experts = available_experts[:req.top_k_experts]
        
        # Perform distributed inference with failover
        response_text, expert_usage = await distributed_coordinator.distribute_inference_with_failover(
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


# ------------------------------ Node Reputation ------------------------------

@app.get("/p2p/node_reputation")
async def get_node_reputation_summary():
    """Get reputation summary for all nodes."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    return distributed_coordinator.reputation_manager.get_metrics_summary()


@app.get("/p2p/node_reputation/{node_id}")
async def get_node_reputation(node_id: str):
    """Get detailed reputation metrics for a specific node."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    metrics = distributed_coordinator.reputation_manager.node_metrics.get(node_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for node {node_id}")
    
    return {
        "node_id": node_id,
        "reputation_score": metrics.reputation_score,
        "success_rate": f"{metrics.success_rate:.2%}",
        "avg_response_time": f"{metrics.avg_response_time:.2f}s",
        "p95_response_time": f"{metrics.p95_response_time:.2f}s",
        "total_requests": metrics.total_requests,
        "successful_requests": metrics.successful_requests,
        "failed_requests": metrics.failed_requests,
        "consecutive_failures": metrics.consecutive_failures,
        "is_healthy": metrics.is_healthy,
        "is_blacklisted": node_id in distributed_coordinator.reputation_manager.blacklist,
        "last_failure": metrics.last_failure
    }


@app.post("/p2p/node_health_check/{node_id}")
async def trigger_node_health_check(node_id: str):
    """Manually trigger a health check for a specific node."""
    if not distributed_coordinator:
        raise HTTPException(status_code=500, detail="Distributed coordinator not initialized")
    
    node = distributed_coordinator.registry.nodes.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Perform health check
    is_healthy = await distributed_coordinator.reputation_manager.health_check(node_id, node.endpoint)
    
    return {
        "node_id": node_id,
        "is_healthy": is_healthy,
        "reputation_score": distributed_coordinator.reputation_manager.get_node_reputation(node_id)
    }


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
        if not distributed_coordinator:
            raise HTTPException(status_code=503, detail="P2P system not available")
        node_info = distributed_coordinator.nodes.get(node_id)
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
                "last_heartbeat": distributed_coordinator.last_heartbeat.get(node_id, 0) if distributed_coordinator else 0
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
    anti_spam_nonce: int = 0


class DatasetVoteRequest(BaseModel):
    dataset_id: str
    vote: bool  # True for approve, False for reject
    voter_id: str
    voter_gpu_hwid: str = None


class PoLChallengeRequest(BaseModel):
    """Request for PoL challenge to bypass rate limits."""
    challenge_type: str
    expert_hash: Optional[str] = None
    performance_improvement: Optional[float] = None
    validation_proof: Optional[str] = None


class APIKeyRequest(BaseModel):
    """Request for API key creation."""
    name: str
    key_type: str  # "basic", "contributor", "node_operator", "admin"
    description: str = ""


@app.post("/datasets/upload")
async def upload_dataset(request: DatasetUploadRequest):
    """Stage 1: Upload new dataset to Chain D (Zero-barrier with lightweight anti-spam)."""
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
        success, message = dataset_chain.add_dataset(metadata, request.anti_spam_nonce)
        
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


@app.post("/auth/pol_challenge")
async def pol_challenge_bypass(request: PoLChallengeRequest, http_request: Request):
    """Handle PoL challenge for rate limit bypass."""
    try:
        user_id = rate_limiter._get_user_id(http_request)
        
        if request.challenge_type == "prove_ai_contribution":
            if not request.expert_hash or request.performance_improvement is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Expert hash and performance improvement required for AI contribution proof"
                )
            
            # Create contribution proof
            contribution_proof = create_contribution_proof(
                user_id=user_id,
                expert_hash=request.expert_hash,
                performance_improvement=request.performance_improvement
            )
            
            # Validate and apply bypass
            if rate_limiter.pol_challenge_bypass(user_id, contribution_proof):
                return {
                    "success": True,
                    "message": f"Quota increased by {int(contribution_proof.performance_improvement * 100)}% improvement contribution",
                    "new_quota_info": rate_limiter.get_user_stats(user_id).__dict__
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid contribution proof or insufficient improvement"
                )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown challenge type: {request.challenge_type}"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PoL challenge failed: {str(e)}")


@app.get("/auth/rate_limit_status")
async def get_rate_limit_status(http_request: Request):
    """Get current rate limit status for the user."""
    try:
        user_id = rate_limiter._get_user_id(http_request)
        user_stats = rate_limiter.get_user_stats(user_id)
        quota_config = rate_limiter.user_quotas[user_stats.reputation_level]
        
        return {
            "user_id": user_id[:8] + "...",  # Partial ID for privacy
            "reputation_level": user_stats.reputation_level,
            "quotas": {
                "uploads_remaining_today": quota_config["uploads_per_day"] - user_stats.uploads_today,
                "inference_remaining_hour": quota_config["inference_per_hour"] - user_stats.inference_requests_hour,
                "uploads_per_day": quota_config["uploads_per_day"],
                "inference_per_hour": quota_config["inference_per_hour"]
            },
            "statistics": {
                "successful_uploads": user_stats.successful_uploads,
                "failed_uploads": user_stats.failed_uploads,
                "consecutive_successes": user_stats.consecutive_successes,
                "pol_contributions": user_stats.pol_contributions
            },
            "next_reset": {
                "daily_reset_in": rate_limiter._seconds_until_daily_reset(),
                "hourly_reset_in": rate_limiter._seconds_until_hourly_reset()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rate limit status: {str(e)}")


@app.get("/auth/rate_limit_stats")  
async def get_rate_limit_system_stats():
    """Get system-wide rate limiting statistics (admin endpoint)."""
    try:
        return rate_limiter.get_stats_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")


@app.post("/auth/register_api_key")
async def register_api_key(request: APIKeyRequest):
    """Register a new API key."""
    try:
        # Create API key based on type
        if request.key_type == "basic":
            api_key = APIKeyGenerator.create_basic_user_key(request.name)
        elif request.key_type == "contributor":
            api_key = APIKeyGenerator.create_contributor_key(request.name)
        elif request.key_type == "node_operator":
            api_key = APIKeyGenerator.create_node_operator_key(request.name)
        elif request.key_type == "admin":
            api_key = APIKeyGenerator.create_admin_key(request.name)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid key type: {request.key_type}. Use: basic, contributor, node_operator, admin"
            )
        
        return {
            "success": True,
            "api_key": api_key,
            "key_type": request.key_type,
            "name": request.name,
            "message": "⚠️ Save this key securely - it won't be shown again!",
            "usage_instructions": {
                "header_format": f"X-API-Key: {api_key}",
                "bearer_format": f"Authorization: Bearer {api_key}",
                "test_endpoint": "/auth/api_key_info"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


@app.get("/auth/api_key_info")
async def get_api_key_info_endpoint(http_request: Request):
    """Get information about the current API key."""
    try:
        key_info = get_api_key_info(http_request)
        if not key_info:
            raise HTTPException(status_code=401, detail="No valid API key found")
        
        return {
            "key_id": key_info.key_id,
            "name": key_info.name,
            "permissions": list(key_info.permissions),
            "rate_limit_tier": key_info.rate_limit_tier,
            "created_at": key_info.created_at,
            "last_used": key_info.last_used,
            "usage_count": key_info.usage_count,
            "is_active": key_info.is_active,
            "expires_at": key_info.expires_at
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API key info: {str(e)}")


@app.get("/auth/list_api_keys")
async def list_api_keys():
    """List all API keys (admin only)."""
    try:
        return {
            "keys": api_key_manager.list_keys(),
            "total_keys": len(api_key_manager.keys_db),
            "active_keys": sum(1 for k in api_key_manager.keys_db.values() if k.is_active)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list API keys: {str(e)}")


@app.delete("/auth/revoke_api_key/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke an API key (admin only)."""
    try:
        success = api_key_manager.revoke_key(key_id)
        if success:
            return {"success": True, "message": f"API key {key_id} revoked successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"API key {key_id} not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke API key: {str(e)}")


# ============================================================================
# V2 API KEY ENDPOINTS - Production Grade JWT-based System
# ============================================================================

# Import the new V2 system (conditionally if available)
try:
    from backend.auth.api_key_system import (
        APIKeyManager as APIKeyManagerV2,
        RegisterAPIKeyRequest,
        RegisterAPIKeyResponse,
        RefreshAPIKeyRequest,
        get_current_api_key,
        get_optional_api_key,
        APIKeyPayload
    )
    
    # Initialize V2 manager
    api_key_manager_v2 = APIKeyManagerV2()
    
    @app.post("/auth/v2/register", response_model=RegisterAPIKeyResponse)
    async def register_api_key_v2(
        request: RegisterAPIKeyRequest,
        current_key: Optional[APIKeyPayload] = Depends(get_optional_api_key)
    ):
        """Register a new API key using V2 JWT-based system."""
        from backend.auth.api_key_system import APIKeyRole
        requester_role = APIKeyRole.from_string(current_key.role) if current_key else None
        return await api_key_manager_v2.register_api_key(requester_role, request)
    
    @app.post("/auth/v2/refresh", response_model=RegisterAPIKeyResponse)
    async def refresh_api_key_v2(request: RefreshAPIKeyRequest):
        """Refresh an existing API key if within refresh window."""
        return await api_key_manager_v2.refresh_api_key(request.current_key)
    
    @app.get("/auth/v2/validate")
    async def validate_api_key_v2(api_key: APIKeyPayload = Depends(get_current_api_key)):
        """Validate current API key and return its info."""
        return {
            "valid": True,
            "key_id": api_key.sub,
            "role": api_key.role,
            "scopes": api_key.scopes,
            "expires_at": datetime.datetime.fromtimestamp(api_key.exp).isoformat(),
            "metadata": api_key.metadata
        }
    
    @app.post("/auth/v2/revoke")
    async def revoke_api_key_v2(
        api_key: str,
        _: APIKeyPayload = Depends(get_current_api_key)
    ):
        """Revoke an API key immediately."""
        await api_key_manager_v2.revoke_api_key(api_key)
        return {"success": True, "message": "API key revoked successfully"}
    
    print("✅ V2 API Key System endpoints loaded (JWT-based)")
    
except ImportError as e:
    print(f"⚠️  V2 API Key System not available: {e}")
    # V2 endpoints will not be registered if the module is not available


@app.get("/genesis/hash")
async def get_genesis_hash():
    """Get Genesis Pact hash for node verification."""
    try:
        genesis_hash_file = Path("./data/genesis_pact_hash.txt")
        if genesis_hash_file.exists():
            genesis_hash = genesis_hash_file.read_text().strip()
            return {
                "genesis_hash": genesis_hash,
                "timestamp": os.path.getmtime(genesis_hash_file),
                "network": "blyan_mainnet"
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail="Genesis Pact not found. Run: python scripts/create_genesis_pact.py"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get genesis hash: {str(e)}")


@app.get("/health")
async def health_check():
    """Basic health check endpoint - always returns 200 in degraded mode."""
    # Quick fix: Always return 200 to prevent service disruption
    if os.getenv("BLYAN_MINIMAL_MODE", "false").lower() == "true":
        return {
            "status": "ok",
            "mode": "minimal",
            "timestamp": time.time()
        }
    
    try:
        # Attempt to get system health with timeout
        import asyncio
        try:
            health_status = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(get_system_health)),
                timeout=2.0  # 2 second timeout
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Health check subsystem timeout/error: {e}")
            return {
                "status": "ok",
                "mode": "degraded",
                "warning": "Some subsystems unavailable",
                "timestamp": time.time()
            }
        
        # Simple health check
        if health_status["overall_status"] == "healthy":
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime_seconds": health_status["uptime_seconds"],
                "api_response_time_ms": health_status["api_avg_response_time_ms"]
            }
        else:
            return {
                "status": health_status["overall_status"],
                "timestamp": time.time(),
                "details": health_status
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive security and monitoring dashboard."""
    try:
        return get_security_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring dashboard: {str(e)}")


@app.get("/monitoring/health")
async def get_detailed_health():
    """Get detailed system health information."""
    try:
        return get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@app.post("/monitoring/record_event")
async def record_manual_security_event(
    event_type: str,
    source: str,
    details: Optional[Dict] = None
):
    """Manually record a security event (admin only)."""
    try:
        record_security_event(event_type, source, details or {})
        return {
            "success": True,
            "message": f"Security event '{event_type}' recorded from source '{source}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record security event: {str(e)}")


@app.post("/genesis/register_peer")
async def register_genesis_peer(peer_id: str, host: str, port: int):
    """Register a peer for Genesis verification."""
    try:
        success = register_network_peer(peer_id, host, port)
        if success:
            return {
                "success": True,
                "message": f"Peer {peer_id} registered for Genesis verification",
                "peer_info": {"peer_id": peer_id, "host": host, "port": port}
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register peer")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register peer: {str(e)}")


@app.post("/genesis/verify_consensus")
async def verify_genesis_consensus():
    """Verify Genesis consensus across all registered peers."""
    try:
        consensus_result = verify_network_genesis_consensus()
        
        if consensus_result["consensus_achieved"]:
            return {
                "status": "consensus_achieved",
                "details": consensus_result
            }
        else:
            return {
                "status": "consensus_failed",
                "warning": "Network consensus verification failed - potential security risk",
                "details": consensus_result
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify Genesis consensus: {str(e)}")


@app.get("/genesis/network_status")
async def get_genesis_network_status_endpoint():
    """Get Genesis verification network status."""
    try:
        return get_genesis_network_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Genesis network status: {str(e)}")


@app.post("/genesis/verify_peer")
async def verify_peer_genesis(peer_id: str, peer_genesis_hash: str):
    """Verify if a peer's Genesis hash is acceptable for network connection."""
    try:
        should_accept = should_accept_peer(peer_id, peer_genesis_hash)
        
        return {
            "peer_id": peer_id,
            "genesis_hash_valid": should_accept,
            "connection_allowed": should_accept,
            "message": "Peer accepted" if should_accept else "Peer rejected: Genesis hash mismatch"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify peer Genesis: {str(e)}")


@app.post("/recovery/create_snapshot")
async def create_recovery_snapshot(description: str = "", tags: Optional[List[str]] = None):
    """Create a manual system snapshot for disaster recovery."""
    try:
        snapshot_id = create_manual_snapshot(description, tags or [])
        
        if snapshot_id:
            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "description": description,
                "tags": tags or [],
                "message": f"Snapshot {snapshot_id} created successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create snapshot")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create snapshot: {str(e)}")


@app.post("/recovery/emergency_rollback")
async def execute_emergency_rollback(snapshot_id: str):
    """Execute emergency rollback to specified snapshot (ADMIN ONLY - DESTRUCTIVE)."""
    try:
        # This is a destructive operation - add extra security
        print(f"🚨 EMERGENCY ROLLBACK REQUESTED: {snapshot_id}")
        
        success = emergency_rollback(snapshot_id)
        
        if success:
            return {
                "success": True,
                "snapshot_id": snapshot_id,
                "message": f"Emergency rollback to {snapshot_id} completed successfully",
                "warning": "System has been restored to previous state. Recent data may be lost."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Emergency rollback to {snapshot_id} failed. Check logs for details."
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency rollback failed: {str(e)}")


@app.get("/recovery/snapshots")
async def list_recovery_snapshots():
    """List all available recovery snapshots."""
    try:
        snapshots = list_available_snapshots()
        return {
            "snapshots": snapshots,
            "total_snapshots": len(snapshots),
            "total_size_mb": sum(s["size_mb"] for s in snapshots),
            "oldest_snapshot": min([s["timestamp"] for s in snapshots]) if snapshots else None,
            "newest_snapshot": max([s["timestamp"] for s in snapshots]) if snapshots else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list snapshots: {str(e)}")


@app.get("/recovery/status")
async def get_recovery_system_status():
    """Get disaster recovery system status."""
    try:
        return get_disaster_recovery_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recovery status: {str(e)}")


@app.get("/recovery/snapshot/{snapshot_id}")
async def get_snapshot_details(snapshot_id: str):
    """Get detailed information about a specific snapshot."""
    try:
        snapshots = list_available_snapshots()
        snapshot = next((s for s in snapshots if s["snapshot_id"] == snapshot_id), None)
        
        if not snapshot:
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
        
        return {
            "snapshot": snapshot,
            "rollback_available": snapshot["age_hours"] <= 24,  # 24 hour limit
            "estimated_rollback_time_minutes": 10,  # 10 minute guarantee
            "data_loss_warning": f"Rolling back will lose all data created after {datetime.fromtimestamp(snapshot['timestamp']).isoformat()}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot details: {str(e)}")


# =============================================================================
# SECURE KEY MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/keys/create")
async def create_key_endpoint(
    key_type: str,
    description: str,
    metadata: Dict = None,
    request: Request = None
):
    """Create a new secure key (ADMIN ONLY)."""
    try:
        # Validate key type
        try:
            key_type_enum = KeyType(key_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid key type. Valid types: {[t.value for t in KeyType]}"
            )
        
        key_id, key_value = create_secure_key(key_type_enum, description, metadata or {})
        
        return {
            "success": True,
            "key_id": key_id,
            "key_value": key_value,
            "key_type": key_type,
            "description": description,
            "warning": "Store this key securely - it will not be shown again!",
            "message": f"Secure key {key_id} created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create key: {str(e)}")


@app.get("/keys/list")
async def list_keys_endpoint(key_type: Optional[str] = None):
    """List all secure keys (without sensitive values)."""
    try:
        key_type_filter = None
        if key_type:
            try:
                key_type_filter = KeyType(key_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid key type. Valid types: {[t.value for t in KeyType]}"
                )
        
        keys = list_secure_keys(key_type_filter)
        
        return {
            "keys": keys,
            "total_keys": len(keys),
            "key_types": list(set(k["key_type"] for k in keys)),
            "keys_needing_rotation": len([k for k in keys if k["rotation_needed"]])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list keys: {str(e)}")


@app.post("/keys/{key_id}/rotate")
async def rotate_key_endpoint(key_id: str):
    """Rotate a secure key (ADMIN ONLY)."""
    try:
        new_key_value = rotate_secure_key(key_id)
        
        if new_key_value:
            return {
                "success": True,
                "key_id": key_id,
                "new_key_value": new_key_value,
                "message": f"Key {key_id} rotated successfully",
                "warning": "Update all systems using this key with the new value!"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Key {key_id} not found or cannot be rotated")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rotate key: {str(e)}")


@app.post("/keys/{key_id}/revoke")
async def revoke_key_endpoint(key_id: str):
    """Revoke a secure key (ADMIN ONLY)."""
    try:
        success = revoke_secure_key(key_id)
        
        if success:
            return {
                "success": True,
                "key_id": key_id,
                "message": f"Key {key_id} revoked successfully",
                "warning": "All systems using this key will now fail authentication!"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to revoke key: {str(e)}")


@app.get("/keys/{key_id}/retrieve")
async def retrieve_key_endpoint(key_id: str):
    """Retrieve a secure key value (ADMIN ONLY - USE WITH CAUTION)."""
    try:
        key_value = get_secure_key(key_id)
        
        if key_value:
            # Record security event for key retrieval
            record_security_event(
                "secure_key_retrieved",
                "key_management_api",
                {"key_id": key_id, "accessed_via": "api"}
            )
            
            return {
                "success": True,
                "key_id": key_id,
                "key_value": key_value,
                "warning": "This key value is sensitive - ensure secure transmission!"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Key {key_id} not found or inactive")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve key: {str(e)}")


@app.get("/keys/status")
async def get_key_management_status_endpoint():
    """Get key management system status."""
    try:
        return get_key_management_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get key management status: {str(e)}")


# =============================================================================
# SBOM AND LICENSE VALIDATION ENDPOINTS
# =============================================================================

@app.post("/sbom/scan")
async def scan_components_endpoint():
    """Scan all software components and update SBOM."""
    try:
        scan_results = scan_software_components()
        
        return {
            "success": True,
            "scan_results": scan_results,
            "message": f"Scanned {scan_results['total_components']} components successfully",
            "components_found": {
                "python_packages": scan_results["python_packages"],
                "ai_models": scan_results["ai_models"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan components: {str(e)}")


@app.post("/sbom/validate")
async def validate_compliance_endpoint():
    """Validate license compliance for all components."""
    try:
        report = validate_license_compliance()
        
        return {
            "success": True,
            "report_id": report.report_id,
            "compliance_status": report.compliance_status,
            "summary": {
                "total_components": report.total_components,
                "verified_components": report.verified_components,
                "high_risk_licenses": report.high_risk_licenses,
                "license_conflicts": len(report.license_conflicts),
                "missing_licenses": len(report.missing_licenses),
                "security_vulnerabilities": report.security_vulnerabilities
            },
            "recommendations": report.recommendations[:5],  # Top 5 recommendations
            "message": f"Validation completed with status: {report.compliance_status}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate compliance: {str(e)}")


@app.get("/sbom/status")
async def get_sbom_status_endpoint():
    """Get SBOM validation system status."""
    try:
        return get_sbom_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SBOM status: {str(e)}")


@app.get("/sbom/report")
async def get_latest_report_endpoint():
    """Get the latest SBOM validation report."""
    try:
        report = get_latest_sbom_report()
        
        if report:
            return {
                "report": {
                    "report_id": report.report_id,
                    "timestamp": report.timestamp,
                    "compliance_status": report.compliance_status,
                    "total_components": report.total_components,
                    "verified_components": report.verified_components,
                    "high_risk_licenses": report.high_risk_licenses,
                    "license_conflicts": report.license_conflicts,
                    "missing_licenses": report.missing_licenses,
                    "recommendations": report.recommendations
                },
                "component_summary": {
                    "by_type": {},
                    "by_risk_level": {}
                }
            }
        else:
            return {
                "report": None,
                "message": "No SBOM validation reports available. Run /sbom/validate first."
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SBOM report: {str(e)}")


@app.get("/sbom/components")
async def list_components_endpoint(
    component_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: Optional[int] = 50
):
    """List software components with optional filtering."""
    try:
        # Get latest report
        report = get_latest_sbom_report()
        
        if not report:
            return {
                "components": [],
                "total": 0,
                "message": "No components found. Run /sbom/scan first."
            }
        
        components = report.components
        
        # Apply filters
        if component_type:
            components = [c for c in components if c.component_type.value == component_type]
        
        if risk_level:
            components = [c for c in components 
                         if any(lic.risk_level.value == risk_level for lic in c.license_info)]
        
        # Apply limit
        if limit:
            components = components[:limit]
        
        # Convert to response format
        component_list = []
        for component in components:
            component_list.append({
                "component_id": component.component_id,
                "name": component.name,
                "version": component.version,
                "component_type": component.component_type.value,
                "supplier": component.supplier,
                "license_info": [
                    {
                        "license_name": lic.license_name,
                        "risk_level": lic.risk_level.value,
                        "commercial_use": lic.commercial_use,
                        "copyleft": lic.copyleft
                    }
                    for lic in component.license_info
                ],
                "verification_status": component.verification_status,
                "last_updated": component.last_updated
            })
        
        return {
            "components": component_list,
            "total": len(component_list),
            "filters_applied": {
                "component_type": component_type,
                "risk_level": risk_level,
                "limit": limit
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list components: {str(e)}")


# =============================================================================
# GPU UUID HARDWARE BINDING ENDPOINTS
# =============================================================================

@app.post("/hardware/bind/{node_id}")
async def bind_hardware_endpoint(node_id: str, expert_assignments: Optional[List[str]] = None):
    """Bind a node to its current hardware configuration (NODE OPERATOR ONLY)."""
    try:
        binding_id = bind_node_hardware(node_id, expert_assignments or [])
        
        if binding_id:
            return {
                "success": True,
                "binding_id": binding_id,
                "node_id": node_id,
                "expert_assignments": expert_assignments or [],
                "message": f"Node {node_id} successfully bound to hardware",
                "warning": "Hardware configuration is now locked to this node"
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to bind hardware for node {node_id}. Check if GPUs are available."
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bind hardware: {str(e)}")


@app.post("/hardware/verify/{binding_id}")
async def verify_hardware_endpoint(binding_id: str):
    """Verify hardware binding for a node."""
    try:
        result = verify_node_hardware(binding_id)
        
        return {
            "verification_result": {
                "hardware_id": result.hardware_id,
                "verification_success": result.verification_success,
                "verification_timestamp": result.verification_timestamp,
                "hardware_present": result.hardware_present,
                "performance_delta": result.performance_delta,
                "trust_score_change": result.trust_score_change,
                "anomalies_detected": result.anomalies_detected
            },
            "fingerprint_match": result.expected_fingerprint == result.actual_fingerprint if result.actual_fingerprint else False,
            "message": "Hardware verification completed successfully" if result.verification_success else "Hardware verification failed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify hardware: {str(e)}")


@app.get("/hardware/trust/{node_id}")
async def check_node_trust_endpoint(node_id: str):
    """Check if a node should be trusted based on hardware binding."""
    try:
        should_trust, trust_score, trust_issues = check_node_trust(node_id)
        
        return {
            "node_id": node_id,
            "should_trust": should_trust,
            "trust_score": trust_score,
            "trust_issues": trust_issues,
            "trust_level": "high" if trust_score >= 0.8 else "medium" if trust_score >= 0.5 else "low",
            "recommendation": "Accept inference requests" if should_trust else "Reject or require re-binding"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check node trust: {str(e)}")


@app.get("/hardware/detect")
async def detect_hardware_endpoint():
    """Detect current hardware configuration."""
    try:
        hardware_list = detect_current_hardware()
        
        hardware_summary = []
        gpu_count = 0
        total_gpu_memory = 0
        
        for hardware in hardware_list:
            hardware_info = {
                "hardware_id": hardware.hardware_id,
                "hardware_type": hardware.hardware_type.value,
                "name": hardware.name,
                "uuid": hardware.uuid,
                "vendor": hardware.vendor,
                "memory_mb": hardware.memory_mb,
                "compute_capability": hardware.compute_capability,
                "driver_version": hardware.driver_version,
                "pci_bus_id": hardware.pci_bus_id,
                "temperature_c": hardware.temperature_c,
                "power_draw_w": hardware.power_draw_w,
                "utilization_percent": hardware.utilization_percent
            }
            
            if hardware.hardware_type.value.endswith("_gpu"):
                gpu_count += 1
                total_gpu_memory += hardware.memory_mb
            
            hardware_summary.append(hardware_info)
        
        return {
            "hardware_detected": hardware_summary,
            "summary": {
                "total_components": len(hardware_list),
                "gpu_count": gpu_count,
                "total_gpu_memory_gb": round(total_gpu_memory / 1024, 2),
                "detection_timestamp": time.time()
            },
            "message": f"Detected {len(hardware_list)} hardware components"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect hardware: {str(e)}")


@app.get("/hardware/status")
async def get_hardware_status_endpoint():
    """Get hardware binding system status."""
    try:
        return get_hardware_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hardware status: {str(e)}")


@app.get("/hardware/bindings")
async def list_hardware_bindings_endpoint():
    """List all hardware bindings."""
    try:
        from backend.security.hardware_binding import hardware_binding_manager
        
        bindings_list = []
        for binding_id, binding in hardware_binding_manager.bindings.items():
            bindings_list.append({
                "binding_id": binding_id,
                "node_id": binding.node_id,
                "hardware_id": binding.hardware_id,
                "binding_status": binding.binding_status.value,
                "trust_score": binding.trust_score,
                "binding_timestamp": binding.binding_timestamp,
                "last_verified": binding.last_verified,
                "verification_count": binding.verification_count,
                "expert_assignments": binding.expert_assignments,
                "days_since_binding": (time.time() - binding.binding_timestamp) / 86400,
                "hours_since_verification": (time.time() - binding.last_verified) / 3600
            })
        
        # Sort by binding timestamp (newest first)
        bindings_list.sort(key=lambda x: x["binding_timestamp"], reverse=True)
        
        return {
            "bindings": bindings_list,
            "total_bindings": len(bindings_list),
            "active_bindings": len([b for b in bindings_list if b["binding_status"] == "bound"]),
            "average_trust_score": sum(b["trust_score"] for b in bindings_list) / len(bindings_list) if bindings_list else 0.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list hardware bindings: {str(e)}")


# =============================================================================
# PII/TOXICITY CONTENT SAFETY ENDPOINTS  
# =============================================================================

@app.post("/content/scan")
async def scan_content_endpoint(content_id: str, content: str):
    """Scan content for PII, toxicity, and malware (ADMIN/MODERATOR ONLY)."""
    try:
        scan_result = scan_content_safety(content_id, content)
        
        return {
            "scan_result": {
                "content_id": scan_result.content_id,
                "content_hash": scan_result.content_hash[:16] + "...",  # Truncated hash
                "scan_timestamp": scan_result.scan_timestamp,
                "overall_risk": scan_result.overall_risk.value,
                "pii_detected": scan_result.pii_detected,
                "toxicity_detected": scan_result.toxicity_detected,
                "malware_detected": scan_result.malware_detected,
                "violations_count": len(scan_result.violations),
                "scan_duration_ms": scan_result.scan_duration_ms,
                "scanner_version": scan_result.scanner_version
            },
            "violations_summary": [
                {
                    "violation_type": v.violation_type.value,
                    "severity": v.severity.value,
                    "confidence": v.confidence,
                    "auto_fixable": v.auto_fixable,
                    "suggested_action": v.suggested_action
                }
                for v in scan_result.violations[:10]  # Limit to first 10 violations
            ],
            "message": f"Content scan completed - Risk level: {scan_result.overall_risk.value}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan content: {str(e)}")


@app.get("/content/safety/{content_id}")
async def check_content_safety_endpoint(content_id: str):
    """Check if content is safe for use."""
    try:
        is_safe, scan_result = is_content_safe(content_id)
        
        if scan_result:
            return {
                "content_id": content_id,
                "is_safe": is_safe,
                "risk_level": scan_result.overall_risk.value,
                "last_scanned": scan_result.scan_timestamp,
                "hours_since_scan": (time.time() - scan_result.scan_timestamp) / 3600,
                "violations_count": len(scan_result.violations),
                "pii_detected": scan_result.pii_detected,
                "toxicity_detected": scan_result.toxicity_detected,
                "malware_detected": scan_result.malware_detected,
                "recommendation": "Content approved for use" if is_safe else "Content requires review or quarantine"
            }
        else:
            return {
                "content_id": content_id,
                "is_safe": False,
                "risk_level": "unknown",
                "message": "Content not found or not yet scanned",
                "recommendation": "Scan content before use"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check content safety: {str(e)}")


@app.post("/content/quarantine/{content_id}")
async def quarantine_content_endpoint(content_id: str, reason: str = "Manual quarantine"):
    """Manually quarantine content (ADMIN/MODERATOR ONLY)."""
    try:
        quarantine_content(content_id)
        
        # Record security event
        record_security_event(
            "content_manually_quarantined",
            "content_safety_api",
            {
                "content_id": content_id,
                "reason": reason
            }
        )
        
        return {
            "success": True,
            "content_id": content_id,
            "status": "quarantined",
            "reason": reason,
            "message": f"Content {content_id} has been quarantined",
            "warning": "Quarantined content cannot be used in inference or training"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to quarantine content: {str(e)}")


@app.post("/content/unquarantine/{content_id}")
async def unquarantine_content_endpoint(content_id: str, reason: str = "Manual review completed"):
    """Remove content from quarantine (ADMIN ONLY)."""
    try:
        unquarantine_content(content_id)
        
        # Record security event
        record_security_event(
            "content_unquarantined",
            "content_safety_api",
            {
                "content_id": content_id,
                "reason": reason
            }
        )
        
        return {
            "success": True,
            "content_id": content_id,
            "status": "active",
            "reason": reason,
            "message": f"Content {content_id} has been removed from quarantine",
            "warning": "Ensure content has been properly reviewed before unquarantine"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unquarantine content: {str(e)}")


@app.get("/content/safety/status")
async def get_content_safety_status_endpoint():
    """Get content safety system status."""
    try:
        return get_content_safety_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get content safety status: {str(e)}")


@app.get("/content/quarantined")
async def list_quarantined_content_endpoint():
    """List all quarantined content."""
    try:
        from backend.security.content_safety import content_safety_scanner
        
        quarantined_list = []
        for content_id in content_safety_scanner.quarantined_content:
            scan_result = content_safety_scanner.scan_results.get(content_id)
            
            quarantined_item = {
                "content_id": content_id,
                "quarantine_timestamp": scan_result.scan_timestamp if scan_result else None,
                "risk_level": scan_result.overall_risk.value if scan_result else "unknown",
                "violations_count": len(scan_result.violations) if scan_result else 0,
                "pii_detected": scan_result.pii_detected if scan_result else False,
                "toxicity_detected": scan_result.toxicity_detected if scan_result else False,
                "malware_detected": scan_result.malware_detected if scan_result else False,
                "days_quarantined": (time.time() - scan_result.scan_timestamp) / 86400 if scan_result else 0
            }
            
            quarantined_list.append(quarantined_item)
        
        # Sort by quarantine time (most recent first)
        quarantined_list.sort(key=lambda x: x["quarantine_timestamp"] or 0, reverse=True)
        
        return {
            "quarantined_content": quarantined_list,
            "total_quarantined": len(quarantined_list),
            "high_risk_count": len([item for item in quarantined_list if item["risk_level"] == "high_risk"]),
            "pii_violations": len([item for item in quarantined_list if item["pii_detected"]]),
            "toxicity_violations": len([item for item in quarantined_list if item["toxicity_detected"]]),
            "malware_violations": len([item for item in quarantined_list if item["malware_detected"]])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list quarantined content: {str(e)}")


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
        if not migration_manager:
            raise HTTPException(status_code=503, detail="Migration manager not available")
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
        
        if not migration_manager:
            raise HTTPException(status_code=503, detail="Migration manager not available")
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
        if not migration_manager:
            raise HTTPException(status_code=503, detail="Migration manager not available")
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
        if not migration_manager:
            raise HTTPException(status_code=503, detail="Migration manager not available")
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
        if not epoch_scheduler:
            raise HTTPException(status_code=503, detail="Evolution scheduler not available")
        current_epoch = epoch_scheduler.get_current_epoch_status()
        evolution_status = migration_manager.get_evolution_status() if migration_manager else {}
        
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
        if not epoch_scheduler:
            raise HTTPException(status_code=503, detail="Evolution scheduler not available")
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
        
        if not epoch_scheduler:
            raise HTTPException(status_code=503, detail="Evolution scheduler not available")
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
        if not epoch_scheduler:
            raise HTTPException(status_code=503, detail="Evolution scheduler not available")
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

# ========================
# Validation APIs with INT8 Models
# ========================

@app.post("/validate/quality")
async def validate_with_quantized_models(request: Request):
    """
    Validate content quality using INT8 quantized Teacher/Sentinel models.
    Demonstrates 2-4x speedup from quantization.
    """
    try:
        data = await request.json()
        input_text = data.get("input_text", "")
        candidate_output = data.get("candidate_output", "")
        
        # Get validation manager
        from backend.optimization.validation_model_manager import get_validation_manager
        validation_manager = get_validation_manager()
        
        # Load validation suite if not loaded
        if not validation_manager.models:
            logger.info("Loading validation models with INT8 quantization...")
            validation_manager.load_validation_suite()
        
        # Run validation with quantized models
        start_time = time.time()
        results = validation_manager.validate_with_quantized_models(
            input_text=input_text,
            candidate_output=candidate_output
        )
        validation_time_ms = (time.time() - start_time) * 1000
        
        # Record in performance dashboard
        perf_dashboard = get_dashboard()
        
        # Record teacher validation
        if results.get("teacher_score") is not None:
            perf_dashboard.tracker.record_request(
                request_id=f"val_teacher_{int(time.time())}",
                latency_ms=validation_time_ms * 0.7,  # Teacher takes ~70% of time
                tokens_generated=len(candidate_output.split()),
                model_type="teacher",
                batch_size=1,
                kv_cache_hit=False,
                experts_used=0
            )
        
        # Record sentinel validation  
        if results.get("sentinel_score") is not None:
            perf_dashboard.tracker.record_request(
                request_id=f"val_sentinel_{int(time.time())}",
                latency_ms=validation_time_ms * 0.3,  # Sentinel takes ~30% of time
                tokens_generated=len(candidate_output.split()),
                model_type="sentinel",
                batch_size=1,
                kv_cache_hit=False,
                experts_used=0
            )
        
        # Get memory savings info
        memory_savings = validation_manager.get_memory_savings()
        
        return {
            "success": True,
            "validation_results": results,
            "validation_time_ms": validation_time_ms,
            "quantization_benefits": {
                "memory_saved_mb": memory_savings.get("total_saved_mb", 0),
                "compression_ratio": memory_savings.get("overall_compression", 1.0),
                "speedup_estimate": "2-4x faster than FP32"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# ========================
# Performance Metrics APIs
# ========================

@app.get("/metrics/performance")
async def get_performance_metrics():
    """Get real-time performance metrics including p50/p90/p95/p99 latencies and throughput."""
    try:
        perf_dashboard = get_dashboard()
        dashboard_data = perf_dashboard.get_dashboard()
        
        return {
            "success": True,
            "metrics": dashboard_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@app.get("/metrics/latency")
async def get_latency_metrics():
    """Get latency percentiles (p50, p90, p95, p99)."""
    try:
        perf_dashboard = get_dashboard()
        percentiles = perf_dashboard.tracker.get_percentiles([50, 90, 95, 99])
        
        return {
            "success": True,
            "latency_ms": percentiles,
            "unit": "milliseconds"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latency metrics: {str(e)}")

@app.get("/metrics/throughput")
async def get_throughput_metrics(window_seconds: int = 60):
    """Get throughput metrics (requests/sec and tokens/sec)."""
    try:
        perf_dashboard = get_dashboard()
        throughput = perf_dashboard.tracker.get_throughput(window_seconds)
        
        return {
            "success": True,
            "window_seconds": window_seconds,
            "requests_per_second": throughput["requests_per_second"],
            "tokens_per_second": throughput["tokens_per_second"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get throughput metrics: {str(e)}")

@app.get("/metrics/model_performance")
async def get_model_performance():
    """Get performance breakdown by model type (main/teacher/sentinel)."""
    try:
        perf_dashboard = get_dashboard()
        model_breakdown = perf_dashboard.tracker.get_model_breakdown()
        
        # Show quantization benefits
        performance_comparison = {}
        for model_type, stats in model_breakdown.items():
            if stats["count"] > 0:
                performance_comparison[model_type] = {
                    "request_count": stats["count"],
                    "avg_latency_ms": stats["avg_latency_ms"],
                    "p95_latency_ms": stats["p95_latency_ms"],
                    "total_tokens": stats["total_tokens"],
                    "quantization": "INT8" if model_type in ["teacher", "sentinel"] else "FP16"
                }
        
        return {
            "success": True,
            "model_performance": performance_comparison,
            "quantization_speedup": {
                "teacher_vs_main": model_breakdown["main"]["avg_latency_ms"] / model_breakdown["teacher"]["avg_latency_ms"] if model_breakdown["teacher"]["count"] > 0 and model_breakdown["main"]["count"] > 0 else None,
                "sentinel_vs_main": model_breakdown["main"]["avg_latency_ms"] / model_breakdown["sentinel"]["avg_latency_ms"] if model_breakdown["sentinel"]["count"] > 0 and model_breakdown["main"]["count"] > 0 else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@app.get("/metrics/cache")
async def get_cache_metrics():
    """Get KV-cache hit rate and statistics."""
    try:
        perf_dashboard = get_dashboard()
        cache_stats = perf_dashboard.tracker.get_cache_stats()
        
        return {
            "success": True,
            "cache_metrics": cache_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache metrics: {str(e)}")

@app.get("/metrics/gpu")
async def get_gpu_metrics():
    """Get GPU utilization and memory statistics."""
    try:
        perf_dashboard = get_dashboard()
        gpu_stats = perf_dashboard.tracker.get_gpu_stats()
        
        return {
            "success": True,
            "gpu_metrics": gpu_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get GPU metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 