"""Integration module for new server infrastructure.

Add this to your api/server.py imports and initialization.
"""

import time
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi import HTTPException

try:
    import yaml
except ImportError:
    yaml = None
    print("⚠️ PyYAML not installed - server config disabled")

# New server component imports
from server.middleware.chain_bridge import ChainBridgeMiddleware, ChainModelResolver
from server.accounting.middleware import AccountingMiddleware
from server.kv.manager import KVCacheManager
from server.moe.expert_cache import ExpertCacheAPI
from server.batching.batcher import Batcher
from server.fallback.oom_policy import OOMFallbackPolicy, FallbackContext
from server.versioning.resolver import VersionResolver
from server.http.routes import setup_streaming_routes


def load_server_config(config_path: Path = Path("config/server.yaml")) -> dict:
    """Load server configuration."""
    if not yaml:
        # Return default config if yaml not available
        return {
            'server': {
                'batching': {'max_batch_tokens': 2048, 'max_batch_size': 32, 'prefill_decode_ratio': 0.3},
                'kv_cache': {'max_memory_gb': 8.0, 'admission_policy': 'always', 'eviction_policy': 'LRU'},
                'accounting': {'pricing': {}, 'bly_usd_rate': 0.1},
                'versioning': {'enforce_hash_validation': False, 'allow_version_fallback': True},
                'chain_bridge': {'model_resolver_cache_ttl': 300, 'receipt_log_path': 'data/receipts/inference_receipts.jsonl'},
                'expert_cache': {'max_memory_gb': 16.0, 'prefetch_queue_size': 10}
            }
        }
    
    if not config_path.exists():
        # Return defaults if config file doesn't exist
        return load_server_config(None)  # Recursive call to get defaults
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_server_components(app, existing_components: dict):
    """Initialize all new server components.
    
    Args:
        app: FastAPI application instance
        existing_components: Dict with existing components like:
            - model_manager
            - distributed_coordinator
            - ledger_client
            - streaming_handler
    """
    # Load configuration
    config = load_server_config()
    server_config = config['server']
    
    # Initialize version resolver
    version_resolver = VersionResolver(
        enforce_hash_validation=server_config['versioning']['enforce_hash_validation'],
        allow_fallback=server_config['versioning']['allow_version_fallback']
    )
    
    # Initialize chain bridge middleware
    model_resolver = ChainModelResolver(cache_ttl=server_config['chain_bridge']['model_resolver_cache_ttl'])
    chain_bridge = ChainBridgeMiddleware(
        model_resolver=model_resolver,
        receipt_log_path=Path(server_config['chain_bridge']['receipt_log_path'])
    )
    
    # Initialize accounting middleware
    accounting = AccountingMiddleware(
        pricing_config=server_config['accounting']['pricing'],
        bly_usd_rate=server_config['accounting']['bly_usd_rate'],
        ledger_client=existing_components.get('ledger_client')
    )
    
    # Initialize KV cache manager
    kv_manager = KVCacheManager(
        max_memory_gb=server_config['kv_cache']['max_memory_gb'],
        admission_policy=server_config['kv_cache']['admission_policy'],
        eviction_policy=server_config['kv_cache']['eviction_policy']
    )
    
    # Initialize expert cache
    expert_cache = ExpertCacheAPI(
        max_memory_gb=server_config['expert_cache']['max_memory_gb'],
        prefetch_queue_size=server_config['expert_cache']['prefetch_queue_size']
    )
    
    # Initialize continuous batcher
    batcher = Batcher(
        max_batch_tokens=server_config['batching']['max_batch_tokens'],
        max_batch_size=server_config['batching']['max_batch_size'],
        prefill_decode_ratio=server_config['batching']['prefill_decode_ratio'],
        eviction_policy=server_config['batching']['eviction_policy']
    )
    
    # Initialize OOM fallback policy
    oom_policy = OOMFallbackPolicy(
        batcher=batcher,
        kv_manager=kv_manager,
        model_manager=existing_components.get('model_manager')
    )
    
    # Setup streaming routes
    setup_streaming_routes(
        app,
        streaming_handler=existing_components.get('streaming_handler')
    )
    
    # Return all components for use in endpoints
    return {
        'version_resolver': version_resolver,
        'chain_bridge': chain_bridge,
        'accounting': accounting,
        'kv_manager': kv_manager,
        'expert_cache': expert_cache,
        'batcher': batcher,
        'oom_policy': oom_policy,
        'config': config
    }


def create_enhanced_chat_endpoint(components: dict):
    """Create enhanced /chat endpoint with all middleware.
    
    This is a template - adapt to your existing endpoint.
    """
    async def enhanced_chat(request: dict):
        # Pre-inference: resolve model version
        request = await components['chain_bridge'].pre_inference(request)
        
        # Check KV cache
        cache_key = components['kv_manager'].make_key(
            model_version=request['_resolved']['model_version'],
            model_hash=request['_resolved']['model_hash'],
            tokenizer_hash=request['_resolved']['tokenizer_hash'],
            tokens=request.get('prompt_tokens', [])
        )
        
        cached_kv = components['kv_manager'].get(cache_key)
        
        try:
            # Run inference (your existing logic)
            # ... inference code ...
            
            response = {}  # Your response
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM with fallback policy
            context = FallbackContext(
                batch_size=components['batcher'].get_metrics()['active_requests'],
                kv_cache_entries=components['kv_manager'].stats()['unique_keys'],
                precision='fp16',
                memory_available=0,
                memory_required=1e9  # Estimate
            )
            
            if await components['oom_policy'].handle_oom(context):
                # Retry inference
                response = {}  # Retry
            else:
                raise HTTPException(status_code=507, detail="Insufficient memory")
        
        # Post-inference: generate receipts
        response = await components['chain_bridge'].post_inference(
            request,
            response,
            latency_ms=time.time() - request.get('start_time', 0)
        )
        
        # Track billing
        await components['accounting'].track_request(
            request_id=request.get('request_id'),
            user_address=request.get('user_address'),
            prompt_tokens=request.get('prompt_tokens', []),
            completion_tokens=response.get('completion_tokens', []),
            inference_receipt=response.get('receipt')
        )
        
        return response
    
    return enhanced_chat