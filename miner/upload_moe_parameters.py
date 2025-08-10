#!/usr/bin/env python
"""Upload MoE model parameters with DAG structure.

This script extracts expert-specific tensors from a Mixture-of-Experts model
and uploads them as individual blocks to the DAG blockchain.

Example:
    python upload_moe_parameters.py --address alice --model-file model.pt --meta-hash abc123 --candidate-loss 0.9
"""

from __future__ import annotations

import argparse
import base64
import json
import hashlib
import io
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib import request, error
import ecdsa  # type: ignore

try:
    import torch
    from transformers import AutoModelForCausalLM
except ImportError:
    print("torch and transformers required for MoE model processing")
    exit(1)

DEFAULT_API_BASE = os.environ.get("AIBLOCK_API_URL", "http://127.0.0.1:8000")


class MoEExpertExtractor:
    """Extract expert tensors from MoE models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.state_dict = None
        
    def load_model(self) -> Dict[str, torch.Tensor]:
        """Load model state dict from file."""
        try:
            # Try loading as PyTorch state dict
            self.state_dict = torch.load(self.model_path, map_location='cpu')
            if hasattr(self.state_dict, 'state_dict'):
                self.state_dict = self.state_dict.state_dict()
        except Exception:
            try:
                # Try loading as HuggingFace model
                model = AutoModelForCausalLM.from_pretrained(self.model_path)
                self.state_dict = model.state_dict()
            except Exception as e:
                raise ValueError(f"Could not load model from {self.model_path}: {e}")
        
        return self.state_dict
    
    def extract_experts(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract expert tensors grouped by expert identity."""
        if self.state_dict is None:
            self.load_model()
        
        experts = {}
        router_weights = {}
        base_weights = {}
        
        # Pattern matching for different MoE architectures
        expert_patterns = [
            r'layer\.(\d+)\.mlp\.experts\.(\d+)\.(.+)',  # Standard MoE
            r'model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)',  # LLaMA-MoE
            r'transformer\.h\.(\d+)\.mlp\.experts\.(\d+)\.(.+)',  # GPT-style MoE
            r'layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(.+)',  # Switch Transformer
            r'encoder\.layer\.(\d+)\.output\.dense_(\d+)\.(.+)',  # BERT-style MoE
        ]
        
        router_patterns = [
            r'layer\.(\d+)\.mlp\.router\.(.+)',
            r'model\.layers\.(\d+)\.mlp\.router\.(.+)',
            r'transformer\.h\.(\d+)\.mlp\.router\.(.+)',
            r'layers\.(\d+)\.feed_forward\.router\.(.+)',
        ]
        
        for param_name, tensor in self.state_dict.items():
            expert_matched = False
            
            # Check expert patterns
            for pattern in expert_patterns:
                match = re.match(pattern, param_name)
                if match:
                    layer_idx, expert_idx, weight_name = match.groups()
                    expert_name = f"layer{layer_idx}.expert{expert_idx}"
                    layer_id = f"layer{layer_idx}"
                    
                    if expert_name not in experts:
                        experts[expert_name] = {
                            'tensors': {},
                            'layer_id': layer_id,
                            'expert_idx': expert_idx
                        }
                    
                    experts[expert_name]['tensors'][weight_name] = tensor
                    expert_matched = True
                    break
            
            if expert_matched:
                continue
                
            # Check router patterns
            router_matched = False
            for pattern in router_patterns:
                match = re.match(pattern, param_name)
                if match:
                    layer_idx, weight_name = match.groups()
                    router_name = f"layer{layer_idx}.router"
                    layer_id = f"layer{layer_idx}"
                    
                    if router_name not in router_weights:
                        router_weights[router_name] = {
                            'tensors': {},
                            'layer_id': layer_id
                        }
                    
                    router_weights[router_name]['tensors'][weight_name] = tensor
                    router_matched = True
                    break
            
            if not router_matched:
                # Base model weights (non-expert, non-router)
                base_weights[param_name] = tensor
        
        # If no experts found, create virtual experts from base weights
        if not experts and base_weights:
            print("⚠️  No MoE patterns found. Creating virtual experts from base model...")
            experts = self._create_virtual_experts(base_weights)
            base_weights = {}  # Move base weights to virtual experts
        
        return {
            'experts': experts,
            'routers': router_weights,
            'base': base_weights
        }
    
    def _create_virtual_experts(self, base_weights: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Create virtual experts from a standard (non-MoE) model."""
        virtual_experts = {}
        
        # Group parameters by layer
        layer_groups = {}
        for param_name, tensor in base_weights.items():
            # Extract layer number from parameter name
            layer_match = re.search(r'\.(\d+)\.', param_name)
            if layer_match:
                layer_idx = layer_match.group(1)
            else:
                layer_idx = "0"  # Default layer
            
            if layer_idx not in layer_groups:
                layer_groups[layer_idx] = {}
            layer_groups[layer_idx][param_name] = tensor
        
        # Create 2 virtual experts per layer
        for layer_idx, layer_params in layer_groups.items():
            param_items = list(layer_params.items())
            mid_point = len(param_items) // 2
            
            for expert_idx in range(2):
                expert_name = f"layer{layer_idx}.expert{expert_idx}"
                
                # Split parameters between experts
                if expert_idx == 0:
                    expert_params = dict(param_items[:mid_point])
                else:
                    expert_params = dict(param_items[mid_point:])
                
                if expert_params:  # Only create if has parameters
                    virtual_experts[expert_name] = {
                        'tensors': expert_params,
                        'layer_id': f"layer{layer_idx}",
                        'expert_idx': str(expert_idx)
                    }
        
        print(f"✓ Created {len(virtual_experts)} virtual experts from standard model")
        return virtual_experts


class ExpertBlockUploader:
    """Upload expert blocks to the DAG blockchain."""
    
    def __init__(
        self,
        miner_address: str,
        private_key: Optional[str] = None,
        reuse_existing: bool = False,
        api_base: str = DEFAULT_API_BASE,
        node_id: Optional[str] = None,
        node_token: Optional[str] = None,
    ):
        self.miner_address = miner_address
        self.reuse_existing = reuse_existing
        self.api_base = api_base.rstrip("/")
        self.node_id = node_id or os.environ.get("BLYAN_NODE_ID")
        self.node_token = node_token or os.environ.get("BLYAN_MAIN_NODE_TOKEN")
        
        # Setup ECDSA signing
        if private_key:
            self.signing_key = ecdsa.SigningKey.from_string(
                bytes.fromhex(private_key), curve=ecdsa.SECP256k1
            )
        else:
            self.signing_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
            print(f"Generated new private key: {self.signing_key.to_string().hex()}")
        
        self.verifying_key = self.signing_key.verifying_key
        self.miner_pub = self.verifying_key.to_string().hex()
        
        # Cache for existing experts to avoid repeated API calls
        self._existing_experts_cache = None
    
    def serialize_expert_tensors(self, tensors: Dict[str, torch.Tensor]) -> bytes:
        """Serialize expert tensors to bytes."""
        buffer = io.BytesIO()
        torch.save(tensors, buffer)
        return buffer.getvalue()
    
    def create_expert_block_payload(
        self,
        expert_name: str,
        tensors: Dict[str, torch.Tensor],
        layer_id: str,
        meta_hash: str,
        candidate_loss: float,
        previous_loss: Optional[float] = None,
        base_block_hash: Optional[str] = None
    ) -> Dict:
        """Create payload for expert block upload."""
        
        # Serialize tensors
        tensor_bytes = self.serialize_expert_tensors(tensors)
        tensor_b64 = base64.b64encode(tensor_bytes).decode()
        
        # Sign the tensor data
        signature = self.signing_key.sign(tensor_bytes, hashfunc=hashlib.sha256)
        
        payload = {
            "miner_address": self.miner_address,
            "miner_pub": self.miner_pub,
            "payload_sig": signature.hex(),
            "expert_name": expert_name,
            "layer_id": layer_id,
            "block_type": "expert",
            # Always depend on the meta block to anchor experts to the current spec
            "depends_on": ([meta_hash] if meta_hash else []),
            "tensor_data_b64": tensor_b64,
            "candidate_loss": candidate_loss,
            "previous_loss": previous_loss,
        }

        # Include base_block_hash for CAS and add to dependencies
        if base_block_hash:
            payload["base_block_hash"] = base_block_hash
            if payload["depends_on"] is None:
                payload["depends_on"] = []
            if base_block_hash not in payload["depends_on"]:
                payload["depends_on"].append(base_block_hash)
        
        return payload
    
    def create_router_block_payload(
        self,
        router_name: str,
        tensors: Dict[str, torch.Tensor],
        layer_id: str,
        meta_hash: str,
        candidate_loss: float,
        previous_loss: Optional[float] = None,
        base_block_hash: Optional[str] = None
    ) -> Dict:
        """Create payload for router block upload."""
        
        # Serialize tensors
        tensor_bytes = self.serialize_expert_tensors(tensors)
        tensor_b64 = base64.b64encode(tensor_bytes).decode()
        
        # Sign the tensor data
        signature = self.signing_key.sign(tensor_bytes, hashfunc=hashlib.sha256)
        
        payload = {
            "miner_address": self.miner_address,
            "miner_pub": self.miner_pub,
            "payload_sig": signature.hex(),
            "expert_name": router_name,
            "layer_id": layer_id,
            "block_type": "router",
            # Anchor routers to the same meta spec
            "depends_on": ([meta_hash] if meta_hash else []),
            "tensor_data_b64": tensor_b64,
            "candidate_loss": candidate_loss,
            "previous_loss": previous_loss,
        }

        if base_block_hash:
            payload["base_block_hash"] = base_block_hash
            if payload["depends_on"] is None:
                payload["depends_on"] = []
            if base_block_hash not in payload["depends_on"]:
                payload["depends_on"].append(base_block_hash)
        
        return payload
    
    def upload_expert_block(self, payload: Dict) -> Dict:
        """Upload a single expert block to the API."""
        url = f"{self.api_base}/upload_moe_experts"
        headers = {"Content-Type": "application/json"}
        # Attach main-node auth headers if available (required by server)
        if self.node_id and self.node_token:
            headers["X-Node-ID"] = self.node_id
            headers["X-Node-Auth-Token"] = self.node_token
        req = request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        
        try:
            with request.urlopen(req) as resp:
                return json.load(resp)
        except error.HTTPError as e:
            error_detail = e.read().decode()
            raise RuntimeError(f"HTTP {e.code}: {error_detail}")
    
    def check_existing_experts(self) -> Dict[str, str]:
        """Check which experts already exist in the parameter chain."""
        if self._existing_experts_cache is not None:
            return self._existing_experts_cache
        
        existing_experts = {}
        
        try:
            # Fetch all blocks from parameter chain (B)
            req = request.Request(f"{self.api_base}/chain/B/blocks")
            with request.urlopen(req) as resp:
                response_data = json.load(resp)
                blocks = response_data.get('blocks', [])
            
            # Extract expert names and their block hashes
            for block in blocks:
                try:
                    # API returns BlockMeta objects with expert_name field
                    expert_name = block.get('expert_name')
                    block_type = block.get('block_type')
                    
                    # Only consider expert and router blocks
                    if expert_name and block_type in ('expert', 'router'):
                        block_hash = block.get('hash', 'unknown')
                        existing_experts[expert_name] = block_hash
                        
                except Exception as e:
                    print(f"Warning: Could not parse block data: {e}")
                    continue
            
            self._existing_experts_cache = existing_experts
            print(f"Found {len(existing_experts)} existing experts in chain")
            
        except Exception as e:
            print(f"Warning: Could not fetch existing experts: {e}")
            # Return empty dict on error - will proceed with upload
            existing_experts = {}
            
        return existing_experts
    
    def upload_all_experts(
        self,
        extracted_data: Dict,
        meta_hash: str,
        candidate_loss: float,
        previous_loss: Optional[float] = None
    ) -> Dict[str, str]:
        """Upload all experts and routers as separate blocks."""
        
        block_hashes = {}
        
        # Check existing experts if reuse_existing is enabled
        existing_experts = {}
        if self.reuse_existing:
            existing_experts = self.check_existing_experts()
            if existing_experts:
                print(f"🔄 Reuse mode: Found {len(existing_experts)} existing experts")
        
        # Upload expert blocks
        for expert_name, expert_data in extracted_data['experts'].items():
            # Check if expert already exists and reuse_existing is enabled
            if self.reuse_existing and expert_name in existing_experts:
                existing_hash = existing_experts[expert_name]
                block_hashes[expert_name] = existing_hash
                print(f"♾️ Expert {expert_name} already exists: {existing_hash[:16]}... (reused)")
                continue
                
            print(f"Uploading expert: {expert_name}")
            
            payload = self.create_expert_block_payload(
                expert_name=expert_name,
                tensors=expert_data['tensors'],
                layer_id=expert_data['layer_id'],
                meta_hash=meta_hash,
                candidate_loss=candidate_loss,
                previous_loss=previous_loss,
                base_block_hash=existing_experts.get(expert_name)
            )
            
            try:
                response = self.upload_expert_block(payload)
                block_hashes[expert_name] = response.get('block_hash', 'unknown')
                print(f"✓ Expert {expert_name} uploaded: {block_hashes[expert_name][:16]}...")
            except Exception as e:
                print(f"✗ Failed to upload expert {expert_name}: {e}")
                continue
        
        # Upload router blocks
        for router_name, router_data in extracted_data['routers'].items():
            # Check if router already exists and reuse_existing is enabled
            if self.reuse_existing and router_name in existing_experts:
                existing_hash = existing_experts[router_name]
                block_hashes[router_name] = existing_hash
                print(f"♾️ Router {router_name} already exists: {existing_hash[:16]}... (reused)")
                continue
                
            print(f"Uploading router: {router_name}")
            
            payload = self.create_router_block_payload(
                router_name=router_name,
                tensors=router_data['tensors'],
                layer_id=router_data['layer_id'],
                meta_hash=meta_hash,
                candidate_loss=candidate_loss,
                previous_loss=previous_loss,
                base_block_hash=existing_experts.get(router_name)
            )
            
            try:
                response = self.upload_expert_block(payload)
                block_hashes[router_name] = response.get('block_hash', 'unknown')
                print(f"✓ Router {router_name} uploaded: {block_hashes[router_name][:16]}...")
            except Exception as e:
                print(f"✗ Failed to upload router {router_name}: {e}")
                continue
        
        return block_hashes


def main():
    parser = argparse.ArgumentParser(description="Upload MoE model parameters to DAG blockchain")
    parser.add_argument("--address", required=True, help="Miner wallet address")
    parser.add_argument("--model-file", required=True, help="Path to MoE model file")
    parser.add_argument("--meta-hash", required=True, help="MetaBlock hash to depend on")
    parser.add_argument("--candidate-loss", type=float, required=False, default=None, help="Model loss score (used only when PoL is disabled)")
    parser.add_argument("--prev-loss", type=float, default=None, help="Previous model loss")
    parser.add_argument("--base-block-hash", type=str, default=None, help="Base expert block hash for CAS (version control)")
    parser.add_argument("--privkey", help="Private key for signing (hex)")
    parser.add_argument("--dry-run", action="store_true", help="Extract experts but don't upload")
    parser.add_argument("--reuse-existing", action="store_true", help="Skip uploading experts that already exist in chain")
    parser.add_argument("--skip-pow", action="store_true", help="Skip proof-of-work mining (development mode)")
    parser.add_argument("--api-url", default=DEFAULT_API_BASE, help="Main node API base URL (e.g., http://MAIN:8000)")
    parser.add_argument("--node-id", default=os.environ.get("BLYAN_NODE_ID"), help="Main node ID for auth (header X-Node-ID)")
    parser.add_argument("--node-token", default=os.environ.get("BLYAN_MAIN_NODE_TOKEN"), help="Main node token for auth (header X-Node-Auth-Token)")
    
    args = parser.parse_args()
    
    # Allow HF repo id OR local file path
    if not (Path(args.model_file).exists() or 
            (isinstance(args.model_file, str) and 
             ("/" not in args.model_file and ":" not in args.model_file))):
        # If it neither exists locally nor looks like a repo id, fail
        if not Path(args.model_file).exists():
            print(f"Error: Model file {args.model_file} not found")
            return 1
    
    try:
        # Extract MoE experts
        print(f"Loading MoE model from {args.model_file}...")
        extractor = MoEExpertExtractor(args.model_file)
        extracted_data = extractor.extract_experts()
        
        expert_count = len(extracted_data['experts'])
        router_count = len(extracted_data['routers'])
        base_param_count = len(extracted_data['base'])
        
        print(f"✓ Extracted {expert_count} experts, {router_count} routers, {base_param_count} base parameters")
        
        if expert_count == 0 and router_count == 0:
            print("Warning: No MoE experts or routers found in model")
            print("This might not be a MoE model, or the extraction patterns need updating")
            return 1
        
        # Print summary
        print("\nExtracted experts:")
        for expert_name in extracted_data['experts'].keys():
            tensor_count = len(extracted_data['experts'][expert_name]['tensors'])
            print(f"  - {expert_name}: {tensor_count} tensors")
        
        print("\nExtracted routers:")
        for router_name in extracted_data['routers'].keys():
            tensor_count = len(extracted_data['routers'][router_name]['tensors'])
            print(f"  - {router_name}: {tensor_count} tensors")
        
        if args.dry_run:
            print("\nDry run completed - no upload performed")
            return 0
        
        # Set environment variables for development mode
        if args.skip_pow:
            os.environ['SKIP_POW'] = 'true'
            print("🚧 PoW mining disabled for this upload")
        
        # Upload to blockchain
        print(f"\nUploading to DAG blockchain (meta_hash: {args.meta_hash})...")
        uploader = ExpertBlockUploader(
            miner_address=args.address,
            private_key=args.privkey,
            reuse_existing=args.reuse_existing,
            api_base=args.api_url,
            node_id=args.node_id,
            node_token=args.node_token,
        )
        
        block_hashes = uploader.upload_all_experts(
            extracted_data=extracted_data,
            meta_hash=args.meta_hash,
            candidate_loss=args.candidate_loss if args.candidate_loss is not None else 0.0,
            previous_loss=args.prev_loss
        )
        
        print(f"\n✓ Upload completed! {len(block_hashes)} blocks created")
        print("\nBlock hashes:")
        for name, hash_val in block_hashes.items():
            print(f"  {name}: {hash_val}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())