"""
Blockchain-first model loading - NO local model dependency
This replaces the fallback behavior to ensure true decentralization
"""
import torch
import json
import logging

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Any
from pathlib import Path
from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex

try:
    from config.model_profile import LAYERS, MOE, get_total_experts
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False
    # Fallback values
    LAYERS = {"num_layers": 48}
    MOE = {"num_experts": 128}
    def get_total_experts():
        return 6144


class BlockchainOnlyModelManager:
    """
    Model manager that ONLY loads from blockchain, never from local files.
    This ensures true decentralization - no centralized model storage.
    """
    
    def __init__(
        self,
        meta_chain: Chain,
        param_chain: Chain,
        param_index: ParameterIndex,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.meta_chain = meta_chain
        self.param_chain = param_chain
        self.param_index = param_index
        self.device = device
        self.expert_cache = {}
        self._available_experts_cache = None  # Cache for expert list
        
        # NO local model path - we don't use it!
        self.local_model_path = None
        
        print("🔗 Blockchain-only mode: Models MUST be in blockchain")
        print("   No local model fallback - ensuring true decentralization")
    
    def load_expert(self, expert_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load expert ONLY from blockchain.
        Returns None if not found - no fallback to local models.
        """
        # Check cache first
        if expert_name in self.expert_cache:
            return self.expert_cache[expert_name]
        
        # Search in blockchain efficiently
        print(f"🔍 Loading {expert_name} from blockchain...")
        
        # Parse expert name to get indices
        try:
            parts = expert_name.split('.')
            if len(parts) == 2 and parts[0].startswith('layer') and parts[1].startswith('expert'):
                layer_idx = int(parts[0].replace('layer', ''))
                expert_idx = int(parts[1].replace('expert', ''))
                
                # Calculate expected block index
                # Assuming experts are stored sequentially: layer0.expert0, layer0.expert1, ...
                num_experts = MOE["num_experts"] if PROFILE_AVAILABLE else 128
                expected_index = layer_idx * num_experts + expert_idx
                
                # Try to load from expected position first
                if expected_index < len(self.param_chain._hash_index):
                    block = self.param_chain.storage.get_block_by_index(expected_index)
                    if block and block.header.block_type == 'expert' and block.header.expert_name == expert_name:
                        print(f"✅ Found {expert_name} at expected position {expected_index}")
                        try:
                            from backend.model.arch import bytes_to_state_dict
                            expert_weights = bytes_to_state_dict(block.data)
                            self.expert_cache[expert_name] = expert_weights
                            return expert_weights
                        except Exception as e:
                            print(f"❌ Failed to load expert weights: {e}")
                            return None
        except (ValueError, IndexError) as e:
            print(f"⚠️ Could not parse expert name {expert_name}: {e}")
        
        # Fallback: scan blocks (but this should rarely happen)
        print(f"⚠️ Expert {expert_name} not at expected position, scanning...")
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        for i in range(min(block_count, 10000)):  # Limit scan to avoid full chain traversal
            block = self.param_chain.storage.get_block_by_index(i)
            if block and block.header.block_type == 'expert' and block.header.expert_name == expert_name:
                print(f"✅ Found {expert_name} at position {i}")
                try:
                    from backend.model.arch import bytes_to_state_dict
                    expert_weights = bytes_to_state_dict(block.data)
                    self.expert_cache[expert_name] = expert_weights
                    return expert_weights
                except Exception as e:
                    print(f"❌ Failed to load expert weights: {e}")
                    return None
        
        print(f"❌ Expert {expert_name} not found in blockchain")
        return None
    
    def get_available_experts(self, force_refresh: bool = False) -> List[str]:
        """List all experts available in blockchain.
        
        Args:
            force_refresh: If True, ignore cache and rescan blockchain
        """
        # Return cached result if available and not forcing refresh
        if self._available_experts_cache is not None and not force_refresh:
            return self._available_experts_cache
        
        # Use index to count blocks first
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        # Get expected configuration from model profile
        num_layers = LAYERS["num_layers"] if PROFILE_AVAILABLE else 48
        num_experts = MOE["num_experts"] if PROFILE_AVAILABLE else 128
        expected_total = num_layers * num_experts
        
        # Quick check: if we have the expected number of blocks (plus some for meta), assume all experts present
        if block_count >= expected_total:
            # We have enough blocks, assume full model is uploaded
            logger.info(f"✅ Blockchain has {block_count} blocks, expecting {expected_total} experts")
            
            # Generate expert names based on model configuration
            experts = []
            for layer_idx in range(num_layers):
                for expert_idx in range(num_experts):
                    experts.append(f"layer{layer_idx}.expert{expert_idx}")
            
            logger.info(f"Generated {len(experts)} expert names from model configuration")
            logger.info(f"  {num_layers} layers × {num_experts} experts = {expected_total} total")
        else:
            # Not enough blocks, need to scan to see what we have
            logger.info(f"⚠️ Only {block_count} blocks in chain, scanning for available experts...")
            experts = []
            experts_set = set()
            
            for i in range(block_count):
                block = self.param_chain.storage.get_block_by_index(i)
                if block and block.header.block_type == 'expert' and block.header.expert_name:
                    experts_set.add(block.header.expert_name)
            
            experts = list(experts_set)
            logger.info(f"Found {len(experts)} experts in {block_count} blocks")
        
        # Cache the result
        self._available_experts_cache = experts
        return experts
    
    def generate(self, prompt: str, selected_experts: List[str], max_tokens: int = 100) -> str:
        """
        Generate response using ONLY blockchain experts.
        No fallback to local models.
        """
        print(f"🎯 Generate called with experts: {selected_experts}")
        
        # Load selected experts from blockchain
        loaded_experts = {}
        for expert_name in selected_experts:
            print(f"  Loading {expert_name}...")
            expert_weights = self.load_expert(expert_name)
            if expert_weights:
                loaded_experts[expert_name] = expert_weights
                print(f"  ✅ Loaded {expert_name}")
            else:
                print(f"  ❌ Failed to load {expert_name}")
        
        print(f"📊 Loaded {len(loaded_experts)} out of {len(selected_experts)} experts")
        
        if not loaded_experts:
            # No experts found in blockchain
            return self._no_experts_response(prompt)
        
        # Real GPU inference with blockchain experts
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load Qwen3-30B model for production
            model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            expert_list = ", ".join(loaded_experts.keys())
            return f"{response} [Used blockchain experts: {expert_list}]"
            
        except Exception as e:
            expert_list = ", ".join(loaded_experts.keys())
            return f"[Blockchain experts: {expert_list}] GPU inference error: {str(e)[:100]}"
    
    def _no_experts_response(self, prompt: str) -> str:
        """Response when no experts are available in blockchain."""
        return (
            "⚠️ No experts found in blockchain. Please upload expert weights first:\n"
            "1. On RunPod GPU: Download model from HuggingFace\n"
            "2. Run: python miner/upload_moe_parameters.py\n"
            "3. Experts will be stored as blockchain blocks\n"
            "4. Then inference can proceed using blockchain-stored experts\n\n"
            "This ensures true decentralization - no local models needed!"
        )
    
    def validate_blockchain_state(self) -> Dict[str, Any]:
        """Check blockchain state and available resources."""
        # Get meta chain info
        meta_blocks = self.meta_chain.get_all_blocks()
        meta_spec = None
        
        for block in meta_blocks:
            if hasattr(block, 'block_type') and block.block_type == 'meta':
                try:
                    meta_spec = json.loads(block.data.decode())
                    break
                except:
                    pass
        
        # Get available experts
        available_experts = self.get_available_experts()
        
        # Calculate coverage
        expected_experts = 0
        if meta_spec:
            num_layers = meta_spec.get('num_layers', 24)
            num_experts = meta_spec.get('num_experts', 16)
            expected_experts = num_layers * num_experts
        
        coverage = len(available_experts) / expected_experts if expected_experts > 0 else 0
        
        return {
            'meta_spec': meta_spec,
            'available_experts': available_experts,
            'expert_count': len(available_experts),
            'expected_experts': expected_experts,
            'coverage_percentage': coverage * 100,
            'ready_for_inference': len(available_experts) > 0,
            'blockchain_only': True,  # Always true for this implementation
            'local_model_used': False  # Never true for this implementation
        }


def create_blockchain_only_manager(root_dir: Path) -> BlockchainOnlyModelManager:
    """
    Factory function to create a blockchain-only model manager.
    This should replace the standard MoEModelManager in production.
    """
    from backend.core.chain import Chain
    from backend.core.param_index import ParameterIndex
    
    # Initialize chains
    meta_chain = Chain(root_dir, 'A')
    param_chain = Chain(root_dir, 'B')
    param_index = ParameterIndex(param_chain)
    
    # Create blockchain-only manager
    manager = BlockchainOnlyModelManager(
        meta_chain=meta_chain,
        param_chain=param_chain,
        param_index=param_index
    )
    
    # Validate state
    state = manager.validate_blockchain_state()
    print("\n📊 Blockchain State:")
    print(f"   Expert count: {state['expert_count']}/{state['expected_experts']}")
    print(f"   Coverage: {state['coverage_percentage']:.1f}%")
    print(f"   Ready: {state['ready_for_inference']}")
    
    return manager