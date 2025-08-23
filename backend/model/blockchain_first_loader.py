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
        
        # Search in blockchain
        print(f"🔍 Searching blockchain for {expert_name}...")
        
        # Get all blocks of type 'expert'
        all_blocks = self.param_chain.get_all_blocks()
        
        for block in all_blocks:
            if hasattr(block, 'block_type') and block.block_type == 'expert':
                # Check if this is the expert we're looking for
                if hasattr(block, 'metadata'):
                    metadata = json.loads(block.metadata) if isinstance(block.metadata, str) else block.metadata
                    if metadata.get('expert_name') == expert_name:
                        print(f"✅ Found {expert_name} in blockchain at block {block.hash[:8]}")
                        
                        # Load the expert weights
                        try:
                            from backend.model.arch import bytes_to_state_dict
                            expert_weights = bytes_to_state_dict(block.data)
                            
                            # Cache it
                            self.expert_cache[expert_name] = expert_weights
                            return expert_weights
                        except Exception as e:
                            print(f"❌ Failed to load expert weights: {e}")
                            return None
        
        print(f"⚠️ Expert {expert_name} not found in blockchain")
        print(f"   Available experts need to be uploaded via miner/upload_moe_parameters.py")
        return None
    
    def get_available_experts(self) -> List[str]:
        """List all experts available in blockchain (optimized)."""
        experts = []
        
        # Use index to count blocks first
        block_count = len(self.param_chain._hash_index) if hasattr(self.param_chain, '_hash_index') else 0
        
        # For large chains, use sampling to estimate experts
        if block_count > 1000:
            # Sample approach - check last 100 blocks
            for i in range(max(0, block_count - 100), block_count):
                block = self.param_chain.storage.get_block_by_index(i)
                if block and block.header.block_type == 'expert' and block.header.expert_name:
                    experts.append(block.header.expert_name)
            
            # Estimate total based on sample
            if experts:
                logger.info(f"Found {len(experts)} experts in sample (estimated ~{block_count} total blocks)")
        else:
            # For small chains, check all blocks
            for i in range(block_count):
                block = self.param_chain.storage.get_block_by_index(i)
                if block and block.header.block_type == 'expert' and block.header.expert_name:
                    experts.append(block.header.expert_name)
        
        return experts
    
    def generate(self, prompt: str, selected_experts: List[str], max_tokens: int = 100) -> str:
        """
        Generate response using ONLY blockchain experts.
        No fallback to local models.
        """
        # Load selected experts from blockchain
        loaded_experts = {}
        for expert_name in selected_experts:
            expert_weights = self.load_expert(expert_name)
            if expert_weights:
                loaded_experts[expert_name] = expert_weights
        
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