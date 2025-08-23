#!/usr/bin/env python3
"""
Upload ALL experts from a MoE model to the blockchain.
This ensures every layer is properly represented.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
from pathlib import Path
from backend.core.chain import Chain
from backend.model.arch import state_dict_to_bytes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_all_experts(model_path: str, meta_hash: str, resume: bool = True):
    """Upload all experts from a model to blockchain."""
    
    # Initialize chains
    data_dir = Path("./data")
    meta_chain = Chain(data_dir, "A")
    param_chain = Chain(data_dir, "B")
    
    # Track progress
    progress_file = data_dir / "upload_progress.json"
    uploaded_experts = set()
    
    if resume and progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            uploaded_experts = set(progress.get('uploaded', []))
            logger.info(f"Resuming upload - {len(uploaded_experts)} experts already uploaded")
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}")
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU to save GPU memory
            trust_remote_code=True
        )
        
        state_dict = model.state_dict()
        logger.info(f"Model loaded with {len(state_dict)} parameters")
        
        # Process each layer
        num_layers = 48  # Qwen3-30B has 48 layers
        num_experts = 128  # 128 experts per layer
        
        for layer_idx in range(num_layers):
            logger.info(f"\nProcessing layer {layer_idx}...")
            
            for expert_idx in range(num_experts):
                expert_name = f"layer{layer_idx}.expert{expert_idx}"
                
                # Skip if already uploaded
                if expert_name in uploaded_experts:
                    continue
                
                # Collect parameters for this expert
                expert_params = {}
                param_count = 0
                
                # Look for MoE expert parameters
                for key, value in state_dict.items():
                    # Match patterns like:
                    # model.layers.0.mlp.experts.0.gate_proj.weight
                    # model.layers.0.mlp.experts.0.up_proj.weight
                    # model.layers.0.mlp.experts.0.down_proj.weight
                    if f"layers.{layer_idx}" in key and f"experts.{expert_idx}" in key:
                        expert_params[key] = value
                        param_count += 1
                
                if not expert_params:
                    # Try alternative naming patterns
                    for key, value in state_dict.items():
                        if f"layer{layer_idx}" in key and f"expert{expert_idx}" in key:
                            expert_params[key] = value
                            param_count += 1
                
                if expert_params:
                    # Serialize and upload
                    logger.info(f"  Uploading {expert_name} with {param_count} parameters...")
                    
                    expert_bytes = state_dict_to_bytes(expert_params)
                    
                    # Create block
                    block_hash = param_chain.add_block(
                        data=expert_bytes,
                        block_type='expert',
                        expert_name=expert_name,
                        layer_id=f"layer{layer_idx}",
                        depends_on=[meta_hash],
                        metadata={
                            'expert_index': expert_idx,
                            'layer_index': layer_idx,
                            'num_parameters': param_count,
                            'model': 'Qwen3-30B'
                        }
                    )
                    
                    logger.info(f"    ✓ Uploaded as block {block_hash[:8]}...")
                    
                    # Update progress
                    uploaded_experts.add(expert_name)
                    with open(progress_file, 'w') as f:
                        json.dump({'uploaded': list(uploaded_experts)}, f)
                else:
                    logger.warning(f"  No parameters found for {expert_name}")
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Upload complete!")
        logger.info(f"Total experts uploaded: {len(uploaded_experts)}")
        logger.info(f"Expected: {num_layers * num_experts}")
        
        # Verify all layers are represented
        layers_with_experts = set()
        for expert in uploaded_experts:
            layer = expert.split('.')[0]
            layers_with_experts.add(layer)
        
        logger.info(f"Layers with experts: {len(layers_with_experts)}/48")
        
        if len(layers_with_experts) < 48:
            missing = []
            for i in range(48):
                if f"layer{i}" not in layers_with_experts:
                    missing.append(f"layer{i}")
            logger.warning(f"Missing layers: {missing}")
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Upload all MoE experts to blockchain')
    parser.add_argument('--model', default='Qwen/Qwen3-30B-A3B-Instruct-2507-FP8',
                        help='Model name or path')
    parser.add_argument('--meta-hash', required=True,
                        help='Meta chain block hash')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore previous progress')
    
    args = parser.parse_args()
    
    upload_all_experts(
        model_path=args.model,
        meta_hash=args.meta_hash,
        resume=not args.no_resume
    )

if __name__ == "__main__":
    main()