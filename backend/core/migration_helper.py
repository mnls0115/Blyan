"""
Migration helper for converting legacy MoE blockchain data to dense model format.
"""
import logging
from typing import Dict, Optional, List
from pathlib import Path
from .chain import Chain
from .param_index import ParameterIndex

logger = logging.getLogger(__name__)


class BlockchainMigrationHelper:
    """
    Helper class for migrating between blockchain data formats.
    """
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.meta_chain = Chain(root_dir, "A")
        self.param_chain = Chain(root_dir, "B")
        self.param_index = ParameterIndex(root_dir / "param_index.json")
    
    def detect_format(self) -> str:
        """
        Detect the current format of blockchain data.
        
        Returns:
            'dense': Dense model format with layer_name
            'moe': Legacy MoE format with expert_name
            'mixed': Both formats present
            'empty': No blockchain data
        """
        blocks = list(self.param_chain.storage.iter_blocks())
        
        if not blocks:
            return 'empty'
        
        has_dense = False
        has_moe = False
        
        for block in blocks:
            if hasattr(block.header, 'layer_name') and block.header.layer_name:
                has_dense = True
            if hasattr(block.header, 'expert_name') and block.header.expert_name:
                has_moe = True
        
        if has_dense and has_moe:
            return 'mixed'
        elif has_dense:
            return 'dense'
        elif has_moe:
            return 'moe'
        else:
            return 'empty'
    
    def get_migration_plan(self) -> Dict:
        """
        Analyze blockchain and create migration plan.
        
        Returns:
            Dictionary with migration information
        """
        format_type = self.detect_format()
        blocks = list(self.param_chain.storage.iter_blocks())
        
        plan = {
            'current_format': format_type,
            'total_blocks': len(blocks),
            'dense_blocks': 0,
            'moe_blocks': 0,
            'unmapped_blocks': 0,
            'mappings': {}
        }
        
        for block in blocks:
            if hasattr(block.header, 'layer_name') and block.header.layer_name:
                plan['dense_blocks'] += 1
            elif hasattr(block.header, 'expert_name') and block.header.expert_name:
                plan['moe_blocks'] += 1
                # Suggest mapping for MoE expert to dense layer
                expert_name = block.header.expert_name
                if 'layer' in expert_name:
                    # Try to extract layer number
                    try:
                        parts = expert_name.split('.')
                        for part in parts:
                            if 'layer' in part:
                                layer_num = ''.join(filter(str.isdigit, part))
                                if layer_num:
                                    suggested_mapping = f"layer_{layer_num}"
                                    plan['mappings'][expert_name] = suggested_mapping
                    except:
                        pass
            else:
                plan['unmapped_blocks'] += 1
        
        return plan
    
    def can_use_for_dense_model(self) -> bool:
        """
        Check if current blockchain data can be used for dense model inference.
        
        Returns:
            True if dense model can work with current data
        """
        format_type = self.detect_format()
        
        if format_type == 'empty':
            logger.info("No blockchain data available")
            return False
        
        if format_type == 'dense':
            logger.info("Blockchain contains dense model format - ready to use")
            return True
        
        if format_type == 'moe':
            # Check if we have essential layers
            all_params = self.param_index.all()
            has_embedding = any('embed' in key.lower() for key in all_params.keys())
            has_layers = any('layer' in key.lower() for key in all_params.keys())
            has_lm_head = any('lm_head' in key.lower() or 'output' in key.lower() for key in all_params.keys())
            
            if has_embedding or has_layers or has_lm_head:
                logger.warning("Blockchain contains MoE format but has some compatible layers")
                return True
            else:
                logger.error("Blockchain contains MoE format without compatible layers for dense model")
                return False
        
        if format_type == 'mixed':
            logger.info("Blockchain contains mixed format - will use dense blocks")
            return True
        
        return False
    
    def create_compatibility_index(self) -> Dict[str, int]:
        """
        Create a compatibility index that maps both old and new formats.
        
        Returns:
            Dictionary mapping layer names to block indices
        """
        compat_index = {}
        blocks = list(self.param_chain.storage.iter_blocks())
        
        for block in blocks:
            block_idx = block.header.index
            
            # Handle dense format
            if hasattr(block.header, 'layer_name') and block.header.layer_name:
                compat_index[block.header.layer_name] = block_idx
            
            # Handle MoE format
            if hasattr(block.header, 'expert_name') and block.header.expert_name:
                # Store under expert name for backward compatibility
                compat_index[f"expert_{block.header.expert_name}"] = block_idx
                
                # Also try to map to dense layer if possible
                if 'layer' in block.header.expert_name:
                    try:
                        layer_num = ''.join(filter(str.isdigit, block.header.expert_name.split('.')[-1]))
                        if layer_num:
                            compat_index[f"layer_{layer_num}"] = block_idx
                    except:
                        pass
        
        return compat_index
    
    def print_migration_report(self):
        """Print a detailed migration report."""
        plan = self.get_migration_plan()
        
        print("\n" + "="*60)
        print("BLOCKCHAIN MIGRATION REPORT")
        print("="*60)
        print(f"Current Format: {plan['current_format']}")
        print(f"Total Blocks: {plan['total_blocks']}")
        print(f"  - Dense Model Blocks: {plan['dense_blocks']}")
        print(f"  - MoE Expert Blocks: {plan['moe_blocks']}")
        print(f"  - Unmapped Blocks: {plan['unmapped_blocks']}")
        
        if plan['mappings']:
            print("\nSuggested MoE → Dense Mappings:")
            for moe_name, dense_name in plan['mappings'].items():
                print(f"  {moe_name} → {dense_name}")
        
        can_use = self.can_use_for_dense_model()
        print(f"\nCan use for dense model: {'✅ Yes' if can_use else '❌ No'}")
        
        if plan['current_format'] == 'moe':
            print("\n⚠️  Warning: Legacy MoE format detected")
            print("   Consider re-uploading model in dense format for optimal performance")
        
        print("="*60 + "\n")


def check_blockchain_compatibility(root_dir: Path = Path("./data")) -> bool:
    """
    Quick check if blockchain is compatible with dense model.
    
    Args:
        root_dir: Root directory for blockchain data
        
    Returns:
        True if compatible
    """
    helper = BlockchainMigrationHelper(root_dir)
    return helper.can_use_for_dense_model()


def print_blockchain_status(root_dir: Path = Path("./data")):
    """
    Print blockchain status and migration information.
    
    Args:
        root_dir: Root directory for blockchain data
    """
    helper = BlockchainMigrationHelper(root_dir)
    helper.print_migration_report()