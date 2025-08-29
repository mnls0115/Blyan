"""
Layer Manifest for Partial Node Support

Purpose: Track which layers a node can serve based on param_index.json,
verify hosted layers' bodies at startup, and provide readiness status.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class LayerEntry:
    """Manifest entry for a hosted layer."""
    layer_name: str
    block_index: int
    block_hash: str
    payload_hash: str
    verified: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerEntry':
        """Create from dictionary."""
        return cls(**data)


class LayerManifest:
    """Manages layer manifest for partial node support."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize layer manifest.
        
        Args:
            data_dir: Data directory containing param_index and chains
        """
        self.data_dir = Path(data_dir)
        self.manifest_file = self.data_dir / "layer_manifest.json"
        self.param_index_file = self.data_dir / "param_index.json"
        self.manifest: Dict[str, LayerEntry] = {}
    
    def build_from_param_index(self, chain_b) -> None:
        """
        Build manifest from param_index.json.
        
        Args:
            chain_b: Parameter chain to get header info from
        """
        if not self.param_index_file.exists():
            logger.info("No param_index.json found, skipping manifest build")
            return
        
        try:
            # Load parameter index
            with open(self.param_index_file, 'r') as f:
                param_index = json.load(f)
            
            # Try to load header index for efficient hash lookup
            from backend.core.header_index import HeaderIndex
            header_index = HeaderIndex(self.data_dir / "B")
            header_records = header_index.load()
            
            # Build manifest entries
            self.manifest = {}
            
            for layer_name, block_index in param_index.items():
                try:
                    # Get header record for this block
                    if block_index < len(header_records):
                        record = header_records[block_index]
                        entry = LayerEntry(
                            layer_name=layer_name,
                            block_index=block_index,
                            block_hash=record.hash,
                            payload_hash=record.payload_hash,
                            verified=False  # Will verify later
                        )
                    else:
                        # Fallback: load block to get hashes
                        logger.debug(f"No header record for {layer_name} at index {block_index}, loading block")
                        block = None
                        if hasattr(chain_b, 'get_block_by_index'):
                            block = chain_b.get_block_by_index(block_index)
                        elif hasattr(chain_b, 'storage'):
                            block = chain_b.storage.get_block_by_index(block_index)
                        
                        if block:
                            entry = LayerEntry(
                                layer_name=layer_name,
                                block_index=block_index,
                                block_hash=block.compute_hash(),
                                payload_hash=hashlib.sha256(block.payload).hexdigest(),
                                verified=False
                            )
                        else:
                            logger.warning(f"Could not find block for {layer_name} at index {block_index}")
                            continue
                    
                    self.manifest[layer_name] = entry
                    
                except Exception as e:
                    logger.error(f"Failed to add {layer_name} to manifest: {e}")
            
            # Save manifest
            self.save()
            logger.info(f"Built layer manifest with {len(self.manifest)} entries")
            
        except Exception as e:
            logger.error(f"Failed to build manifest from param_index: {e}")
    
    def verify_hosted_layers(self, chain_b, max_verify: int = 100) -> Dict:
        """
        Verify hosted layers by checking payload hashes.
        
        Args:
            chain_b: Parameter chain to load blocks from
            max_verify: Maximum number of layers to verify (bounded)
        
        Returns:
            Verification summary dict
        """
        if not self.manifest:
            self.load()
        
        if not self.manifest:
            return {
                'hosted_count': 0,
                'verified_count': 0,
                'missing_layers': [],
                'ready_to_serve': False
            }
        
        # Get block loader
        loader = None
        if hasattr(chain_b, 'get_block_by_index'):
            loader = chain_b.get_block_by_index
        elif hasattr(chain_b, 'storage'):
            loader = chain_b.storage.get_block_by_index
        else:
            logger.warning("No block loader available for verification")
            return {
                'hosted_count': len(self.manifest),
                'verified_count': 0,
                'missing_layers': [],
                'ready_to_serve': False
            }
        
        # Verify each hosted layer (bounded)
        verified_count = 0
        missing_layers = []
        layers_to_verify = list(self.manifest.items())[:max_verify]
        
        logger.info(f"Verifying {len(layers_to_verify)} hosted layers...")
        
        for layer_name, entry in layers_to_verify:
            try:
                # Load block body
                block = loader(entry.block_index)
                if not block:
                    logger.warning(f"Missing block for {layer_name} at index {entry.block_index}")
                    missing_layers.append(layer_name)
                    entry.verified = False
                    continue
                
                # Verify payload hash
                payload_hash = hashlib.sha256(block.payload).hexdigest()
                if payload_hash != entry.payload_hash:
                    logger.error(f"Payload hash mismatch for {layer_name}")
                    entry.verified = False
                    continue
                
                # Verify block hash
                block_hash = block.compute_hash()
                if block_hash != entry.block_hash:
                    logger.error(f"Block hash mismatch for {layer_name}")
                    entry.verified = False
                    continue
                
                entry.verified = True
                verified_count += 1
                
            except Exception as e:
                logger.error(f"Failed to verify {layer_name}: {e}")
                entry.verified = False
        
        # Update manifest with verification status
        self.save()
        
        # Calculate readiness
        ready_to_serve = (verified_count == len(layers_to_verify)) and len(missing_layers) == 0
        
        summary = {
            'hosted_count': len(self.manifest),
            'verified_count': verified_count,
            'missing_layers': missing_layers[:10],  # First 10 missing
            'ready_to_serve': ready_to_serve
        }
        
        # Log summary
        logger.info(f"Layer verification complete:")
        logger.info(f"  Hosted: {summary['hosted_count']}")
        logger.info(f"  Verified: {summary['verified_count']}")
        logger.info(f"  Ready: {'✅' if ready_to_serve else '❌'}")
        
        if missing_layers:
            logger.warning(f"  Missing: {missing_layers[:5]}")
        
        return summary
    
    def get_missing_layers(self, expected_layers: List[str]) -> List[str]:
        """
        Get list of layers that are expected but not hosted.
        
        Args:
            expected_layers: List of expected layer names
        
        Returns:
            List of missing layer names
        """
        if not self.manifest:
            self.load()
        
        hosted = set(self.manifest.keys())
        expected = set(expected_layers)
        missing = expected - hosted
        
        return sorted(list(missing))
    
    def get_verified_layers(self) -> List[str]:
        """Get list of verified layer names."""
        if not self.manifest:
            self.load()
        
        return [name for name, entry in self.manifest.items() if entry.verified]
    
    def save(self) -> None:
        """Save manifest to file."""
        try:
            data = {
                name: entry.to_dict() 
                for name, entry in self.manifest.items()
            }
            
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Write atomically
            temp_file = self.manifest_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.manifest_file)
            
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def load(self) -> None:
        """Load manifest from file."""
        if not self.manifest_file.exists():
            self.manifest = {}
            return
        
        try:
            with open(self.manifest_file, 'r') as f:
                data = json.load(f)
            
            self.manifest = {
                name: LayerEntry.from_dict(entry_data)
                for name, entry_data in data.items()
            }
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            self.manifest = {}
    
    def is_ready(self) -> bool:
        """Check if all hosted layers are verified and ready."""
        if not self.manifest:
            return False
        
        return all(entry.verified for entry in self.manifest.values())
    
    def get_summary(self) -> Dict:
        """Get manifest summary for health reporting."""
        if not self.manifest:
            self.load()
        
        verified = [name for name, entry in self.manifest.items() if entry.verified]
        unverified = [name for name, entry in self.manifest.items() if not entry.verified]
        
        return {
            'total_layers': len(self.manifest),
            'verified_layers': len(verified),
            'unverified_layers': len(unverified),
            'ready': self.is_ready(),
            'sample_verified': verified[:5] if verified else [],
            'sample_unverified': unverified[:5] if unverified else []
        }