from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime


class ParameterIndex:
    """Persistent mapping from parameter name -> block index with model versioning.

    Stored as a JSON file on disk with metadata to ensure consistency across models.
    Includes model_id, config hash, and layer count to prevent cross-model reuse.
    """

    VERSION = "2.0"  # Index format version

    def __init__(self, path: Path):
        self.path = path
        self.checksum_path = path.with_suffix('.sha256')
        self.metadata_path = path.with_suffix('.meta')
        self._index: Dict[str, int] = {}
        self._metadata: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                # Check integrity first
                if not self._verify_checksum():
                    import logging
                    logging.warning(f"Param index checksum mismatch - rebuilding from blockchain")
                    self._index = {}
                    self._metadata = {}
                    return
                
                # Load index
                with self.path.open() as fp:
                    data = json.load(fp)
                    
                    # Handle both old format (direct dict) and new format (with metadata)
                    if isinstance(data, dict) and "version" in data and "index" in data:
                        # New format with metadata
                        self._metadata = data.get("metadata", {})
                        self._index = {k: int(v) for k, v in data["index"].items()}
                    else:
                        # Old format - just the index
                        self._index = {k: int(v) for k, v in data.items()}
                        self._metadata = {}
                        
                # Load separate metadata file if exists (for backward compatibility)
                if self.metadata_path.exists():
                    try:
                        with self.metadata_path.open() as fp:
                            file_meta = json.load(fp)
                            # File metadata takes precedence
                            self._metadata.update(file_meta)
                    except Exception:
                        pass
                        
            except Exception as e:
                import logging
                logging.error(f"Failed to load param index: {e}")
                self._index = {}
                self._metadata = {}
        else:
            self._index = {}
            self._metadata = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in new format with metadata
        data = {
            "version": self.VERSION,
            "metadata": self._metadata,
            "index": self._index,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Write index with metadata
        with self.path.open("w") as fp:
            json.dump(data, fp, indent=2)
        
        # Also write separate metadata file for compatibility
        with self.metadata_path.open("w") as fp:
            json.dump(self._metadata, fp, indent=2)
        
        # Write checksum
        self._write_checksum()

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------
    def set_metadata(self, model_id: str, num_hidden_layers: int, 
                     config_hash: Optional[str] = None, **kwargs) -> None:
        """Set model metadata for versioning and validation.
        
        Args:
            model_id: Model identifier (e.g., "Qwen/Qwen3-8B")
            num_hidden_layers: Number of hidden layers in the model
            config_hash: Hash of the model config for validation
            **kwargs: Additional metadata fields
        """
        self._metadata.update({
            "model_id": model_id,
            "num_hidden_layers": num_hidden_layers,
            "config_hash": config_hash or self._compute_config_hash(model_id, num_hidden_layers),
            "version": self.VERSION,
            "created_at": self._metadata.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": datetime.utcnow().isoformat(),
            **kwargs
        })
        self._save()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata."""
        return dict(self._metadata)
    
    def validate_metadata(self, model_id: str, num_hidden_layers: int,
                         config_hash: Optional[str] = None) -> bool:
        """Validate that index matches the expected model configuration.
        
        Args:
            model_id: Expected model identifier
            num_hidden_layers: Expected number of layers
            config_hash: Expected config hash (optional)
            
        Returns:
            True if metadata matches, False otherwise
        """
        if not self._metadata:
            # No metadata - consider invalid for safety
            return False
        
        # Check model ID
        if self._metadata.get("model_id") != model_id:
            import logging
            logging.warning(f"Model ID mismatch: expected {model_id}, got {self._metadata.get('model_id')}")
            return False
        
        # Check layer count
        if self._metadata.get("num_hidden_layers") != num_hidden_layers:
            import logging
            logging.warning(f"Layer count mismatch: expected {num_hidden_layers}, got {self._metadata.get('num_hidden_layers')}")
            return False
        
        # Check config hash if provided
        if config_hash and self._metadata.get("config_hash") != config_hash:
            import logging
            logging.warning(f"Config hash mismatch")
            return False
        
        return True
    
    def _compute_config_hash(self, model_id: str, num_hidden_layers: int) -> str:
        """Compute a hash based on model configuration."""
        config_str = f"{model_id}:{num_hidden_layers}:{self.VERSION}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set(self, name: str, block_index: int) -> None:
        self._index[name] = block_index
        self._save()

    def bulk_set(self, mapping: Dict[str, int]) -> None:
        self._index.update(mapping)
        self._save()

    def get(self, name: str) -> Optional[int]:
        return self._index.get(name)

    def all(self) -> Dict[str, int]:
        return dict(self._index)
    
    def get_all_layers(self) -> list[str]:
        """Get all layer names from the index."""
        return list(self._index.keys())
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self._index = {}
        self._metadata = {}
        self._save()
        
        import logging
        logging.info("Cleared parameter index and metadata")
    
    # ------------------------------------------------------------------
    # Integrity methods
    # ------------------------------------------------------------------
    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of the index file."""
        if not self.path.exists():
            return ""
        
        sha256 = hashlib.sha256()
        with open(self.path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _write_checksum(self) -> None:
        """Write checksum file for the index."""
        try:
            checksum = self._compute_checksum()
            with open(self.checksum_path, 'w') as f:
                f.write(checksum)
        except Exception as e:
            import logging
            logging.warning(f"Failed to write param index checksum: {e}")
    
    def _verify_checksum(self) -> bool:
        """Verify the index file against its checksum."""
        import logging
        
        if not self.checksum_path.exists():
            # No checksum file yet - this is OK for migration
            if self.path.exists():
                # But warn if index exists without checksum
                logging.warning(f"⚠️ Parameter index exists without checksum file - creating one")
                self._write_checksum()
            return True
        
        try:
            with open(self.checksum_path, 'r') as f:
                expected = f.read().strip()
            
            actual = self._compute_checksum()
            
            if actual != expected:
                logging.error(f"❌ CHECKSUM MISMATCH for param_index!")
                logging.error(f"   Expected: {expected[:16]}...")
                logging.error(f"   Got: {actual[:16]}...")
                return False
            
            return True
        except Exception as e:
            # Log warning but don't break existing systems
            logging.warning(f"⚠️ Could not verify param_index checksum: {e}")
            # In production, you might want stricter behavior:
            if os.getenv("STRICT_CHECKSUM", "false").lower() == "true":
                return False
            return True
    
    def get_index_hash(self) -> str:
        """Get a hash of the current index for snapshot validation."""
        # Include both metadata and index content
        content = {
            "metadata": self._metadata,
            "index": sorted(self._index.items())  # Sort for consistency
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def rebuild_if_corrupted(self, chain_b) -> bool:
        """Rebuild index from blockchain if corrupted."""
        if self._verify_checksum():
            return False  # No rebuild needed
        
        import logging
        logging.warning("Param index corrupted - rebuilding from blockchain")
        
        try:
            # Clear current index
            self._index = {}
            
            # Rebuild from chain B
            if hasattr(chain_b, 'get_blocks_by_type'):
                layer_blocks = chain_b.get_blocks_by_type('dense_layer')
                for block in layer_blocks:
                    if hasattr(block.header, 'layer_name'):
                        self._index[block.header.layer_name] = block.header.index
                
                self._save()
                logging.info(f"Rebuilt param index with {len(self._index)} entries")
                return True
        except Exception as e:
            logging.error(f"Failed to rebuild param index: {e}")
        
        return False
    
    def invalidate_if_model_changed(self, model_id: str, num_hidden_layers: int) -> bool:
        """Invalidate index if model configuration changed.
        
        Args:
            model_id: Current model identifier
            num_hidden_layers: Current number of layers
            
        Returns:
            True if index was invalidated, False if still valid
        """
        if not self.validate_metadata(model_id, num_hidden_layers):
            import logging
            logging.warning(f"Model configuration changed - invalidating param index")
            logging.info(f"  Old: {self._metadata.get('model_id')}, {self._metadata.get('num_hidden_layers')} layers")
            logging.info(f"  New: {model_id}, {num_hidden_layers} layers")
            
            # Clear index and update metadata
            self._index = {}
            self.set_metadata(model_id, num_hidden_layers)
            return True
        
        return False