"""
Manifest/catalog module for describing blockchain objects without reading full bodies.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack not available, using JSON fallback")


@dataclass
class ManifestEntry:
    """Single entry in the manifest."""
    key: Tuple[str, str, str]  # (layer, expert, part)
    cid: str                   # Content identifier
    size: int                  # Size in bytes
    shard: int                 # Shard index
    offset: int                # Offset within shard
    length: int                # Length of data
    merkle_root: str           # Merkle tree root hash
    parent_hash: str           # Previous block hash
    timestamp: float           # Creation timestamp
    block_index: Optional[int] = None  # Original block index
    block_type: Optional[str] = None   # Block type (meta, expert, etc.)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'cid': self.cid,
            'size': self.size,
            'shard': self.shard,
            'offset': self.offset,
            'length': self.length,
            'merkle_root': self.merkle_root,
            'parent_hash': self.parent_hash,
            'ts': self.timestamp,
            'block_index': self.block_index,
            'block_type': self.block_type
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ManifestEntry':
        """Create from dictionary."""
        return cls(
            key=tuple(d['key']),
            cid=d['cid'],
            size=d['size'],
            shard=d['shard'],
            offset=d['offset'],
            length=d['length'],
            merkle_root=d['merkle_root'],
            parent_hash=d['parent_hash'],
            timestamp=d['ts'],
            block_index=d.get('block_index'),
            block_type=d.get('block_type')
        )


class Manifest:
    """
    Blockchain manifest for fast metadata access.
    """
    
    VERSION = "1.0.0"
    SCHEMA_VERSION = 1
    
    def __init__(self, chain_id: str = "B"):
        """Initialize empty manifest."""
        self.chain_id = chain_id
        self.height = 0
        self.head = ""  # Hash of latest block
        self.objects: List[ManifestEntry] = []
        self._index: Dict[Tuple, ManifestEntry] = {}  # key -> entry
        self._cid_index: Dict[str, ManifestEntry] = {}  # cid -> entry
        self.created_at = time.time()
        self.hash_algo = "blake3" if HAS_MSGPACK else "sha256"
    
    def add_entry(self, entry: ManifestEntry):
        """Add entry to manifest."""
        self.objects.append(entry)
        self._index[entry.key] = entry
        self._cid_index[entry.cid] = entry
        
        if entry.block_index is not None:
            self.height = max(self.height, entry.block_index + 1)
    
    def find(self, key: Tuple[str, str, str]) -> Optional[ManifestEntry]:
        """Find entry by key."""
        return self._index.get(key)
    
    def find_by_cid(self, cid: str) -> Optional[ManifestEntry]:
        """Find entry by CID."""
        return self._cid_index.get(cid)
    
    def get_entries_by_layer(self, layer: str) -> List[ManifestEntry]:
        """Get all entries for a specific layer."""
        return [e for e in self.objects if e.key[0] == layer]
    
    def get_entries_by_expert(self, layer: str, expert: str) -> List[ManifestEntry]:
        """Get all entries for a specific expert."""
        return [e for e in self.objects if e.key[0] == layer and e.key[1] == expert]
    
    def compute_head_hash(self) -> str:
        """Compute hash of manifest head."""
        # Create deterministic representation
        head_data = {
            'version': self.VERSION,
            'schema_version': self.SCHEMA_VERSION,
            'chain_id': self.chain_id,
            'height': self.height,
            'object_count': len(self.objects),
            'created_at': self.created_at
        }
        
        if self.hash_algo == "blake3":
            import blake3
            return blake3.blake3(json.dumps(head_data, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.sha256(json.dumps(head_data, sort_keys=True).encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert manifest to dictionary."""
        return {
            'version': self.VERSION,
            'schema_version': self.SCHEMA_VERSION,
            'chain_id': self.chain_id,
            'height': self.height,
            'head': self.head,
            'created_at': self.created_at,
            'hash_algo': self.hash_algo,
            'objects': [e.to_dict() for e in self.objects],
            'counts': self.get_counts()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Manifest':
        """Create manifest from dictionary."""
        manifest = cls(d.get('chain_id', 'B'))
        manifest.height = d['height']
        manifest.head = d['head']
        manifest.created_at = d.get('created_at', time.time())
        manifest.hash_algo = d.get('hash_algo', 'sha256')
        
        for obj_dict in d['objects']:
            entry = ManifestEntry.from_dict(obj_dict)
            manifest.add_entry(entry)
        
        return manifest
    
    def save(self, path: Path):
        """Save manifest to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update head hash
        self.head = self.compute_head_hash()
        
        data = self.to_dict()
        
        if HAS_MSGPACK and path.suffix == '.msgpack':
            # Use msgpack for efficiency
            with open(path, 'wb') as f:
                msgpack.pack(data, f, use_bin_type=True)
        else:
            # Fallback to JSON
            if path.suffix != '.json':
                path = path.with_suffix('.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Manifest':
        """Load manifest from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        
        if HAS_MSGPACK and path.suffix == '.msgpack':
            with open(path, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate manifest integrity.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check version
        if not self.VERSION:
            errors.append("Missing version")
        
        # Check for duplicate keys
        seen_keys = set()
        for entry in self.objects:
            if entry.key in seen_keys:
                errors.append(f"Duplicate key: {entry.key}")
            seen_keys.add(entry.key)
        
        # Check for duplicate CIDs (allowed but worth noting)
        cid_counts = {}
        for entry in self.objects:
            cid_counts[entry.cid] = cid_counts.get(entry.cid, 0) + 1
        
        # Check shard/offset validity
        for entry in self.objects:
            if entry.shard < 0:
                errors.append(f"Invalid shard index: {entry.shard}")
            if entry.offset < 0:
                errors.append(f"Invalid offset: {entry.offset}")
            if entry.length <= 0:
                errors.append(f"Invalid length: {entry.length}")
        
        return (len(errors) == 0, errors)
    
    def get_counts(self) -> Dict[str, Any]:
        """Get counts by layer and expert."""
        counts = {
            'total': len(self.objects),
            'by_layer': {},
            'by_expert': {},
            'by_type': {}
        }
        
        for entry in self.objects:
            layer = entry.key[0]
            expert = entry.key[1]
            
            # Count by layer
            counts['by_layer'][layer] = counts['by_layer'].get(layer, 0) + 1
            
            # Count by expert
            expert_key = f"{layer}.{expert}"
            counts['by_expert'][expert_key] = counts['by_expert'].get(expert_key, 0) + 1
            
            # Count by type
            if entry.block_type:
                counts['by_type'][entry.block_type] = counts['by_type'].get(entry.block_type, 0) + 1
        
        return counts
    
    def print_summary(self):
        """Print manifest summary."""
        print(f"Manifest Summary")
        print("=" * 60)
        print(f"Chain ID:     {self.chain_id}")
        print(f"Height:       {self.height}")
        print(f"Head:         {self.head[:16]}...")
        print(f"Objects:      {len(self.objects)}")
        print(f"Created:      {time.ctime(self.created_at)}")
        print(f"Hash Algo:    {self.hash_algo}")
        
        counts = self.get_counts()
        
        print("\nCounts by Layer:")
        for layer, count in sorted(counts['by_layer'].items()):
            print(f"  {layer:20} {count:6}")
        
        print("\nTop Experts by Count:")
        expert_counts = sorted(counts['by_expert'].items(), key=lambda x: x[1], reverse=True)
        for expert, count in expert_counts[:10]:
            print(f"  {expert:30} {count:6}")
        
        if counts['by_type']:
            print("\nCounts by Type:")
            for block_type, count in sorted(counts['by_type'].items()):
                print(f"  {block_type:20} {count:6}")


def main():
    """CLI for manifest validation."""
    if len(sys.argv) < 3:
        print("Usage: python -m backend.storage.manifest validate <path>")
        sys.exit(1)
    
    command = sys.argv[1]
    path = Path(sys.argv[2])
    
    if command == "validate":
        try:
            manifest = Manifest.load(path)
            is_valid, errors = manifest.validate()
            
            manifest.print_summary()
            
            if is_valid:
                print("\n✅ Manifest is valid")
            else:
                print(f"\n❌ Manifest has {len(errors)} errors:")
                for error in errors[:10]:
                    print(f"  - {error}")
                sys.exit(1)
            
        except Exception as e:
            print(f"Error loading manifest: {e}")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()