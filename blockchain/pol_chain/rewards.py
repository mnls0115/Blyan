"""
PoL chain reward management and receipt generation
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import hashlib
import time
import json

@dataclass
class RewardReceipt:
    proof_id: bytes
    earner: str
    amount: int
    epoch: int
    meta_hash: bytes
    timestamp: int
    inference_type: str  # "expert", "router", "full"
    quality_score: float

class RewardManager:
    """Manages reward generation and epoching on PoL chain"""
    
    def __init__(self, epoch_duration: int = 300):  # 5 minute epochs
        self.epoch_duration = epoch_duration
        self.current_epoch = 0
        self.epoch_start_time = int(time.time())
        self.pending_receipts: List[RewardReceipt] = []
        self.finalized_epochs: Dict[int, Dict] = {}
        self.epoch_roots: Dict[int, bytes] = {}
        
    def issue_receipt(self, earner: str, amount: int, meta_hash: bytes,
                     inference_type: str, quality_score: float) -> RewardReceipt:
        """
        Issue a new reward receipt
        
        Args:
            earner: Address to receive reward
            amount: BLY tokens earned
            meta_hash: Reference to model/inference metadata
            inference_type: Type of contribution
            quality_score: Quality metric (0-1)
        """
        # Generate unique proof ID
        proof_data = f"{earner}{amount}{time.time()}{meta_hash.hex()}"
        proof_id = hashlib.sha256(proof_data.encode()).digest()
        
        receipt = RewardReceipt(
            proof_id=proof_id,
            earner=earner,
            amount=amount,
            epoch=self.current_epoch,
            meta_hash=meta_hash,
            timestamp=int(time.time()),
            inference_type=inference_type,
            quality_score=quality_score
        )
        
        self.pending_receipts.append(receipt)
        return receipt
    
    def check_epoch_transition(self) -> bool:
        """Check if we should transition to next epoch"""
        current_time = int(time.time())
        if current_time - self.epoch_start_time >= self.epoch_duration:
            self.finalize_epoch()
            return True
        return False
    
    def finalize_epoch(self):
        """Finalize current epoch and prepare for next"""
        if not self.pending_receipts:
            # Empty epoch
            self.epoch_roots[self.current_epoch] = hashlib.sha256(b'empty').digest()
        else:
            # Build Merkle tree
            root, proofs = self.build_merkle_tree(self.pending_receipts)
            
            # Store finalized epoch data
            self.finalized_epochs[self.current_epoch] = {
                'receipts': self.pending_receipts[:],
                'proofs': proofs,
                'root': root,
                'timestamp': int(time.time()),
                'receipt_count': len(self.pending_receipts),
                'total_rewards': sum(r.amount for r in self.pending_receipts)
            }
            
            self.epoch_roots[self.current_epoch] = root
        
        # Reset for next epoch
        self.pending_receipts = []
        self.current_epoch += 1
        self.epoch_start_time = int(time.time())
    
    def build_merkle_tree(self, receipts: List[RewardReceipt]) -> tuple[bytes, Dict]:
        """Build Merkle tree from receipts"""
        if not receipts:
            return hashlib.sha256(b'').digest(), {}
        
        # Create leaves
        leaves = []
        for receipt in receipts:
            leaf_data = json.dumps({
                'proof_id': receipt.proof_id.hex(),
                'earner': receipt.earner,
                'amount': receipt.amount,
                'epoch': receipt.epoch,
                'meta_hash': receipt.meta_hash.hex()
            }, sort_keys=True)
            leaves.append(hashlib.sha256(leaf_data.encode()).digest())
        
        # Build tree
        proofs = {}
        current_level = leaves[:]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    left = current_level[i]
                    right = current_level[i + 1]
                    parent = hashlib.sha256(left + right).digest()
                else:
                    parent = current_level[i]
                next_level.append(parent)
            current_level = next_level
        
        return current_level[0], proofs
    
    def get_epoch_root(self, epoch: int) -> Optional[bytes]:
        """Get Merkle root for finalized epoch"""
        return self.epoch_roots.get(epoch)
    
    def is_epoch_finalized(self, epoch: int) -> bool:
        """Check if epoch is finalized"""
        return epoch in self.finalized_epochs
    
    def get_receipt_proof(self, proof_id: bytes, epoch: int) -> Optional[List]:
        """Get Merkle proof for a receipt"""
        if epoch not in self.finalized_epochs:
            return None
        
        epoch_data = self.finalized_epochs[epoch]
        receipts = epoch_data['receipts']
        
        # Find receipt
        receipt_idx = None
        for i, r in enumerate(receipts):
            if r.proof_id == proof_id:
                receipt_idx = i
                break
        
        if receipt_idx is None:
            return None
        
        # Generate proof
        return self._generate_proof(receipts, receipt_idx)
    
    def _generate_proof(self, receipts: List[RewardReceipt], index: int) -> List:
        """Generate Merkle proof for receipt at index"""
        if not receipts:
            return []
        
        # Create leaves
        leaves = []
        for receipt in receipts:
            leaf_data = json.dumps({
                'proof_id': receipt.proof_id.hex(),
                'earner': receipt.earner,
                'amount': receipt.amount,
                'epoch': receipt.epoch,
                'meta_hash': receipt.meta_hash.hex()
            }, sort_keys=True)
            leaves.append(hashlib.sha256(leaf_data.encode()).digest())
        
        proof = []
        current_level = leaves[:]
        current_index = index
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    left = current_level[i]
                    right = current_level[i + 1]
                    
                    # Add to proof if needed
                    if i == current_index or i + 1 == current_index:
                        if i == current_index:
                            proof.append((right, False))  # Right sibling
                        else:
                            proof.append((left, True))  # Left sibling
                        current_index = i // 2
                    
                    parent = hashlib.sha256(left + right).digest()
                else:
                    if i == current_index:
                        current_index = i // 2
                    parent = current_level[i]
                
                next_level.append(parent)
            
            current_level = next_level
        
        return proof

class AnchoringTx:
    """Anchor transaction chain headers to PoL chain"""
    
    def __init__(self):
        self.anchored_headers: Dict[int, Dict] = {}
        self.latest_height = 0
        
    def commit(self, header: Dict, signatures: List[bytes], finality_proof: bytes) -> bool:
        """
        Commit tx-chain header to PoL chain
        
        Args:
            header: Transaction chain block header
            signatures: Validator signatures
            finality_proof: Proof of finalization
        """
        height = header['height']
        
        # Verify height sequence
        if height <= self.latest_height:
            raise ValueError(f"Height {height} already anchored or out of sequence")
        
        # Verify signatures (simplified)
        if len(signatures) < 67:  # Need 2f+1
            raise ValueError("Insufficient signatures")
        
        # Store header
        self.anchored_headers[height] = {
            'header': header,
            'signatures': signatures,
            'finality_proof': finality_proof,
            'timestamp': int(time.time())
        }
        
        self.latest_height = height
        return True
    
    def get_anchored_header(self, height: int) -> Optional[Dict]:
        """Get anchored header by height"""
        return self.anchored_headers.get(height)