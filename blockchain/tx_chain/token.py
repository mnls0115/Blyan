"""
BLY Token contract with claim and transfer functionality
"""
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import json

@dataclass
class RewardReceipt:
    proof_id: bytes
    earner: str
    amount: int
    epoch: int
    meta_hash: bytes

class MerkleTree:
    """Merkle tree for receipt verification"""
    
    @staticmethod
    def hash_receipt(receipt: RewardReceipt) -> bytes:
        data = f"{receipt.proof_id.hex()}{receipt.earner}{receipt.amount}{receipt.epoch}"
        return hashlib.sha256(data.encode()).digest()
    
    @staticmethod
    def hash_pair(left: bytes, right: bytes) -> bytes:
        return hashlib.sha256(left + right).digest()
    
    @staticmethod
    def verify_proof(leaf: bytes, proof: List[Tuple[bytes, bool]], root: bytes) -> bool:
        """Verify Merkle proof. Each proof element is (hash, is_left)"""
        current = leaf
        for sibling, is_left in proof:
            if is_left:
                current = MerkleTree.hash_pair(sibling, current)
            else:
                current = MerkleTree.hash_pair(current, sibling)
        return current == root
    
    @staticmethod
    def build_tree(receipts: List[RewardReceipt]) -> Tuple[bytes, Dict[bytes, List[Tuple[bytes, bool]]]]:
        """Build tree and return (root, proofs_map)"""
        if not receipts:
            return hashlib.sha256(b'').digest(), {}
        
        # Hash all receipts
        leaves = [MerkleTree.hash_receipt(r) for r in receipts]
        proofs = {r.proof_id: [] for r in receipts}
        
        # Build tree level by level
        current_level = leaves[:]
        leaf_indices = {r.proof_id: i for i, r in enumerate(receipts)}
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    left = current_level[i]
                    right = current_level[i + 1]
                    parent = MerkleTree.hash_pair(left, right)
                    
                    # Update proofs
                    for proof_id, idx in leaf_indices.items():
                        if idx // (2 ** len(proofs[proof_id])) == i // 2:
                            if idx % 2 == 0:
                                proofs[proof_id].append((right, False))
                            else:
                                proofs[proof_id].append((left, True))
                else:
                    # Odd number of nodes, promote last one
                    parent = current_level[i]
                
                next_level.append(parent)
            
            current_level = next_level
        
        return current_level[0], proofs

class BLYToken:
    """BLY Token with claim and transfer"""
    
    def __init__(self, anchoring_pol):
        self.balances: Dict[str, int] = {}
        self.claimed: Dict[bytes, bool] = {}
        self.total_supply = 0
        self.anchoring_pol = anchoring_pol
        self.nonces: Dict[str, int] = {}
        
    def claim(self, receipt: RewardReceipt, merkle_proof: List[Tuple[bytes, bool]], 
              epoch_root: bytes, sender: str) -> bool:
        """
        Claim tokens from PoL receipt
        
        Args:
            receipt: The reward receipt
            merkle_proof: Merkle proof path
            epoch_root: Root hash for the epoch
            sender: Transaction sender (must match earner)
        """
        # Check sender is earner
        if sender != receipt.earner:
            raise ValueError("Only earner can claim")
        
        # Check not already claimed
        if receipt.proof_id in self.claimed:
            raise ValueError("Already claimed")
        
        # Verify epoch is anchored
        if not self.anchoring_pol.is_epoch_anchored(receipt.epoch):
            raise ValueError("Epoch not anchored")
        
        # Verify epoch root matches
        anchored_root = self.anchoring_pol.get_epoch_root(receipt.epoch)
        if anchored_root != epoch_root:
            raise ValueError("Invalid epoch root")
        
        # Verify Merkle proof
        leaf = MerkleTree.hash_receipt(receipt)
        if not MerkleTree.verify_proof(leaf, merkle_proof, epoch_root):
            raise ValueError("Invalid Merkle proof")
        
        # Mark as claimed
        self.claimed[receipt.proof_id] = True
        
        # Mint tokens
        self._mint(receipt.earner, receipt.amount)
        
        return True
    
    def transfer(self, sender: str, recipient: str, amount: int, nonce: int) -> bool:
        """
        Transfer tokens between accounts
        
        Args:
            sender: Sender address
            recipient: Recipient address  
            amount: Amount to transfer
            nonce: Transaction nonce
        """
        # Check nonce
        expected_nonce = self.nonces.get(sender, 0)
        if nonce != expected_nonce:
            raise ValueError(f"Invalid nonce: expected {expected_nonce}, got {nonce}")
        
        # Check balance
        if self.balances.get(sender, 0) < amount:
            raise ValueError("Insufficient balance")
        
        # Update nonce
        self.nonces[sender] = nonce + 1
        
        # Transfer
        self.balances[sender] -= amount
        self.balances[recipient] = self.balances.get(recipient, 0) + amount
        
        return True
    
    def _mint(self, address: str, amount: int):
        """Internal mint function"""
        self.balances[address] = self.balances.get(address, 0) + amount
        self.total_supply += amount
    
    def balance_of(self, address: str) -> int:
        """Get account balance"""
        return self.balances.get(address, 0)
    
    def get_nonce(self, address: str) -> int:
        """Get account nonce"""
        return self.nonces.get(address, 0)

class AnchoringPoL:
    """Anchoring contract for PoL epoch roots"""
    
    def __init__(self):
        self.epoch_roots: Dict[int, bytes] = {}
        self.finalized_epochs: Dict[int, bool] = {}
        self.last_anchored_epoch = 0
        
    def commit(self, epoch: int, root: bytes, signatures: List[bytes], 
               finality_proof: bytes) -> bool:
        """
        Commit PoL epoch root to tx-chain
        
        Args:
            epoch: Epoch number
            root: Merkle root of receipts
            signatures: Validator signatures (2f+1 required)
            finality_proof: Proof of epoch finalization
        """
        # Check epoch is next in sequence
        if epoch != self.last_anchored_epoch + 1:
            raise ValueError(f"Invalid epoch sequence: expected {self.last_anchored_epoch + 1}")
        
        # Verify signatures (simplified - real impl would check validator set)
        if len(signatures) < 67:  # Assuming 100 validators, need 67
            raise ValueError("Insufficient signatures")
        
        # Store epoch root
        self.epoch_roots[epoch] = root
        self.finalized_epochs[epoch] = True
        self.last_anchored_epoch = epoch
        
        return True
    
    def is_epoch_anchored(self, epoch: int) -> bool:
        """Check if epoch is anchored"""
        return epoch in self.finalized_epochs
    
    def get_epoch_root(self, epoch: int) -> Optional[bytes]:
        """Get anchored epoch root"""
        return self.epoch_roots.get(epoch)