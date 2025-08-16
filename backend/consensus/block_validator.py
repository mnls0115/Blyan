"""Block validation and verification logic"""
import time
from typing import Optional, List, Dict, Any
from backend.crypto import verify_block_signature, hash_block, merkle_root


class BlockValidator:
    """Validates blocks according to consensus rules"""
    
    def __init__(self, chain_state):
        self.chain_state = chain_state
        self.max_block_time_drift = 600  # 10 minutes
        self.min_block_time = 1  # 1 second minimum between blocks
        self.max_transactions_per_block = 10000
        
    def verify_block(self, block: dict, parent: Optional[dict] = None) -> tuple[bool, str]:
        """
        Comprehensive block verification
        
        Args:
            block: Block to verify
            parent: Parent block (if None, fetched from chain)
            
        Returns:
            (is_valid, error_message)
        """
        # 1. Check block structure
        required_fields = ['height', 'parent_hash', 'timestamp', 'miner', 'signature']
        for field in required_fields:
            if field not in block:
                return False, f"Missing required field: {field}"
        
        # 2. Verify signature
        if not verify_block_signature(block):
            return False, "Invalid block signature"
        
        # 3. Check timestamp
        current_time = time.time()
        if block['timestamp'] > current_time + self.max_block_time_drift:
            return False, "Block timestamp too far in future"
        
        # 4. Get parent block if not provided
        if parent is None and block['height'] > 0:
            parent = self.chain_state.get_block_by_hash(block['parent_hash'])
            if parent is None:
                return False, "Parent block not found"
        
        # 5. Verify parent relationship
        if block['height'] > 0:
            if parent['hash'] != block['parent_hash']:
                return False, "Parent hash mismatch"
            
            if parent['height'] != block['height'] - 1:
                return False, "Invalid block height"
            
            if block['timestamp'] < parent['timestamp'] + self.min_block_time:
                return False, "Block timestamp too close to parent"
        
        # 6. Verify transactions
        if 'transactions' in block:
            if len(block['transactions']) > self.max_transactions_per_block:
                return False, f"Too many transactions (max {self.max_transactions_per_block})"
            
            # Verify each transaction
            for tx in block['transactions']:
                valid, err = self.verify_transaction(tx)
                if not valid:
                    return False, f"Invalid transaction: {err}"
        
        # 7. Verify Merkle root if present
        if 'merkle_root' in block and 'transactions' in block:
            tx_hashes = [tx.get('hash', '') for tx in block['transactions']]
            expected_root = merkle_root(tx_hashes)
            if block['merkle_root'] != expected_root:
                return False, "Invalid Merkle root"
        
        # 8. Verify state transition (placeholder for state validation)
        if 'state_root' in block:
            valid, err = self.verify_state_transition(block, parent)
            if not valid:
                return False, f"Invalid state transition: {err}"
        
        # 9. Verify block hash
        calculated_hash = hash_block(block)
        if 'hash' in block and block['hash'] != calculated_hash:
            return False, "Block hash mismatch"
        
        return True, ""
    
    def verify_transaction(self, tx: dict) -> tuple[bool, str]:
        """
        Verify a single transaction
        
        Args:
            tx: Transaction to verify
            
        Returns:
            (is_valid, error_message)
        """
        # Basic transaction structure check
        required_fields = ['from', 'to', 'value', 'nonce', 'signature']
        for field in required_fields:
            if field not in tx:
                return False, f"Missing field: {field}"
        
        # Verify signature
        from backend.crypto import verify_signature
        if not verify_signature(tx):
            return False, "Invalid transaction signature"
        
        # Check value is non-negative
        if tx['value'] < 0:
            return False, "Negative value"
        
        # Check nonce is sequential (would need account state)
        # This is a placeholder - real implementation needs state lookup
        
        return True, ""
    
    def verify_state_transition(self, block: dict, parent: Optional[dict]) -> tuple[bool, str]:
        """
        Verify state transition is valid
        
        Args:
            block: New block
            parent: Parent block
            
        Returns:
            (is_valid, error_message)
        """
        # Placeholder for state validation
        # Real implementation would:
        # 1. Apply transactions to parent state
        # 2. Calculate new state root
        # 3. Compare with block's state_root
        
        # For now, just check state_root exists and is valid hex
        if 'state_root' in block:
            try:
                bytes.fromhex(block['state_root'])
                if len(block['state_root']) != 64:  # SHA256 is 32 bytes = 64 hex chars
                    return False, "Invalid state root length"
            except ValueError:
                return False, "Invalid state root format"
        
        return True, ""
    
    def validate_block_sequence(self, blocks: List[dict]) -> tuple[bool, str]:
        """
        Validate a sequence of blocks
        
        Args:
            blocks: List of blocks in order
            
        Returns:
            (is_valid, error_message)
        """
        if not blocks:
            return True, ""
        
        # Verify first block
        valid, err = self.verify_block(blocks[0])
        if not valid:
            return False, f"Block 0: {err}"
        
        # Verify chain
        for i in range(1, len(blocks)):
            valid, err = self.verify_block(blocks[i], blocks[i-1])
            if not valid:
                return False, f"Block {i}: {err}"
        
        return True, ""


class ForkDetector:
    """Detects and handles blockchain forks"""
    
    def __init__(self, chain_state):
        self.chain_state = chain_state
        self.known_forks: Dict[str, List[dict]] = {}
        
    def detect_fork(self, new_block: dict) -> Optional[str]:
        """
        Detect if new block creates a fork
        
        Args:
            new_block: Incoming block
            
        Returns:
            Fork ID if fork detected, None otherwise
        """
        # Check if parent exists
        parent = self.chain_state.get_block_by_hash(new_block['parent_hash'])
        if not parent:
            return None
        
        # Check if we already have a different block at this height
        existing = self.chain_state.get_block_by_height(new_block['height'])
        if existing and existing['hash'] != new_block.get('hash'):
            # Fork detected!
            fork_id = f"fork_{new_block['height']}_{time.time()}"
            
            # Track both branches
            if fork_id not in self.known_forks:
                self.known_forks[fork_id] = []
            
            self.known_forks[fork_id].append(existing)
            self.known_forks[fork_id].append(new_block)
            
            return fork_id
        
        return None
    
    def resolve_fork(self, fork_id: str) -> dict:
        """
        Resolve a fork using longest chain rule
        
        Args:
            fork_id: Fork identifier
            
        Returns:
            Winning branch head block
        """
        if fork_id not in self.known_forks:
            raise ValueError(f"Unknown fork: {fork_id}")
        
        branches = self.known_forks[fork_id]
        
        # Find longest branch (most work)
        longest = None
        max_height = -1
        
        for block in branches:
            chain_height = self.get_chain_height(block)
            if chain_height > max_height:
                max_height = chain_height
                longest = block
        
        return longest
    
    def get_chain_height(self, head_block: dict) -> int:
        """
        Get total height of chain ending at block
        
        Args:
            head_block: Chain head
            
        Returns:
            Total chain height
        """
        # For now, just return block height
        # Real implementation would calculate cumulative difficulty
        return head_block['height']