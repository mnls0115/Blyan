"""Chain management with fork handling and finality"""
import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading
from backend.crypto import hash_block


@dataclass 
class ChainTip:
    """Represents a chain tip (potential fork)"""
    head_hash: str
    height: int
    total_difficulty: int
    blocks: List[dict]  # Recent blocks in this chain


class ChainManager:
    """Manages blockchain state with fork handling"""
    
    def __init__(self, genesis_block: dict = None):
        """
        Initialize chain manager
        
        Args:
            genesis_block: Genesis block (created if None)
        """
        self.blocks: Dict[str, dict] = {}  # hash -> block
        self.height_index: Dict[int, Set[str]] = defaultdict(set)  # height -> hashes
        self.children: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.tips: Dict[str, ChainTip] = {}  # Active chain tips
        self.main_chain: List[str] = []  # Main chain block hashes
        self.finalized_height = 0
        self.lock = threading.RLock()
        
        # Consensus parameters
        self.max_reorg_depth = 100  # Max blocks to reorg
        self.finality_depth = 10  # Blocks until finality
        
        # Initialize with genesis
        if genesis_block is None:
            genesis_block = self.create_genesis_block()
        
        self.add_block(genesis_block)
        self.main_chain = [genesis_block['hash']]
    
    def create_genesis_block(self) -> dict:
        """Create genesis block"""
        genesis = {
            'height': 0,
            'parent_hash': '0' * 64,
            'timestamp': int(time.time()),
            'transactions': [],
            'miner': '0' * 40,
            'difficulty': 1,
            'nonce': 0
        }
        genesis['hash'] = hash_block(genesis)
        return genesis
    
    def add_block(self, block: dict) -> Tuple[bool, str]:
        """
        Add block to chain
        
        Args:
            block: Block to add
            
        Returns:
            (success, message)
        """
        with self.lock:
            # Calculate hash if not present
            if 'hash' not in block:
                block['hash'] = hash_block(block)
            
            block_hash = block['hash']
            
            # Check if already have it
            if block_hash in self.blocks:
                return True, "Already have block"
            
            # Verify parent exists (except genesis)
            if block['height'] > 0:
                if block['parent_hash'] not in self.blocks:
                    return False, "Parent not found"
            
            # Store block
            self.blocks[block_hash] = block
            self.height_index[block['height']].add(block_hash)
            self.children[block['parent_hash']].add(block_hash)
            
            # Update chain tips
            self._update_tips(block)
            
            # Check for reorg
            if self._should_reorg(block):
                self._perform_reorg(block)
            
            # Update finality
            self._update_finality()
            
            return True, "Block added"
    
    def _update_tips(self, block: dict):
        """Update chain tips with new block"""
        block_hash = block['hash']
        parent_hash = block['parent_hash']
        
        # Remove parent from tips if it was a tip
        if parent_hash in self.tips:
            parent_tip = self.tips[parent_hash]
            del self.tips[parent_hash]
            
            # Create new tip extending parent
            new_tip = ChainTip(
                head_hash=block_hash,
                height=block['height'],
                total_difficulty=parent_tip.total_difficulty + block.get('difficulty', 1),
                blocks=parent_tip.blocks[-99:] + [block]  # Keep last 100 blocks
            )
            self.tips[block_hash] = new_tip
        else:
            # New fork or extending existing chain
            # Calculate total difficulty
            total_diff = self._calculate_total_difficulty(block)
            
            new_tip = ChainTip(
                head_hash=block_hash,
                height=block['height'],
                total_difficulty=total_diff,
                blocks=[block]
            )
            self.tips[block_hash] = new_tip
    
    def _calculate_total_difficulty(self, block: dict) -> int:
        """Calculate total difficulty up to block"""
        total = 0
        current = block
        
        while current and current['height'] > 0:
            total += current.get('difficulty', 1)
            parent_hash = current['parent_hash']
            current = self.blocks.get(parent_hash)
        
        return total
    
    def _should_reorg(self, block: dict) -> bool:
        """Check if new block should trigger reorg"""
        # Don't reorg if block extends main chain
        if self.main_chain and block['parent_hash'] == self.main_chain[-1]:
            return False
        
        # Check if new chain is heavier
        block_hash = block['hash']
        if block_hash not in self.tips:
            return False
        
        new_tip = self.tips[block_hash]
        main_tip_hash = self.main_chain[-1] if self.main_chain else None
        
        if main_tip_hash and main_tip_hash in self.tips:
            main_tip = self.tips[main_tip_hash]
            return new_tip.total_difficulty > main_tip.total_difficulty
        
        return True
    
    def _perform_reorg(self, new_head: dict):
        """Perform chain reorganization"""
        print(f"⚠️ Reorg to block {new_head['hash'][:8]} at height {new_head['height']}")
        
        # Find common ancestor
        new_chain = self._get_chain_to_genesis(new_head)
        old_chain = self.main_chain.copy()
        
        # Find fork point
        fork_point = 0
        for i, (new_hash, old_hash) in enumerate(zip(new_chain, old_chain)):
            if new_hash == old_hash:
                fork_point = i
            else:
                break
        
        # Update main chain
        self.main_chain = new_chain
        
        # Notify about reorg depth
        reorg_depth = len(old_chain) - fork_point
        if reorg_depth > 0:
            print(f"  Reorg depth: {reorg_depth} blocks")
    
    def _get_chain_to_genesis(self, block: dict) -> List[str]:
        """Get chain from block to genesis"""
        chain = []
        current = block
        
        while current:
            chain.append(current['hash'])
            if current['height'] == 0:
                break
            current = self.blocks.get(current['parent_hash'])
        
        chain.reverse()
        return chain
    
    def _update_finality(self):
        """Update finalized height based on depth"""
        if len(self.main_chain) > self.finality_depth:
            self.finalized_height = len(self.main_chain) - self.finality_depth
    
    def get_block(self, block_hash: str) -> Optional[dict]:
        """Get block by hash"""
        return self.blocks.get(block_hash)
    
    def get_block_by_height(self, height: int) -> Optional[dict]:
        """Get block at height on main chain"""
        if height < len(self.main_chain):
            return self.blocks.get(self.main_chain[height])
        return None
    
    def get_latest_block(self) -> Optional[dict]:
        """Get latest block on main chain"""
        if self.main_chain:
            return self.blocks.get(self.main_chain[-1])
        return None
    
    def is_finalized(self, block_hash: str) -> bool:
        """Check if block is finalized"""
        block = self.blocks.get(block_hash)
        if not block:
            return False
        
        return block['height'] <= self.finalized_height
    
    def get_fork_chains(self) -> List[ChainTip]:
        """Get all fork chains"""
        forks = []
        
        for tip_hash, tip in self.tips.items():
            if tip_hash != self.main_chain[-1]:
                forks.append(tip)
        
        return forks
    
    def prune_old_forks(self):
        """Remove old abandoned forks"""
        with self.lock:
            current_height = len(self.main_chain) - 1
            
            # Remove tips too far behind
            old_tips = []
            for tip_hash, tip in self.tips.items():
                if tip.height < current_height - self.max_reorg_depth:
                    old_tips.append(tip_hash)
            
            for tip_hash in old_tips:
                del self.tips[tip_hash]
                
                # Could also remove blocks, but keep for now
    
    def validate_chain(self) -> Tuple[bool, str]:
        """Validate entire main chain"""
        if not self.main_chain:
            return False, "Empty chain"
        
        prev_block = None
        for block_hash in self.main_chain:
            block = self.blocks.get(block_hash)
            if not block:
                return False, f"Missing block {block_hash}"
            
            if prev_block:
                if block['parent_hash'] != prev_block['hash']:
                    return False, f"Broken chain at {block['height']}"
                
                if block['height'] != prev_block['height'] + 1:
                    return False, f"Invalid height at {block['height']}"
            
            prev_block = block
        
        return True, "Chain valid"


class FinalityGadget:
    """Provides finality for blocks (simplified BFT)"""
    
    def __init__(self, chain_manager: ChainManager, validator_set: Set[str]):
        """
        Initialize finality gadget
        
        Args:
            chain_manager: Chain manager instance
            validator_set: Set of validator addresses
        """
        self.chain = chain_manager
        self.validators = validator_set
        self.votes: Dict[str, Set[str]] = defaultdict(set)  # block_hash -> voters
        self.finalized_blocks: Set[str] = set()
        
    def vote(self, block_hash: str, validator: str) -> bool:
        """
        Cast finality vote
        
        Args:
            block_hash: Block to vote for
            validator: Validator address
            
        Returns:
            True if block becomes final
        """
        if validator not in self.validators:
            return False
        
        self.votes[block_hash].add(validator)
        
        # Check if we have supermajority (2/3+)
        if len(self.votes[block_hash]) > len(self.validators) * 2 / 3:
            self.finalize_block(block_hash)
            return True
        
        return False
    
    def finalize_block(self, block_hash: str):
        """Mark block as final"""
        self.finalized_blocks.add(block_hash)
        
        # Also finalize all ancestors
        block = self.chain.get_block(block_hash)
        while block and block['height'] > 0:
            parent_hash = block['parent_hash']
            if parent_hash not in self.finalized_blocks:
                self.finalized_blocks.add(parent_hash)
            block = self.chain.get_block(parent_hash)
    
    def is_finalized(self, block_hash: str) -> bool:
        """Check if block is finalized"""
        return block_hash in self.finalized_blocks