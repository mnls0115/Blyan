"""
Tendermint-style BFT consensus for transaction chain
"""
import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum

class VoteType(Enum):
    PREVOTE = 1
    PRECOMMIT = 2

@dataclass
class Block:
    height: int
    parent_hash: bytes
    timestamp: int
    transactions: List[bytes]
    proposer: str
    state_root: bytes
    
    def hash(self) -> bytes:
        data = f"{self.height}{self.parent_hash.hex()}{self.timestamp}{self.proposer}"
        return hashlib.sha256(data.encode()).digest()

@dataclass
class Vote:
    vote_type: VoteType
    height: int
    round: int
    block_hash: Optional[bytes]
    validator: str
    signature: bytes
    timestamp: int

class ConsensusState:
    def __init__(self, validator_id: str, validators: List[str], stake_weights: Dict[str, int]):
        self.validator_id = validator_id
        self.validators = validators
        self.stake_weights = stake_weights
        self.height = 0
        self.round = 0
        self.step = "propose"
        self.locked_block: Optional[Block] = None
        self.locked_round = -1
        self.valid_block: Optional[Block] = None
        self.valid_round = -1
        self.votes: Dict[str, List[Vote]] = {"prevote": [], "precommit": []}
        
    def total_stake(self) -> int:
        return sum(self.stake_weights.values())
    
    def vote_power(self, votes: List[Vote]) -> int:
        validators = set(v.validator for v in votes)
        return sum(self.stake_weights.get(v, 0) for v in validators)
    
    def has_2f_plus_1(self, votes: List[Vote]) -> bool:
        power = self.vote_power(votes)
        return power > (2 * self.total_stake()) // 3

class BFTConsensus:
    def __init__(self, state: ConsensusState, network, storage):
        self.state = state
        self.network = network
        self.storage = storage
        self.block_time = 1.0  # 1 second blocks
        self.timeout_propose = 3.0
        self.timeout_prevote = 1.0
        self.timeout_precommit = 1.0
        
    async def run(self):
        """Main consensus loop"""
        while True:
            await self.new_round(self.state.height, 0)
            
    async def new_round(self, height: int, round: int):
        """Start new consensus round"""
        self.state.height = height
        self.state.round = round
        self.state.step = "propose"
        
        proposer = self.select_proposer(height, round)
        
        if proposer == self.state.validator_id:
            block = await self.create_block()
            await self.broadcast_proposal(block)
        
        # Wait for proposal
        block = await self.wait_for_proposal(proposer)
        
        # Prevote
        await self.prevote(block)
        
        # Wait for prevotes
        if await self.wait_for_prevotes():
            # Precommit
            await self.precommit()
            
            # Wait for precommits
            if await self.wait_for_precommits():
                await self.commit_block()
                self.state.height += 1
                return
        
        # Round failed, try next round
        await self.new_round(height, round + 1)
    
    def select_proposer(self, height: int, round: int) -> str:
        """VRF-based proposer selection"""
        seed = hashlib.sha256(f"{height}{round}".encode()).digest()
        vrf_output = int.from_bytes(seed[:8], 'big')
        
        cumulative = 0
        total = self.state.total_stake()
        threshold = vrf_output % total
        
        for validator in self.state.validators:
            cumulative += self.state.stake_weights[validator]
            if cumulative > threshold:
                return validator
        
        return self.state.validators[-1]
    
    async def create_block(self) -> Block:
        """Create new block proposal"""
        parent = await self.storage.get_block(self.state.height - 1)
        txs = await self.get_pending_transactions()
        
        return Block(
            height=self.state.height,
            parent_hash=parent.hash() if parent else b'',
            timestamp=int(time.time()),
            transactions=txs[:1000],  # Max 1000 txs
            proposer=self.state.validator_id,
            state_root=await self.compute_state_root(txs)
        )
    
    async def prevote(self, block: Optional[Block]):
        """Send prevote"""
        vote_block = None
        
        if self.state.locked_block:
            vote_block = self.state.locked_block
        elif block and await self.validate_block(block):
            vote_block = block
        
        vote = Vote(
            vote_type=VoteType.PREVOTE,
            height=self.state.height,
            round=self.state.round,
            block_hash=vote_block.hash() if vote_block else None,
            validator=self.state.validator_id,
            signature=b'',  # Sign with validator key
            timestamp=int(time.time())
        )
        
        await self.network.broadcast_vote(vote)
        self.state.votes["prevote"].append(vote)
    
    async def precommit(self):
        """Send precommit"""
        # Check if we have 2f+1 prevotes for a block
        prevotes_by_block = {}
        for vote in self.state.votes["prevote"]:
            if vote.block_hash:
                key = vote.block_hash.hex()
                if key not in prevotes_by_block:
                    prevotes_by_block[key] = []
                prevotes_by_block[key].append(vote)
        
        commit_block = None
        for block_hash_hex, votes in prevotes_by_block.items():
            if self.state.has_2f_plus_1(votes):
                commit_block = bytes.fromhex(block_hash_hex)
                break
        
        vote = Vote(
            vote_type=VoteType.PRECOMMIT,
            height=self.state.height,
            round=self.state.round,
            block_hash=commit_block,
            validator=self.state.validator_id,
            signature=b'',
            timestamp=int(time.time())
        )
        
        await self.network.broadcast_vote(vote)
        self.state.votes["precommit"].append(vote)
    
    async def commit_block(self):
        """Commit finalized block"""
        # Find block with 2f+1 precommits
        precommits_by_block = {}
        for vote in self.state.votes["precommit"]:
            if vote.block_hash:
                key = vote.block_hash.hex()
                if key not in precommits_by_block:
                    precommits_by_block[key] = []
                precommits_by_block[key].append(vote)
        
        for block_hash_hex, votes in precommits_by_block.items():
            if self.state.has_2f_plus_1(votes):
                block = await self.storage.get_block_by_hash(bytes.fromhex(block_hash_hex))
                if block:
                    await self.storage.commit_block(block)
                    await self.execute_transactions(block.transactions)
                    print(f"✅ Committed block {block.height} with finality")
                    return
    
    async def validate_block(self, block: Block) -> bool:
        """Validate block proposal"""
        # Check parent
        if block.height > 0:
            parent = await self.storage.get_block(block.height - 1)
            if not parent or parent.hash() != block.parent_hash:
                return False
        
        # Validate transactions
        for tx in block.transactions:
            if not await self.validate_transaction(tx):
                return False
        
        return True
    
    async def validate_transaction(self, tx: bytes) -> bool:
        """Validate individual transaction"""
        # Implement transaction validation
        return True
    
    async def execute_transactions(self, txs: List[bytes]):
        """Execute committed transactions"""
        for tx in txs:
            await self.storage.apply_transaction(tx)
    
    async def get_pending_transactions(self) -> List[bytes]:
        """Get transactions from mempool"""
        return await self.network.get_mempool_transactions()
    
    async def compute_state_root(self, txs: List[bytes]) -> bytes:
        """Compute new state root after transactions"""
        # Simplified - real implementation would use Merkle Patricia Trie
        data = b''.join(txs)
        return hashlib.sha256(data).digest()
    
    async def wait_for_proposal(self, proposer: str) -> Optional[Block]:
        """Wait for block proposal with timeout"""
        try:
            return await asyncio.wait_for(
                self.network.receive_proposal(proposer),
                timeout=self.timeout_propose
            )
        except asyncio.TimeoutError:
            return None
    
    async def wait_for_prevotes(self) -> bool:
        """Wait for 2f+1 prevotes"""
        try:
            await asyncio.wait_for(
                self._wait_for_votes("prevote"),
                timeout=self.timeout_prevote
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def wait_for_precommits(self) -> bool:
        """Wait for 2f+1 precommits"""
        try:
            await asyncio.wait_for(
                self._wait_for_votes("precommit"),
                timeout=self.timeout_precommit
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    async def _wait_for_votes(self, vote_type: str):
        """Wait until we have 2f+1 votes"""
        while not self.state.has_2f_plus_1(self.state.votes[vote_type]):
            vote = await self.network.receive_vote(vote_type)
            if vote:
                self.state.votes[vote_type].append(vote)
    
    async def broadcast_proposal(self, block: Block):
        """Broadcast block proposal to network"""
        await self.network.broadcast_block(block)