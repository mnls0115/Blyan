"""
Production database layer for learning cycle persistence
Uses PostgreSQL for reliable state management
"""

import os
import json
import uuid
import asyncio
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import asyncpg
import logging

logger = logging.getLogger(__name__)


class RoundState(Enum):
    """Learning round states"""
    TRIGGERED = "TRIGGERED"
    NOTIFYING = "NOTIFYING"
    DATA_ALLOC = "DATA_ALLOC"
    TRAINING = "TRAINING"
    CONSENSUS = "CONSENSUS"
    DELTA_CREATION = "DELTA_CREATION"
    REWARD_DIST = "REWARD_DIST"
    FAILED = "FAILED"


class LearningDatabase:
    """
    PostgreSQL database interface for learning cycle persistence.
    Handles all state management with ACID guarantees.
    """
    
    def __init__(self, connection_url: str = None):
        self.connection_url = connection_url or os.getenv(
            "DATABASE_URL",
            "postgresql://localhost/blyan_learning"
        )
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Create tables if not exist
            await self._create_tables()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _create_tables(self):
        """Create all required tables"""
        async with self.pool.acquire() as conn:
            # Read and execute schema
            schema_path = os.path.join(
                os.path.dirname(__file__),
                'learning_schema.sql'
            )
            
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema = f.read()
                    try:
                        await conn.execute(schema)
                    except Exception as e:
                        # If tables/triggers already exist, that's OK
                        if "already exists" not in str(e):
                            logger.warning(f"Schema execution warning: {e}")
            else:
                # Create minimal schema if file doesn't exist
                await self._create_minimal_schema(conn)
    
    async def _create_minimal_schema(self, conn):
        """Create minimal required tables"""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS burns_ledger (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                user_addr TEXT NOT NULL,
                request_id TEXT NOT NULL,
                bly_amount NUMERIC(38, 18) NOT NULL,
                round_candidate BOOLEAN DEFAULT FALSE
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_rounds (
                round_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                state TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                bly_sum NUMERIC(38,18) NOT NULL,
                min_interval_ok BOOLEAN NOT NULL,
                seed BYTEA NOT NULL,
                config JSONB NOT NULL,
                target_expert TEXT,
                base_version TEXT
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_nodes (
                node_id TEXT PRIMARY KEY,
                base_url TEXT NOT NULL,
                pubkey TEXT NOT NULL,
                last_seen TIMESTAMPTZ,
                capabilities JSONB,
                status TEXT DEFAULT 'active',
                reputation_score NUMERIC DEFAULT 1.0
            )
        """)
    
    # Burn tracking methods
    async def record_burn(self, user_addr: str, request_id: str, amount: Decimal) -> None:
        """Record a BLY burn from inference"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO burns_ledger (user_addr, request_id, bly_amount)
                VALUES ($1, $2, $3)
            """, user_addr, request_id, amount)
    
    async def get_burn_accumulator(self) -> Tuple[Decimal, Optional[datetime]]:
        """Get total accumulated burns and last round timestamp"""
        async with self.pool.acquire() as conn:
            # Get sum of non-allocated burns
            row = await conn.fetchrow("""
                SELECT 
                    COALESCE(SUM(bly_amount), 0) as total,
                    MAX(ts) as last_burn
                FROM burns_ledger
                WHERE round_candidate = FALSE
            """)
            
            total = Decimal(str(row['total']))
            
            # Get last round time
            last_round = await conn.fetchrow("""
                SELECT created_at
                FROM learning_rounds
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            last_round_time = last_round['created_at'] if last_round else None
            
            return total, last_round_time
    
    async def mark_burns_allocated(self, round_id: str) -> None:
        """Mark burns as allocated to a round"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE burns_ledger
                SET round_candidate = TRUE
                WHERE round_candidate = FALSE
            """)
    
    # Round management methods
    async def create_round(self, 
                          bly_sum: Decimal,
                          seed: bytes,
                          config: Dict,
                          target_expert: str = "layer0.expert0") -> str:
        """Create a new learning round"""
        async with self.pool.acquire() as conn:
            round_id = str(uuid.uuid4())
            
            await conn.execute("""
                INSERT INTO learning_rounds 
                (round_id, state, bly_sum, min_interval_ok, seed, config, target_expert, base_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, round_id, RoundState.TRIGGERED.value, bly_sum, True, 
                seed, json.dumps(config), target_expert, "")
            
            # Mark burns as allocated
            await self.mark_burns_allocated(round_id)
            
            logger.info(f"Created learning round {round_id}")
            return round_id
    
    async def update_round_state(self, round_id: str, state: RoundState) -> None:
        """Update round state"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE learning_rounds
                SET state = $1, updated_at = now()
                WHERE round_id = $2
            """, state.value, round_id)
            
            logger.info(f"Round {round_id} state updated to {state.value}")
    
    async def get_round(self, round_id: str) -> Optional[Dict]:
        """Get round details"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM learning_rounds
                WHERE round_id = $1
            """, round_id)
            
            if row:
                return dict(row)
            return None
    
    async def get_active_rounds(self) -> List[Dict]:
        """Get all non-finalized rounds"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM learning_rounds
                WHERE state NOT IN ('REWARD_DIST', 'FAILED')
                ORDER BY created_at DESC
            """)
            
            # Convert non-JSON-serializable types
            result = []
            for row in rows:
                round_data = dict(row)
                # Convert UUID to string
                if 'round_id' in round_data and round_data['round_id']:
                    round_data['round_id'] = str(round_data['round_id'])
                # Convert BYTEA seed to hex string
                if 'seed' in round_data and round_data['seed']:
                    round_data['seed'] = round_data['seed'].hex()
                # Convert datetime to ISO string
                for field in ['created_at', 'updated_at']:
                    if field in round_data and round_data[field]:
                        round_data[field] = round_data[field].isoformat()
                result.append(round_data)
            
            return result
    
    # Node registry methods
    async def register_node(self,
                           node_id: str,
                           base_url: str,
                           pubkey: str,
                           capabilities: Dict) -> None:
        """Register or update a GPU node"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO gpu_nodes (node_id, base_url, pubkey, capabilities, last_seen)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (node_id) DO UPDATE
                SET base_url = $2, pubkey = $3, capabilities = $4, last_seen = now()
            """, node_id, base_url, pubkey, json.dumps(capabilities))
    
    async def get_active_nodes(self) -> List[Dict]:
        """Get all active GPU nodes"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM gpu_nodes
                WHERE status = 'active'
                AND last_seen > now() - INTERVAL '5 minutes'
                ORDER BY reputation_score DESC
            """)
            
            return [dict(row) for row in rows]
    
    # Data assignment methods
    async def save_assignments(self, 
                              round_id: str,
                              assignments: Dict[str, List[str]]) -> None:
        """Save data shard assignments"""
        async with self.pool.acquire() as conn:
            for node_id, shards in assignments.items():
                for shard_id in shards:
                    # Generate shard URI and checksum
                    shard_uri = f"ipfs://{shard_id}"
                    checksum = hashlib.sha256(shard_id.encode()).hexdigest()
                    
                    await conn.execute("""
                        INSERT INTO data_assignments 
                        (round_id, node_id, shard_id, shard_uri, checksum)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                    """, round_id, node_id, shard_id, shard_uri, checksum)
    
    async def get_node_assignments(self, round_id: str, node_id: str) -> List[Dict]:
        """Get data assignments for a specific node"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM data_assignments
                WHERE round_id = $1 AND node_id = $2
            """, round_id, node_id)
            
            return [dict(row) for row in rows]
    
    # Delta submission methods
    async def save_delta(self,
                        round_id: str,
                        node_id: str,
                        delta_data: Dict) -> str:
        """Save a delta submission"""
        async with self.pool.acquire() as conn:
            delta_id = hashlib.sha256(
                f"{round_id}:{node_id}:{datetime.utcnow()}".encode()
            ).hexdigest()[:16]
            
            await conn.execute("""
                INSERT INTO deltas 
                (round_id, node_id, delta_id, parent_model_hash, delta_hash, 
                 train_meta, metrics, proof_hash, sig)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, round_id, node_id, delta_id,
                delta_data.get('parent_model_hash', ''),
                delta_data.get('delta_hash', ''),
                json.dumps(delta_data.get('train_meta', {})),
                json.dumps(delta_data.get('metrics', {})),
                delta_data.get('proof_hash', ''),
                delta_data.get('sig', ''))
            
            return delta_id
    
    async def get_round_deltas(self, round_id: str) -> List[Dict]:
        """Get all deltas for a round"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM deltas
                WHERE round_id = $1
                ORDER BY submitted_at
            """, round_id)
            
            return [dict(row) for row in rows]
    
    # Idempotency methods
    async def check_idempotency(self, key: str) -> Optional[Dict]:
        """Check if request was already processed"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT response FROM idempotency_keys
                WHERE key = $1 AND expires_at > now()
            """, key)
            
            if row and row['response']:
                return json.loads(row['response'])
            return None
    
    async def save_idempotency(self, key: str, request_id: str, response: Dict) -> None:
        """Save idempotency key and response"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO idempotency_keys (key, request_id, response)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO NOTHING
            """, key, request_id, json.dumps(response))
    
    # Cleanup methods
    async def cleanup_expired(self) -> None:
        """Clean up expired idempotency keys"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DELETE FROM idempotency_keys
                WHERE expires_at < now()
            """)
    
    async def get_last_round_time(self) -> Optional[datetime]:
        """Get the timestamp of the last completed round"""
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT MAX(created_at) 
                FROM rounds 
                WHERE state = 'completed'
            """)
            return result
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()


# Singleton instance
_db_instance = None

async def get_learning_db() -> LearningDatabase:
    """Get or create database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = LearningDatabase()
        await _db_instance.initialize()
    return _db_instance