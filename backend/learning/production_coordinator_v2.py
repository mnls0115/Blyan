"""
Production Learning Coordinator V2
With timeout handling, automatic recovery, and fault tolerance
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx
from backend.database.learning_db import LearningDatabase

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    RECOVERING = "recovering"


@dataclass
class StageConfig:
    """Configuration for each learning stage"""
    timeout_seconds: int
    max_retries: int
    min_nodes_required: int
    allow_partial_success: bool


class FaultTolerantLearningCoordinator:
    """
    Production learning coordinator with enterprise-grade fault tolerance.
    
    Key improvements:
    - Stage timeouts with automatic progression
    - Node health tracking and degradation handling
    - Task reassignment for failed nodes
    - Crash recovery with state persistence
    - Comprehensive SLA metrics
    """
    
    DEFAULT_STAGE_CONFIGS = {
        "NOTIFYING": StageConfig(30, 3, 1, True),
        "DATA_ALLOC": StageConfig(60, 2, 1, True),
        "TRAINING": StageConfig(600, 1, 1, True),
        "CONSENSUS": StageConfig(120, 2, 2, False),
        "DELTA_CREATION": StageConfig(60, 2, 1, False),
        "REWARD_DIST": StageConfig(30, 3, 1, True)
    }
    
    def __init__(self,
                 db: LearningDatabase,
                 dataset_chain,
                 param_chain,
                 budget_controller,
                 config: Dict = None):
        
        self.db = db
        self.dataset_chain = dataset_chain
        self.param_chain = param_chain
        self.budget_controller = budget_controller
        
        # Configuration with defaults
        self.config = config or {}
        self.config.setdefault("threshold_bly", Decimal("1000"))
        self.config.setdefault("min_interval_seconds", 3600)
        self.config.setdefault("quorum_threshold", 0.67)
        self.config.setdefault("heartbeat_interval", 30)
        self.config.setdefault("max_consecutive_failures", 3)
        self.config.setdefault("node_recovery_cooldown", 300)
        
        # Stage configurations
        self.stage_configs = self.DEFAULT_STAGE_CONFIGS.copy()
        if "stage_configs" in self.config:
            self.stage_configs.update(self.config["stage_configs"])
        
        # Runtime state
        self.is_running = False
        self.processing_task = None
        self.heartbeat_task = None
        self.http_client = None
        
        # Metrics
        self.metrics = {
            "rounds_processed": 0,
            "nodes_degraded": 0,
            "tasks_reassigned": 0,
            "timeouts_occurred": 0
        }
    
    async def start(self):
        """Start the coordinator with all background tasks"""
        if self.is_running:
            return
        
        logger.info("Starting fault-tolerant learning coordinator...")
        self.is_running = True
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Start background tasks
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Resume any unfinished rounds after restart
        await self._resume_unfinished_rounds()
        
        logger.info("✅ Fault-tolerant coordinator started")
    
    async def stop(self):
        """Stop the coordinator gracefully"""
        if not self.is_running:
            return
        
        logger.info("Stopping coordinator...")
        self.is_running = False
        
        # Cancel background tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("Coordinator stopped")
    
    async def _processing_loop(self):
        """Main processing loop with automatic round progression"""
        while self.is_running:
            try:
                # Check for triggered burns
                await self._check_burn_trigger()
                
                # Process active rounds with timeout handling
                await self._process_active_rounds()
                
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(10)
    
    async def _heartbeat_loop(self):
        """Monitor node health with periodic heartbeats"""
        while self.is_running:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.config["heartbeat_interval"])
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(30)
    
    async def _check_node_health(self):
        """Send heartbeats to all active nodes and update health status"""
        nodes = await self.db.get_active_nodes()
        
        for node_data in nodes:
            node_id = node_data['node_id']
            base_url = node_data['base_url']
            
            try:
                start_time = time.time()
                response = await self.http_client.get(
                    f"{base_url}/health",
                    timeout=5.0
                )
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    await self._update_node_health(node_id, True, response_time_ms)
                else:
                    await self._update_node_health(node_id, False)
                    
            except Exception as e:
                logger.warning(f"Node {node_id} health check failed: {e}")
                await self._update_node_health(node_id, False)
    
    async def _update_node_health(self, node_id: str, success: bool, response_time_ms: float = None):
        """Update node health status in database"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                "SELECT update_node_health($1, $2, $3)",
                node_id, success, response_time_ms
            )
            
            # Check if node needs to be degraded
            result = await conn.fetchrow(
                "SELECT health_status, consecutive_failures FROM node_heartbeats WHERE node_id = $1",
                node_id
            )
            
            if result:
                if result['health_status'] in ('degraded', 'unreachable'):
                    self.metrics["nodes_degraded"] += 1
                    logger.warning(f"Node {node_id} marked as {result['health_status']}")
    
    async def _resume_unfinished_rounds(self):
        """Resume processing of rounds that were interrupted by restart"""
        async with self.db.pool.acquire() as conn:
            unfinished = await conn.fetch("""
                SELECT * FROM learning_rounds
                WHERE state NOT IN ('REWARD_DIST', 'FAILED')
                ORDER BY created_at ASC
            """)
            
            for round_data in unfinished:
                round_id = str(round_data['round_id'])
                state = round_data['state']
                logger.info(f"Resuming round {round_id} from state {state}")
                
                # Check if stage has timed out
                stage_timeout = await self._check_stage_timeout(round_id, state)
                if stage_timeout:
                    await self._handle_stage_timeout(round_id, state)
    
    async def _check_burn_trigger(self):
        """Check if burn threshold reached and trigger new round"""
        burn_result = await self.db.get_burn_accumulator()
        
        # Handle tuple return from database (sum, count)
        if isinstance(burn_result, tuple):
            burn_sum = burn_result[0] if burn_result[0] else Decimal("0")
        else:
            burn_sum = burn_result if burn_result else Decimal("0")
        
        if burn_sum >= self.config["threshold_bly"]:
            # Check minimum interval
            last_round = await self.db.get_last_round_time()
            if last_round:
                elapsed = (datetime.utcnow() - last_round).total_seconds()
                if elapsed < self.config["min_interval_seconds"]:
                    return
            
            # Get healthy nodes
            healthy_nodes = await self._get_healthy_nodes()
            if len(healthy_nodes) < 1:
                logger.warning("No healthy nodes available for learning")
                return
            
            # Create new round
            await self._create_learning_round()
    
    async def _get_healthy_nodes(self) -> List[Dict]:
        """Get list of healthy nodes for task assignment"""
        async with self.db.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM get_healthy_nodes($1)",
                0.8  # Minimum 80% success rate
            )
            return [dict(row) for row in rows]
    
    async def _create_learning_round(self):
        """Create a new learning round with timeout tracking"""
        round_id = await self.db.create_round(
            bly_sum=self.config["threshold_bly"],
            min_interval_ok=True,
            config={
                "threshold": str(self.config["threshold_bly"]),
                "quorum": self.config["quorum_threshold"]
            }
        )
        
        # Initialize stage metrics
        await self._init_stage_metrics(round_id, "NOTIFYING")
        
        # Mark burns as used
        await self.db.mark_burns_used(self.config["threshold_bly"])
        
        logger.info(f"Created learning round {round_id}")
        self.metrics["rounds_processed"] += 1
    
    async def _init_stage_metrics(self, round_id: str, stage: str):
        """Initialize metrics for a round stage"""
        config = self.stage_configs[stage]
        timeout_at = datetime.utcnow() + timedelta(seconds=config.timeout_seconds)
        
        healthy_nodes = await self._get_healthy_nodes()
        
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO round_stage_metrics 
                (round_id, stage, timeout_at, nodes_total)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (round_id, stage) DO UPDATE
                SET timeout_at = $3, nodes_total = $4
            """, round_id, stage, timeout_at, len(healthy_nodes))
    
    async def _process_active_rounds(self):
        """Process all active rounds with timeout and recovery"""
        rounds = await self.db.get_active_rounds()
        
        for round_data in rounds:
            round_id = round_data['round_id']
            state = round_data['state']
            
            try:
                # Check for stage timeout
                timed_out = await self._check_stage_timeout(round_id, state)
                if timed_out:
                    await self._handle_stage_timeout(round_id, state)
                    continue
                
                # Process based on current state
                await self._process_round_state(round_id, state, round_data)
                
            except Exception as e:
                logger.error(f"Error processing round {round_id}: {e}")
                await self._handle_round_error(round_id, state, str(e))
    
    async def _check_stage_timeout(self, round_id: str, stage: str) -> bool:
        """Check if current stage has timed out"""
        async with self.db.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT timeout_at, timeout_occurred
                FROM round_stage_metrics
                WHERE round_id = $1 AND stage = $2
            """, round_id, stage)
            
            if not result:
                return False
            
            if result['timeout_occurred']:
                return False  # Already handled
            
            if datetime.utcnow() > result['timeout_at']:
                # Mark timeout occurred
                await conn.execute("""
                    UPDATE round_stage_metrics
                    SET timeout_occurred = true
                    WHERE round_id = $1 AND stage = $2
                """, round_id, stage)
                
                self.metrics["timeouts_occurred"] += 1
                return True
            
            return False
    
    async def _handle_stage_timeout(self, round_id: str, stage: str):
        """Handle stage timeout with automatic progression or reassignment"""
        logger.warning(f"Round {round_id} stage {stage} timed out")
        
        config = self.stage_configs[stage]
        
        # Get stage metrics
        async with self.db.pool.acquire() as conn:
            metrics = await conn.fetchrow("""
                SELECT nodes_total, nodes_responded, nodes_failed
                FROM round_stage_metrics
                WHERE round_id = $1 AND stage = $2
            """, round_id, stage)
            
            if not metrics:
                return
            
            # Check if we have minimum nodes
            active_nodes = metrics['nodes_responded']
            if active_nodes >= config.min_nodes_required:
                # Proceed with partial success if allowed
                if config.allow_partial_success:
                    logger.info(f"Proceeding with {active_nodes}/{metrics['nodes_total']} nodes")
                    await self._advance_round_state(round_id, stage)
                else:
                    # Fail the round
                    await self._fail_round(round_id, f"Insufficient nodes: {active_nodes}/{config.min_nodes_required}")
            else:
                # Try reassignment or fail
                reassigned = await self._reassign_failed_tasks(round_id, stage)
                if reassigned:
                    # Reset timeout for retry
                    await self._init_stage_metrics(round_id, stage)
                else:
                    await self._fail_round(round_id, f"Stage {stage} failed after timeout")
    
    async def _reassign_failed_tasks(self, round_id: str, stage: str) -> bool:
        """Reassign tasks from failed nodes to healthy ones"""
        healthy_nodes = await self._get_healthy_nodes()
        if not healthy_nodes:
            return False
        
        async with self.db.pool.acquire() as conn:
            # Find failed assignments
            if stage == "DATA_ALLOC":
                failed = await conn.fetch("""
                    SELECT da.node_id, da.shard_id
                    FROM data_assignments da
                    LEFT JOIN node_heartbeats nh ON da.node_id = nh.node_id
                    WHERE da.round_id = $1
                    AND da.fetched_at IS NULL
                    AND (nh.health_status IN ('degraded', 'unreachable') OR nh.health_status IS NULL)
                """, round_id)
                
                for task in failed:
                    # Reassign to healthy node
                    new_node = healthy_nodes[0]['node_id']  # Simple assignment
                    
                    # Update assignment
                    await conn.execute("""
                        UPDATE data_assignments
                        SET node_id = $1
                        WHERE round_id = $2 AND shard_id = $3
                    """, new_node, round_id, task['shard_id'])
                    
                    # Track reassignment
                    await conn.execute("""
                        INSERT INTO task_reassignments
                        (round_id, original_node_id, new_node_id, task_type, task_id, reason)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, round_id, task['node_id'], new_node, 'data_shard', task['shard_id'], 'timeout')
                    
                    self.metrics["tasks_reassigned"] += 1
                
                return len(failed) > 0
        
        return False
    
    async def _process_round_state(self, round_id: str, state: str, round_data: Dict):
        """Process round based on current state"""
        handlers = {
            "TRIGGERED": self._handle_triggered,
            "NOTIFYING": self._handle_notifying,
            "DATA_ALLOC": self._handle_data_alloc,
            "TRAINING": self._handle_training,
            "CONSENSUS": self._handle_consensus,
            "DELTA_CREATION": self._handle_delta_creation,
            "REWARD_DIST": self._handle_reward_dist
        }
        
        handler = handlers.get(state)
        if handler:
            await handler(round_id, round_data)
    
    async def _handle_triggered(self, round_id: str, round_data: Dict):
        """Handle triggered state - move to notifying"""
        await self._advance_round_state(round_id, "TRIGGERED")
    
    async def _handle_notifying(self, round_id: str, round_data: Dict):
        """Notify nodes with timeout and retry"""
        healthy_nodes = await self._get_healthy_nodes()
        
        notifications_sent = 0
        notifications_failed = 0
        
        for node in healthy_nodes:
            success = await self._notify_node_with_retry(
                round_id, node['node_id'], round_data
            )
            
            if success:
                notifications_sent += 1
            else:
                notifications_failed += 1
        
        # Update stage metrics
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                UPDATE round_stage_metrics
                SET nodes_responded = $1, nodes_failed = $2
                WHERE round_id = $3 AND stage = $4
            """, notifications_sent, notifications_failed, round_id, "NOTIFYING")
        
        # Check if we have enough nodes
        config = self.stage_configs["NOTIFYING"]
        if notifications_sent >= config.min_nodes_required:
            await self._advance_round_state(round_id, "NOTIFYING")
    
    async def _notify_node_with_retry(self, round_id: str, node_id: str, round_data: Dict) -> bool:
        """Notify a node with retry logic"""
        config = self.stage_configs["NOTIFYING"]
        
        for attempt in range(config.max_retries):
            try:
                # Get node URL
                async with self.db.pool.acquire() as conn:
                    node = await conn.fetchrow(
                        "SELECT base_url FROM gpu_nodes WHERE node_id = $1",
                        node_id
                    )
                    
                    if not node:
                        return False
                
                # Send notification
                response = await self.http_client.post(
                    f"{node['base_url']}/learning/start",
                    json={
                        "round_id": round_id,
                        "target_expert": round_data.get('target_expert', 'layer0.expert0'),
                        "base_version": round_data.get('base_version', ''),
                        "seed": round_data.get('seed', '')
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    await self._update_node_health(node_id, True)
                    return True
                    
            except Exception as e:
                logger.warning(f"Node {node_id} notification attempt {attempt+1} failed: {e}")
                await self._update_node_health(node_id, False)
                
                if attempt < config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    async def _advance_round_state(self, round_id: str, current_state: str):
        """Advance round to next state"""
        state_progression = {
            "TRIGGERED": "NOTIFYING",
            "NOTIFYING": "DATA_ALLOC",
            "DATA_ALLOC": "TRAINING",
            "TRAINING": "CONSENSUS",
            "CONSENSUS": "DELTA_CREATION",
            "DELTA_CREATION": "REWARD_DIST"
        }
        
        next_state = state_progression.get(current_state)
        if not next_state:
            return
        
        await self.db.update_round_state(round_id, next_state)
        await self._init_stage_metrics(round_id, next_state)
        
        logger.info(f"Round {round_id} advanced: {current_state} -> {next_state}")
    
    async def _fail_round(self, round_id: str, reason: str):
        """Mark round as failed"""
        await self.db.update_round_state(round_id, "FAILED")
        logger.error(f"Round {round_id} failed: {reason}")
        
        # Return unused burns
        async with self.db.pool.acquire() as conn:
            round_data = await conn.fetchrow(
                "SELECT bly_sum FROM learning_rounds WHERE round_id = $1",
                round_id
            )
            
            if round_data:
                # Return burns to pool
                await conn.execute("""
                    UPDATE burns_ledger
                    SET round_candidate = false
                    WHERE round_candidate = true
                    AND bly_amount <= $1
                """, round_data['bly_sum'])
    
    # Placeholder methods for remaining states
    async def _handle_data_alloc(self, round_id: str, round_data: Dict):
        """Handle data allocation with reassignment"""
        # Implementation similar to notifying with reassignment logic
        await self._advance_round_state(round_id, "DATA_ALLOC")
    
    async def _handle_training(self, round_id: str, round_data: Dict):
        """Monitor training progress with timeout"""
        await self._advance_round_state(round_id, "TRAINING")
    
    async def _handle_consensus(self, round_id: str, round_data: Dict):
        """Run consensus with quorum checking"""
        await self._advance_round_state(round_id, "CONSENSUS")
    
    async def _handle_delta_creation(self, round_id: str, round_data: Dict):
        """Create delta block"""
        await self._advance_round_state(round_id, "DELTA_CREATION")
    
    async def _handle_reward_dist(self, round_id: str, round_data: Dict):
        """Distribute rewards"""
        await self.db.update_round_state(round_id, "REWARD_DIST")
    
    async def _handle_round_error(self, round_id: str, state: str, error: str):
        """Handle round processing error"""
        logger.error(f"Round {round_id} error in state {state}: {error}")
        
        # Update retry count
        async with self.db.pool.acquire() as conn:
            await conn.execute("""
                UPDATE round_stage_metrics
                SET retry_count = retry_count + 1
                WHERE round_id = $1 AND stage = $2
            """, round_id, state)
    
    async def record_inference_burn(self, amount: Decimal, user_addr: str = "manual", request_id: str = "manual"):
        """Record burn from inference request (compatibility method)"""
        await self.db.record_burn(user_addr, request_id, amount)
        logger.info(f"Recorded burn: {amount} BLY from {user_addr}")
    
    def get_status(self) -> Dict:
        """Get coordinator status with metrics"""
        return {
            "is_running": self.is_running,
            "config": {
                "threshold_bly": float(self.config["threshold_bly"]),
                "min_interval_seconds": self.config["min_interval_seconds"],
                "quorum_threshold": self.config["quorum_threshold"],
                "heartbeat_interval": self.config["heartbeat_interval"]
            },
            "metrics": self.metrics,
            "stage_timeouts": {
                stage: {
                    "timeout_seconds": config.timeout_seconds,
                    "max_retries": config.max_retries,
                    "min_nodes": config.min_nodes_required
                }
                for stage, config in self.stage_configs.items()
            }
        }
    
    async def get_sla_metrics(self) -> Dict:
        """Get SLA metrics for monitoring"""
        async with self.db.pool.acquire() as conn:
            # Get node health summary
            node_health = await conn.fetch("""
                SELECT health_status, COUNT(*) as count
                FROM node_heartbeats
                GROUP BY health_status
            """)
            
            # Get recent timeout events
            timeouts = await conn.fetchval("""
                SELECT COUNT(*)
                FROM round_stage_metrics
                WHERE timeout_occurred = true
                AND started_at > now() - INTERVAL '1 hour'
            """)
            
            # Get reassignment stats
            reassignments = await conn.fetchval("""
                SELECT COUNT(*)
                FROM task_reassignments
                WHERE reassigned_at > now() - INTERVAL '1 hour'
            """)
            
            # Get average stage durations
            stage_durations = await conn.fetch("""
                SELECT stage,
                       AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration,
                       MAX(EXTRACT(EPOCH FROM (completed_at - started_at))) as max_duration
                FROM round_stage_metrics
                WHERE completed_at IS NOT NULL
                GROUP BY stage
            """)
            
            return {
                "node_health_summary": {row['health_status']: row['count'] for row in node_health},
                "recent_timeouts": timeouts,
                "recent_reassignments": reassignments,
                "stage_durations": [dict(row) for row in stage_durations],
                "coordinator_metrics": self.metrics
            }


# Singleton instance management
_coordinator_instance = None

async def get_fault_tolerant_coordinator(dataset_chain, param_chain, budget_controller) -> FaultTolerantLearningCoordinator:
    """Get or create fault-tolerant coordinator singleton"""
    global _coordinator_instance
    if _coordinator_instance is None:
        from backend.database.learning_db import get_learning_db
        db = await get_learning_db()
        _coordinator_instance = FaultTolerantLearningCoordinator(
            db=db,
            dataset_chain=dataset_chain,
            param_chain=param_chain,
            budget_controller=budget_controller
        )
    return _coordinator_instance