"""
Enhanced Learning API endpoints with fault tolerance and SLA metrics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Optional
from datetime import datetime
import json

router = APIRouter(prefix="/learning/v2", tags=["learning_v2"])

# Will be set by server initialization
learning_coordinator = None


@router.get("/status")
async def get_coordinator_status():
    """Get coordinator status with configuration and metrics."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    return learning_coordinator.get_status()


@router.get("/health/nodes")
async def get_node_health():
    """Get detailed node health status."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    async with learning_coordinator.db.pool.acquire() as conn:
        nodes = await conn.fetch("""
            SELECT 
                n.node_id,
                n.base_url,
                n.status,
                n.reputation_score,
                h.last_heartbeat,
                h.consecutive_failures,
                h.health_status,
                h.response_time_ms,
                h.success_rate,
                h.last_failure_time,
                h.last_recovery_time
            FROM gpu_nodes n
            LEFT JOIN node_heartbeats h ON n.node_id = h.node_id
            ORDER BY n.reputation_score DESC, h.success_rate DESC NULLS LAST
        """)
        
        return {
            "nodes": [dict(row) for row in nodes],
            "summary": {
                "total": len(nodes),
                "healthy": sum(1 for n in nodes if n.get('health_status') == 'healthy'),
                "degraded": sum(1 for n in nodes if n.get('health_status') == 'degraded'),
                "unreachable": sum(1 for n in nodes if n.get('health_status') == 'unreachable'),
                "recovering": sum(1 for n in nodes if n.get('health_status') == 'recovering')
            }
        }


@router.get("/rounds")
async def get_rounds_with_sla():
    """Get learning rounds with SLA metrics."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    async with learning_coordinator.db.pool.acquire() as conn:
        rounds = await conn.fetch("""
            SELECT 
                r.round_id,
                r.state,
                r.created_at,
                r.updated_at,
                r.bly_sum,
                r.target_expert,
                r.base_version,
                m.stage,
                m.started_at as stage_started,
                m.completed_at as stage_completed,
                m.timeout_at,
                m.nodes_total,
                m.nodes_responded,
                m.nodes_failed,
                m.nodes_reassigned,
                m.retry_count,
                m.timeout_occurred
            FROM learning_rounds r
            LEFT JOIN round_stage_metrics m ON r.round_id = m.round_id AND m.stage = r.state
            WHERE r.state NOT IN ('REWARD_DIST', 'FAILED')
            ORDER BY r.created_at DESC
            LIMIT 10
        """)
        
        # Format results
        formatted_rounds = []
        for row in rounds:
            round_data = dict(row)
            # Convert non-serializable types
            for field in ['round_id', 'created_at', 'updated_at', 'stage_started', 'stage_completed', 'timeout_at']:
                if field in round_data and round_data[field]:
                    round_data[field] = str(round_data[field])
            formatted_rounds.append(round_data)
        
        return {
            "active_rounds": formatted_rounds,
            "total": len(formatted_rounds)
        }


@router.get("/sla/metrics")
async def get_sla_metrics():
    """Get comprehensive SLA metrics for monitoring."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    return await learning_coordinator.get_sla_metrics()


@router.get("/sla/prometheus")
async def get_prometheus_metrics():
    """Export metrics in Prometheus format."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    metrics = await learning_coordinator.get_sla_metrics()
    
    # Format as Prometheus metrics
    lines = []
    
    # Node health metrics
    lines.append("# HELP blyan_nodes_health Node health status")
    lines.append("# TYPE blyan_nodes_health gauge")
    for status, count in metrics['node_health_summary'].items():
        lines.append(f'blyan_nodes_health{{status="{status}"}} {count}')
    
    # Timeout metrics
    lines.append("# HELP blyan_learning_timeouts_total Total number of stage timeouts")
    lines.append("# TYPE blyan_learning_timeouts_total counter")
    lines.append(f"blyan_learning_timeouts_total {metrics['recent_timeouts']}")
    
    # Reassignment metrics
    lines.append("# HELP blyan_task_reassignments_total Total number of task reassignments")
    lines.append("# TYPE blyan_task_reassignments_total counter")
    lines.append(f"blyan_task_reassignments_total {metrics['recent_reassignments']}")
    
    # Stage duration metrics
    lines.append("# HELP blyan_stage_duration_seconds Stage processing duration")
    lines.append("# TYPE blyan_stage_duration_seconds summary")
    for stage_data in metrics['stage_durations']:
        stage = stage_data['stage']
        if stage_data['avg_duration']:
            lines.append(f'blyan_stage_duration_seconds{{stage="{stage}",quantile="0.5"}} {stage_data["avg_duration"]}')
            lines.append(f'blyan_stage_duration_seconds{{stage="{stage}",quantile="1.0"}} {stage_data["max_duration"]}')
    
    # Coordinator metrics
    coord_metrics = metrics['coordinator_metrics']
    lines.append("# HELP blyan_rounds_processed_total Total rounds processed")
    lines.append("# TYPE blyan_rounds_processed_total counter")
    lines.append(f"blyan_rounds_processed_total {coord_metrics['rounds_processed']}")
    
    lines.append("# HELP blyan_nodes_degraded_total Total nodes marked as degraded")
    lines.append("# TYPE blyan_nodes_degraded_total counter")
    lines.append(f"blyan_nodes_degraded_total {coord_metrics['nodes_degraded']}")
    
    return "\n".join(lines)


@router.post("/nodes/{node_id}/heartbeat")
async def manual_heartbeat(node_id: str, background_tasks: BackgroundTasks):
    """Manually trigger a heartbeat check for a specific node."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    background_tasks.add_task(
        learning_coordinator._check_node_health_single,
        node_id
    )
    
    return {"status": "heartbeat_scheduled", "node_id": node_id}


@router.post("/nodes/{node_id}/recover")
async def recover_node(node_id: str):
    """Attempt to recover a degraded node."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    async with learning_coordinator.db.pool.acquire() as conn:
        # Reset node health status
        await conn.execute("""
            UPDATE node_heartbeats
            SET health_status = 'recovering',
                consecutive_failures = 0,
                last_recovery_time = now()
            WHERE node_id = $1
        """, node_id)
        
        # Update node status
        await conn.execute("""
            UPDATE gpu_nodes
            SET status = 'active'
            WHERE node_id = $1
        """, node_id)
    
    return {"status": "recovery_initiated", "node_id": node_id}


@router.get("/reassignments/{round_id}")
async def get_round_reassignments(round_id: str):
    """Get task reassignment history for a specific round."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    async with learning_coordinator.db.pool.acquire() as conn:
        reassignments = await conn.fetch("""
            SELECT 
                original_node_id,
                new_node_id,
                task_type,
                task_id,
                reason,
                reassigned_at
            FROM task_reassignments
            WHERE round_id = $1
            ORDER BY reassigned_at DESC
        """, round_id)
        
        return {
            "round_id": round_id,
            "reassignments": [dict(row) for row in reassignments],
            "total": len(reassignments)
        }


@router.post("/rounds/{round_id}/retry")
async def retry_round_stage(round_id: str):
    """Retry the current stage of a round."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    async with learning_coordinator.db.pool.acquire() as conn:
        # Get current round state
        round_data = await conn.fetchrow(
            "SELECT state FROM learning_rounds WHERE round_id = $1",
            round_id
        )
        
        if not round_data:
            raise HTTPException(status_code=404, detail="Round not found")
        
        current_state = round_data['state']
        
        # Reset stage metrics for retry
        await conn.execute("""
            UPDATE round_stage_metrics
            SET timeout_occurred = false,
                retry_count = retry_count + 1,
                timeout_at = now() + INTERVAL '%s seconds'
            WHERE round_id = $1 AND stage = $2
        """, round_id, current_state)
    
    return {
        "round_id": round_id,
        "stage": current_state,
        "status": "retry_initiated"
    }


@router.get("/config/timeouts")
async def get_timeout_configuration():
    """Get timeout configuration for all stages."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    return {
        "stage_timeouts": {
            stage: {
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "min_nodes_required": config.min_nodes_required,
                "allow_partial_success": config.allow_partial_success
            }
            for stage, config in learning_coordinator.stage_configs.items()
        }
    }


@router.put("/config/timeouts/{stage}")
async def update_stage_timeout(stage: str, timeout_seconds: int, max_retries: int = None):
    """Update timeout configuration for a specific stage."""
    if not learning_coordinator:
        raise HTTPException(status_code=503, detail="Learning coordinator not initialized")
    
    if stage not in learning_coordinator.stage_configs:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")
    
    config = learning_coordinator.stage_configs[stage]
    config.timeout_seconds = timeout_seconds
    
    if max_retries is not None:
        config.max_retries = max_retries
    
    return {
        "stage": stage,
        "updated_config": {
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
            "min_nodes_required": config.min_nodes_required,
            "allow_partial_success": config.allow_partial_success
        }
    }


def init_router(coordinator):
    """Initialize router with coordinator instance."""
    global learning_coordinator
    learning_coordinator = coordinator
    return router