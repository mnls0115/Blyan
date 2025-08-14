#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for Blyan Network
Exposes system metrics in Prometheus format
"""

from typing import Dict, List, Any
import time
import psutil
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and formats metrics for Prometheus."""
    
    def __init__(self):
        self.start_time = time.time()
        self._cache = {}
        self._cache_ttl = 5  # seconds
        
    async def collect_all_metrics(self) -> str:
        """Collect all metrics and format for Prometheus."""
        metrics = []
        
        # System metrics
        metrics.extend(await self.collect_system_metrics())
        
        # Database metrics
        metrics.extend(await self.collect_database_metrics())
        
        # Application metrics
        metrics.extend(await self.collect_application_metrics())
        
        # Business metrics
        metrics.extend(await self.collect_business_metrics())
        
        return '\n'.join(metrics)
    
    async def collect_system_metrics(self) -> List[str]:
        """Collect system-level metrics."""
        metrics = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(f'# HELP blyan_cpu_usage_percent CPU usage percentage')
        metrics.append(f'# TYPE blyan_cpu_usage_percent gauge')
        metrics.append(f'blyan_cpu_usage_percent {cpu_percent}')
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics.append(f'# HELP blyan_memory_usage_bytes Memory usage in bytes')
        metrics.append(f'# TYPE blyan_memory_usage_bytes gauge')
        metrics.append(f'blyan_memory_usage_bytes{{type="used"}} {memory.used}')
        metrics.append(f'blyan_memory_usage_bytes{{type="available"}} {memory.available}')
        metrics.append(f'blyan_memory_usage_bytes{{type="total"}} {memory.total}')
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics.append(f'# HELP blyan_disk_usage_bytes Disk usage in bytes')
        metrics.append(f'# TYPE blyan_disk_usage_bytes gauge')
        metrics.append(f'blyan_disk_usage_bytes{{mount="/"}} {disk.used}')
        
        # Network I/O
        net_io = psutil.net_io_counters()
        metrics.append(f'# HELP blyan_network_bytes_total Network I/O bytes')
        metrics.append(f'# TYPE blyan_network_bytes_total counter')
        metrics.append(f'blyan_network_bytes_total{{direction="sent"}} {net_io.bytes_sent}')
        metrics.append(f'blyan_network_bytes_total{{direction="recv"}} {net_io.bytes_recv}')
        
        # Process info
        process = psutil.Process()
        metrics.append(f'# HELP blyan_process_cpu_seconds_total Process CPU time')
        metrics.append(f'# TYPE blyan_process_cpu_seconds_total counter')
        metrics.append(f'blyan_process_cpu_seconds_total {process.cpu_times().user + process.cpu_times().system}')
        
        metrics.append(f'# HELP blyan_process_memory_bytes Process memory usage')
        metrics.append(f'# TYPE blyan_process_memory_bytes gauge')
        metrics.append(f'blyan_process_memory_bytes {process.memory_info().rss}')
        
        # Uptime
        uptime = time.time() - self.start_time
        metrics.append(f'# HELP blyan_uptime_seconds Uptime in seconds')
        metrics.append(f'# TYPE blyan_uptime_seconds counter')
        metrics.append(f'blyan_uptime_seconds {uptime}')
        
        return metrics
    
    async def collect_database_metrics(self) -> List[str]:
        """Collect PostgreSQL metrics."""
        metrics = []
        
        try:
            from backend.accounting.postgres_ledger import get_postgres_ledger
            ledger = get_postgres_ledger()
            
            if ledger.pool:
                # Connection pool stats
                pool_size = ledger.pool._size
                pool_free = ledger.pool._queue.qsize() if hasattr(ledger.pool, '_queue') else 0
                
                metrics.append(f'# HELP blyan_db_pool_size Database connection pool size')
                metrics.append(f'# TYPE blyan_db_pool_size gauge')
                metrics.append(f'blyan_db_pool_size{{state="total"}} {pool_size}')
                metrics.append(f'blyan_db_pool_size{{state="free"}} {pool_free}')
                metrics.append(f'blyan_db_pool_size{{state="used"}} {pool_size - pool_free}')
                
                # Query database stats
                async with ledger.pool.acquire() as conn:
                    # Table sizes
                    result = await conn.fetch("""
                        SELECT 
                            schemaname,
                            tablename,
                            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                            n_live_tup as row_count
                        FROM pg_stat_user_tables
                        WHERE schemaname = 'public'
                    """)
                    
                    metrics.append(f'# HELP blyan_db_table_size_bytes Database table sizes')
                    metrics.append(f'# TYPE blyan_db_table_size_bytes gauge')
                    
                    for row in result:
                        metrics.append(f'blyan_db_table_size_bytes{{table="{row["tablename"]}"}} {row["size_bytes"]}')
                    
                    metrics.append(f'# HELP blyan_db_table_rows Database table row counts')
                    metrics.append(f'# TYPE blyan_db_table_rows gauge')
                    
                    for row in result:
                        metrics.append(f'blyan_db_table_rows{{table="{row["tablename"]}"}} {row["row_count"]}')
                    
                    # Transaction stats
                    tx_stats = await conn.fetchrow("""
                        SELECT 
                            COUNT(*) FILTER (WHERE status = 'credited') as credited,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed,
                            COUNT(*) FILTER (WHERE status = 'pending') as pending
                        FROM transactions
                        WHERE created_at > NOW() - INTERVAL '1 hour'
                    """)
                    
                    if tx_stats:
                        metrics.append(f'# HELP blyan_transactions_hourly Transaction counts (last hour)')
                        metrics.append(f'# TYPE blyan_transactions_hourly gauge')
                        metrics.append(f'blyan_transactions_hourly{{status="credited"}} {tx_stats["credited"] or 0}')
                        metrics.append(f'blyan_transactions_hourly{{status="failed"}} {tx_stats["failed"] or 0}')
                        metrics.append(f'blyan_transactions_hourly{{status="pending"}} {tx_stats["pending"] or 0}')
                    
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            metrics.append(f'# Database metrics unavailable: {e}')
        
        return metrics
    
    async def collect_application_metrics(self) -> List[str]:
        """Collect application-specific metrics."""
        metrics = []
        
        try:
            # Transaction manager metrics
            from backend.core.transaction_manager import get_transaction_manager
            tx_manager = get_transaction_manager()
            
            metrics.append(f'# HELP blyan_active_transactions Active transaction contexts')
            metrics.append(f'# TYPE blyan_active_transactions gauge')
            metrics.append(f'blyan_active_transactions {len(tx_manager.active_transactions)}')
            
            # Streaming metrics
            from api.streaming import get_streaming_handler
            streaming = get_streaming_handler()
            
            metrics.append(f'# HELP blyan_active_streams Active streaming sessions')
            metrics.append(f'# TYPE blyan_active_streams gauge')
            metrics.append(f'blyan_active_streams {len(streaming.active_streams)}')
            
            # Reward distribution metrics
            from backend.rewards.automatic_distribution import get_reward_distributor
            distributor = get_reward_distributor()
            
            pending_total = sum(
                sum(r.amount for r in rewards)
                for rewards in distributor.pending_rewards.values()
            )
            
            metrics.append(f'# HELP blyan_pending_rewards_total Pending reward amount')
            metrics.append(f'# TYPE blyan_pending_rewards_total gauge')
            metrics.append(f'blyan_pending_rewards_total {float(pending_total)}')
            
            metrics.append(f'# HELP blyan_pending_rewards_recipients Recipients with pending rewards')
            metrics.append(f'# TYPE blyan_pending_rewards_recipients gauge')
            metrics.append(f'blyan_pending_rewards_recipients {len(distributor.pending_rewards)}')
            
            # Rate limiting metrics
            from backend.security.rate_limiting import rate_limiter
            
            metrics.append(f'# HELP blyan_rate_limit_violations Rate limit violations')
            metrics.append(f'# TYPE blyan_rate_limit_violations counter')
            # This would need actual tracking in rate_limiter
            metrics.append(f'blyan_rate_limit_violations 0')

            # Pipeline metrics
            try:
                from backend.learning.pipeline_metrics import get_pipeline_metrics
                pm = get_pipeline_metrics()
                snap = await pm.export_snapshot()

                # Stage occupancy
                metrics.append('# HELP blyan_pipeline_stage_occupancy Stage occupancy (0..1)')
                metrics.append('# TYPE blyan_pipeline_stage_occupancy gauge')
                for stage_idx, occ in snap["stage_occupancy"].items():
                    metrics.append(f'blyan_pipeline_stage_occupancy{{stage="{stage_idx}"}} {occ}')

                # RPC error rate (counter)
                metrics.append('# HELP blyan_pipeline_rpc_errors Total RPC errors')
                metrics.append('# TYPE blyan_pipeline_rpc_errors counter')
                metrics.append(f'blyan_pipeline_rpc_errors {snap["rpc_errors"]}')

                # Round failures/pipeline resets/fallbacks
                metrics.append('# HELP blyan_pipeline_round_failures Total round failures')
                metrics.append('# TYPE blyan_pipeline_round_failures counter')
                metrics.append(f'blyan_pipeline_round_failures {snap["round_failures"]}')

                metrics.append('# HELP blyan_pipeline_resets Total pipeline resets')
                metrics.append('# TYPE blyan_pipeline_resets counter')
                metrics.append(f'blyan_pipeline_resets {snap["pipeline_resets"]}')

                metrics.append('# HELP blyan_pipeline_fallback_activations Total fallback activations')
                metrics.append('# TYPE blyan_pipeline_fallback_activations counter')
                metrics.append(f'blyan_pipeline_fallback_activations {snap["fallback_activations"]}')

                # Fallback mode gauge / stage count
                metrics.append('# HELP blyan_pipeline_fallback_mode_active Fallback mode active (0/1)')
                metrics.append('# TYPE blyan_pipeline_fallback_mode_active gauge')
                metrics.append(f'blyan_pipeline_fallback_mode_active {snap.get("fallback_mode_active", 0)}')

                metrics.append('# HELP blyan_pipeline_current_stage_count Current number of stages in plan')
                metrics.append('# TYPE blyan_pipeline_current_stage_count gauge')
                metrics.append(f'blyan_pipeline_current_stage_count {snap.get("current_stage_count", 0)}')

                # Microbatch wait histogram (export as buckets)
                hist = snap["microbatch_wait_hist"]
                metrics.append('# HELP blyan_pipeline_microbatch_wait_seconds Microbatch wait time')
                metrics.append('# TYPE blyan_pipeline_microbatch_wait_seconds histogram')
                for b, c in hist["buckets"].items():
                    # translate label: le_X_Y -> X.Y
                    label = b.replace('le_', '').replace('_', '.')
                    metrics.append(f'blyan_pipeline_microbatch_wait_seconds_bucket{{le="{label}"}} {c}')
                metrics.append(f'blyan_pipeline_microbatch_wait_seconds_count {hist["count"]}')
                metrics.append(f'blyan_pipeline_microbatch_wait_seconds_sum {hist["sum"]}')

                # Partition drift and plan id
                plan_id = snap.get("partition_plan_id") or ""
                metrics.append('# HELP blyan_partition_drift Partition drift (0..1)')
                metrics.append('# TYPE blyan_partition_drift gauge')
                metrics.append(f'blyan_partition_drift{{plan_id="{plan_id}"}} {snap.get("partition_drift", 0.0)}')

                # Device profile staleness
                metrics.append('# HELP blyan_device_profile_staleness_seconds Seconds since last profile update')
                metrics.append('# TYPE blyan_device_profile_staleness_seconds gauge')
                for nid, st in snap["device_profile_staleness"].items():
                    metrics.append(f'blyan_device_profile_staleness_seconds{{node="{nid}"}} {st}')
            except Exception as e:
                logger.warning(f"Pipeline metrics unavailable: {e}")

            # Security alerts thresholds for pipeline-specific metrics (export counts)
            try:
                from backend.security.monitoring import security_monitor
                alerts = security_monitor.alert_manager.alert_history[-50:]
                metrics.append('# HELP blyan_security_alerts_total Number of security alerts recorded')
                metrics.append('# TYPE blyan_security_alerts_total counter')
                metrics.append(f'blyan_security_alerts_total {len(alerts)}')
            except Exception as e:
                logger.debug(f"Security alerts unavailable: {e}")
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            metrics.append(f'# Application metrics unavailable: {e}')
        
        return metrics
    
    async def collect_business_metrics(self) -> List[str]:
        """Collect business/usage metrics."""
        metrics = []
        
        try:
            from backend.accounting.postgres_ledger import get_postgres_ledger
            ledger = get_postgres_ledger()
            
            if ledger.pool:
                async with ledger.pool.acquire() as conn:
                    # Daily active users
                    dau = await conn.fetchval("""
                        SELECT COUNT(DISTINCT user_address)
                        FROM transactions
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                    
                    metrics.append(f'# HELP blyan_daily_active_users Daily active users')
                    metrics.append(f'# TYPE blyan_daily_active_users gauge')
                    metrics.append(f'blyan_daily_active_users {dau or 0}')
                    
                    # Revenue metrics
                    revenue = await conn.fetchrow("""
                        SELECT 
                            SUM(amount) FILTER (WHERE tx_type = 'charge') as charges,
                            SUM(amount) FILTER (WHERE tx_type = 'reward') as rewards
                        FROM transactions
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                        AND status = 'credited'
                    """)
                    
                    if revenue:
                        metrics.append(f'# HELP blyan_daily_revenue_bly Daily revenue in BLY')
                        metrics.append(f'# TYPE blyan_daily_revenue_bly gauge')
                        metrics.append(f'blyan_daily_revenue_bly{{type="charges"}} {float(revenue["charges"] or 0)}')
                        metrics.append(f'blyan_daily_revenue_bly{{type="rewards"}} {float(revenue["rewards"] or 0)}')
                    
                    # Dataset contributions
                    from backend.data.podl_score_system import get_podl_recorder
                    recorder = get_podl_recorder()
                    
                    validated_count = sum(
                        1 for c in recorder.contributions.values()
                        if c.validation_status == "validated"
                    )
                    
                    metrics.append(f'# HELP blyan_datasets_total Total dataset contributions')
                    metrics.append(f'# TYPE blyan_datasets_total counter')
                    metrics.append(f'blyan_datasets_total{{status="validated"}} {validated_count}')
                    metrics.append(f'blyan_datasets_total{{status="all"}} {len(recorder.contributions)}')
                    
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
            metrics.append(f'# Business metrics unavailable: {e}')
        
        return metrics

# FastAPI Router
from fastapi import APIRouter, Response

metrics_router = APIRouter(prefix="/metrics", tags=["monitoring"])

@metrics_router.get("", response_class=Response)
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Scrape this endpoint for monitoring.
    """
    collector = MetricsCollector()
    metrics_text = await collector.collect_all_metrics()
    
    return Response(
        content=metrics_text,
        media_type="text/plain; version=0.0.4",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@metrics_router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns 200 if healthy, 503 if unhealthy.
    """
    checks = {
        "database": False,
        "api": True,
        "disk_space": False,
        "memory": False
    }
    
    # Check database
    try:
        from backend.accounting.postgres_ledger import get_postgres_ledger
        ledger = get_postgres_ledger()
        if ledger.pool:
            async with ledger.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
                checks["database"] = True
    except:
        pass
    
    # Check disk space
    disk = psutil.disk_usage('/')
    if disk.percent < 90:  # Less than 90% used
        checks["disk_space"] = True
    
    # Check memory
    memory = psutil.virtual_memory()
    if memory.percent < 90:  # Less than 90% used
        checks["memory"] = True
    
    # Overall health
    is_healthy = all(checks.values())
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "uptime_seconds": int(time.time() - MetricsCollector().start_time)
    }

@metrics_router.get("/readiness")
async def readiness_check():
    """
    Readiness check for Kubernetes.
    Returns 200 if ready to serve traffic.
    """
    try:
        from backend.accounting.postgres_ledger import get_postgres_ledger
        ledger = get_postgres_ledger()
        
        if not ledger.pool:
            return {"ready": False, "reason": "Database not connected"}, 503
        
        # Check if we can query database
        async with ledger.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {"ready": True}
        
    except Exception as e:
        return {"ready": False, "reason": str(e)}, 503

@metrics_router.get("/liveness")
async def liveness_check():
    """
    Liveness check for Kubernetes.
    Returns 200 if the application is alive.
    """
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}