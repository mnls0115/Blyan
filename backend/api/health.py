"""
Health Check Endpoints for Production Monitoring
Comprehensive status checks for all critical components
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import asyncio
try:
    from redis import asyncio as aioredis
except ImportError:
    # Fallback for older versions
    try:
        import aioredis
    except ImportError:
        aioredis = None
import asyncpg
import os
import time
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

class HealthChecker:
    """Comprehensive health checking for all services"""
    
    def __init__(self):
        self.checks_performed = 0
        self.last_check = None
        
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        if aioredis is None:
            return {
                "status": "unavailable",
                "message": "Redis client not installed"
            }
            
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_password = os.getenv("REDIS_PASSWORD")
            
            start_time = time.time()
            
            # Handle both new redis.asyncio and old aioredis API
            if hasattr(aioredis, 'Redis'):
                # New redis.asyncio API
                redis = aioredis.Redis.from_url(
                    redis_url,
                    password=redis_password,
                    decode_responses=True
                )
            else:
                # Old aioredis API
                redis = await aioredis.from_url(
                    redis_url,
                    password=redis_password,
                    decode_responses=True
                )
            
            # Test basic operations
            await redis.set("health_check", str(time.time()), ex=60)
            value = await redis.get("health_check")
            
            # Get Redis info
            info = await redis.info()
            
            latency = (time.time() - start_time) * 1000
            
            await redis.close()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "uptime_days": info.get("uptime_in_days", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_postgresql(self) -> Dict[str, Any]:
        """Check PostgreSQL connectivity and status"""
        try:
            db_url = os.getenv(
                "DATABASE_URL",
                "postgresql://blyan_user:changeMe456@localhost:5432/blyan_db"
            )
            
            start_time = time.time()
            conn = await asyncpg.connect(db_url)
            
            # Test query
            result = await conn.fetchval("SELECT COUNT(*) FROM ledger.accounts")
            
            # Check ledger balance
            balance_check = await conn.fetchrow("""
                SELECT 
                    COALESCE(SUM(debit), 0) as total_debits,
                    COALESCE(SUM(credit), 0) as total_credits
                FROM ledger.entries
            """)
            
            latency = (time.time() - start_time) * 1000
            
            await conn.close()
            
            is_balanced = (
                balance_check['total_debits'] == balance_check['total_credits']
                if balance_check else True
            )
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "accounts_count": result,
                "ledger_balanced": is_balanced,
                "total_transactions": balance_check['total_debits'] if balance_check else 0
            }
            
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_teacher_model(self) -> Dict[str, Any]:
        """Check Teacher Model availability"""
        try:
            from backend.model.teacher_loader import get_teacher_loader
            
            teacher = get_teacher_loader()
            health = teacher.health_check()
            
            return {
                "status": health.get("status", "unknown"),
                "model_loaded": health.get("model_loaded", False),
                "version": health.get("version", "unknown"),
                "metrics": health.get("metrics", {})
            }
            
        except Exception as e:
            logger.error(f"Teacher model health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False
            }
    
    async def check_external_apis(self) -> Dict[str, Any]:
        """Check external API configuration"""
        apis = {}
        
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "")
        apis["openai"] = {
            "configured": bool(openai_key and not openai_key.startswith("sk-...")),
            "rate_limit": os.getenv("OPENAI_RATE_LIMIT", "100"),
            "monthly_quota": os.getenv("OPENAI_MONTHLY_QUOTA", "10000")
        }
        
        # Perspective API
        perspective_key = os.getenv("PERSPECTIVE_API_KEY", "")
        apis["perspective"] = {
            "configured": bool(perspective_key and perspective_key != "..."),
            "rate_limit": os.getenv("PERSPECTIVE_RATE_LIMIT", "1000"),
            "monthly_quota": os.getenv("PERSPECTIVE_MONTHLY_QUOTA", "100000")
        }
        
        # Stripe
        stripe_key = os.getenv("STRIPE_SECRET_KEY", "")
        apis["stripe"] = {
            "configured": bool(stripe_key and not stripe_key.startswith("sk_live_...")),
            "webhook_configured": bool(os.getenv("STRIPE_WEBHOOK_SECRET"))
        }
        
        return apis
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Process info
            process = psutil.Process()
            process_info = {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "network_connections": connections,
                "process": process_info
            }
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return {"error": str(e)}
    
    async def comprehensive_check(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()
        
        # Run all checks in parallel
        results = await asyncio.gather(
            self.check_redis(),
            self.check_postgresql(),
            self.check_teacher_model(),
            self.check_external_apis(),
            self.check_system_resources(),
            return_exceptions=True
        )
        
        # Process results
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {
                "redis": results[0] if not isinstance(results[0], Exception) else {"status": "error", "error": str(results[0])},
                "postgresql": results[1] if not isinstance(results[1], Exception) else {"status": "error", "error": str(results[1])},
                "teacher_model": results[2] if not isinstance(results[2], Exception) else {"status": "error", "error": str(results[2])},
                "external_apis": results[3] if not isinstance(results[3], Exception) else {"status": "error", "error": str(results[3])},
                "system": results[4] if not isinstance(results[4], Exception) else {"status": "error", "error": str(results[4])}
            },
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Determine overall status
        critical_services = ["redis", "postgresql"]
        for service in critical_services:
            if health_status["checks"][service].get("status") != "healthy":
                health_status["overall_status"] = "unhealthy"
                break
        
        # Check for degraded status
        if health_status["overall_status"] == "healthy":
            if health_status["checks"]["teacher_model"].get("status") != "healthy":
                health_status["overall_status"] = "degraded"
        
        self.checks_performed += 1
        self.last_check = datetime.utcnow()
        
        return health_status

# Create singleton instance
health_checker = HealthChecker()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Blyan Network API",
        "timestamp": datetime.utcnow().isoformat(),
        "checks_performed": health_checker.checks_performed
    }

@router.get("/comprehensive")
async def comprehensive_health_check() -> Dict[str, Any]:
    """Comprehensive health check of all services"""
    return await health_checker.comprehensive_check()

@router.get("/redis")
async def redis_health() -> Dict[str, Any]:
    """Check Redis health"""
    return await health_checker.check_redis()

@router.get("/postgresql")
async def postgresql_health() -> Dict[str, Any]:
    """Check PostgreSQL health"""
    return await health_checker.check_postgresql()

@router.get("/teacher")
async def teacher_health() -> Dict[str, Any]:
    """Check Teacher Model health"""
    return await health_checker.check_teacher_model()

@router.get("/apis")
async def external_apis_health() -> Dict[str, Any]:
    """Check external API configuration"""
    return await health_checker.check_external_apis()

@router.get("/system")
async def system_health() -> Dict[str, Any]:
    """Check system resource usage"""
    return await health_checker.check_system_resources()

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint"""
    health = await health_checker.comprehensive_check()
    
    if health["overall_status"] == "unhealthy":
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "ready": True,
        "status": health["overall_status"]
    }

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint"""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }