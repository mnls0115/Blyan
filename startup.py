#!/usr/bin/env python3
"""
Blyan Network Startup Script
Pre-loads models and initializes services before API starts

### PRODUCTION STARTUP SEQUENCE ###
1. Load environment variables
2. Pre-load Teacher model with health check
3. Initialize metrics collector
4. Start background tasks
5. Launch API server
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check and set required environment variables."""
    logger.info("Checking environment variables...")
    
    # Set defaults if not present
    defaults = {
        "BLYAN_TEACHER_CKPT": "/models/teacher_v17-int8.safetensors",
        "MIN_APPROVAL_RATE": "0.30",
        "L1_THRESH_START": "0.65",
        "PYTHONPATH": str(Path(__file__).parent.absolute())
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
            
    # Check optional API keys
    api_keys = ["OPENAI_API_KEY", "PERSPECTIVE_API_KEY", "STRIPE_SECRET_KEY"]
    for key in api_keys:
        if key in os.environ:
            logger.info(f"✓ {key} configured")
        else:
            logger.warning(f"✗ {key} not configured (some features disabled)")
            
def pre_load_models():
    """Pre-load all models to avoid first-call latency."""
    logger.info("Pre-loading models...")
    
    try:
        from backend.data.teacher_loader import get_teacher_loader
        
        # Pre-load Teacher model
        teacher = get_teacher_loader()
        success = teacher.pre_load()
        
        if success:
            health = teacher.get_health()
            logger.info(f"✅ Teacher model loaded successfully")
            logger.info(f"   Version: v{health['version']}")
            logger.info(f"   Memory: {health['memory_mb']:.1f}MB")
            logger.info(f"   Load time: {health['load_time_ms']:.1f}ms")
            logger.info(f"   Warmup time: {health['warmup_time_ms']:.1f}ms")
        else:
            logger.warning("Teacher model failed to load, running in fallback mode")
            
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")
        logger.warning("Continuing with lazy loading (may cause first-request delays)")
        
def initialize_metrics():
    """Initialize metrics collector and Prometheus export."""
    logger.info("Initializing metrics collector...")
    
    try:
        from backend.data.metrics_collector import get_metrics_collector
        
        collector = get_metrics_collector()
        logger.info("✅ Metrics collector initialized")
        
        # Check if Prometheus is available
        try:
            from prometheus_client import start_http_server
            # Start Prometheus metrics server on port 9090
            start_http_server(9090)
            logger.info("✅ Prometheus metrics server started on :9090/metrics")
        except ImportError:
            logger.warning("Prometheus client not installed, metrics export disabled")
            
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        
def initialize_api_clients():
    """Initialize external API clients with circuit breakers."""
    logger.info("Initializing API clients...")
    
    try:
        from backend.data.resilient_api import get_perspective_client, get_openai_client
        
        # Initialize clients (they're singletons)
        if os.environ.get("PERSPECTIVE_API_KEY"):
            perspective = get_perspective_client()
            logger.info("✅ Perspective API client initialized")
            
        if os.environ.get("OPENAI_API_KEY"):
            openai = get_openai_client()
            logger.info("✅ OpenAI API client initialized")
            
    except Exception as e:
        logger.error(f"Failed to initialize API clients: {e}")
        
async def start_background_tasks():
    """Start background tasks for monitoring and maintenance."""
    logger.info("Starting background tasks...")
    
    tasks = []
    
    # Start pipeline monitoring
    try:
        from backend.data.pipeline_monitor import monitoring_loop
        task = asyncio.create_task(monitoring_loop())
        tasks.append(task)
        logger.info("✅ Pipeline monitoring started")
    except Exception as e:
        logger.error(f"Failed to start pipeline monitoring: {e}")
        
    # Start queue processing for resilient APIs
    try:
        from backend.data.resilient_api import get_perspective_client
        
        async def process_api_queues():
            while True:
                try:
                    perspective = get_perspective_client()
                    await perspective.process_queue()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Queue processing error: {e}")
                    await asyncio.sleep(60)
                    
        task = asyncio.create_task(process_api_queues())
        tasks.append(task)
        logger.info("✅ API queue processor started")
    except Exception as e:
        logger.error(f"Failed to start queue processor: {e}")
        
    return tasks

def run_health_checks():
    """Run comprehensive health checks."""
    logger.info("Running health checks...")
    
    health_status = {
        "teacher_model": False,
        "metrics": False,
        "api_clients": False,
        "database": False
    }
    
    # Check Teacher model
    try:
        from backend.data.teacher_loader import get_teacher_loader
        teacher = get_teacher_loader()
        health = teacher.get_health()
        health_status["teacher_model"] = health["loaded"]
    except:
        pass
        
    # Check metrics
    try:
        from backend.data.metrics_collector import get_metrics_collector
        collector = get_metrics_collector()
        health_status["metrics"] = True
    except:
        pass
        
    # Check API clients
    try:
        from backend.data.resilient_api import get_perspective_client
        client = get_perspective_client()
        stats = client.get_metrics()
        health_status["api_clients"] = stats["circuit_breaker"]["state"] != "open"
    except:
        pass
        
    # Print health summary
    logger.info("=== HEALTH CHECK SUMMARY ===")
    for component, status in health_status.items():
        icon = "✅" if status else "❌"
        logger.info(f"{icon} {component}: {'healthy' if status else 'unhealthy'}")
        
    # Overall status
    all_healthy = all(health_status.values())
    if all_healthy:
        logger.info("✅ All systems operational")
    else:
        logger.warning("⚠️ Some components unhealthy, but continuing...")
        
    return all_healthy

async def main():
    """Main startup sequence."""
    logger.info("=== BLYAN NETWORK STARTUP ===")
    start_time = time.time()
    
    # 1. Check environment
    check_environment()
    
    # 2. Pre-load models
    pre_load_models()
    
    # 3. Initialize metrics
    initialize_metrics()
    
    # 4. Initialize API clients
    initialize_api_clients()
    
    # 5. Run health checks
    health_ok = run_health_checks()
    
    # 6. Start background tasks
    background_tasks = await start_background_tasks()
    
    # Calculate startup time
    startup_time = time.time() - start_time
    logger.info(f"✅ Startup complete in {startup_time:.1f}s")
    
    # Now start the API server
    logger.info("Starting API server...")
    
    # Import and run the API server
    try:
        # Check if running from server.sh
        if os.environ.get("BLYAN_API_MODE") == "true":
            from api.server import app
            import uvicorn
            
            # Run with production settings
            await uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=8000,
                    log_level="info",
                    access_log=True,
                    reload=False  # No reload in production
                )
            ).serve()
        else:
            # Just keep background tasks running
            logger.info("Running in background mode (no API server)")
            await asyncio.gather(*background_tasks)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for task in background_tasks:
            task.cancel()
            
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the startup sequence
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Startup interrupted")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)