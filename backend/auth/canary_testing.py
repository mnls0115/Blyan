"""
Canary Testing System for V2 API Keys
=====================================

Automated testing of V2 flows in production with synthetic users.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TestResult(str, Enum):
    """Test result states"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

@dataclass
class CanaryTest:
    """Individual canary test definition"""
    test_id: str
    name: str
    endpoint: str
    method: str
    payload: Optional[Dict[str, Any]]
    expected_status: int
    timeout: float
    critical: bool  # If true, failure triggers alert
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TestRun:
    """Test run result"""
    test_id: str
    timestamp: str
    result: TestResult
    latency_ms: float
    status_code: Optional[int]
    error: Optional[str]
    metadata: Dict[str, Any]

class CanaryTestSuite:
    """
    Automated canary testing for V2 API key system.
    
    Runs synthetic tests continuously to detect issues early.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.canary_user_id = "canary_test_user_" + hashlib.md5(
            f"{base_url}_canary".encode()
        ).hexdigest()[:8]
        
        self.canary_api_key = None
        self.test_results = []
        self.last_run = None
        self.consecutive_failures = 0
        
        # Define test suite
        self.tests = self._define_tests()
        
        # Test configuration
        self.run_interval = 60  # Run every minute
        self.alert_threshold = 3  # Alert after 3 consecutive failures
        self.is_running = False
    
    def _define_tests(self) -> List[CanaryTest]:
        """Define the canary test suite"""
        return [
            # V2 Registration
            CanaryTest(
                test_id="v2_register",
                name="V2 API Key Registration",
                endpoint="/api/auth/v2/register",
                method="POST",
                payload={
                    "name": f"canary_{int(time.time())}",
                    "key_type": "basic",
                    "metadata": {"canary": True}
                },
                expected_status=200,
                timeout=5.0,
                critical=True
            ),
            
            # V2 Validation
            CanaryTest(
                test_id="v2_validate",
                name="V2 API Key Validation",
                endpoint="/api/auth/v2/validate",
                method="GET",
                payload=None,
                expected_status=200,
                timeout=2.0,
                critical=True
            ),
            
            # V2 Refresh (will fail early, but should return 400 not 500)
            CanaryTest(
                test_id="v2_refresh",
                name="V2 API Key Refresh",
                endpoint="/api/auth/v2/refresh",
                method="POST",
                payload=None,  # Will be set dynamically
                expected_status=400,  # Too early to refresh
                timeout=5.0,
                critical=False
            ),
            
            # Chat with V2 Auth
            CanaryTest(
                test_id="chat_v2",
                name="Chat with V2 Authentication",
                endpoint="/api/chat",
                method="POST",
                payload={
                    "prompt": "Canary test message",
                    "max_tokens": 10
                },
                expected_status=200,
                timeout=10.0,
                critical=True
            ),
            
            # Rate Limit Check
            CanaryTest(
                test_id="rate_limit",
                name="Rate Limit Enforcement",
                endpoint="/api/auth/v2/validate",
                method="GET",
                payload=None,
                expected_status=200,  # Should pass under limit
                timeout=1.0,
                critical=False
            )
        ]
    
    async def setup_canary_user(self) -> bool:
        """Set up canary test user with V2 API key"""
        try:
            async with aiohttp.ClientSession() as session:
                # Register canary API key
                async with session.post(
                    f"{self.base_url}/api/auth/v2/register",
                    json={
                        "name": f"canary_{self.canary_user_id}",
                        "key_type": "basic",
                        "metadata": {
                            "canary": True,
                            "created_at": datetime.now().isoformat()
                        }
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.canary_api_key = data.get("api_key")
                        logger.info(f"Canary user setup complete: {self.canary_user_id}")
                        return True
                    else:
                        logger.error(f"Failed to setup canary user: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Canary setup error: {e}")
            return False
    
    async def run_test(self, test: CanaryTest) -> TestRun:
        """Run a single canary test"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare request
                headers = {"Content-Type": "application/json"}
                
                # Add auth for tests that need it
                if test.test_id != "v2_register" and self.canary_api_key:
                    headers["Authorization"] = f"Bearer {self.canary_api_key}"
                
                # Special handling for refresh test
                if test.test_id == "v2_refresh":
                    test.payload = {"current_key": self.canary_api_key}
                
                # Make request
                async with session.request(
                    test.method,
                    f"{self.base_url}{test.endpoint}",
                    json=test.payload if test.method in ["POST", "PUT"] else None,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=test.timeout)
                ) as resp:
                    latency = (time.time() - start_time) * 1000
                    
                    # Check result
                    if resp.status == test.expected_status:
                        result = TestResult.PASS
                        error = None
                        
                        # Store new key if registration test
                        if test.test_id == "v2_register":
                            data = await resp.json()
                            self.canary_api_key = data.get("api_key")
                    else:
                        result = TestResult.FAIL
                        error = f"Expected {test.expected_status}, got {resp.status}"
                        
                        # Try to get error details
                        try:
                            error_data = await resp.json()
                            error = error_data.get("detail", error)
                        except:
                            pass
                    
                    return TestRun(
                        test_id=test.test_id,
                        timestamp=datetime.now().isoformat(),
                        result=result,
                        latency_ms=latency,
                        status_code=resp.status,
                        error=error,
                        metadata={"test_name": test.name}
                    )
                    
        except asyncio.TimeoutError:
            return TestRun(
                test_id=test.test_id,
                timestamp=datetime.now().isoformat(),
                result=TestResult.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                status_code=None,
                error="Timeout",
                metadata={"test_name": test.name}
            )
        except Exception as e:
            return TestRun(
                test_id=test.test_id,
                timestamp=datetime.now().isoformat(),
                result=TestResult.ERROR,
                latency_ms=(time.time() - start_time) * 1000,
                status_code=None,
                error=str(e),
                metadata={"test_name": test.name}
            )
    
    async def run_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting canary test suite")
        
        # Ensure canary user exists
        if not self.canary_api_key:
            if not await self.setup_canary_user():
                return {
                    "success": False,
                    "error": "Failed to setup canary user",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Run all tests
        results = []
        for test in self.tests:
            result = await self.run_test(test)
            results.append(result)
            
            # Log result
            if result.result == TestResult.PASS:
                logger.debug(f"✓ {test.name}: {result.latency_ms:.1f}ms")
            else:
                logger.warning(f"✗ {test.name}: {result.error}")
        
        # Calculate statistics
        total = len(results)
        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        errors = sum(1 for r in results if r.result == TestResult.ERROR)
        avg_latency = sum(r.latency_ms for r in results) / total if total > 0 else 0
        
        # Check for critical failures
        critical_failures = [
            r for r in results
            if r.result != TestResult.PASS and
            any(t.test_id == r.test_id and t.critical for t in self.tests)
        ]
        
        # Update consecutive failure counter
        if critical_failures:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Trigger alert if threshold reached
        if self.consecutive_failures >= self.alert_threshold:
            await self.trigger_alert(critical_failures)
        
        # Store results
        self.test_results.extend(results)
        # Keep only last 1000 results
        if len(self.test_results) > 1000:
            self.test_results = self.test_results[-1000:]
        
        self.last_run = datetime.now()
        
        return {
            "success": len(critical_failures) == 0,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
                "avg_latency_ms": avg_latency
            },
            "results": [asdict(r) for r in results],
            "critical_failures": [asdict(r) for r in critical_failures],
            "consecutive_failures": self.consecutive_failures
        }
    
    async def trigger_alert(self, failures: List[TestRun]):
        """Trigger alert for critical failures"""
        alert_message = f"""
        🚨 CANARY TEST ALERT 🚨
        
        Critical failures detected in V2 API Key system:
        
        Consecutive failures: {self.consecutive_failures}
        Failed tests: {', '.join(set(r.test_id for r in failures))}
        
        Details:
        """
        
        for failure in failures:
            alert_message += f"\n- {failure.test_id}: {failure.error}"
        
        logger.critical(alert_message)
        
        # In production, this would:
        # - Send PagerDuty alert
        # - Post to Slack #alerts channel
        # - Create incident ticket
        # - Potentially trigger automatic rollback
    
    async def start_continuous_testing(self):
        """Start continuous canary testing"""
        if self.is_running:
            logger.warning("Canary tests already running")
            return
        
        self.is_running = True
        logger.info("Starting continuous canary testing")
        
        while self.is_running:
            try:
                # Run test suite
                result = await self.run_suite()
                
                # Log summary
                stats = result["statistics"]
                logger.info(
                    f"Canary run complete: {stats['passed']}/{stats['total']} passed "
                    f"({stats['pass_rate']:.1f}%), avg latency: {stats['avg_latency_ms']:.1f}ms"
                )
                
                # Wait before next run
                await asyncio.sleep(self.run_interval)
                
            except Exception as e:
                logger.error(f"Canary test error: {e}")
                await asyncio.sleep(self.run_interval)
    
    def stop_testing(self):
        """Stop continuous testing"""
        self.is_running = False
        logger.info("Stopping canary tests")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get canary health report"""
        if not self.test_results:
            return {
                "status": "no_data",
                "message": "No canary tests have been run yet"
            }
        
        # Get recent results (last 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        recent_results = [
            r for r in self.test_results
            if datetime.fromisoformat(r.timestamp) > cutoff
        ]
        
        if not recent_results:
            return {
                "status": "stale",
                "message": "No recent test results",
                "last_run": self.last_run.isoformat() if self.last_run else None
            }
        
        # Calculate health
        total = len(recent_results)
        passed = sum(1 for r in recent_results if r.result == TestResult.PASS)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Determine status
        if pass_rate >= 95:
            status = "healthy"
        elif pass_rate >= 80:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "pass_rate": pass_rate,
            "recent_tests": total,
            "consecutive_failures": self.consecutive_failures,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "canary_user": self.canary_user_id
        }

# Global instance
canary_suite = CanaryTestSuite()

"""
Usage:

# Start canary testing in background
asyncio.create_task(canary_suite.start_continuous_testing())

# Get health report
health = canary_suite.get_health_report()
print(f"Canary health: {health['status']} ({health['pass_rate']:.1f}% pass rate)")

# Run single test suite manually
result = await canary_suite.run_suite()
if not result["success"]:
    print(f"Canary failures: {result['critical_failures']}")
"""