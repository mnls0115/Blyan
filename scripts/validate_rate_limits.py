#!/usr/bin/env python3
"""
Rate Limit and Free Tier Consistency Validator
Ensures SSOT (Single Source of Truth) alignment
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from decimal import Decimal

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class RateLimitValidator:
    """Validates rate limit consistency across the system."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.config_sources = {}
    
    def load_configurations(self):
        """Load all rate limit configurations from various sources."""
        
        # 1. Load rate limiter config
        try:
            from backend.security.rate_limiting import rate_limiter
            self.config_sources['rate_limiter'] = {
                'burst_limit': getattr(rate_limiter, 'burst_limit', None),
                'sustained_limit': getattr(rate_limiter, 'sustained_limit', None),
                'window_size': getattr(rate_limiter, 'window_size', None)
            }
        except ImportError as e:
            self.warnings.append(f"Could not load rate_limiter: {e}")
        
        # 2. Load abuse prevention config
        try:
            from backend.security.abuse_prevention import get_abuse_prevention_system
            abuse_system = get_abuse_prevention_system()
            self.config_sources['abuse_prevention'] = {
                'max_requests_per_minute': 60,  # Default
                'max_requests_per_hour': 1000,  # Default
                'max_requests_per_day': 10000   # Default
            }
        except ImportError as e:
            self.warnings.append(f"Could not load abuse_prevention: {e}")
        
        # 3. Load free tier config from Redis SSOT
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            
            # Get sample user to check structure
            test_key = "user:limits:test"
            if r.exists(test_key):
                limits = r.hgetall(test_key)
                self.config_sources['redis_ssot'] = {
                    'daily_free_requests': int(limits.get(b'daily_free_requests', 5)),
                    'trust_level': limits.get(b'trust_level', b'new').decode(),
                    'burst_limit': int(limits.get(b'burst_limit', 10))
                }
            else:
                # Set default structure
                self.config_sources['redis_ssot'] = {
                    'daily_free_requests': 5,
                    'trust_level': 'new',
                    'burst_limit': 10
                }
        except Exception as e:
            self.warnings.append(f"Could not connect to Redis: {e}")
        
        # 4. Load environment variables
        import os
        self.config_sources['environment'] = {
            'RATE_LIMIT_BURST': int(os.getenv('RATE_LIMIT_BURST', '10')),
            'RATE_LIMIT_SUSTAINED': int(os.getenv('RATE_LIMIT_SUSTAINED', '60')),
            'FREE_TIER_DAILY': int(os.getenv('FREE_TIER_DAILY', '5')),
            'FREE_TIER_TRUSTED': int(os.getenv('FREE_TIER_TRUSTED', '50'))
        }
        
        # 5. Load from config files if they exist
        config_file = Path("config/rate_limits.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config_sources['config_file'] = json.load(f)
        else:
            self.warnings.append(f"Config file not found: {config_file}")
    
    def check_free_tier_consistency(self):
        """Check free tier limits are consistent."""
        
        # Define expected tiers
        expected_tiers = {
            'new': {'daily': 5, 'burst': 10},
            'regular': {'daily': 20, 'burst': 20},
            'trusted': {'daily': 50, 'burst': 50},
            'contributor': {'daily': 200, 'burst': 100},
            'premium': {'daily': 500, 'burst': 200}
        }
        
        # Check Redis SSOT
        if 'redis_ssot' in self.config_sources:
            redis_config = self.config_sources['redis_ssot']
            trust_level = redis_config.get('trust_level', 'new')
            daily_limit = redis_config.get('daily_free_requests', 0)
            
            expected_daily = expected_tiers.get(trust_level, {}).get('daily', 5)
            
            if daily_limit != expected_daily:
                self.issues.append(
                    f"Free tier mismatch: {trust_level} user has {daily_limit} daily requests, "
                    f"expected {expected_daily}"
                )
        
        # Check environment vs config
        env_daily = self.config_sources.get('environment', {}).get('FREE_TIER_DAILY', 5)
        
        if 'config_file' in self.config_sources:
            config_daily = self.config_sources['config_file'].get('free_tier', {}).get('daily', 5)
            
            if env_daily != config_daily:
                self.issues.append(
                    f"Environment FREE_TIER_DAILY ({env_daily}) doesn't match "
                    f"config file ({config_daily})"
                )
    
    def check_burst_limits(self):
        """Check burst limits are properly configured."""
        
        burst_values = []
        
        # Collect all burst limit values
        if 'rate_limiter' in self.config_sources:
            burst = self.config_sources['rate_limiter'].get('burst_limit')
            if burst:
                burst_values.append(('rate_limiter', burst))
        
        if 'redis_ssot' in self.config_sources:
            burst = self.config_sources['redis_ssot'].get('burst_limit')
            if burst:
                burst_values.append(('redis_ssot', burst))
        
        env_burst = self.config_sources.get('environment', {}).get('RATE_LIMIT_BURST')
        if env_burst:
            burst_values.append(('environment', env_burst))
        
        # Check for inconsistencies
        if burst_values:
            unique_values = set(v for _, v in burst_values)
            
            if len(unique_values) > 1:
                self.issues.append(
                    f"Inconsistent burst limits found: {burst_values}"
                )
            
            # Check burst is reasonable
            for source, value in burst_values:
                if value < 5:
                    self.warnings.append(
                        f"Burst limit too low in {source}: {value} (may impact UX)"
                    )
                elif value > 1000:
                    self.warnings.append(
                        f"Burst limit too high in {source}: {value} (potential abuse risk)"
                    )
    
    def check_daily_limits(self):
        """Check daily request limits."""
        
        # Check progression makes sense
        if 'redis_ssot' in self.config_sources:
            redis_config = self.config_sources['redis_ssot']
            daily = redis_config.get('daily_free_requests', 0)
            burst = redis_config.get('burst_limit', 0)
            
            if burst > daily:
                self.warnings.append(
                    f"Burst limit ({burst}) exceeds daily limit ({daily}) - "
                    "users could exhaust daily quota in one burst"
                )
            
            if daily > 0 and burst == 0:
                self.issues.append(
                    "Daily limit set but burst limit is 0 - requests will be blocked"
                )
    
    def generate_report(self) -> Tuple[bool, str]:
        """Generate validation report."""
        
        report = []
        report.append("=" * 60)
        report.append("RATE LIMIT CONSISTENCY VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Configuration sources
        report.append("CONFIGURATION SOURCES FOUND:")
        for source, config in self.config_sources.items():
            report.append(f"  ✓ {source}")
            for key, value in config.items():
                if value is not None:
                    report.append(f"    - {key}: {value}")
        report.append("")
        
        # Issues (critical)
        if self.issues:
            report.append(f"CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                report.append(f"  ✗ {issue}")
            report.append("")
        else:
            report.append("✓ No critical issues found")
            report.append("")
        
        # Warnings
        if self.warnings:
            report.append(f"WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                report.append(f"  ⚠ {warning}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        if not self.config_sources.get('redis_ssot'):
            report.append("  1. Set up Redis SSOT for centralized rate limit management")
        
        if 'config_file' not in self.config_sources:
            report.append("  2. Create config/rate_limits.json for version-controlled limits")
        
        if self.issues:
            report.append("  3. Fix critical issues before production deployment")
        
        report.append("  4. Implement automated tests for rate limit changes")
        report.append("  5. Set up monitoring alerts for rate limit violations")
        report.append("")
        
        # Summary
        is_valid = len(self.issues) == 0
        status = "PASSED ✓" if is_valid else "FAILED ✗"
        
        report.append(f"VALIDATION STATUS: {status}")
        report.append(f"  Issues: {len(self.issues)}")
        report.append(f"  Warnings: {len(self.warnings)}")
        report.append("")
        report.append("=" * 60)
        
        return is_valid, '\n'.join(report)
    
    def write_config_template(self):
        """Write a template configuration file."""
        
        template = {
            "rate_limits": {
                "burst": 10,
                "sustained": 60,
                "window_seconds": 60
            },
            "free_tier": {
                "tiers": {
                    "new": {
                        "daily_requests": 5,
                        "burst_limit": 10,
                        "days_to_advance": 7
                    },
                    "regular": {
                        "daily_requests": 20,
                        "burst_limit": 20,
                        "days_to_advance": 30
                    },
                    "trusted": {
                        "daily_requests": 50,
                        "burst_limit": 50,
                        "days_to_advance": 90
                    },
                    "contributor": {
                        "daily_requests": 200,
                        "burst_limit": 100,
                        "requirement": "5_contributions"
                    },
                    "premium": {
                        "daily_requests": 500,
                        "burst_limit": 200,
                        "requirement": "paid_subscription"
                    }
                }
            },
            "abuse_prevention": {
                "max_requests_per_minute": 60,
                "max_requests_per_hour": 1000,
                "max_requests_per_day": 10000,
                "challenge_threshold": 0.8,
                "ban_threshold": 0.95
            }
        }
        
        config_file = Path("config/rate_limits.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✓ Template configuration written to {config_file}")

def main():
    """Main validation entry point."""
    
    validator = RateLimitValidator()
    
    print("Loading configurations...")
    validator.load_configurations()
    
    print("Checking consistency...")
    validator.check_free_tier_consistency()
    validator.check_burst_limits()
    validator.check_daily_limits()
    
    # Generate report
    is_valid, report = validator.generate_report()
    
    print(report)
    
    # Write template if needed
    if 'config_file' not in validator.config_sources:
        print("\nWriting configuration template...")
        validator.write_config_template()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()