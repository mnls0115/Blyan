#!/usr/bin/env python3
"""
Supply consistency verification script
Runs daily to ensure all token supply numbers add up correctly
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.reward_engine_secure import get_secure_reward_engine

def main():
    """Verify supply consistency and log results."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        engine = get_secure_reward_engine()
        result = engine.verify_supply_consistency()
        
        if result['is_consistent']:
            logging.info("✅ Supply consistency check PASSED")
            logging.info(f"Total cap: {result['total_cap']:,}")
            logging.info(f"Total accounted: {result['total_accounted']:,}")
            logging.info(f"Difference: {result['difference']:,}")
        else:
            logging.error("❌ Supply consistency check FAILED")
            logging.error(f"Difference: {result['difference']:,}")
            logging.error(f"Breakdown: {result['breakdown']}")
            
            # Send alert (in production, use proper alerting)
            print("ALERT: Supply inconsistency detected!")
            sys.exit(1)
            
        # Log breakdown for monitoring
        breakdown = result['breakdown']
        logging.info(f"Circulating: {breakdown['circulating']:,}")
        logging.info(f"Locked: {breakdown['locked']:,}")
        logging.info(f"Foundation: {breakdown['foundation']:,}")
        logging.info(f"Remaining rewards: {breakdown['remaining_rewards']:,}")
        
    except Exception as e:
        logging.error(f"Supply verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()