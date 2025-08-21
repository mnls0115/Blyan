#!/usr/bin/env python3
"""
Real-time monitoring dashboard for learning cycles
Shows burn accumulation, round progress, and node status
"""

import os
import sys
import time
import json
import asyncio
import httpx
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class LearningMonitor:
    """Real-time learning cycle monitor"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.last_metrics = None
        
    async def fetch_metrics(self):
        """Fetch current metrics from API"""
        try:
            response = await self.client.get(f"{self.api_url}/learning/metrics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Failed to fetch metrics: {e}")
        return None
    
    async def fetch_rounds(self):
        """Fetch active rounds"""
        try:
            response = await self.client.get(f"{self.api_url}/learning/rounds")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Failed to fetch rounds: {e}")
        return None
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_time_ago(self, timestamp_str: str) -> str:
        """Format timestamp as time ago"""
        if not timestamp_str:
            return "Never"
        
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            diff = datetime.utcnow() - dt.replace(tzinfo=None)
            
            if diff.total_seconds() < 60:
                return f"{int(diff.total_seconds())}s ago"
            elif diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() / 60)}m ago"
            elif diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds() / 3600)}h ago"
            else:
                return f"{int(diff.total_seconds() / 86400)}d ago"
        except:
            return timestamp_str
    
    def render_dashboard(self, metrics: dict, rounds: dict):
        """Render the monitoring dashboard"""
        
        self.clear_screen()
        
        print("=" * 80)
        print("🎓 BLYAN LEARNING CYCLE MONITOR".center(80))
        print("=" * 80)
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if not metrics:
            print("⚠️  No metrics available - API might be down")
            return
        
        # Burn Accumulator Section
        burn = metrics.get("burn_accumulator", {})
        print("🔥 BURN ACCUMULATOR")
        print("-" * 40)
        
        total = burn.get("total", 0)
        threshold = burn.get("threshold", 1000)
        progress = burn.get("progress_percent", 0)
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"Progress: [{bar}] {progress:.1f}%")
        print(f"Burned:   {total:.2f} / {threshold:.2f} BLY")
        print(f"Last Round: {self.format_time_ago(burn.get('last_round_time'))}")
        print()
        
        # Nodes Section
        nodes = metrics.get("nodes", {})
        print("🖥️  GPU NODES")
        print("-" * 40)
        print(f"Active Nodes: {nodes.get('active', 0)}")
        
        node_list = nodes.get("list", [])[:5]  # Show top 5
        if node_list:
            print("\nTop Nodes by Reputation:")
            for node in node_list:
                rep = node.get("reputation", 1.0)
                stars = "⭐" * int(rep * 5)
                print(f"  {node['node_id']}: {stars} ({rep:.2f})")
        print()
        
        # Active Rounds Section
        rounds_data = metrics.get("rounds", {})
        print("📚 LEARNING ROUNDS")
        print("-" * 40)
        print(f"Active Rounds: {rounds_data.get('active', 0)}")
        
        states = rounds_data.get("states", {})
        if states:
            print("\nRound States:")
            state_icons = {
                "TRIGGERED": "🎯",
                "NOTIFYING": "📢",
                "DATA_ALLOC": "📊",
                "TRAINING": "🏃",
                "CONSENSUS": "🤝",
                "DELTA_CREATION": "📦",
                "REWARD_DIST": "💰"
            }
            
            for state, count in states.items():
                if count > 0:
                    icon = state_icons.get(state, "•")
                    print(f"  {icon} {state}: {count}")
        
        # Detailed rounds info
        if rounds and rounds.get("active_rounds"):
            print("\nActive Round Details:")
            for round_info in rounds["active_rounds"][:3]:  # Show top 3
                round_id = round_info["round_id"][:8]
                state = round_info["state"]
                created = self.format_time_ago(round_info.get("created_at"))
                bly = float(round_info.get("bly_sum", 0))
                
                print(f"  [{round_id}] {state} - {bly:.2f} BLY - Started {created}")
        print()
        
        # Configuration Section
        config = metrics.get("config", {})
        if config:
            print("⚙️  CONFIGURATION")
            print("-" * 40)
            print(f"BLY Threshold:    {config.get('threshold_bly', 1000):.2f}")
            print(f"Min Interval:     {config.get('min_interval_seconds', 3600)}s")
            print(f"Quorum Required:  {config.get('quorum_threshold', 0.67)*100:.0f}%")
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to exit | Refreshing every 5 seconds...")
    
    async def run(self):
        """Run the monitoring loop"""
        try:
            while True:
                metrics = await self.fetch_metrics()
                rounds = await self.fetch_rounds()
                
                self.render_dashboard(metrics, rounds)
                
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
        finally:
            await self.client.aclose()


async def test_trigger(api_url: str):
    """Test function to manually trigger a learning round"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{api_url}/learning/trigger")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Learning round triggered!")
                print(f"   Amount: {data['amount']} BLY")
                print(f"   Status: {data['status']}")
            else:
                print(f"❌ Failed to trigger: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main entry point"""
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    if "--trigger" in sys.argv:
        # Manual trigger mode
        print("🔥 Manually triggering learning round...")
        await test_trigger(api_url)
    else:
        # Monitoring mode
        monitor = LearningMonitor(api_url)
        await monitor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)