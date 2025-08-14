#!/usr/bin/env python3
"""Setup cron jobs for pipeline maintenance"""
from __future__ import annotations

import subprocess
import os

def setup_cron():
    cron_entries = [
        # TLS cert renewal (daily 2am)
        "0 2 * * * /usr/bin/python3 /path/to/aiblock/scripts/ssl_manager.py",
        # Memory monitoring (every 5 min)
        "*/5 * * * * /usr/bin/python3 /path/to/aiblock/scripts/monitor_memory_vram.py >> /var/log/blyan_memory.log",
        # Pipeline metrics snapshot (hourly)
        "0 * * * * curl -s http://localhost:8000/metrics > /tmp/blyan_metrics_$(date +\\%Y\\%m\\%d_\\%H).txt",
    ]
    
    # Get current cron
    try:
        current = subprocess.check_output(["crontab", "-l"], text=True)
    except subprocess.CalledProcessError:
        current = ""
    
    # Add entries if not present
    new_entries = []
    for entry in cron_entries:
        if entry.split("* * *")[1].strip() not in current:
            new_entries.append(entry)
    
    if new_entries:
        all_cron = current + "\n" + "\n".join(new_entries) + "\n"
        proc = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
        proc.communicate(input=all_cron)
        print(f"Added {len(new_entries)} cron entries")
    else:
        print("All cron entries already present")

if __name__ == "__main__":
    setup_cron()