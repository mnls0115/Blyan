"""
Progress tracking utility for long-running operations.
"""

import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and report progress for long operations."""
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 5.0  # Report every 5 seconds
        
    def update(self, items_done: int = 1, force_report: bool = False):
        """Update progress and report if needed."""
        self.processed_items += items_done
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Report if interval passed or forced
        if force_report or (current_time - self.last_report_time) >= self.report_interval:
            self.report_progress(elapsed)
            self.last_report_time = current_time
    
    def report_progress(self, elapsed: Optional[float] = None):
        """Report current progress with ETA."""
        if elapsed is None:
            elapsed = time.time() - self.start_time
        
        if self.processed_items == 0:
            logger.info(f"⏳ {self.operation_name}: Starting...")
            return
        
        # Calculate rate and ETA
        rate = self.processed_items / elapsed if elapsed > 0 else 0
        remaining = self.total_items - self.processed_items
        eta_seconds = remaining / rate if rate > 0 else 0
        
        # Format times
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)
        
        # Calculate percentage
        percentage = (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0
        
        # Create progress bar
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        logger.info(
            f"⏳ {self.operation_name}: [{bar}] {percentage:.1f}% "
            f"({self.processed_items}/{self.total_items}) "
            f"[{elapsed_str} / ETA: {eta_str}] "
            f"({rate:.1f} items/sec)"
        )
    
    def finish(self):
        """Report final completion."""
        total_time = time.time() - self.start_time
        time_str = self._format_time(total_time)
        
        rate = self.processed_items / total_time if total_time > 0 else 0
        
        logger.info(
            f"✅ {self.operation_name} complete: "
            f"{self.processed_items} items in {time_str} "
            f"({rate:.1f} items/sec)"
        )
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 0:
            return "??:??"
        
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class StepTracker:
    """Track multi-step operations."""
    
    def __init__(self, steps: list, name: str = "Operation"):
        self.steps = steps
        self.name = name
        self.current_step = 0
        self.total_steps = len(steps)
        self.start_time = time.time()
        self.step_times = []
        
    def next_step(self) -> str:
        """Move to next step and return its name."""
        if self.current_step < self.total_steps:
            step_name = self.steps[self.current_step]
            self.current_step += 1
            
            elapsed = time.time() - self.start_time
            elapsed_str = ProgressTracker._format_time(elapsed)
            
            # Estimate remaining time based on average step time
            if self.step_times:
                avg_step_time = sum(self.step_times) / len(self.step_times)
                remaining_steps = self.total_steps - self.current_step + 1
                eta = avg_step_time * remaining_steps
                eta_str = ProgressTracker._format_time(eta)
                
                logger.info(
                    f"\n📍 Step {self.current_step}/{self.total_steps}: {step_name} "
                    f"[{elapsed_str} / ETA: {eta_str}]"
                )
            else:
                logger.info(
                    f"\n📍 Step {self.current_step}/{self.total_steps}: {step_name} "
                    f"[{elapsed_str}]"
                )
            
            logger.info("-" * 60)
            
            # Track step timing
            if self.current_step > 1:
                step_time = elapsed - sum(self.step_times)
                self.step_times.append(step_time)
            
            return step_name
        return ""
    
    def finish(self):
        """Report completion."""
        total_time = time.time() - self.start_time
        time_str = ProgressTracker._format_time(total_time)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✅ {self.name} COMPLETE")
        logger.info(f"⏱️  Total time: {time_str}")
        logger.info(f"📊 Steps completed: {self.current_step}/{self.total_steps}")
        logger.info("=" * 60)