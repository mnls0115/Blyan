"""Structured logging for P2P network"""
import logging
import json
import sys
import time
from typing import Any, Dict
from pathlib import Path
import traceback


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_obj = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'peer_id'):
            log_obj['peer_id'] = record.peer_id
        
        if hasattr(record, 'block_hash'):
            log_obj['block_hash'] = record.block_hash
            
        if hasattr(record, 'chain_height'):
            log_obj['chain_height'] = record.chain_height
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_obj)


class P2PLogger:
    """P2P network logger with structured output"""
    
    def __init__(self, name: str, level: str = 'INFO', log_file: Path = None):
        """
        Initialize logger
        
        Args:
            name: Logger name
            level: Log level
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Debug level log"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level log"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level log"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Error level log"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical level log"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)


# Component loggers
def get_logger(component: str, level: str = None) -> P2PLogger:
    """
    Get logger for component
    
    Args:
        component: Component name
        level: Optional log level override
        
    Returns:
        Configured logger
    """
    # Determine log level from environment or default
    import os
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Determine log file
    log_dir = Path(os.getenv('LOG_DIR', 'logs'))
    log_file = log_dir / f"{component}.log"
    
    return P2PLogger(f"blyan.{component}", level, log_file)


# Pre-configured loggers
chain_logger = get_logger('chain')
network_logger = get_logger('network')
consensus_logger = get_logger('consensus')
dht_logger = get_logger('dht')
security_logger = get_logger('security')
sync_logger = get_logger('sync')


# Log aggregator for analysis
class LogAggregator:
    """Aggregates logs for analysis"""
    
    def __init__(self):
        self.events: Dict[str, list] = {
            'connections': [],
            'blocks': [],
            'messages': [],
            'errors': [],
            'security': [],
        }
        self.max_events = 1000
    
    def add_event(self, category: str, event: Dict[str, Any]):
        """Add event to aggregator"""
        if category not in self.events:
            self.events[category] = []
        
        self.events[category].append({
            'timestamp': time.time(),
            **event
        })
        
        # Trim old events
        if len(self.events[category]) > self.max_events:
            self.events[category] = self.events[category][-self.max_events:]
    
    def get_recent_events(self, category: str, count: int = 100) -> list:
        """Get recent events in category"""
        if category not in self.events:
            return []
        return self.events[category][-count:]
    
    def get_error_rate(self, window: int = 300) -> float:
        """Get error rate in time window"""
        cutoff = time.time() - window
        recent_errors = [
            e for e in self.events.get('errors', [])
            if e['timestamp'] > cutoff
        ]
        return len(recent_errors) / window if window > 0 else 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_events': sum(len(events) for events in self.events.values()),
            'categories': {
                cat: len(events) for cat, events in self.events.items()
            },
            'error_rate': self.get_error_rate(),
        }


# Global aggregator
log_aggregator = LogAggregator()