"""
Model Adapter Registry
======================
Model-agnostic adapter system for loading any causal LLM from blockchain.
"""

from .base import ModelAdapter
from .registry import get_adapter, register_adapter, AUTO_DETECT

# Import all adapters to register them
from . import qwen_adapter
from . import llama_adapter
from . import mixtral_adapter
from . import default_adapter

__all__ = [
    'ModelAdapter',
    'get_adapter',
    'register_adapter',
    'AUTO_DETECT'
]