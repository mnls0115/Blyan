"""
Dense Runtime Module - Clean implementation without MoE complexity
==================================================================
This module provides a straightforward dense model runtime that bypasses
all MoE routing/gating complexity while reusing existing infrastructure.
"""

from .runtime import DenseRuntime
from .selector import select_runtime

__all__ = ['DenseRuntime', 'select_runtime']