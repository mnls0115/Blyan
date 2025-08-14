#!/usr/bin/env python3
"""
Layer ↔ Index mapping utilities.

Supports verification of stage boundaries including MoE router/expert blocks,
and provides stable mapping from model structure to integer layer indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class LayerSpec:
    name: str
    kind: str  # e.g., "transformer_block", "moe_router", "moe_expert_group"
    extra: Dict[str, Any]


class LayerIndexMapper:
    def __init__(self, layers: List[LayerSpec]):
        self.layers = layers
        self.name_to_index: Dict[str, int] = {spec.name: i for i, spec in enumerate(layers)}

    def index_of(self, layer_name: str) -> Optional[int]:
        return self.name_to_index.get(layer_name)

    def get_spec(self, index: int) -> LayerSpec:
        return self.layers[index]

    def validate_stage_boundaries(self, start_idx: int, end_idx: int) -> bool:
        """
        Validate stage boundaries with MoE constraints:
        - router and its experts should not be split across stages if tightly coupled
        - disallow negative or inverted ranges
        """
        if start_idx < 0 or end_idx >= len(self.layers) or end_idx < start_idx:
            return False

        # Check MoE cohesion: if a router exists in range, ensure its expert group fully inside
        moe_router_indices = [i for i in range(start_idx, end_idx + 1)
                              if self.layers[i].kind == "moe_router"]
        for ridx in moe_router_indices:
            group_name = self.layers[ridx].extra.get("group_name")
            if not group_name:
                continue
            # Find corresponding expert group indices
            expert_indices = [i for i, spec in enumerate(self.layers)
                              if spec.kind == "moe_expert_group" and spec.extra.get("group_name") == group_name]
            if not expert_indices:
                continue
            # Require all experts of the group be within the same stage as the router
            for eidx in expert_indices:
                if not (start_idx <= eidx <= end_idx):
                    return False
        return True


def build_mapper_from_model_structure(model_structure: List[Dict[str, Any]]) -> LayerIndexMapper:
    """
    Build mapper from a high-level structure description, e.g.,
    [
      {"name": "block_0", "kind": "transformer_block"},
      {"name": "moe_router_0", "kind": "moe_router", "extra": {"group_name": "g0"}},
      {"name": "moe_experts_0", "kind": "moe_expert_group", "extra": {"group_name": "g0", "experts": 4}},
      ...
    ]
    """
    specs: List[LayerSpec] = []
    for item in model_structure:
        specs.append(LayerSpec(
            name=item.get("name"),
            kind=item.get("kind", "transformer_block"),
            extra=item.get("extra", {}),
        ))
    return LayerIndexMapper(specs)

