#!/usr/bin/env python3
"""
Partition Plan Registry

Stores per-epoch/per-round partition plans built from device profiles and
cost/partitioning results. Provides immutable snapshots for training rounds
and supports fallback to previous stable plan on failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StagePlan:
    device_id: str
    start_layer: int
    end_layer: int


@dataclass
class PartitionPlan:
    plan_id: str
    epoch_id: str
    round_id: str
    created_at: float
    stage_plans: List[StagePlan]
    metadata: Dict[str, float]


class PartitionPlanRegistry:
    def __init__(self, root_dir: str = "./data/partition_plans") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index: Dict[str, str] = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def save_plan(self, plan: PartitionPlan) -> None:
        path = self.root / f"{plan.plan_id}.json"
        doc = asdict(plan)
        doc["stage_plans"] = [asdict(s) for s in plan.stage_plans]
        path.write_text(json.dumps(doc, indent=2))
        key = f"{plan.epoch_id}:{plan.round_id}"
        self._index[key] = plan.plan_id
        self._save_index()

    def load_plan(self, plan_id: str) -> Optional[PartitionPlan]:
        path = self.root / f"{plan_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        stages = [StagePlan(**s) for s in data["stage_plans"]]
        return PartitionPlan(
            plan_id=data["plan_id"],
            epoch_id=data["epoch_id"],
            round_id=data["round_id"],
            created_at=data["created_at"],
            stage_plans=stages,
            metadata=data.get("metadata", {}),
        )

    def resolve_for_round(self, epoch_id: str, round_id: str) -> Optional[PartitionPlan]:
        key = f"{epoch_id}:{round_id}"
        pid = self._index.get(key)
        if not pid:
            return None
        return self.load_plan(pid)

    def latest_stable(self) -> Optional[PartitionPlan]:
        # Best-effort: return the newest by created_at
        latest: Optional[PartitionPlan] = None
        for pid in self._index.values():
            plan = self.load_plan(pid)
            if not plan:
                continue
            if latest is None or plan.created_at > latest.created_at:
                latest = plan
        return latest

    def freeze_rounds(self, epoch_id: str, plans: List[PartitionPlan]) -> None:
        """Save a list of round plans for an epoch in one shot."""
        for p in plans:
            assert p.epoch_id == epoch_id
            self.save_plan(p)

    # CLI helper methods
    def save_draft(self, plan: PartitionPlan) -> Path:
        """Save a plan as a draft without indexing (for offline validation)."""
        draft_path = self.root / f"draft_{plan.plan_id}.json"
        doc = asdict(plan)
        doc["stage_plans"] = [asdict(s) for s in plan.stage_plans]
        draft_path.write_text(json.dumps(doc, indent=2))
        return draft_path

    def validate_plan(self, plan: PartitionPlan) -> bool:
        """Basic structural validation of a partition plan."""
        if not plan.stage_plans:
            return False
        # Layers must be contiguous and non-overlapping
        expected = None
        for sp in plan.stage_plans:
            if expected is None:
                expected = sp.start_layer
            if sp.start_layer != expected:
                return False
            if sp.end_layer < sp.start_layer:
                return False
            expected = sp.end_layer + 1
        return True

