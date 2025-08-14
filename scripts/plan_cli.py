#!/usr/bin/env python3
"""
Partition Plan CLI

Features:
 - Generate snapshot/draft plans from simple inputs
 - Validate plans
 - Promote drafts to active (indexed) plans

Environment variables control behavior:
 - PLAN_ROOT: directory for plans (default ./data/partition_plans)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from backend.learning.partition_plan_registry import (
    PartitionPlanRegistry,
    PartitionPlan,
    StagePlan,
)


def cmd_snapshot(args: argparse.Namespace) -> None:
    reg = PartitionPlanRegistry(args.root)
    now = time.time()
    stages: List[StagePlan] = []
    for spec in args.stages:
        device_id, start, end = spec.split(":")
        stages.append(StagePlan(device_id=device_id, start_layer=int(start), end_layer=int(end)))
    plan = PartitionPlan(
        plan_id=args.plan_id or f"plan_{int(now)}",
        epoch_id=args.epoch,
        round_id=args.round,
        created_at=now,
        stage_plans=stages,
        metadata={
            "use_zero1": 1.0 if args.zero1 else 0.0,
            "use_activation_checkpointing": 1.0 if args.act_ckpt else 0.0,
        },
    )
    draft = reg.save_draft(plan)
    print(str(draft))


def cmd_validate(args: argparse.Namespace) -> None:
    reg = PartitionPlanRegistry(args.root)
    data = json.loads(Path(args.file).read_text())
    stages = [StagePlan(**s) for s in data["stage_plans"]]
    plan = PartitionPlan(
        plan_id=data["plan_id"],
        epoch_id=data["epoch_id"],
        round_id=data["round_id"],
        created_at=data.get("created_at", time.time()),
        stage_plans=stages,
        metadata=data.get("metadata", {}),
    )
    ok = reg.validate_plan(plan)
    print("valid" if ok else "invalid")


def cmd_promote(args: argparse.Namespace) -> None:
    reg = PartitionPlanRegistry(args.root)
    data = json.loads(Path(args.file).read_text())
    stages = [StagePlan(**s) for s in data["stage_plans"]]
    plan = PartitionPlan(
        plan_id=data["plan_id"],
        epoch_id=data["epoch_id"],
        round_id=data["round_id"],
        created_at=data.get("created_at", time.time()),
        stage_plans=stages,
        metadata=data.get("metadata", {}),
    )
    if not reg.validate_plan(plan):
        raise SystemExit("Invalid plan; refusing to promote")
    reg.save_plan(plan)
    print(plan.plan_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Partition Plan CLI")
    p.add_argument("command", choices=["snapshot", "validate", "promote"])
    p.add_argument("--root", dest="root", default=str(Path("./data/partition_plans")))
    # snapshot
    p.add_argument("--epoch", dest="epoch", help="epoch id")
    p.add_argument("--round", dest="round", help="round id")
    p.add_argument("--plan-id", dest="plan_id", default=None)
    p.add_argument("--stages", nargs="*", default=[], help="stage specs device:start:end e.g. n1:0:11")
    p.add_argument("--zero1", action="store_true")
    p.add_argument("--act-ckpt", action="store_true")
    # validate/promote
    p.add_argument("--file", dest="file", default=None)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "snapshot":
        if not args.epoch or not args.round or not args.stages:
            raise SystemExit("snapshot requires --epoch --round --stages")
        cmd_snapshot(args)
    elif args.command == "validate":
        if not args.file:
            raise SystemExit("validate requires --file")
        cmd_validate(args)
    elif args.command == "promote":
        if not args.file:
            raise SystemExit("promote requires --file")
        cmd_promote(args)


if __name__ == "__main__":
    main()

