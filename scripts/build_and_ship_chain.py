#!/usr/bin/env python3
"""
Build-and-Ship Chain

Runs on a GPU node (e.g., Runpod) to:
  1) Fetch/load GPT-OSS-20B (or given HF repo / local ckpt)
  2) Construct a local blockchain dataset under build root:
     - Meta-chain A with a meta spec block that references the model
     - Parameter-chain B with expert/router blocks (virtual experts if non-MoE)
  3) Package and transfer the built chain to a main node's data folder via rsync/ssh

Usage (example):
  python scripts/build_and_ship_chain.py \
    --model-id Qwen/Qwen1.5-MoE-A2.7B \
    --remote user@MAIN_HOST \
    --remote-path /root/aiblock/data \
    --ssh-key ~/.ssh/id_rsa \
    --meta-arch gpt-neox

Notes:
  - This script avoids using the HTTP API for bulk payloads and instead ships the on-disk chain.
  - For safety, the remote data directory is optionally backed up before replacement.
  - If your main node API is running during swap, stop it briefly to avoid race conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.chain import Chain
from miner.upload_moe_parameters import MoEExpertExtractor  # Reuse robust extractor


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_meta_block(meta_chain: Chain, model_id: str, meta_arch: str) -> str:
    """Create a simple meta spec block and return its hash."""
    spec: Dict[str, Any] = {
        "model_name": model_id,
        "architecture": meta_arch or "gpt-oss",
        # Minimal fields for reference/anchor; extend if you have exact layer counts
        "routing_strategy": "single",
        "created_at": int(time.time()),
    }
    payload = json.dumps(spec, ensure_ascii=False).encode()
    blk = meta_chain.add_block(payload, block_type="meta")
    return blk.compute_hash()


def add_param_block(
    param_chain: Chain,
    tensor_bytes: bytes,
    block_type: str,
    expert_name: str,
    layer_id: str,
    meta_hash: str,
) -> str:
    blk = param_chain.add_block(
        payload=tensor_bytes,
        points_to=meta_hash,
        miner_pub=None,
        payload_sig=None,
        depends_on=[meta_hash],
        block_type=block_type,  # 'expert' or 'router'
        expert_name=expert_name,
        layer_id=layer_id,
    )
    return blk.compute_hash()


def serialize_tensors(state_dict: Dict[str, "torch.Tensor"]) -> bytes:
    import io
    import torch
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def package_dir_to_tar_gz(src_dir: Path, out_tar_gz: Path) -> None:
    with tarfile.open(out_tar_gz, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def rsync_to_remote(
    local_path: Path,
    remote: str,
    remote_path: str,
    ssh_key: str | None,
    backup: bool = True,
) -> None:
    """Copy local_path to remote_path using rsync over ssh, optional backup on remote."""
    # Upload tar to /tmp on remote
    remote_tmp = f"/tmp/{local_path.name}"
    scp_cmd = [
        "rsync", "-avz", "-e",
        f"ssh -o StrictHostKeyChecking=no{(' -i '+ssh_key) if ssh_key else ''}",
        str(local_path), f"{remote}:{remote_tmp}"
    ]
    subprocess.run(" ".join(scp_cmd), shell=True, check=True)

    # Prepare remote commands: backup and extract
    timestamp = int(time.time())
    backup_cmd = (
        f"if [ -d '{remote_path}' ]; then mv '{remote_path}' '{remote_path}_backup_{timestamp}'; fi"
        if backup else "true"
    )
    extract_cmd = (
        f"mkdir -p '{remote_path}' && tar -xzf '{remote_tmp}' -C '{Path(remote_path).parent}' && "
        f"rm -f '{remote_tmp}' && mv '{Path(remote_path).parent}/{Path(local_path.name).stem}' '{remote_path}'"
    )
    ssh_prefix = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if ssh_key:
        ssh_prefix += ["-i", ssh_key]
    ssh_cmd = ssh_prefix + [remote, f"bash -lc \"{backup_cmd} && {extract_cmd}\""]
    subprocess.run(ssh_cmd, check=True)


def try_fetch_remote_genesis(remote: str, remote_path: str, ssh_key: str | None, build_root: Path) -> bool:
    """Attempt to fetch remote genesis hash and chain A into local build root.
    Returns True if fetched, False otherwise.
    """
    # Fetch genesis_pact_hash.txt (if exists)
    genesis_local = build_root / "genesis_pact_hash.txt"
    remote_genesis = f"{remote}:{remote_path}/genesis_pact_hash.txt"
    rsync_opts = f"-e 'ssh -o StrictHostKeyChecking=no{(' -i '+ssh_key) if ssh_key else ''}'"
    try:
        cmd = f"rsync -avz {rsync_opts} {remote_genesis} {genesis_local}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # No genesis file found
        return False

    # Fetch chain A directory
    try:
        local_A = build_root / "A"
        local_A.mkdir(parents=True, exist_ok=True)
        cmd = f"rsync -avz {rsync_opts} {remote}:{remote_path}/A/ {local_A}/"
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        # Could not fetch A
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MoE DAG chain locally and ship to main node")
    parser.add_argument("--model-id", default="Qwen/Qwen1.5-MoE-A2.7B", help="HF repo id or local ckpt path")
    parser.add_argument("--build-root", default="./data_build", help="Local build root for chain data")
    parser.add_argument("--meta-arch", default="gpt-oss", help="Architecture label for meta block")
    parser.add_argument("--skip-pol", action="store_true", help="Initialize chains with SKIP_POL for speed")
    parser.add_argument("--remote", required=True, help="SSH target, e.g., user@host")
    parser.add_argument("--remote-path", required=True, help="Remote data path, e.g., /root/aiblock/data")
    parser.add_argument("--ssh-key", default=os.environ.get("SSH_KEY"), help="SSH private key path")
    parser.add_argument("--no-remote-backup", action="store_true", help="Do not backup remote data dir")
    parser.add_argument("--no-remote-genesis", action="store_true", help="Do not pull remote genesis/A; build without it (not recommended)")
    args = parser.parse_args()

    build_root = Path(args.build_root).resolve()
    ensure_empty_dir(build_root)

    # Optionally pull genesis + chain A from remote to ensure new blocks anchor to genesis
    pulled = False
    if not args.no_remote_genesis:
        log("[0/4] Pulling remote genesis/A to anchor new blocks...")
        pulled = try_fetch_remote_genesis(args.remote, args.remote_path, args.ssh_key, build_root)
        log(f"    pulled={pulled}")

    # Create chains locally
    os.environ.setdefault("CHAIN_DIFFICULTY", "1")  # Keep fast
    meta_chain = Chain(build_root, "A", skip_pol=args.skip_pol)
    param_chain = Chain(build_root, "B", skip_pol=args.skip_pol)

    # 1) Create meta block
    log("[1/4] Creating meta block...")
    meta_hash = create_meta_block(meta_chain, args.model_id, args.meta_arch)
    log(f"    meta_hash={meta_hash[:16]}...")

    # 2) Load model and extract experts (virtual experts if non-MoE)
    log("[2/4] Loading model & extracting experts...")
    extractor = MoEExpertExtractor(args.model_id)
    extracted = extractor.extract_experts()
    experts = extracted.get('experts', {})
    routers = extracted.get('routers', {})
    log(f"    experts={len(experts)}, routers={len(routers)}")

    # 3) Build parameter chain blocks
    import torch  # local import to avoid top-level dependency issues
    log("[3/4] Creating parameter blocks...")
    created = 0
    for expert_name, data in experts.items():
        tensors = data['tensors']
        layer_id = data.get('layer_id', 'layer0')
        payload = serialize_tensors(tensors)
        h = add_param_block(param_chain, payload, 'expert', expert_name, layer_id, meta_hash)
        created += 1
        if created % 10 == 0:
            log(f"    +{created} blocks... (last: {expert_name} -> {h[:16]}...)")

    for router_name, data in routers.items():
        tensors = data['tensors']
        layer_id = data.get('layer_id', 'layer0')
        payload = serialize_tensors(tensors)
        h = add_param_block(param_chain, payload, 'router', router_name, layer_id, meta_hash)
        created += 1
        if created % 10 == 0:
            log(f"    +{created} blocks... (last: {router_name} -> {h[:16]}...)")

    log(f"    total blocks created (B): {created}")

    # 4) Package and ship
    log("[4/4] Packaging and transferring...")
    out_tar = build_root.parent / f"{build_root.name}.tar.gz"
    package_dir_to_tar_gz(build_root, out_tar)
    rsync_to_remote(
        local_path=out_tar,
        remote=args.remote,
        remote_path=args.remote_path,
        ssh_key=args.ssh_key,
        backup=not args.no_remote_backup,
    )

    log("✅ Done! Remote data directory replaced with built chain.")
    log("   If your API was running, restart it now to pick up the new data.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

