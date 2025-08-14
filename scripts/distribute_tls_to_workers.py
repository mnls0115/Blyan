#!/usr/bin/env python3
"""
Distribute TLS certificates to worker nodes via scp/ssh.

Usage:
  python scripts/distribute_tls_to_workers.py --cert /etc/letsencrypt/live/example.com/fullchain.pem \
      --key /etc/letsencrypt/live/example.com/privkey.pem \
      --nodes worker1@10.0.0.2:22 worker2@10.0.0.3:22 --dest /etc/blyan/tls
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--nodes", nargs="+", help="user@host:port")
    p.add_argument("--dest", default="/etc/blyan/tls")
    args = p.parse_args()

    cert = Path(args.cert)
    key = Path(args.key)
    if not cert.exists() or not key.exists():
        raise SystemExit("cert or key not found")

    for node in args.nodes:
        if ":" in node:
            host, port = node.split(":", 1)
        else:
            host, port = node, "22"
        # Create dest dir
        run(["ssh", "-p", port, host, "sudo", "mkdir", "-p", args.dest])
        run(["scp", "-P", port, str(cert), f"{host}:{args.dest}/fullchain.pem"]) 
        run(["scp", "-P", port, str(key), f"{host}:{args.dest}/privkey.pem"]) 
        run(["ssh", "-p", port, host, "sudo", "chmod", "600", f"{args.dest}/privkey.pem"]) 
        print(f"Deployed certs to {host}:{args.dest}")


if __name__ == "__main__":
    main()

