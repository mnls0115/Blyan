#!/usr/bin/env python
"""Simple miner submission script.

Usage:
    python submit_payload.py --address alice --file weights.pt --candidate-loss 0.95 --prev-loss 1.0

The script encodes the weight file as base64 and POSTs to /mine.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib import request, error
import hashlib
import ecdsa  # type: ignore

API_URL = "http://127.0.0.1:8000/mine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True, help="Miner wallet address")
    parser.add_argument("--file", required=True, help="Path to weights file (.pt)")
    parser.add_argument("--candidate-loss", type=float, required=True)
    parser.add_argument("--prev-loss", type=float, default=None)
    parser.add_argument("--privkey", required=False, help="Hex-encoded ECDSA private key. If omitted, new key is generated and printed.")
    args = parser.parse_args()

    file_bytes = Path(args.file).read_bytes()
    payload_b64 = base64.b64encode(file_bytes).decode()

    # load/generate key
    if args.privkey:
        sk_bytes = bytes.fromhex(args.privkey)
        sk = ecdsa.SigningKey.from_string(sk_bytes, curve=ecdsa.SECP256k1)
    else:
        sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        print("Generated new private key:", sk.to_string().hex())
    vk = sk.verifying_key
    signature = sk.sign(file_bytes, hashfunc=hashlib.sha256)

    body = {
        "miner_address": args.address,
        "miner_pub": vk.to_string().hex(),
        "payload_sig": signature.hex(),
        "candidate_payload_b64": payload_b64,
        "candidate_loss": args.candidate_loss,
        "previous_loss": args.prev_loss,
    }

    data = json.dumps(body).encode()
    req = request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})

    try:
        with request.urlopen(req) as resp:
            resp_data = json.load(resp)
            print("✅ Block mined", resp_data)
    except error.HTTPError as e:
        print("❌ HTTP error", e.read().decode())
    except Exception as e:
        print("❌ Error", str(e))


if __name__ == "__main__":
    main() 