#!/usr/bin/env python
"""Upload full weight file which server will split into 1MB blocks.

Example:
    python upload_parameters.py --address alice --file weights.pt --candidate-loss 0.9 --prev-loss 1.0
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib import request, error
import hashlib
import ecdsa  # type: ignore

API_URL = "http://127.0.0.1:8000/upload_parameters"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--candidate-loss", type=float, required=True)
    parser.add_argument("--prev-loss", type=float, default=None)
    parser.add_argument("--privkey", required=False)
    args = parser.parse_args()

    data_bytes = Path(args.file).read_bytes()
    file_b64 = base64.b64encode(data_bytes).decode()

    # signing
    if args.privkey:
        sk = ecdsa.SigningKey.from_string(bytes.fromhex(args.privkey), curve=ecdsa.SECP256k1)
    else:
        sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        print("Generated new private key:", sk.to_string().hex())
    vk = sk.verifying_key
    sig = sk.sign(data_bytes, hashfunc=hashlib.sha256)

    body = {
        "miner_address": args.address,
        "miner_pub": vk.to_string().hex(),
        "payload_sig": sig.hex(),
        "file_b64": file_b64,
        "candidate_loss": args.candidate_loss,
        "previous_loss": args.prev_loss,
    }

    req = request.Request(
        API_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )

    try:
        with request.urlopen(req) as resp:
            print(json.load(resp))
    except error.HTTPError as e:
        print("HTTP error", e.status, e.read().decode())


if __name__ == "__main__":
    main() 