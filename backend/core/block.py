from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class BlockHeader:
    """Metadata for a block stored on-chain."""

    index: int
    timestamp: float
    prev_hash: str
    chain_id: str  # "A" for meta, "B" for parameter, or other future chains
    points_to: Optional[str]  # hash in the sister chain this block is bound to
    payload_hash: str
    payload_size: int
    nonce: int = 0  # proof-of-work nonce

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def to_json(self) -> str:
        """Stable JSON representation used for hashing."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """SHA-256 of the header JSON."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class Block:
    """A full block: header + payload + optional miner signature."""

    header: BlockHeader
    payload: bytes
    miner_pub: Optional[str] = None  # hex-encoded compressed public key
    payload_sig: Optional[str] = None  # hex ECDSA signature of payload

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def compute_hash(self) -> str:
        header_json = self.header.to_json()
        return hashlib.sha256(header_json.encode() + self.payload).hexdigest()

    def to_dict(self) -> dict:
        d = {
            "header": asdict(self.header),
            "payload": self.payload.hex(),
        }
        if self.miner_pub is not None:
            d["miner_pub"] = self.miner_pub
        if self.payload_sig is not None:
            d["payload_sig"] = self.payload_sig
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        header_data = data["header"]
        payload = bytes.fromhex(data["payload"])
        header = BlockHeader(**header_data)
        return cls(
            header=header,
            payload=payload,
            miner_pub=data.get("miner_pub"),
            payload_sig=data.get("payload_sig"),
        ) 