from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class Ledger:
    """Very simple token ledger stored as a JSON dict {address: balance}."""

    def __init__(self, file_path: Path):
        self.path = file_path
        self._balances: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open() as fp:
                    self._balances = json.load(fp)
            except Exception:
                self._balances = {}
        else:
            self._balances = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fp:
            json.dump(self._balances, fp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_balance(self, address: str) -> float:
        return float(self._balances.get(address, 0.0))

    def credit(self, address: str, amount: float) -> None:
        if amount < 0:
            raise ValueError("Amount must be non-negative")
        self._balances[address] = self.get_balance(address) + amount
        self._save()

    def transfer(self, sender: str, receiver: str, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        if self.get_balance(sender) < amount:
            raise ValueError("Insufficient balance")
        self._balances[sender] = self.get_balance(sender) - amount
        self._balances[receiver] = self.get_balance(receiver) + amount
        self._save() 