from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from backend.core.chain import Chain
from backend.core.param_index import ParameterIndex

from .arch import ModelWrapper, bytes_to_state_dict


def _extract_model_name(meta_chain: Chain) -> str:
    """Parse the latest meta-chain block payload to get the model name.

    For the prototype we assume the payload is a JSON string like
    {"model_name": "distilbert-base-uncased"}.
    """
    latest = meta_chain.storage.get_latest_block()
    if latest is None:
        # default fallback
        return "distilbert-base-uncased"
    try:
        spec = json.loads(latest.payload.decode())
        return spec.get("model_name", "distilbert-base-uncased")
    except Exception:
        return "distilbert-base-uncased"


class ModelManager:
    """Lazy-loads the model from chains and caches it."""

    def __init__(self, meta_chain: Chain, param_chain: Chain, param_index: ParameterIndex, device: Optional[str] = None):
        self.meta_chain = meta_chain
        self.param_chain = param_chain
        self.param_index = param_index
        self.device = device
        self._model: Optional[ModelWrapper] = None
        self._current_meta_hash: Optional[str] = None
        self._current_param_hash: Optional[str] = None

    def _refresh_if_needed(self):
        latest_meta = self.meta_chain.storage.get_latest_block()
        if latest_meta is None:
            raise RuntimeError("Meta chain empty; cannot load model")
        latest_meta_hash = latest_meta.compute_hash()

        latest_param = self.param_chain.storage.get_latest_block()
        latest_param_hash = latest_param.compute_hash() if latest_param else None

        # Check if model architecture or parameters have changed
        if (
            self._model is None
            or latest_meta_hash != self._current_meta_hash
            or latest_param_hash != self._current_param_hash
        ):
            # Load base model architecture if it has changed
            if self._model is None or latest_meta_hash != self._current_meta_hash:
                model_name = _extract_model_name(self.meta_chain)
                self._model = ModelWrapper(model_name, device=self.device)
                self._current_meta_hash = latest_meta_hash

            # Load parameters from param_chain if it has blocks
            if latest_param_hash is not None:
                combined: dict = {}

                # Prefer index-based fast path
                if self.param_index.all():
                    for name, idx in self.param_index.all().items():
                        blk = self.param_chain.storage.load_block(idx)
                        if blk is None:
                            continue
                        try:
                            piece = bytes_to_state_dict(blk.payload)
                            if isinstance(piece, dict):
                                combined.update(piece)
                        except Exception:
                            continue
                else:
                    # fallback full scan
                    for block in self.param_chain.storage.iter_blocks():
                        try:
                            piece = bytes_to_state_dict(block.payload)
                            if isinstance(piece, dict):
                                combined.update(piece)
                        except Exception:
                            continue

                if combined:
                    assert self._model is not None
                    self._model.apply_weights(combined)

            self._current_param_hash = latest_param_hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        self._refresh_if_needed()
        assert self._model is not None  # for mypy
        return self._model.generate(prompt, max_new_tokens=max_new_tokens) 