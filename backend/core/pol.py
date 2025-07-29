"""Proof-of-Learning (PoL) evaluation helpers.

In the prototype, PoL simply checks that the candidate model achieves at least a
`threshold` fractional improvement in loss on a small validation set.
This is a stub; replace with real evaluation when integrating a full dataset.
"""

from __future__ import annotations

import random
from typing import Callable


def mock_validation_loss() -> float:
    """Placeholder: returns a random loss between 0.5 and 2.0."""
    return random.uniform(0.5, 2.0)


def evaluate_candidate(
    candidate_loss_fn: Callable[[], float] | None = None,
    previous_loss: float | None = None,
    threshold: float = 0.005,
) -> bool:
    """Return True if candidate improves by `threshold` fraction.

    Parameters
    ----------
    candidate_loss_fn: Callable that returns the candidate model's loss on the
        validation set. If None, `mock_validation_loss` is used.
    previous_loss: Baseline loss to improve upon. If None, candidate always
        wins (e.g. first block).
    threshold: Minimum relative improvement required (e.g. 0.005 => 0.5%).
    """
    if candidate_loss_fn is None:
        candidate_loss_fn = mock_validation_loss

    cand_loss = candidate_loss_fn()
    if previous_loss is None:
        return True  # first model wins by default

    improvement = previous_loss - cand_loss
    required = threshold * previous_loss
    return improvement >= required 