"""Determinism. Every training run calls set_global_seed before touching data."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed every RNG we can reach so a rerun reproduces the same numbers.

    Covers random, NumPy, and PYTHONHASHSEED for the process. Torch is seeded
    only when importable, so the core install does not depend on the vision extra.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    try:  # optional vision extra
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
