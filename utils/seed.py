"""
utils/seed.py
-------------
Global reproducibility seeding.
"""
from __future__ import annotations

__all__ = ["set_seed"]

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed:          Integer seed.
        deterministic: If True, sets torch to use deterministic algorithms
                       (may be slower on some ops but is fully reproducible).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some ops may not have deterministic implementations — skip silently
            pass
