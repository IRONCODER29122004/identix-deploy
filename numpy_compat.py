"""Runtime compatibility helpers for loading legacy/new NumPy pickles."""

import sys

import numpy as np


def ensure_numpy_pickle_compat():
    """Register NumPy module aliases expected by checkpoints across versions."""
    np_core = getattr(np, "_core", None) or np.core

    # Checkpoints may reference internal NumPy module paths from different versions.
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy.core", np.core)
    sys.modules.setdefault("numpy_core", np_core)

    multiarray = getattr(np_core, "multiarray", None) or getattr(np.core, "multiarray", None)
    if multiarray is not None:
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy.core.multiarray", multiarray)
