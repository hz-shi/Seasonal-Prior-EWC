"""Import stubs so pure-logic tests can run without torch / PyTorch Lightning.

On a full environment (e.g. the L20 server) the real packages are present and
no stubbing happens. On a lightweight machine the heavy third-party modules and
the project submodules that pull them in are replaced with ``MagicMock`` so the
pure date/selection logic in ``src.rolling`` and the argument parser in
``src.cli`` can still be exercised.
"""

import importlib.util
import sys
from unittest import mock


def _has(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def install_stubs() -> bool:
    """Install stubs for missing heavy deps. Returns True when stubbing was used."""
    if _has("torch") and _has("pytorch_lightning"):
        return False

    for name in [
        "torch",
        "pytorch_lightning",
        "pytorch_lightning.strategies",
        "pytorch_lightning.loggers",
        "pytorch_lightning.callbacks",
    ]:
        if name not in sys.modules:
            sys.modules[name] = mock.MagicMock(name=name)

    pl = sys.modules["pytorch_lightning"]
    pl.Callback = type("Callback", (), {})
    pl.Trainer = type("Trainer", (), {})
    pl.LightningModule = type("LightningModule", (), {})
    pl.LightningDataModule = type("LightningDataModule", (), {})
    pl.seed_everything = lambda *args, **kwargs: None

    # Project submodules that import torch/PL at module scope. ``src.normalization``
    # is intentionally NOT stubbed: it only needs numpy/pandas/xarray.
    for name in [
        "src.data",
        "src.lightning_module",
        "src.reg",
        "src.reg.condition_aware_ewc",
    ]:
        if name not in sys.modules:
            sys.modules[name] = mock.MagicMock(name=name)

    return True
