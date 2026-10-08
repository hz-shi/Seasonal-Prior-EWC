"""Cosine-annealing T_max must follow the real Trainer.max_epochs.

Skipped automatically when torch / PyTorch Lightning are unavailable.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
    import pytorch_lightning  # noqa: F401

    # Guard against the MagicMock stubs installed by sibling tests: a real
    # torch exposes a string __version__, a stub does not.
    _HAS_TORCH = isinstance(getattr(torch, "__version__", None), str)
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch/pytorch_lightning not installed")
class CosineTMaxTest(unittest.TestCase):
    def _module(self):
        from src.lightning_module import PM25ForecastLitModule

        return PM25ForecastLitModule(
            model_name="unet",
            in_len=5,
            out_len=3,
            patch_h=8,
            patch_w=8,
            in_channels=3,
            lr=1e-3,
            weight_decay=1e-4,
            model_kwargs={},
        )

    def test_uses_trainer_max_epochs(self):
        module = self._module()

        class _Trainer:
            max_epochs = 20

        module._trainer = _Trainer()
        cfg = module.configure_optimizers()
        self.assertEqual(cfg["lr_scheduler"].T_max, 20)

    def test_fallback_without_trainer(self):
        module = self._module()
        module._trainer = None
        cfg = module.configure_optimizers()
        self.assertEqual(cfg["lr_scheduler"].T_max, 50)


if __name__ == "__main__":
    unittest.main()
