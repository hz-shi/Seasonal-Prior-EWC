"""Regression test for best-validation-loss checkpoint selection.

The manuscript retains the checkpoint with the lowest validation prediction
loss. This test drives the real ``BestValStateTracker`` with two epochs where
the second epoch is worse and asserts the first epoch's weights are the ones
kept -- it is not a mirror of the implementation.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._heavy_stubs import install_stubs  # noqa: E402

install_stubs()

from src import rolling  # noqa: E402

_REAL_TORCH = isinstance(getattr(rolling.torch, "__version__", None), str)


class _FakeTensor:
    """Stand-in for a torch tensor supporting the tracker's clone calls."""

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        return _FakeTensor(self.value)


class BestValTrackerTest(unittest.TestCase):
    def test_keeps_first_epoch_when_second_is_worse(self):
        tracker = rolling.BestValStateTracker()
        self.assertTrue(tracker.update(1.0, {"w": _FakeTensor("epoch1")}, epoch=0))
        self.assertFalse(tracker.update(2.0, {"w": _FakeTensor("epoch2")}, epoch=1))
        self.assertEqual(tracker.best_val_loss, 1.0)
        self.assertEqual(tracker.best_epoch, 0)
        self.assertEqual(tracker.best_state["w"].value, "epoch1")

    def test_replaces_when_later_epoch_improves(self):
        tracker = rolling.BestValStateTracker()
        tracker.update(2.0, {"w": _FakeTensor("epoch1")}, epoch=0)
        self.assertTrue(tracker.update(0.5, {"w": _FakeTensor("epoch2")}, epoch=1))
        self.assertEqual(tracker.best_val_loss, 0.5)
        self.assertEqual(tracker.best_state["w"].value, "epoch2")

    def test_ignores_non_finite_loss(self):
        tracker = rolling.BestValStateTracker()
        self.assertFalse(tracker.update(float("nan"), {"w": _FakeTensor("x")}))
        self.assertFalse(tracker.update(float("inf"), {"w": _FakeTensor("x")}))
        self.assertIsNone(tracker.best_state)
        self.assertIsNone(tracker.best_val_loss)

    def test_stored_state_is_a_copy(self):
        tracker = rolling.BestValStateTracker()
        source = {"w": _FakeTensor("epoch1")}
        tracker.update(1.0, source, epoch=0)
        # Mutating the source mapping must not affect the stored best state.
        source["w"] = _FakeTensor("mutated")
        self.assertEqual(tracker.best_state["w"].value, "epoch1")


@unittest.skipUnless(_REAL_TORCH, "requires torch and PyTorch Lightning")
class BestValCallbackIntegrationTest(unittest.TestCase):
    def test_lightning_keeps_best_epoch_and_restores_without_replacing_model(self):
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class WorseningModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.condition_ewc = object()
                self.register_buffer("raw_scale", torch.tensor(3.0), persistent=False)

            def on_train_epoch_start(self):
                with torch.no_grad():
                    self.weight.fill_(self.current_epoch + 1.0)

            def training_step(self, batch, batch_idx):
                return self.weight * 0.0

            def validation_step(self, batch, batch_idx):
                self.log("val_loss", self.weight.square(), on_epoch=True, batch_size=1)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        model = WorseningModel()
        anchor = model.condition_ewc
        tracker = rolling.BestValStateTracker()
        loader = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
        trainer = pl.Trainer(
            accelerator="cpu", devices=1, max_epochs=2,
            limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[rolling.BestValCheckpointCallback(tracker)],
        )
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        self.assertEqual(model.weight.item(), 2.0)
        self.assertEqual(tracker.best_epoch, 0)
        self.assertEqual(tracker.best_val_loss, 1.0)
        self.assertEqual(tracker.best_state["weight"].device.type, "cpu")
        model.load_state_dict(tracker.best_state)
        self.assertEqual(model.weight.item(), 1.0)
        self.assertIs(model.condition_ewc, anchor)
        self.assertEqual(model.raw_scale.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
