"""Offline checks for multi-GPU model placement and round failure handling.

Author: Nihal M
Date: 2026
License: MIT
"""

import os
import unittest
from unittest import mock

from learning import inputs, runtime
from learning.pipeline import models
from learning.training import fit
from learning.training import round as training_round

DDP_WORKER_1 = {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1"}
DDP_WORKER_0 = {"WORLD_SIZE": "2", "LOCAL_RANK": "0", "RANK": "0"}


class _FastModel:
    """Record the keyword arguments each base-model load receives."""

    def __init__(self) -> None:
        self.loads: list[dict] = []
        self.model = mock.MagicMock()

    def from_pretrained(self, **kwargs):
        self.loads.append(kwargs)
        return self.model, mock.MagicMock()

    def get_peft_model(self, model, **_kwargs):
        return model

    def for_training(self, model):
        return model


class TestModelPlacement(unittest.TestCase):
    """Every DDP worker loads its quantized base model on its own GPU."""

    def setUp(self) -> None:
        for module in (runtime, inputs):
            patcher = mock.patch.object(module, "configure_tokenizer", lambda t: t)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _dependencies(self) -> tuple[_FastModel, dict]:
        """Return a recording fake and the dependency map that carries it."""
        fast = _FastModel()
        return fast, {"FastLanguageModel": fast}

    def test_fresh_model_loads_on_the_worker_gpu(self) -> None:
        fast, dependencies = self._dependencies()
        with mock.patch.dict(os.environ, DDP_WORKER_1):
            runtime.prepare_model(dependencies)
        self.assertEqual(fast.loads[0]["device_map"], {"": 1})

    def test_continuation_model_loads_on_the_worker_gpu(self) -> None:
        fast, dependencies = self._dependencies()
        with mock.patch.dict(os.environ, DDP_WORKER_1):
            models.load_round_model(dependencies, "outputs/run/round-01/checkpoint-50")
        self.assertEqual(fast.loads[0]["device_map"], {"": 1})

    def test_single_gpu_keeps_the_default_placement(self) -> None:
        """Leave single-GPU placement to Unsloth."""
        fast, dependencies = self._dependencies()
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            os.environ.pop("LOCAL_RANK", None)
            runtime.prepare_model(dependencies)
        self.assertNotIn("device_map", fast.loads[0])


class _Config:

    __dataclass_fields__ = dict.fromkeys((
        "dataset_text_field", "max_seq_length", "eval_strategy", "eval_steps",
        "load_best_model_at_end", "metric_for_best_model", "save_steps",
        "ddp_find_unused_parameters",
    ))

    def __init__(self, **kwargs) -> None:
        """Record the settings."""
        self.settings = kwargs


class _Torch:
    class cuda:
        @staticmethod
        def is_bf16_supported() -> bool:
            return True


class TestTrainerConfiguration(unittest.TestCase):

    def _settings(self, environment: dict) -> dict:
        with mock.patch.dict(os.environ, environment), \
                mock.patch("builtins.print"):
            return fit._configuration(_Config, [1], _Torch).settings

    def test_ddp_turns_off_the_unused_parameter_search(self) -> None:
        self.assertIs(self._settings(DDP_WORKER_0)["ddp_find_unused_parameters"], False)

    def test_single_gpu_leaves_the_setting_alone(self) -> None:
        self.assertNotIn("ddp_find_unused_parameters", self._settings({"WORLD_SIZE": "1"}))


class TestRoundFailure(unittest.TestCase):

    def _round(self):
        return training_round.train_round(
            {}, None, None, None, None, [], None, False, "continuation-1",
            50, "round-01", 0.0, 0.0, 1e-4,
        )

    def test_ddp_round_failure_is_raised(self) -> None:
        failure = RuntimeError("DDP expects same model across all ranks")
        with mock.patch.dict(os.environ, DDP_WORKER_0), mock.patch.object(
            training_round, "train_model", side_effect=failure
        ), self.assertRaises(RuntimeError):
            self._round()

    def test_single_gpu_round_failure_is_returned_for_recovery(self) -> None:
        failure = RuntimeError("CUDA out of memory")
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}), mock.patch.object(
            training_round, "train_model", side_effect=failure
        ):
            trainer, interrupted = self._round()
        self.assertIsNone(trainer)
        self.assertIs(interrupted, failure)


if __name__ == "__main__":
    unittest.main()
