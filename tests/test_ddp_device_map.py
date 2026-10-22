"""Pin each DDP worker's quantized base model to its own GPU.

Unsloth's default ``device_map="sequential"`` spreads the 4-bit model across every
visible GPU. Under DDP that gives workers different parameter sets, which makes DDP
abort with "expects same model across all ranks". These tests check the option the
loaders pass, using the environment variables torchrun sets, without loading a model.
"""

import os
import unittest
from unittest import mock

from learning import distributed, runtime


class ModelLoadOptionsTests(unittest.TestCase):
    """Check the device placement each training worker requests."""

    def test_single_gpu_keeps_unsloth_default(self) -> None:
        """A single process leaves Unsloth's default placement untouched."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime.model_load_options(), {})

    def test_ddp_worker_pins_model_to_its_local_rank(self) -> None:
        """A DDP worker loads its replica onto the GPU for its local rank."""
        env = {"WORLD_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(distributed.active())
            self.assertEqual(
                runtime.model_load_options(), {"device_map": {"": 1}}
            )

    def test_every_rank_gets_a_distinct_device(self) -> None:
        """Each rank of a two-GPU job gets its own device, not a shared one."""
        devices = []
        for rank in range(2):
            env = {"WORLD_SIZE": "2", "LOCAL_RANK": str(rank), "RANK": str(rank)}
            with mock.patch.dict(os.environ, env, clear=True):
                devices.append(runtime.model_load_options()["device_map"][""])
        self.assertEqual(devices, [0, 1])


if __name__ == "__main__":
    unittest.main()
