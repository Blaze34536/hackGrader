"""Choose the safe continuation action for a training run.

Every decision here is read from disk: the claim marker, saved checkpoints, the released
adapter, and the merge stamp. The tests build those artifacts by hand so each action is
checked without a model, a GPU, or a served Ollama instance.
"""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli
import learning.continuation as continuation
from learning import resume_runtime
from learning.pipeline import resume as pipeline_resume
from learning.pipeline.recovery_paths import recorded_dataset
from learning.training import artifacts

RUN_ID = "20260101-000000"


def _write(path: Path, payload: dict) -> None:
    """Write one JSON object, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _adapter(path: Path) -> None:
    """Write the two files that make a directory a saved PEFT adapter."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")


def _run(
    root: Path,
    *,
    started_at: float = 0.0,
    stride: int = 500,
    inputs: list[str] | None = None,
) -> Path:
    """Create a run directory with metadata and return it."""
    run_dir = root / "outputs" / RUN_ID
    metadata = {"started_at": started_at, "max_steps_per_round": stride}
    if inputs is not None:
        metadata["inputs"] = inputs
    _write(run_dir / "run_metadata.json", metadata)
    return run_dir


def _diagnostics(run_dir: Path, step: int, metrics: dict) -> None:
    """Write the metadata line of one validation diagnostics file."""
    path = run_dir / f"validation-checkpoint-{step}.jsonl"
    path.write_text(
        json.dumps({"metadata": {"checkpoint": step, "gate_metrics": metrics}}) + "\n",
        encoding="utf-8",
    )


class TestRunInspection(unittest.TestCase):
    """Read logical checkpoint steps, selections, and release evidence from disk."""

    def test_round_checkpoints_become_logical_steps(self) -> None:
        """Number rounds by their stride so two rounds never share a step."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            _adapter(run_dir / "round-01" / "checkpoint-50" / "continuation-1")
            _adapter(run_dir / "round-02" / "checkpoint-100" / "continuation-2")

            state = continuation.inspect(RUN_ID, root=str(root / "outputs"))

            steps = [step for step, _path in state.checkpoints]
            self.assertEqual(steps, [50, 600])
            self.assertTrue(all(isinstance(step, int) for step in steps))
            self.assertTrue(state.checkpoints[1][1].endswith("continuation-2"))
            self.assertIsNone(state.selected)

    def test_diagnostics_selection_wins_over_the_newest_checkpoint(self) -> None:
        """Resume the checkpoint the run's own validation sweep picked."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            best = run_dir / "round-01" / "checkpoint-50" / "continuation-1"
            latest = run_dir / "round-01" / "checkpoint-100" / "continuation-1"
            _adapter(best)
            _adapter(latest)
            _diagnostics(run_dir, 50, {"passed": True, "failed_gates": []})
            _diagnostics(
                run_dir,
                600,
                {"passed": False, "failed_gates": ["reliability"], "balanced_accuracy": 0.4},
            )

            state = continuation.inspect(RUN_ID, root=str(root / "outputs"))

            self.assertIsNotNone(state.selected)
            self.assertEqual(state.selected[0], 50)
            self.assertEqual(Path(state.selected[1]), best)

    def test_release_evidence_needs_the_same_adapter_digest(self) -> None:
        """Report a run as deployed only when the served record names its adapter."""
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            adapter_dir = root / "adapter"
            merge_dir = root / "merge"
            released = root / "data" / "deployed_model.json"
            _adapter(adapter_dir)
            _write(merge_dir / "merge_stamp.json", {"adapter_sha256": digest})
            _write(released, {"adapter_sha256": digest})

            state = continuation.inspect(
                RUN_ID,
                root=str(root / "outputs"),
                adapter_dir=str(adapter_dir),
                merge_dir=str(merge_dir),
                released=str(released),
            )

            self.assertTrue(state.merged)
            self.assertTrue(state.deployed)
            _write(released, {"adapter_sha256": "b" * 64})
            stale = continuation.inspect(
                RUN_ID,
                root=str(root / "outputs"),
                adapter_dir=str(adapter_dir),
                merge_dir=str(merge_dir),
                released=str(released),
            )
            self.assertTrue(stale.merged)
            self.assertFalse(stale.deployed)
            del run_dir


class TestContinuationChoice(unittest.TestCase):
    """Pick resume, recover, complete, spent, or empty for one run state."""

    def _state(self, run_dir: Path, **overrides):
        """Inspect a temporary run with explicit artifact locations."""
        root = run_dir.parent.parent
        return continuation.inspect(
            RUN_ID,
            root=str(run_dir.parent),
            adapter_dir=str(root / "adapter"),
            merge_dir=str(root / "merge"),
            released=str(root / "data" / "deployed_model.json"),
            **overrides,
        )

    def test_an_unclaimed_run_resumes_from_its_checkpoint(self) -> None:
        """Resume a run that still has budget and never claimed its test."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory), started_at=1_000.0)
            _adapter(run_dir / "round-01" / "checkpoint-50" / "continuation-1")

            state = self._state(run_dir)
            decision = continuation.choose(
                state, hours=20.0, reserves_hours=2.0, now=1_000.0 + 3_600
            )

            self.assertEqual(decision.action, "resume")
            self.assertEqual(decision.code, 0)
            self.assertTrue(decision.checkpoint.endswith("continuation-1"))

    def test_an_expired_run_points_at_a_new_run(self) -> None:
        """Stop a resume whose original cap is spent, and name the fork command."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory), started_at=0.0)
            _adapter(run_dir / "round-01" / "checkpoint-50" / "continuation-1")

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=19 * 3600
            )

            self.assertEqual(decision.action, "expired")
            self.assertEqual(decision.code, 1)
            self.assertIn("--fork-checkpoint", decision.message)

    def test_a_claimed_run_with_a_saved_adapter_recovers(self) -> None:
        """Rebuild the release when the adapter was written after the claim."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            marker = run_dir / continuation.ATTEMPT_MARKER
            marker.mkdir()
            _adapter(root / "adapter")
            os.utime(marker, (1_000.0, 1_000.0))
            os.utime(root / "adapter" / "adapter_model.safetensors", (2_000.0, 2_000.0))

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=3_000.0
            )

            self.assertEqual(decision.action, "recover")
            self.assertEqual(decision.code, 0)

    def test_a_claimed_run_without_an_adapter_is_spent(self) -> None:
        """Refuse to guess a release that was never saved."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory))
            (run_dir / continuation.ATTEMPT_MARKER).mkdir()

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=3_000.0
            )

            self.assertEqual(decision.action, "spent")
            self.assertEqual(decision.code, 1)
            self.assertIn("reserves its own test", decision.message)

    def test_a_released_run_needs_nothing(self) -> None:
        """Report a finished run instead of rebuilding it."""
        digest = "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)
            (run_dir / continuation.ATTEMPT_MARKER).mkdir()
            _adapter(root / "adapter")
            os.utime(run_dir / continuation.ATTEMPT_MARKER, (1_000.0, 1_000.0))
            os.utime(root / "adapter" / "adapter_model.safetensors", (2_000.0, 2_000.0))
            _write(root / "merge" / "merge_stamp.json", {"adapter_sha256": digest})
            _write(root / "data" / "deployed_model.json", {"adapter_sha256": digest})

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=3_000.0
            )

            self.assertEqual(decision.action, "complete")
            self.assertEqual(decision.code, 0)

    def test_a_resume_carries_the_data_the_run_started_with(self) -> None:
        """Resume on the rows the run recorded, not on whatever is configured now."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(
                Path(directory),
                inputs=["Modelfile", "data/dataset.recovery-a.candidate.jsonl"],
            )
            _adapter(run_dir / "round-01" / "checkpoint-50" / "continuation-1")

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=3_600.0
            )

            self.assertEqual(decision.action, "resume")
            self.assertEqual(
                decision.inputs, ("data/dataset.recovery-a.candidate.jsonl",)
            )

    def test_a_run_without_recorded_inputs_carries_none(self) -> None:
        """Older runs keep the configured inputs instead of inventing a path."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory))
            _adapter(run_dir / "round-01" / "checkpoint-50" / "continuation-1")

            decision = continuation.choose(
                self._state(run_dir), hours=20.0, reserves_hours=2.0, now=3_600.0
            )

            self.assertEqual(decision.inputs, ())

    def test_a_run_without_checkpoints_cannot_resume(self) -> None:
        """Report an empty run rather than inventing a starting checkpoint."""
        with tempfile.TemporaryDirectory() as directory:
            decision = continuation.choose(
                self._state(_run(Path(directory))),
                hours=20.0,
                reserves_hours=2.0,
                now=10.0,
            )

            self.assertEqual(decision.action, "empty")
            self.assertEqual(decision.code, 1)


class TestRecordedInputs(unittest.TestCase):
    """Record the data a run starts with and restore it when resuming."""

    def test_a_new_run_records_the_files_it_trains_on(self) -> None:
        """Save the inputs beside the stride so a resume can reuse them."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "outputs" / RUN_ID

            artifacts.initialize_run_directory(
                str(run_dir), inputs=("Modelfile", "data/dataset.a.jsonl")
            )

            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            self.assertEqual(metadata["inputs"], ["Modelfile", "data/dataset.a.jsonl"])

    def test_continue_restores_one_combined_dataset(self) -> None:
        """Resume with the combined dataset the run recorded."""
        options = argparse.Namespace(
            dataset=None,
            candidates=False,
            train_file=None,
            validation_file=None,
            test_file=None,
        )

        cli._restore_inputs(options, ("data/dataset.a.jsonl",))

        self.assertEqual(options.dataset, "data/dataset.a.jsonl")

    def test_continue_restores_three_split_files(self) -> None:
        """Resume with the split files the run recorded."""
        options = argparse.Namespace(
            dataset=None,
            candidates=False,
            train_file=None,
            validation_file=None,
            test_file=None,
        )

        cli._restore_inputs(options, ("train.jsonl", "validation.jsonl", "test.jsonl"))

        self.assertEqual(options.train_file, "train.jsonl")
        self.assertEqual(options.validation_file, "validation.jsonl")
        self.assertEqual(options.test_file, "test.jsonl")
        self.assertIsNone(options.dataset)

    def test_explicit_data_wins_over_the_recorded_inputs(self) -> None:
        """Never silently replace data the caller named on the command line."""
        options = argparse.Namespace(
            dataset="data/other.jsonl",
            candidates=False,
            train_file=None,
            validation_file=None,
            test_file=None,
        )

        cli._restore_inputs(options, ("data/dataset.a.jsonl",))

        self.assertEqual(options.dataset, "data/other.jsonl")


class TestClaimedRunGuard(unittest.TestCase):
    """Refuse to resume a run whose one-time independent test is already spent."""

    def test_resuming_a_claimed_run_stops_before_training(self) -> None:
        """Name the recovery and fork commands instead of spending GPU time."""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory))
            checkpoint = run_dir / "round-01" / "checkpoint-50" / "continuation-1"
            _adapter(checkpoint)
            (run_dir / continuation.ATTEMPT_MARKER).mkdir()

            patched = mock.patch.object(
                resume_runtime, "validate_resume_checkpoint", side_effect=lambda path: path
            )
            with patched, self.assertRaises(SystemExit) as raised:
                pipeline_resume.prepare_checkpoint_run(
                    lambda *args, **options: ([], [], []),
                    train_file=None,
                    validation_file=None,
                    test_file=None,
                    dataset_file=None,
                    resume_checkpoint=str(checkpoint),
                    fork_checkpoint=None,
                    run_dir=str(run_dir),
                )

            message = str(raised.exception)
            self.assertIn("already claimed", message)
            self.assertIn("--continue-run", message)


class TestRecordedRecoveryDataset(unittest.TestCase):
    """Count a recovered release's rows from the dataset the run recorded."""

    def test_a_recorded_combined_dataset_is_preferred(self) -> None:
        """Use the run's own dataset instead of whatever is configured now."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root, inputs=["Modelfile", "data/run-data.jsonl"])

            self.assertEqual(
                recorded_dataset(str(run_dir), "data/other.jsonl", str(root)),
                str(root / "data" / "run-data.jsonl"),
            )

    def test_an_unrecorded_run_keeps_the_configured_dataset(self) -> None:
        """Fall back for runs that predate the record."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(root)

            self.assertEqual(
                recorded_dataset(str(run_dir), "data/other.jsonl", str(root)),
                "data/other.jsonl",
            )

    def test_split_files_keep_the_configured_dataset(self) -> None:
        """A split-file run has no single combined dataset to count."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _run(
                root, inputs=["Modelfile", "train.jsonl", "valid.jsonl", "test.jsonl"]
            )

            self.assertEqual(
                recorded_dataset(str(run_dir), "data/other.jsonl", str(root)),
                "data/other.jsonl",
            )


class TestRunDiscovery(unittest.TestCase):
    """Find the newest run and the newest run that claimed its test."""

    def test_continue_skips_a_run_that_never_saved_a_checkpoint(self) -> None:
        """Continue the newest run that has something to continue, not a dead one."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = _run(root)
            _adapter(older / "round-01" / "checkpoint-50" / "continuation-1")
            newer = root / "outputs" / "20260102-000000"
            _write(
                newer / "run_metadata.json",
                {"started_at": 0.0, "max_steps_per_round": 500},
            )

            self.assertEqual(
                continuation.continuable_run(str(root / "outputs")), RUN_ID
            )

    def test_continue_stops_at_a_run_that_spent_its_test(self) -> None:
        """A claimed run is the newest thing to act on even without checkpoints."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = _run(root)
            _adapter(older / "round-01" / "checkpoint-50" / "continuation-1")
            newer = root / "outputs" / "20260102-000000"
            _write(
                newer / "run_metadata.json",
                {"started_at": 0.0, "max_steps_per_round": 500},
            )
            (newer / continuation.ATTEMPT_MARKER).mkdir()

            self.assertEqual(
                continuation.continuable_run(str(root / "outputs")),
                "20260102-000000",
            )

    def test_no_continuable_run_reports_none(self) -> None:
        """Report nothing to continue when no run wrote metadata."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                continuation.continuable_run(str(Path(directory) / "missing"))
            )

    def test_newest_claimed_run_skips_later_unclaimed_runs(self) -> None:
        """Claim status decides recovery, not directory order alone."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run(root)
            for run_id in ("20260102-000000", "20260103-000000"):
                run_dir = root / "outputs" / run_id
                _write(
                    run_dir / "run_metadata.json",
                    {"started_at": 0.0, "max_steps_per_round": 500},
                )
            (root / "outputs" / "20260102-000000" / continuation.ATTEMPT_MARKER).mkdir()

            self.assertEqual(
                continuation.newest_run(str(root / "outputs")), "20260103-000000"
            )
            self.assertEqual(
                continuation.newest_claimed_run(str(root / "outputs")),
                "20260102-000000",
            )
            self.assertIsNone(
                continuation.newest_claimed_run(str(root / "outputs" / "missing"))
            )


if __name__ == "__main__":
    unittest.main()
