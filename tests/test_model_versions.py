"""Sequential versioning for released model adapters.

The ledger is the history: a released adapter keeps the version it was given, a retry of
the same digest reuses it, and a version is only written after the release is verified.
"""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import learning.deployment as deployment
from learning.deployment import versioning

DIGEST_ONE = "a" * 64
DIGEST_TWO = "b" * 64
RELEASED = "2026-10-19T12:00:00"


def _assign(path: Path, digest: str, *, expected: int | None = None) -> int:
    """Record one verified release through the public ledger API."""
    return versioning.assign_version(
        digest,
        run_id="20261019-120000",
        tag=f"hackgrader:v{expected or 1}",
        gguf="models/ai_detector.gguf",
        released_at=RELEASED,
        expected_version=expected,
        path=path,
    )


class TestVersionLedger(unittest.TestCase):
    """Keep one monotonic version per released adapter digest."""

    def test_a_missing_ledger_starts_at_version_one(self) -> None:
        """A fresh checkout previews version one without creating the ledger."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versions.json"
            self.assertEqual(versioning.preview_version(DIGEST_ONE, path=path), 1)
            self.assertFalse(path.exists())

    def test_assigning_writes_the_entry_and_advances_the_sequence(self) -> None:
        """Two distinct adapters become v1 and v2 with their tags recorded."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versions.json"
            first = _assign(path, DIGEST_ONE, expected=1)
            second = versioning.assign_version(
                DIGEST_TWO,
                run_id="20261019-130000",
                tag="hackgrader:v2",
                gguf="models/second.gguf",
                released_at=RELEASED,
                expected_version=2,
                path=path,
            )
            ledger = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((first, second), (1, 2))
            self.assertEqual(ledger["latest"], 2)
            self.assertEqual([entry["version"] for entry in ledger["models"]], [1, 2])
            self.assertEqual(ledger["models"][1]["tag"], "hackgrader:v2")
            self.assertEqual(ledger["models"][1]["gguf"], "models/second.gguf")

    def test_the_same_digest_reuses_its_version_instead_of_advancing(self) -> None:
        """A retried release keeps its original number and entry."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versions.json"
            _assign(path, DIGEST_ONE, expected=1)
            reused = versioning.assign_version(
                DIGEST_ONE,
                run_id="20261019-140000",
                tag="hackgrader:v9",
                gguf="models/retry.gguf",
                released_at=RELEASED,
                path=path,
            )
            ledger = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reused, 1)
            self.assertEqual(len(ledger["models"]), 1)
            self.assertEqual(ledger["latest"], 1)

    def test_a_pinned_expected_version_refuses_a_raced_allocation(self) -> None:
        """A number recorded by another deploy stops this release cleanly."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versions.json"
            _assign(path, DIGEST_ONE, expected=1)
            with self.assertRaises(SystemExit) as caught:
                versioning.assign_version(
                    DIGEST_TWO,
                    run_id="20261019-150000",
                    tag="hackgrader:v2",
                    gguf="models/second.gguf",
                    released_at=RELEASED,
                    expected_version=1,
                    path=path,
                )
            self.assertIn("another deployment", str(caught.exception).lower())

    def test_a_malformed_ledger_stops_instead_of_overwriting_history(self) -> None:
        """Inconsistent history is reported rather than silently replaced."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versions.json"
            path.write_text('{"latest": 4, "models": []}', encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                versioning.preview_version(DIGEST_ONE, path=path)
            self.assertIn("latest", str(caught.exception))

    def test_version_tags_are_human_readable(self) -> None:
        """The tag names the base model and the version alone."""
        self.assertEqual(versioning.version_tag("hackgrader", 4), "hackgrader:v4")
        self.assertEqual(versioning.version_tag("custom", 12), "custom:v12")


class TestReleaseFlow(unittest.TestCase):
    """Register the version tag and record it only after verification."""

    @contextmanager
    def _modelfile(self, *_args, **_kwargs):
        """Yield a stand-in generated Modelfile path."""
        yield Path("Modelfile")

    def _run(self, directory: str, *, dry_run: bool):
        """Run the deployment flow with the external tools replaced by fakes."""
        with (
            patch.object(deployment, "MODEL_DIR", Path(directory)),
            patch.object(versioning, "MODEL_DIR", Path(directory)),
            patch.object(
                deployment,
                "adapter_stamp",
                return_value={"adapter_sha256": DIGEST_ONE},
            ),
            patch.object(
                deployment,
                "_versioned_gguf_name",
                return_value="models/probe.gguf",
            ),
            patch.object(deployment, "convert"),
            patch.object(deployment, "register") as register,
            patch.object(deployment, "_versioned_modelfile", self._modelfile),
            patch.object(deployment, "served_gguf", return_value=True),
        ):
            result = deployment.run(
                dry_run=dry_run, merge_dir="models/merge", modelfile="Modelfile"
            )
        return result, register

    def test_a_verified_release_registers_and_records_version_one(self) -> None:
        """The first successful deploy becomes hackgrader:v1."""
        with tempfile.TemporaryDirectory() as directory:
            result, register = self._run(directory, dry_run=False)
            self.assertTrue(result)
            register.assert_called_once()
            self.assertEqual(register.call_args.args[1], "hackgrader:v1")
            ledger = json.loads(
                (Path(directory) / "data" / "model_versions.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(ledger["latest"], 1)
            record = json.loads(
                (Path(directory) / "data" / "deployed_model.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["version"], 1)
            self.assertEqual(record["tag"], "hackgrader:v1")

    def test_a_dry_run_previews_the_version_without_recording_it(self) -> None:
        """Dry-run names v1 but writes no ledger or release record."""
        with tempfile.TemporaryDirectory() as directory:
            result, register = self._run(directory, dry_run=True)
            self.assertTrue(result)
            self.assertEqual(register.call_args.args[1], "hackgrader:v1")
            self.assertFalse((Path(directory) / "data" / "model_versions.json").exists())
            self.assertFalse((Path(directory) / "data" / "deployed_model.json").exists())


class TestReleaseRecord(unittest.TestCase):
    """Write the version beside the release pointer the grader reads."""

    def test_record_release_carries_the_version(self) -> None:
        """The pointer the grader reads names the released version."""
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(deployment, "MODEL_DIR", Path(directory)):
                deployment.record_release(
                    "hackgrader:v4", DIGEST_TWO, "models/x.gguf", version=4
                )
                record = json.loads(
                    (Path(directory) / "data" / "deployed_model.json").read_text(
                        encoding="utf-8"
                    )
                )
            self.assertEqual(record["version"], 4)
            self.assertEqual(record["tag"], "hackgrader:v4")
            self.assertEqual(record["gguf"], "models/x.gguf")

    def test_an_unversioned_record_omits_the_version(self) -> None:
        """Provisional releases keep their previous record shape."""
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(deployment, "MODEL_DIR", Path(directory)):
                deployment.record_release(
                    "hackgrader-provisional:abc", DIGEST_TWO, "models/x.gguf"
                )
                record = json.loads(
                    (Path(directory) / "data" / "deployed_model.json").read_text(
                        encoding="utf-8"
                    )
                )
            self.assertNotIn("version", record)
