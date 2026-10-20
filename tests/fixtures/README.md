# Detector test fixtures

`detector.py` contains fixed repository-dump strings for detector regression tests. The examples are source text embedded in Python fixtures; they are not fetched from the network during a test run.

Keep these fixtures small, stable, and clearly labeled. A test should exercise the detector behavior it names, not depend on incidental wording from a changing public repository. Do not add private repository dumps, credentials, personal data, or downloaded training datasets.

Run the offline detector tests from the repository root with:

```bash
uv run --project . --group dev pytest -q tests/test_detector.py
```
