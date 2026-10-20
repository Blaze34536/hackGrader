# Learning pipeline reference

`learning.pipeline` connects input validation, training rounds, holdout evaluation, and gated release. The public `main(**options)` coordinator is in the facade/runner path; `run_training_rounds(...)` remains the lower-level round entry point.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Exposes the coordinator and round-level entry point. |
| `facade.py` | Connects the public interface to pipeline components and defaults. |
| `runner.py` | Orders preflight, run setup, training, final test, and release. |
| `inputs.py` | Loads data and validates row shape, provenance, coverage, and destinations. |
| `resume.py` | Prepares resume/fork checkpoint selection and run metadata. |
| `training.py` | Runs optimizer rounds and hands off the selected adapter. |
| `models.py` | Loads base, continuation, and selected adapters. |
| `final_test.py` | Controls the one-time independent-test evaluation. |
| `release_only.py` | Scores and releases a saved fork without an optimizer round. |
| `release.py` | Saves adapters and coordinates verified merge/deployment. |
| `merge_recovery.py` | Rebuilds the merge and registration of an interrupted release without re-scoring the test. |
| `recovery_paths.py` | Resolves the run, adapter, merge, and dataset paths a recovery may touch. |
| `recovery_plan.py` | Verifies a recoverable interrupted release and describes what would be rebuilt. |
| `recovery_evidence.py` | Recovers the selected checkpoint and proves the saved adapter still matches it. |
| `recovery_outputs.py` | Renders a recovery plan and records how a rebuilt release was produced. |

Optimizer loops, checkpoint selection, budgets, callbacks, and recovery live in the sibling [`learning.training`](../training/) package.

## Execution order

1. Inputs and output destinations are checked before the GPU runtime loads.
2. A normal run estimates its budget and initializes fresh output directories; resumes and forks validate their source checkpoint and preserve it.
3. Training data is prepared, then bounded rounds train and score candidates against validation.
4. Validation chooses continuation and release candidates. The independent test remains unavailable to training and ordinary diagnostics.
5. The gated release path can score the selected candidate once on the independent test. A failed gate blocks normal merge and Ollama registration.

`--preflight-only` exits after input/path validation and a run estimate. It does not load the model or train. `--release-only` is different: it skips optimizer rounds but can load a model, score holdouts, and produce release artifacts. `--resume-checkpoint` retains the original run and deadline; `--fork-checkpoint` starts a separate run and time cap.

`recover --run-id <run-id>` is the exception for a release that was interrupted after the one-time independent test was claimed, for example by a power loss during the merge. It re-scores nothing: the claim is single-use, so the reserve test is never evaluated again. It requires the saved selected adapter to postdate the claim and to match the checkpoint recovered from the run's validation diagnostics, then reuses the ordinary gated merge and deployment. The merged directory gains a `merge_recovery.json` recording that the claimed test's gate metrics were never archived.

## Interface and related references

Import `main` from `learning.pipeline` to run the coordinator. Import specialized operations from their owning modules, for example `learning.pipeline.inputs.validate_inputs`, `learning.pipeline.final_test.run_final_test`, `learning.pipeline.release.merge_and_deploy`, or `learning.training.checkpoints.Selection`.

- [Training preflight](../preflight/README.md) describes validation and output-path checks.
- [Holdout gates](../holdout/README.md) describes sample requirements, metrics, and uncertainty.
- [Saved-checkpoint scoring](../scoring/README.md) describes validation-only diagnostics.
- [Deployment](../deployment/README.md) describes artifact conversion and Ollama registration.

The model-side learning and training-data suites that covered these contracts are no longer part of the repository, so changes here are verified through the application contracts under `tests/` instead.
