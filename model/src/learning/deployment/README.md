# Model deployment

The `deployment/` package converts a verified merged checkpoint to a versioned GGUF artifact and can register a matching versioned Ollama tag. This is a consequential operation: it may run a converter, write a large model file, and change locally available Ollama tags. Run it only after the gated training/release workflow allows deployment.

## Inspect before deploying

Run from `model/`:

```bash
uv run src/cli.py deploy -h
uv run src/cli.py deploy --dry-run
uv run src/cli.py deploy --check
uv run src/cli.py deploy --convert-only
```

- `--dry-run` prints planned conversion and registration commands without executing them.
- `--check` verifies that the configured, run-specific GGUF corresponds to the saved provenance and Ollama's served model, and prints the version already recorded for that adapter; it does not convert or register.
- `--convert-only` creates/verifies the converted artifact but skips Ollama registration.
- A plain `deploy` performs conversion and attempts registration. It should be run only when that side effect is intended.

The artifact is named from the configured GGUF base and adapter digest. Registration uses a digest-specific model tag rather than silently replacing the default tag. The deployment path verifies the resulting Ollama blob against the expected GGUF before reporting success. Choose the verified version explicitly with `HACKGRADER_MODEL=<versioned-tag>` when appropriate.

## Sequential versions

Every normal release takes the next number from the committed ledger at `data/model_versions.json` and registers it as `<tag>:vN`, for example `hackgrader:v4`. The entry is written only after Ollama's blob is verified, so a version always names a build that passed its gates. A retried release of the same adapter digest reuses its version instead of advancing the sequence, and an interrupted or unverified deployment writes nothing.

`deploy --dry-run` prints the tag and version it would record without touching the ledger. `deploy --check` prints the version already recorded for the configured adapter. Provisional releases keep the separate `hackgrader-provisional:<digest>` tag and do not consume a version.

## Package map

| File | Responsibility |
| --- | --- |
| `cli.py` | Parses dry-run, check, conversion-only, and path options. |
| `provenance.py` | Reads adapter provenance and verifies GGUF/Ollama artifact identity. |
| `modelfile.py` | Creates a temporary Ollama Modelfile referencing the exact artifact. |
| `versioning.py` | Assigns and records the sequential version for each verified release. |
| `commands.py` | Runs conversion and Ollama registration commands. |
| `__init__.py` | Coordinates conversion and registration. |

The source merged checkpoint is not modified by conversion. Generated GGUF files and model tags are separate deployment artifacts. See the [learning workflow](../README.md) for release gates and the [model workspace guide](../../../../README.md) for output-path rules.
