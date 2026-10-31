# Model weights

The weights (~52 GB) are too large for git, so they live as 28 parts of one tar archive
on the release `model-weights-2026-10-31`.

Contents: `model/models/` (merged Qwen3 AI-detector weights and the q8 GGUF) and
`model/outputs/` (training run outputs).

Restore: `bash model_weights/restore.sh`. It downloads every part, checks it against
`SHA256SUMS`, and unpacks into `model/`.
