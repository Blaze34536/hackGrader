# Design records

This directory holds HackGrader design documents. Each one is a dated record of a direction and its reasoning, not reference documentation for the shipped code. For current behavior, read the [application source map](../src/README.md) and the [model workspace guide](../model/README.md).

| Spec | Date | Status | What it argues |
| --- | --- | --- | --- |
| [Judge-owned scores](superpowers/specs/2026-10-14-judge-owned-scores-design.md) | 2026-10-14 | Historical design; flag behavior superseded | Originally proposed retaining detector verdicts and turning score divergence into flags. Current behavior passes factual metrics only, publishes judge-owned scores, and generates no project flags. |
| [Repository AI-code percentage: restart design](superpowers/specs/2026-10-12-repository-ai-percentage-design.md) | 2026-10-12 | Draft, awaiting review | Defines `AI%` as a coverage-relative fraction, shows why band-midpoint targets and style-derived labels cap achievable accuracy, and proposes measured provenance as the target. |
| [Provenance-carrying evidence view](superpowers/specs/2026-10-12-provenance-evidence-view-design.md) | 2026-10-12 | Superseded by the restart design | Would have carried commit-trailer provenance from collection into the bounded evidence view and the documented prompt. |

Both records cite diagnostic scripts, logs, and test suites that have since been removed from the workspace. Those citations are kept because the measurements they recorded are quoted inline; the paths are historical records rather than links to files you can open.

## Conventions

- A spec states its date and status in the header, including the document it supersedes.
- Specs describe intent and trade-offs; the package READMEs beside the code describe current behavior.
