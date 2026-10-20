# Deterministic repository analysis

The `detection/` package provides deterministic diagnostics without calling Ollama. It combines explicit text markers, code/documentation metrics, and quality findings into diagnostic scores and citations. The grading runtime does not use its verdicts to score projects; it may use only factual counts and coverage as prompt context. These heuristics are not proof of authorship.

## Analysis interface

Use `detection.analyze(repository_dump)` to obtain a mapping with:

| Field | Meaning |
| --- | --- |
| `ai_score` | Heuristic diagnostic estimate from 0 to 100; it is not used as a project score. |
| `ai_findings`, `ai_evidence` | Authorship-related findings and readable evidence citations. |
| `human_findings`, `human_evidence` | Human-style traces; absence of such traces is not proof of generated code. |
| `quality_score`, `quality_findings`, `quality_deductions` | Separate code-quality measurement and its cited deductions. |
| `metrics` | Counts and structural measurements used by the analysis. |
| `coverage` | Files shown, found, fetched, and whether the dump is truncated. |

A low score means this detector found fewer of its configured signals; it does not prove code was written by a person. A high score indicates reviewable signals, not certainty. Repo-level statements explicitly claiming AI generation receive strong heuristic weight, but project claims can also be inaccurate or outdated. Review the cited source and repository context.

## Package map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports `analyze`, `build_report`, `score_ai`, `score_quality`, `files_in`, the signal catalogues, and the tuning constants. |
| `base.py` | Defines score thresholds, weights, file types, and bounds. |
| `signals.py` | Catalogues text patterns used as authorship and quality clues. |
| `metrics.py` | Measures repository structure, comments, documentation, and coverage. |
| `findings.py` | Attributes analysis findings to evidence files. |
| `quality.py` | Builds quality findings and citations. |
| `scoring.py` | Combines findings into bounded scores. |
| `analysis.py` | Coordinates metrics, findings, scores, and coverage. |
| `reports.py` | Renders findings in the shared report format. |

The candidate builder uses detector measurements to create some training targets. That data-generation role does not make the detector a live grader: runtime project scores come from the judge alone. Agreement with detector-derived targets cannot establish real-world authorship accuracy; evaluate against independent human-reviewed labels.
