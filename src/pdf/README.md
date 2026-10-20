# PDF reports

The `pdf/` package turns saved HackGrader judge reports into a printable PDF. It does not crawl repositories, read GitHub, call Ollama, or re-annotate scores.

## Command

Run from the repository root:

```bash
uv run hackgrader-pdf -h
uv run hackgrader-pdf
uv run hackgrader-pdf scores.json report.pdf
```

The optional positional arguments are `input` and `output`. Defaults are root `scores.json` and root `report.pdf`. The input must be valid JSON containing project results; missing input or PDF-generation errors return a non-zero status. The command reports layout issues and page/project count differences when detected; a report holds one cover page plus at least one page per project.

## Page contents

The report opens with a cover page stating when it was generated, which input file was read, the grader build tag, and the project count. The index lists each project's judge-reported scores and repository coverage, with no project-flag column.

Each project page shows the AI-involvement and code-quality scores from the saved report, the judge's ranked AI evidence, and its code-quality deductions. AI signs are shown in the strongest-to-weakest order given by the judge, with its explanation and citation preserved. A project the grader could not score prints "Evaluation failed" with the opening of the returned reason and a pointer to `scores.json`, which holds the full output.

Only scores stated by the judge are rendered. Detector verdicts and findings are not used to score projects; factual counts and coverage may appear as prompt context during grading. Old flags and detector-verdict fields in previously saved JSON are removed during annotation and are not rendered.

Each evidence panel is bounded to six entries. Quality deductions lead with the largest impact, first: the panel orders them by the percentage each one states. A line under a panel counts any entries left out and points to `scores.json`, where all recorded entries remain. The judge's evidence entries are never re-ranked by a heuristic.

Every page carries its number in `Page N of M` form, and each project's pages carry a running header naming that project, so a printed stack can be reassembled. The cover's build tag is read from the local release record at `model/data/deployed_model.json`; the line is omitted when that record is missing or unreadable.

Panels are drawn without a background or border. Rendering uses reportlab's platypus layer, which draws no filled rectangles for this report, and `tests/test_report.py` asserts that no page draws one. Page numbers are stamped by a canvas subclass, because the HTML-based renderer this package used previously could not draw them.

The input is typically produced by the [gallery grading workflow](../hackgrader/README.md). A PDF is a presentation of the saved judge reports, not a new evaluation of the repositories or a verification of authorship.

## Package map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker; the command resolves to `pdf.cli:main`. |
| `cli.py` | Reads the input JSON and coordinates rendering. |
| `context.py` | Prepares the cover record and one printable record per project. |
| `layout.py` | Cleans and wraps text to fit the page. |
| `renderer.py` | Builds the cover and project pages, stamps headers and page numbers, and writes the PDF. |