"""Build the printable PDF report as a reportlab platypus document.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import LayoutError
from reportlab.platypus.flowables import HRFlowable

from .layout import (
    BODY_FONT,
    BODY_SIZE,
    BODY_WIDTH,
    COLUMN_PT,
    HORIZONTAL_MARGIN_PT,
    TITLE_FONT,
    TITLE_SIZE,
    TITLE_WIDTH,
    sanitize,
)

PAGE_WIDTH_PT, PAGE_HEIGHT_PT = letter
HEADER_FOOTER_FONT = BODY_FONT
HEADER_FOOTER_SIZE = 8
FOOTER_BASELINE_PT = 30
HEADER_BASELINE_PT = PAGE_HEIGHT_PT - 40
REPORT_TITLE = "Hackathon Code Review Report"
RULE_COLOUR = colors.HexColor("#e5e7e9")
GRID_COLOUR = colors.HexColor("#d5d8dc")
MUTED_COLOUR = colors.HexColor("#566573")
BODY_COLOUR = colors.HexColor("#424949")
HEADING_COLOUR = colors.HexColor("#2c3e50")
ACCENT_COLOUR = colors.HexColor("#2980b9")
ALERT_COLOUR = colors.HexColor("#922b21")

_REPORT_TITLE = ParagraphStyle(
    "ReportTitle",
    fontName="Helvetica-Bold",
    fontSize=24,
    leading=28,
    alignment=1,
    textColor=HEADING_COLOUR,
    spaceAfter=12,
)
_COVER_LINE = ParagraphStyle(
    "CoverLine",
    fontName=BODY_FONT,
    fontSize=10,
    leading=14,
    alignment=1,
    textColor=MUTED_COLOUR,
)
_SECTION = ParagraphStyle(
    "Section",
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=HEADING_COLOUR,
    spaceBefore=10,
    spaceAfter=6,
)
_PROJECT_TITLE = ParagraphStyle(
    "ProjectTitle",
    fontName=TITLE_FONT,
    fontSize=TITLE_SIZE,
    leading=TITLE_SIZE + 4,
    alignment=1,
    textColor=ACCENT_COLOUR,
    spaceAfter=4,
)
_STATUS = ParagraphStyle(
    "Status",
    fontName=BODY_FONT,
    fontSize=11,
    leading=14,
    alignment=1,
    textColor=ALERT_COLOUR,
    spaceAfter=8,
)
_SCORE_LABEL = ParagraphStyle(
    "ScoreLabel",
    fontName=BODY_FONT,
    fontSize=9,
    leading=12,
    alignment=1,
    textColor=MUTED_COLOUR,
)
_SCORE_VALUE = ParagraphStyle(
    "ScoreValue",
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    alignment=1,
    textColor=HEADING_COLOUR,
)
_COVERAGE = ParagraphStyle(
    "Coverage",
    fontName=BODY_FONT,
    fontSize=10,
    leading=13,
    alignment=1,
    textColor=MUTED_COLOUR,
)
_PANEL_TITLE = ParagraphStyle(
    "PanelTitle",
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    textColor=HEADING_COLOUR,
    # Panels are separated by their own leading, as the previous HTML panel padding did.
    spaceBefore=7,
    keepWithNext=1,
)
_LINE = ParagraphStyle(
    "Line",
    fontName=BODY_FONT,
    fontSize=BODY_SIZE,
    leading=round(BODY_SIZE * 1.45, 1),
    textColor=BODY_COLOUR,
)
_EMPTY = ParagraphStyle(
    "Empty",
    parent=_LINE,
    fontName="Helvetica-Oblique",
    textColor=colors.HexColor("#7f8c8d"),
)
_CELL = ParagraphStyle(
    "Cell",
    fontName=BODY_FONT,
    fontSize=9,
    leading=11.5,
    textColor=BODY_COLOUR,
)
_CELL_HEAD = ParagraphStyle("CellHead", parent=_CELL, fontName="Helvetica-Bold")
_CELL_CENTER = ParagraphStyle("CellCenter", parent=_CELL, alignment=1)

PANEL_GAP_PT = 9


def _text(value: Any) -> str:
    """Sanitize stored text and escape it for the paragraph markup."""
    return escape(sanitize(value))


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    """Build one paragraph from sanitized and escaped stored text."""
    return Paragraph(_text(value), style)


def _project_title(project: dict[str, Any]) -> Paragraph:
    """Build a project heading and tag it so the page header can name the project."""
    lines = project.get("title_lines") or [project.get("name") or ""]
    paragraph = Paragraph("<br/>".join(_text(line) for line in lines), _PROJECT_TITLE)
    paragraph.project_name = str(project.get("name") or "")
    return paragraph


def _rule() -> HRFlowable:
    """Draw the hairline that separates a panel title from its lines."""
    rule = HRFlowable(width="100%", thickness=0.5, color=RULE_COLOUR, spaceAfter=4)
    rule.keepWithNext = 1
    return rule


def _remainder(more: Any) -> str:
    """Word the pointer to the lines a panel left out, which stay in the scores file."""
    count = int(more or 0)
    if not count:
        return ""
    plural = "s" if count != 1 else ""
    return f"and {count} more item{plural} recorded in scores.json."


def _panel(
    title: str,
    lines: list[str],
    *,
    fallback: str = "",
    note: str = "",
) -> list[Any]:
    """Build one titled panel without a tint or box around it."""
    flowables: list[Any] = [_paragraph(title, _PANEL_TITLE), _rule()]
    rendered = [_paragraph(line, _LINE) for line in lines]
    if not rendered and fallback:
        rendered = [_paragraph(fallback, _EMPTY)]
    flowables.extend(rendered)
    if note:
        flowables.append(_paragraph(note, _EMPTY))
    return flowables


def _score_table(project: dict[str, Any]) -> list[Any]:
    """Build the reported-score columns for one project."""
    header = [_paragraph("AI involvement reported", _SCORE_LABEL)]
    header.append(_paragraph("Code quality reported", _SCORE_LABEL))
    values = [_paragraph(project.get("ai", "n/a"), _SCORE_VALUE)]
    values.append(_paragraph(project.get("code", "n/a"), _SCORE_VALUE))
    table = Table([header, values], colWidths=[COLUMN_PT / 2, COLUMN_PT / 2])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOUR),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return [table, Spacer(1, PANEL_GAP_PT)]


def _summary_table(rows: list[dict[str, str]]) -> Table:
    """Build the project index: reported scores and coverage."""
    headings = ["Project", "AI reported", "Quality reported", "Coverage"]
    data = [[_paragraph(heading, _CELL_HEAD) for heading in headings]]
    for row in rows:
        data.append(
            [
                _paragraph(row.get("name", ""), _CELL),
                _paragraph(row.get("ai", "n/a"), _CELL_CENTER),
                _paragraph(row.get("code", "n/a"), _CELL_CENTER),
                _paragraph(row.get("coverage", "") or "-", _CELL_CENTER),
            ]
        )
    # Widths retain the former project/score/coverage proportions without a flags column.
    table = Table(data, colWidths=[168.4, 62, 84, 168], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, GRID_COLOUR),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _cover_flowables(cover: dict[str, Any] | None) -> list[Any]:
    """Build the summary and provenance page that opens the report."""
    if not cover:
        return []
    flowables: list[Any] = [_paragraph(REPORT_TITLE, _REPORT_TITLE)]
    for line in cover.get("provenance") or []:
        flowables.append(_paragraph(line, _COVER_LINE))
    flowables.append(_paragraph("Projects", _SECTION))
    flowables.append(_summary_table(cover.get("rows") or []))
    flowables.append(PageBreak())
    return flowables


def _project_flowables(project: dict[str, Any]) -> list[Any]:
    """Build one project's page: scores, ranked evidence, and deductions."""
    flowables: list[Any] = [_project_title(project)]
    if project.get("status"):
        flowables.append(_paragraph(project["status"], _STATUS))
    flowables.extend(_score_table(project))
    if project.get("coverage"):
        flowables.append(_paragraph(project["coverage"], _COVERAGE))
    if project.get("failure"):
        flowables.extend(
            _panel(
                "Evaluation failed",
                project["failure"],
                note="The full grader output is recorded in scores.json.",
            )
        )
    else:
        flowables.extend(
            _panel(
                f"AI involvement evidence ({project.get('ai', 'n/a')})",
                project.get("ai_items") or [],
                fallback="The judge's report recorded no evidence entry for this score.",
                note=_remainder(project.get("ai_more")),
            )
        )
        flowables.append(Spacer(1, PANEL_GAP_PT))
        flowables.extend(
            _panel(
                f"Code quality deductions ({project.get('code', 'n/a')})",
                project.get("quality_items") or [],
                fallback="No deductions applied.",
                note=_remainder(project.get("quality_more")),
            )
        )
    return flowables


def build_story(
    projects: list[dict[str, Any]], cover: dict[str, Any] | None = None
) -> list[Any]:
    """Build the document story: the cover page, then one page per project."""
    story = _cover_flowables(cover)
    for index, project in enumerate(projects):
        if index:
            story.append(PageBreak())
        story.extend(_project_flowables(project))
    return story


class _ReportDoc(SimpleDocTemplate):
    """Record which project each page belongs to so the header can name it."""

    def __init__(self, *args: Any, page_headers: dict[int, str], **kwargs: Any) -> None:
        """Build the document and keep the page-to-project map it fills in."""
        super().__init__(*args, **kwargs)
        self._page_headers = page_headers

    def afterFlowable(self, flowable: Any) -> None:
        """Note the project whose heading was just drawn on the current page."""
        name = getattr(flowable, "project_name", "")
        if name:
            self._page_headers[self.page] = name


def _fitted(value: str, width: float) -> str:
    """Trim a running header value to the space the page actually has."""
    if stringWidth(value, HEADER_FOOTER_FONT, HEADER_FOOTER_SIZE) <= width:
        return value
    trimmed = value
    while trimmed and stringWidth(
        f"{trimmed}...", HEADER_FOOTER_FONT, HEADER_FOOTER_SIZE
    ) > width:
        trimmed = trimmed[:-1]
    return f"{trimmed}..." if trimmed else ""


def _canvas_maker(
    page_headers: dict[int, str], stamp: str
) -> type[pdf_canvas.Canvas]:
    """Build the canvas class that stamps headers and page numbers.

    ``canvasmaker`` is honoured only by ``DocTemplate.build``, and it is created before
    any page exists, so the class reads the page map and the run stamp from a closure.
    """

    class _StampedCanvas(pdf_canvas.Canvas):
        """Add the running header and page numbers once every page is known."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Create the canvas and start recording per-page drawing states."""
            super().__init__(*args, **kwargs)
            self._states: list[dict[str, Any]] = []

        def showPage(self) -> None:
            """Record the finished page's state instead of writing it out yet."""
            self._states.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            """Stamp each recorded page with its header and number, then write it out."""
            total = len(self._states)
            for number, state in enumerate(self._states, 1):
                self.__dict__.update(state)
                self._stamp(number, total)
                super().showPage()
            super().save()

        def _stamp(self, number: int, total: int) -> None:
            """Draw the footer stamp and, on project pages, the running header."""
            left = HORIZONTAL_MARGIN_PT
            right = PAGE_WIDTH_PT - HORIZONTAL_MARGIN_PT
            self.setFont(HEADER_FOOTER_FONT, HEADER_FOOTER_SIZE)
            self.setFillColor(MUTED_COLOUR)
            self.drawString(left, FOOTER_BASELINE_PT, stamp)
            self.drawRightString(right, FOOTER_BASELINE_PT, f"Page {number} of {total}")
            project = page_headers.get(number)
            if project:
                self.drawString(left, HEADER_BASELINE_PT, REPORT_TITLE)
                room = COLUMN_PT - stringWidth(
                    REPORT_TITLE, HEADER_FOOTER_FONT, HEADER_FOOTER_SIZE
                )
                self.drawRightString(
                    right,
                    HEADER_BASELINE_PT,
                    _fitted(project, max(0.0, room - 12)),
                )
            self.setFillColor(colors.black)

    return _StampedCanvas


def overlong_lines(projects: list[dict[str, Any]]) -> int:
    """Count display lines whose measured width still exceeds their layout column."""
    checks = [
        ("title_lines", TITLE_FONT, TITLE_SIZE, TITLE_WIDTH),
        ("ai_items", BODY_FONT, BODY_SIZE, BODY_WIDTH),
        ("quality_items", BODY_FONT, BODY_SIZE, BODY_WIDTH),
        ("failure", BODY_FONT, BODY_SIZE, BODY_WIDTH),
    ]
    return sum(
        stringWidth(line, font, size) > width
        for project in projects
        for key, font, size, width in checks
        for line in project.get(key) or []
    )


def count_pages(path: Path) -> int | None:
    """Count PDF page objects, returning none when the output cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    return pages if pages > 0 else None


def write_pdf(
    path: Path,
    projects: list[dict[str, Any]],
    cover: dict[str, Any] | None = None,
) -> int:
    """Write the report PDF and return the number of problems, or zero on success."""
    story = build_story(projects, cover)
    page_headers: dict[int, str] = {}
    stamp = str((cover or {}).get("generated_at", "")) or "Hackathon Code Review Report"
    try:
        document = _ReportDoc(
            str(path),
            pagesize=letter,
            leftMargin=HORIZONTAL_MARGIN_PT,
            rightMargin=HORIZONTAL_MARGIN_PT,
            topMargin=HORIZONTAL_MARGIN_PT,
            bottomMargin=HORIZONTAL_MARGIN_PT,
            title=REPORT_TITLE,
            page_headers=page_headers,
        )
        document.build(story, canvasmaker=_canvas_maker(page_headers, stamp))
    except (OSError, ValueError, LayoutError) as exc:
        print(f"[!] PDF generation failed: {exc}")
        return 1
    return 0
