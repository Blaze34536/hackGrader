"""Measure, sanitize, and wrap text for the PDF page layout.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import unicodedata
from typing import Any

from reportlab.pdfbase.pdfmetrics import stringWidth

PAGE_WIDTH_PT = 8.5 * 72
HORIZONTAL_MARGIN_PT = 0.9 * 72
COLUMN_PT = PAGE_WIDTH_PT - 2 * HORIZONTAL_MARGIN_PT
BODY_FONT = "Helvetica"
BODY_SIZE = 11
BODY_WIDTH = COLUMN_PT - 30
TITLE_FONT = "Helvetica-Bold"
TITLE_SIZE = 16
TITLE_WIDTH = COLUMN_PT

# En and em dashes are part of the WinAnsi set the base-14 Helvetica font encodes, so they are
# kept as written; folding them to hyphens changed how review prose reads.
KEPT_DASHES = ("\u2013", "\u2014")

CHARACTER_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...", "\u00a0": " ",
    "\u2022": "-", "\u2192": "to",
}


def sanitize(text: Any) -> str:
    """Fold text to characters the PDF font can display reliably."""
    folded = unicodedata.normalize("NFKD", str(text))
    output: list[str] = []
    for char in folded:
        if unicodedata.combining(char):
            continue
        if char == "\n":
            output.append(char)
            continue
        if ord(char) < 32 or ord(char) == 127:
            # Evidence text can carry control characters from binary artifacts such as
            # `.DS_Store`, which the base-14 fonts have no glyph for: they would print as
            # blank boxes. Keep the line break handled above and render the rest as a
            # separator, so words that were adjacent do not run together.
            output.append(" ")
            continue
        if ord(char) < 128 or char in KEPT_DASHES:
            output.append(char)
        elif char in CHARACTER_REPLACEMENTS:
            output.append(CHARACTER_REPLACEMENTS[char])
        elif unicodedata.category(char) not in ("So", "Sk", "Cs"):
            output.append("?")
    return "".join(output)


def _split_token(token: str, font: str, size: float, width: float) -> tuple[str, str]:
    """Split an overlong token at a natural punctuation boundary when possible."""
    cut = len(token)
    while cut > 1 and stringWidth(token[:cut], font, size) > width:
        cut -= 1
    for separator in ("/", "-", "_", ".", ","):
        position = token.rfind(separator, 1, cut)
        if position + 1 >= cut * 0.6:
            cut = position + 1
            break
    return token[:cut], token[cut:]


def wrap_by_width(
    text: Any,
    width: float = BODY_WIDTH,
    font: str = BODY_FONT,
    size: float = BODY_SIZE,
) -> list[str]:
    """Wrap sanitized text to the selected font's measured point width."""
    lines: list[str] = []
    for paragraph in sanitize(text).split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            while stringWidth(word, font, size) > width:
                head, word = _split_token(word, font, size, width)
                if current:
                    lines.append(current)
                    current = ""
                lines.append(head)
            candidate = f"{current} {word}".strip()
            if stringWidth(candidate, font, size) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def title_lines(value: Any) -> list[str]:
    """Wrap a project title using the larger heading font."""
    return wrap_by_width(value, TITLE_WIDTH, TITLE_FONT, TITLE_SIZE)


def percent(value: Any) -> str:
    """Format a parsed percentage or show a neutral missing-value marker."""
    return f"{value}%" if isinstance(value, int) else "n/a"
