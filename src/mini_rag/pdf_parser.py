import logging
import re
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

_logger: logging.Logger = logging.getLogger(__name__)


def parse(
    file_path: Path,
    footer_line_patterns: list[str] | None = None,
    skip_pages: list[int] | None = None,
) -> str:
    compiled_footer_patterns = [
        re.compile(pattern) for pattern in (footer_line_patterns or []) if pattern
    ]
    skip_page_numbers = set(skip_pages or [])

    _logger.info("Parsing document: %s", file_path)
    page_texts = []
    with pdfplumber.open(str(file_path)) as pdf:
        page_count = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages, start=1):
            if page_number in skip_page_numbers:
                continue
            page_texts.append(_extract_page(page, compiled_footer_patterns))
    text = "\n".join(page_texts)
    _logger.info(
        "Parsed document: %s (%d pages, %d characters)",
        file_path,
        page_count,
        len(text),
    )
    return text


def _is_footer_line(text: str, footer_patterns: list[re.Pattern[str]]) -> bool:
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in footer_patterns)


def _extract_page(page: Page, footer_patterns: list[re.Pattern[str]]) -> str:
    tables = page.find_tables()

    prose_page = page
    for table in tables:
        prose_page = prose_page.outside_bbox(table.bbox)
    prose_lines = prose_page.extract_text_lines()

    segments: list[tuple[float, str]] = [
        (line["top"], line["text"])
        for line in prose_lines
        if not _is_footer_line(line["text"], footer_patterns)
    ]

    for table in tables:
        serialized = _serialize_table(table.extract())
        if serialized:
            segments.append((table.bbox[1], serialized))

    segments.sort(key=lambda segment: segment[0])

    output_parts: list[str] = [text for _, text in segments]
    return "\n".join(output_parts)


def _serialize_table(table_rows: list[list[str | None]]) -> str:
    lines = []
    for row in table_rows:
        cells = [(cell or "").replace("\n", " ").strip() for cell in row]
        cells = [cell for cell in cells if cell]  # drop empty cells
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)
