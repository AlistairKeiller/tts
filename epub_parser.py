"""Extract chapter text from EPUB files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

import ebooklib
from ebooklib import epub


# ---------------------------------------------------------------------------
# Lightweight HTML → plain-text converter
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and collapse whitespace into readable text."""

    _BLOCK_TAGS = frozenset(
        {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "tr"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # collapse runs of whitespace / blank lines
        raw = re.sub(r"[^\S\n]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Chapter dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chapter:
    title: str
    text: str


# ---------------------------------------------------------------------------
# EPUB reader
# ---------------------------------------------------------------------------


def parse_epub(path: str) -> list[Chapter]:
    """Return a list of chapters with their titles and plain-text content.

    Uses the EPUB spine (reading order) rather than raw document order.
    """
    book = epub.read_epub(path, options={"ignore_ncx": True})

    # Build an id→item map so we can walk the spine in reading order
    items_by_id = {
        item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    }

    chapters: list[Chapter] = []
    for idx, (spine_id, _linear) in enumerate(book.spine):
        item = items_by_id.get(spine_id)
        if item is None:
            continue
        body = item.get_body_content()
        if body is None:
            continue

        body_str = body.decode("utf-8", errors="replace")
        text = html_to_text(body_str)
        if not text or len(text.strip()) < 20:
            continue

        title = item.get_name() or f"Chapter {idx + 1}"
        heading_match = re.search(r"<h[1-3][^>]*>(.*?)</h[1-3]>", body_str, re.I | re.S)
        if heading_match:
            title = re.sub(r"<[^>]+>", "", heading_match.group(1)).strip() or title

        chapters.append(Chapter(title=title, text=text))

    return chapters
