"""Parse EPUB files into a list of chapters."""

import re
from dataclasses import dataclass

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

_AD_RE = re.compile(
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}[^◈✪★☆●■▶►⊛⊕✦✧☛➤⌘\n]{1,60}"
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}(?:\s*\([^)]{0,30}\))?",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = _AD_RE.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class Chapter:
    title: str
    text: str


def parse_epub(path: str) -> list[Chapter]:
    book = epub.read_epub(path)
    items = {i.get_id(): i for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    chapters: list[Chapter] = []
    for idx, (sid, _) in enumerate(book.spine):
        item = items.get(sid)
        if not item or not item.get_body_content():
            continue
        soup = BeautifulSoup(
            item.get_body_content().decode("utf-8", errors="replace"), "html.parser"
        )
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 20:
            continue
        h = soup.find(["h1", "h2", "h3"])
        title = h.get_text(strip=True) if h else f"Chapter {idx + 1}"
        chapters.append(Chapter(title=title, text=_clean(text)))
    return chapters
