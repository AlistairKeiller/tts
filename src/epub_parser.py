import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from nemo_text_processing.text_normalization.normalize import Normalizer

_normalizer = Normalizer(input_case="cased", lang="en", post_process=True)

_AD_PATTERNS = [
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}"  # symbol-delimited site ads:
    r"[^◈✪★☆●■▶►⊛⊕✦✧☛➤⌘\n]{1,60}"  # site name (homoglyphs etc)
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}"
    r"(?:\s*\([^)]{0,30}\))?",  # optional (Continue reading) / (Official version)
    r"[\U0001f300-\U0001f9ff][\s\S]{0,200}?"  # emoji-prefixed inline ads:
    r"(?:click away|sign up|start meeting|"  # match common ad phrases
    r"download now|join free|register today|"
    r"limited offer|subscribe now)"
    r"[^.!?\n]*[.!?]?",
]
_AD_RE = re.compile("|".join(_AD_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    text = _AD_RE.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)  # footnote markers
    text = re.sub(r"&[a-z]+;", " ", text)  # stray HTML entities
    text = re.sub(r"https?://\S+", "", text)  # URLs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    parts = []
    for p in text.split("\n"):
        p = p.strip()
        if not p:
            parts.append("")
            continue
        try:
            parts.append(_normalizer.normalize(p))
        except Exception:
            parts.append(p)
    return "\n".join(parts)


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
        chapters.append(Chapter(title=title, text=clean_text(text)))
    return chapters
