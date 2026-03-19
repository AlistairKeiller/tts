"""Parse EPUB files into a list of chapters with metadata."""

import os
import re
import zipfile
from dataclasses import dataclass

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from lxml import etree

_AD_RE = re.compile(
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}[^◈✪★☆●■▶►⊛⊕✦✧☛➤⌘\n]{1,60}"
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}(?:\s*\([^)]{0,30}\))?",
    re.IGNORECASE,
)

_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "u": "urn:oasis:names:tc:opendocument:xmlns:container",
}


def _clean(text: str) -> str:
    text = _AD_RE.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_footnotes(soup: BeautifulSoup) -> None:
    """Remove footnote markers: <a> with only digits, <sup> with only digits."""
    for a in soup.find_all("a", href=True):
        if not any(c.isalpha() for c in a.get_text()):
            a.decompose()
    for sup in soup.find_all("sup"):
        if sup.get_text(strip=True).isdigit():
            sup.decompose()


def _build_toc_map(book) -> dict[str, str]:
    """Map filenames → TOC titles for better chapter naming."""
    toc_map: dict[str, str] = {}

    def _walk(items):
        for item in items:
            if hasattr(item, "href") and hasattr(item, "title"):
                toc_map[item.href.split("#")[0]] = item.title
            elif hasattr(item, "__iter__"):
                _walk(item)

    _walk(getattr(book, "toc", []))
    return toc_map


@dataclass
class Chapter:
    title: str
    text: str


@dataclass
class BookMeta:
    title: str
    author: str
    cover_path: str | None


def _extract_cover(epub_path: str) -> str | None:
    """Extract cover image to a .png next to the epub, return path or None."""
    try:
        with zipfile.ZipFile(epub_path) as z:
            t = etree.fromstring(z.read("META-INF/container.xml"))
            rootfile = t.xpath("/u:container/u:rootfiles/u:rootfile", namespaces=_NS)[
                0
            ].get("full-path")
            t = etree.fromstring(z.read(rootfile))
            meta = t.xpath("//opf:metadata/opf:meta[@name='cover']", namespaces=_NS)
            if not meta:
                return None
            cover_id = meta[0].get("content")
            item = t.xpath(f"//opf:manifest/opf:item[@id='{cover_id}']", namespaces=_NS)
            if not item:
                return None
            href = item[0].get("href")
            cover_zip_path = os.path.join(os.path.dirname(rootfile), href)
            out = epub_path.rsplit(".", 1)[0] + ".png"
            from PIL import Image

            Image.open(z.open(cover_zip_path)).save(out)
            return out
    except Exception:
        return None


def parse_epub(path: str) -> tuple[list[Chapter], BookMeta]:
    book = epub.read_epub(path)

    # Metadata
    t = book.get_metadata("DC", "title")
    a = book.get_metadata("DC", "creator")
    meta = BookMeta(
        title=t[0][0] if t else "Unknown Title",
        author=a[0][0] if a else "Unknown Author",
        cover_path=_extract_cover(path),
    )

    toc_map = _build_toc_map(book)
    items = {i.get_id(): i for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    chapters: list[Chapter] = []

    for idx, (sid, _) in enumerate(book.spine):
        item = items.get(sid)
        if not item or not item.get_body_content():
            continue
        soup = BeautifulSoup(
            item.get_body_content().decode("utf-8", errors="replace"), "html.parser"
        )
        _strip_footnotes(soup)
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 20:
            continue

        # Title: TOC > heading tag > fallback
        title = toc_map.get(item.get_name())
        if not title:
            h = soup.find(["h1", "h2", "h3"])
            title = h.get_text(strip=True) if h else f"Chapter {idx + 1}"

        chapters.append(Chapter(title=title, text=_clean(text)))

    return chapters, meta
