from dataclasses import dataclass

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub


@dataclass
class Chapter:
    title: str
    text: str


def parse_epub(path: str) -> list[Chapter]:
    """Extract chapters from an EPUB in spine (reading) order."""
    book = epub.read_epub(path, options={"ignore_ncx": True})
    items = {
        item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
    }

    chapters: list[Chapter] = []
    for idx, (spine_id, _) in enumerate(book.spine):
        item = items.get(spine_id)
        if not item or not item.get_body_content():
            continue

        soup = BeautifulSoup(
            item.get_body_content().decode("utf-8", errors="replace"), "html.parser"
        )
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 20:
            continue

        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(strip=True) if heading else f"Chapter {idx + 1}"
        chapters.append(Chapter(title=title, text=text))

    return chapters
