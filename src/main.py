import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from epub_parser import parse_epub
from m4b import build_m4b, check_ffmpeg
from tts import synthesise_chapters

app = typer.Typer(help="Convert an EPUB to an M4B audiobook via Fish Speech S2 Pro.")


@app.command()
def main(
    epub: Annotated[Path, typer.Argument(help="Input .epub file.")],
    output: Annotated[
        Optional[Path], typer.Option("-o", help="Output .m4b path.")
    ] = None,
    base_url: Annotated[
        str, typer.Option("--url", help="Fish Speech server URL.")
    ] = "http://127.0.0.1:8080",
    reference_id: Annotated[
        Optional[str], typer.Option("--ref", help="Voice reference ID.")
    ] = None,
    bitrate: Annotated[str, typer.Option(help="AAC bitrate.")] = "48k",
    starting_chapter: Annotated[
        int, typer.Option(help="Starting chapter index (0-based).")
    ] = 0,
    ending_chapter: Annotated[
        Optional[int], typer.Option(help="Ending chapter (exclusive, 0-based).")
    ] = None,
    list_chapters: Annotated[
        bool, typer.Option("--list-chapters", help="List chapter titles and exit.")
    ] = False,
) -> None:
    """Convert an EPUB to an M4B audiobook."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    check_ffmpeg()

    chapters = parse_epub(str(epub))
    assert chapters, "No chapters found in EPUB"

    if list_chapters:
        for i, ch in enumerate(
            chapters[starting_chapter:ending_chapter], start=starting_chapter
        ):
            print(f"{i}. {ch.title}")
        return

    wav_dir = Path(tempfile.mkdtemp(prefix="epub2ab_"))
    wav_paths = synthesise_chapters(
        chapters,
        wav_dir,
        base_url=base_url,
        reference_id=reference_id,
        starting_chapter=starting_chapter,
        ending_chapter=ending_chapter,
    )
    assert wav_paths, "No chapters were synthesised"

    selected = chapters[starting_chapter:ending_chapter]
    titles = [c.title for c in selected][: len(wav_paths)]

    out = output or epub.with_suffix(".m4b")
    build_m4b(wav_paths, titles, out, book_title=epub.stem, bitrate=bitrate)
    shutil.rmtree(wav_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
