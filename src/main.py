import logging, shutil, tempfile
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
    output: Annotated[Optional[Path], typer.Option("-o")] = None,
    base_url: Annotated[str, typer.Option("--url")] = "http://127.0.0.1:8081",
    temperature: Annotated[float, typer.Option("--temp")] = 0.5,
    repetition_penalty: Annotated[float, typer.Option("--rep-penalty")] = 1.3,
    bitrate: Annotated[str, typer.Option()] = "64k",
    starting_chapter: Annotated[int, typer.Option()] = 0,
    ending_chapter: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option("--workers", "-j")] = 4,
    list_chapters: Annotated[bool, typer.Option("--list-chapters")] = False,
) -> None:
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
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        starting_chapter=starting_chapter,
        ending_chapter=ending_chapter,
        max_workers=workers,
    )
    assert wav_paths, "No chapters were synthesised"
    titles = [c.title for c in chapters[starting_chapter:ending_chapter]][
        : len(wav_paths)
    ]
    out = output or epub.with_suffix(".m4b")
    build_m4b(wav_paths, titles, out, book_title=epub.stem, bitrate=bitrate)
    shutil.rmtree(wav_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
