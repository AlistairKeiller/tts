"""CLI: convert an EPUB to an M4B audiobook via Chatterbox TTS."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from .epub_parser import parse_epub
from .m4b import build_m4b, check_ffmpeg
from .tts import synthesise_chapters

app = typer.Typer(help="Convert an EPUB to an M4B audiobook via Chatterbox TTS.")


@app.command()
def main(
    epub: Annotated[Path, typer.Argument(help="Input .epub file.")],
    output: Annotated[Optional[Path], typer.Option("-o")] = None,
    reference_audio: Annotated[
        Optional[Path],
        typer.Option("--ref", "-r", help="WAV file for voice cloning."),
    ] = None,
    turbo: Annotated[
        bool, typer.Option("--turbo/--full", help="Use Turbo (fast) or full model.")
    ] = True,
    exaggeration: Annotated[
        float, typer.Option(help="Emotion exaggeration (full model only).")
    ] = 0.5,
    cfg_weight: Annotated[
        float, typer.Option(help="Speaker similarity CFG (full model only).")
    ] = 0.5,
    bitrate: Annotated[str, typer.Option()] = "128k",
    starting_chapter: Annotated[int, typer.Option("--start")] = 0,
    ending_chapter: Annotated[Optional[int], typer.Option("--end")] = None,
    list_chapters: Annotated[bool, typer.Option("--list")] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    check_ffmpeg()
    chapters = parse_epub(str(epub))
    assert chapters, "No chapters found in EPUB"

    if list_chapters:
        for i, ch in enumerate(chapters, start=0):
            print(f"{i}. {ch.title}")
        return

    wav_dir = Path(tempfile.mkdtemp(prefix="epub2ab_"))
    try:
        wav_paths = synthesise_chapters(
            chapters,
            wav_dir,
            ref_audio=reference_audio,
            starting_chapter=starting_chapter,
            ending_chapter=ending_chapter,
            turbo=turbo,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        assert wav_paths, "No chapters were synthesised"

        titles = [c.title for c in chapters[starting_chapter:ending_chapter]][
            : len(wav_paths)
        ]
        out = output or epub.with_suffix(".m4b")
        build_m4b(wav_paths, titles, out, book_title=epub.stem, bitrate=bitrate)
        print(f"✅ Done — {out}")
    finally:
        shutil.rmtree(wav_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
