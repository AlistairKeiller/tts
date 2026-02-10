"""CLI entry point – convert an EPUB to an M4B audiobook using Qwen3-TTS."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="epub2audiobook",
        description="Convert an EPUB e-book to an M4B audiobook using Qwen3-TTS.",
    )
    parser.add_argument("epub", type=Path, help="Path to the input .epub file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .m4b path (default: <epub-stem>.m4b alongside the input).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="HuggingFace model id or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--speaker",
        default="Aiden",
        help="Speaker voice name (default: %(default)s).",
    )
    parser.add_argument(
        "--language",
        default="Auto",
        help="Target language or 'Auto' (default: %(default)s).",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Optional style instruction for the TTS model.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: %(default)s).",
    )
    parser.add_argument(
        "--bitrate",
        default="64k",
        help="AAC bitrate for M4B encoding (default: %(default)s).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Book title for M4B metadata (default: EPUB filename).",
    )
    parser.add_argument(
        "--author",
        default="",
        help="Book author for M4B metadata.",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep intermediate chapter WAV files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging.",
    )

    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("epub2audiobook")

    # Validate input
    if not args.epub.is_file():
        log.error("File not found: %s", args.epub)
        sys.exit(1)

    output_path: Path = args.output or args.epub.with_suffix(".m4b")
    book_title = args.title or args.epub.stem

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    # --- Step 1: Parse EPUB ---
    from epub_parser import parse_epub  # noqa: local import to defer heavy deps

    log.info("Parsing EPUB: %s", args.epub)
    chapters = parse_epub(str(args.epub))
    if not chapters:
        log.error("No chapters found in the EPUB.")
        sys.exit(1)
    log.info("Found %d chapter(s).", len(chapters))

    # --- Step 2: Synthesise audio ---
    from tts import synthesise_chapters

    wav_dir = (
        Path(tempfile.mkdtemp(prefix="epub2audiobook_"))
        if not args.keep_wav
        else (args.epub.parent / "wav_chapters")
    )
    log.info("WAV output directory: %s", wav_dir)

    wav_paths = synthesise_chapters(
        chapters,
        wav_dir,
        model_name=args.model,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )

    if not wav_paths:
        log.error("No audio was generated.")
        sys.exit(1)

    # --- Step 3: Build M4B ---
    from m4b import build_m4b

    # Align titles to the WAV files that were actually produced
    titles = [ch.title for ch in chapters[: len(wav_paths)]]

    log.info("Building M4B: %s", output_path)
    build_m4b(
        wav_paths,
        titles,
        output_path,
        book_title=book_title,
        book_author=args.author,
        bitrate=args.bitrate,
    )

    # Cleanup temp wavs
    if not args.keep_wav:
        import shutil

        shutil.rmtree(wav_dir, ignore_errors=True)

    log.info("Done! Audiobook saved to %s", output_path)


if __name__ == "__main__":
    main()
