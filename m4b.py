"""Assemble per-chapter WAV files into a single M4B audiobook with chapter markers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


def _escape_ffmeta(value: str) -> str:
    """Escape special characters for FFMETADATA1 format."""
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, f"\\{ch}")
    return value


def _probe_duration_ms(wav_path: Path) -> int:
    info = sf.info(str(wav_path))
    return int(info.duration * 1000)


def check_ffmpeg() -> None:
    """Raise early if ffmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on $PATH – install it (e.g. `brew install ffmpeg`) "
            "before building the M4B."
        )


def build_m4b(
    wav_paths: list[Path],
    chapter_titles: list[str],
    output_path: Path,
    *,
    book_title: str = "Audiobook",
    book_author: str = "",
    bitrate: str = "64k",
) -> Path:
    """Merge per-chapter WAVs into a single ``.m4b`` audiobook.

    Requires ``ffmpeg`` on ``$PATH``.

    Parameters
    ----------
    wav_paths : list[Path]
        Ordered list of chapter WAV files.
    chapter_titles : list[str]
        Human-readable chapter titles (same length as *wav_paths*).
    output_path : Path
        Destination ``.m4b`` file.
    book_title : str
        Metadata title embedded in the file.
    book_author : str
        Metadata author.
    bitrate : str
        AAC encoding bitrate (e.g. ``"64k"``, ``"128k"``).
    """
    if len(wav_paths) != len(chapter_titles):
        raise ValueError("wav_paths and chapter_titles must have the same length")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build chapter timestamps
    cursor_ms = 0
    chapter_spans: list[tuple[str, int, int]] = []  # (title, start, end)
    for wp, title in zip(wav_paths, chapter_titles):
        dur = _probe_duration_ms(wp)
        chapter_spans.append((title, cursor_ms, cursor_ms + dur))
        cursor_ms += dur

    # 2. Write ffmpeg chapter metadata file (FFMETADATA1 format)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as meta_fp:
        meta_fp.write(";FFMETADATA1\n")
        meta_fp.write(f"title={_escape_ffmeta(book_title)}\n")
        if book_author:
            meta_fp.write(f"artist={_escape_ffmeta(book_author)}\n")
        meta_fp.write("\n")
        for title, start, end in chapter_spans:
            meta_fp.write("[CHAPTER]\nTIMEBASE=1/1000\n")
            meta_fp.write(f"START={start}\nEND={end}\n")
            meta_fp.write(f"title={_escape_ffmeta(title)}\n\n")
        metadata_path = Path(meta_fp.name)

    # 3. Write a concat list for ffmpeg
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as concat_fp:
        for wp in wav_paths:
            safe = str(wp.resolve()).replace("'", "'\\''")
            concat_fp.write(f"file '{safe}'\n")
        concat_path = Path(concat_fp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-i",
            str(metadata_path),
            "-map_metadata",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", result.stderr)
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}")
    finally:
        metadata_path.unlink(missing_ok=True)
        concat_path.unlink(missing_ok=True)

    logger.info("✓ M4B written to %s", output_path)
    return output_path
