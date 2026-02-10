"""Assemble per-chapter WAV files into a single M4B audiobook with chapter markers."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class _ChapterMeta:
    title: str
    start_ms: int
    end_ms: int


def _probe_duration_ms(wav_path: Path) -> int:
    """Return the duration of a WAV file in milliseconds using soundfile."""
    info = sf.info(str(wav_path))
    return int(info.duration * 1000)


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

    # 1. Build the chapter metadata
    chapters: list[_ChapterMeta] = []
    cursor_ms = 0
    for wp, title in zip(wav_paths, chapter_titles):
        dur = _probe_duration_ms(wp)
        chapters.append(
            _ChapterMeta(title=title, start_ms=cursor_ms, end_ms=cursor_ms + dur)
        )
        cursor_ms += dur

    # 2. Write ffmpeg chapter metadata file (FFMETADATA1 format)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as meta_fp:
        meta_fp.write(";FFMETADATA1\n")
        meta_fp.write(f"title={book_title}\n")
        if book_author:
            meta_fp.write(f"artist={book_author}\n")
        meta_fp.write("\n")
        for ch in chapters:
            meta_fp.write("[CHAPTER]\n")
            meta_fp.write("TIMEBASE=1/1000\n")
            meta_fp.write(f"START={ch.start_ms}\n")
            meta_fp.write(f"END={ch.end_ms}\n")
            meta_fp.write(f"title={ch.title}\n\n")
        metadata_path = meta_fp.name

    # 3. Write a concat list for ffmpeg
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as concat_fp:
        for wp in wav_paths:
            # ffmpeg concat demuxer needs escaped single-quotes in paths
            safe = str(wp.resolve()).replace("'", "'\\''")
            concat_fp.write(f"file '{safe}'\n")
        concat_path = concat_fp.name

    # 4. Run ffmpeg to concatenate + encode + embed chapters
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_path,
        "-i",
        metadata_path,
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

    # Clean up temp files
    Path(metadata_path).unlink(missing_ok=True)
    Path(concat_path).unlink(missing_ok=True)

    logger.info("✓ M4B written to %s", output_path)
    return output_path
