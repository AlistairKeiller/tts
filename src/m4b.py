"""Merge per-chapter WAVs into a single M4B with chapter markers and cover art."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
import soundfile as sf


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on $PATH")


def _embed_cover(m4b_path: Path, cover_path: str | None) -> None:
    """Embed cover image into M4B using mutagen."""
    if not cover_path or not Path(cover_path).exists():
        return
    try:
        from mutagen import mp4

        m4b = mp4.MP4(str(m4b_path))
        m4b["covr"] = [mp4.MP4Cover(Path(cover_path).read_bytes())]
        m4b.save()
    except Exception as e:
        print(f"Warning: could not embed cover: {e}")


def build_m4b(
    wav_paths: list[Path],
    titles: list[str],
    output: Path,
    *,
    book_title: str = "Audiobook",
    book_author: str = "Unknown",
    cover_path: str | None = None,
    bitrate: str = "128k",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    cursor = 0
    spans: list[tuple[str, int, int]] = []
    for wp, t in zip(wav_paths, titles):
        dur = int(sf.info(str(wp)).duration * 1000)
        spans.append((t, cursor, cursor + dur))
        cursor += dur

    meta = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    meta.write(
        f";FFMETADATA1\ntitle={book_title}\nartist={book_author}\nalbum={book_title}\n\n"
    )
    for t, s, e in spans:
        meta.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={s}\nEND={e}\ntitle={t}\n\n")
    meta.close()

    concat = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for wp in wav_paths:
        concat.write(f"file '{wp.resolve()}'\n")
    concat.close()

    try:
        codec = (
            "libfdk_aac"
            if "libfdk_aac" in subprocess.getoutput("ffmpeg -encoders")
            else "aac"
        )
        if codec == "aac":
            print(
                "Warning: using built-in aac encoder. Install libfdk_aac for better quality."
            )

        (
            ffmpeg.input(concat.name, f="concat", safe=0)
            .audio.filter("loudnorm", I=-16, TP=-1.5, LRA=11)
            .output(
                str(output),
                map_metadata=1,
                acodec=codec,
                audio_bitrate=bitrate,
                ac=1,
                movflags="+faststart",
            )
            .global_args("-i", meta.name)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    finally:
        Path(meta.name).unlink(missing_ok=True)
        Path(concat.name).unlink(missing_ok=True)

    _embed_cover(output, cover_path)
