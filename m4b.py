import shutil
import tempfile
from pathlib import Path

import ffmpeg
import soundfile as sf


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on $PATH")


def build_m4b(
    wav_paths: list[Path],
    titles: list[str],
    output: Path,
    *,
    book_title: str = "Audiobook",
    bitrate: str = "64k",
) -> None:
    """Merge per-chapter WAVs into a single M4B with chapter markers."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build chapter timestamps
    cursor = 0
    spans: list[tuple[str, int, int]] = []
    for wp, t in zip(wav_paths, titles):
        dur = int(sf.info(str(wp)).duration * 1000)
        spans.append((t, cursor, cursor + dur))
        cursor += dur

    # Write FFMETADATA1 chapter file
    meta = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    meta.write(f";FFMETADATA1\ntitle={book_title}\n\n")
    for t, s, e in spans:
        meta.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={s}\nEND={e}\ntitle={t}\n\n")
    meta.close()

    # Write concat file list
    concat = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for wp in wav_paths:
        concat.write(f"file '{wp.resolve()}'\n")
    concat.close()

    try:
        concat_input = ffmpeg.input(concat.name, f="concat", safe=0)
        (
            ffmpeg.output(
                concat_input.audio,
                str(output),
                map_metadata=1,
                acodec="aac",
                audio_bitrate=bitrate,
                movflags="+faststart",
            )
            .global_args("-i", meta.name)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=False)
        )
    finally:
        Path(meta.name).unlink(missing_ok=True)
        Path(concat.name).unlink(missing_ok=True)
