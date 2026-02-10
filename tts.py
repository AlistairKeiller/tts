"""Synthesise speech for each chapter using Qwen3-TTS."""

from __future__ import annotations

import re
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

logger = logging.getLogger(__name__)

# Qwen3-TTS can handle long text but very long passages may OOM.
# We split into chunks of roughly this many characters.
_MAX_CHARS = 500


def _split_text(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
    """Split text into chunks on sentence boundaries."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if length + len(sent) > max_chars and buf:
            chunks.append(" ".join(buf))
            buf = []
            length = 0
        buf.append(sent)
        length += len(sent)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    speaker: str = "Aiden",
    language: str = "Auto",
    instruct: str = "",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> list[Path]:
    """Generate a WAV file per chapter and return the list of paths.

    Parameters
    ----------
    chapters : list[Chapter]
        Parsed chapters from the EPUB.
    output_dir : Path
        Directory to write intermediate WAV files.
    model_name : str
        HuggingFace model id or local path.
    speaker : str
        Speaker name (see ``model.get_supported_speakers()``).
    language : str
        Target language, or ``"Auto"`` for auto-detection.
    instruct : str
        Optional style instruction (e.g. "Read calmly like an audiobook narrator.").
    device : str
        Torch device string.
    dtype : torch.dtype
        Model precision.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Qwen3-TTS model %s …", model_name)
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    wav_paths: list[Path] = []
    for ch_idx, chapter in enumerate(chapters):
        wav_path = output_dir / f"chapter_{ch_idx:04d}.wav"
        logger.info(
            "Chapter %d/%d  '%s'  (%d chars)",
            ch_idx + 1,
            len(chapters),
            chapter.title[:40],
            len(chapter.text),
        )

        chunks = _split_text(chapter.text)
        all_audio: list[np.ndarray] = []
        sr: int | None = None

        for i, chunk in enumerate(chunks):
            logger.info("  chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            wavs, sample_rate = model.generate_custom_voice(
                text=chunk,
                language=language,
                speaker=speaker,
                instruct=instruct if instruct else None,
            )
            all_audio.append(wavs[0])
            sr = sample_rate

        if not all_audio or sr is None:
            logger.warning(
                "  ⚠ No audio produced for chapter %d – skipping", ch_idx + 1
            )
            continue

        combined = np.concatenate(all_audio)
        sf.write(str(wav_path), combined, sr)
        wav_paths.append(wav_path)
        logger.info("  ✓ saved %s (%.1f s)", wav_path.name, len(combined) / sr)

    return wav_paths
