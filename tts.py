"""Synthesise speech for each chapter using Qwen3-TTS."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

logger = logging.getLogger(__name__)

_MAX_CHARS = 500
_BATCH_SIZE = 4  # chunks per batch call – tune for your VRAM
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CLAUSE_RE = re.compile(r"(?<=[,;:，；：])\s+")


def _split_text(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
    """Split *text* into chunks of ≤ *max_chars* on sentence boundaries,
    falling back to clause boundaries for oversized sentences."""
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # If a single sentence exceeds the limit, break on clauses
        if len(sent) > max_chars:
            if buf:
                chunks.append(" ".join(buf))
                buf, length = [], 0
            for clause in _CLAUSE_RE.split(sent):
                clause = clause.strip()
                if not clause:
                    continue
                if length + len(clause) > max_chars and buf:
                    chunks.append(" ".join(buf))
                    buf, length = [], 0
                buf.append(clause)
                length += len(clause)
            continue
        if length + len(sent) > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, length = [], 0
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
    batch_size: int = _BATCH_SIZE,
) -> list[Path]:
    """Generate a WAV file per chapter and return the list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Qwen3-TTS model %s …", model_name)
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    inst = instruct or None
    wav_paths: list[Path] = []

    for ch_idx, chapter in enumerate(chapters):
        wav_path = output_dir / f"chapter_{ch_idx:04d}.wav"

        # Resume: skip chapters already on disk
        if wav_path.exists():
            logger.info(
                "Chapter %d/%d  already exists – skipping", ch_idx + 1, len(chapters)
            )
            wav_paths.append(wav_path)
            continue

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

        # Process chunks in batches for GPU throughput
        for b_start in range(0, len(chunks), batch_size):
            batch = chunks[b_start : b_start + batch_size]
            logger.info(
                "  chunks %d–%d / %d", b_start + 1, b_start + len(batch), len(chunks)
            )
            try:
                wavs, sample_rate = model.generate_custom_voice(
                    text=batch,
                    language=[language] * len(batch),
                    speaker=[speaker] * len(batch),
                    instruct=[inst] * len(batch) if inst else None,
                )
                all_audio.extend(wavs)
                sr = sample_rate
            except Exception:
                logger.exception(
                    "  ⚠ TTS failed on chunks %d–%d – skipping batch",
                    b_start + 1,
                    b_start + len(batch),
                )

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
