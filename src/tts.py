"""Synthesise chapters to WAV files using Chatterbox TTS."""

import logging
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 700
MERGE_THRESHOLD = 200
SILENCE_BODY = 0.25
SILENCE_TITLE = 1.0
SILENCE_CHAPTER_END = 0.5


# ── Text chunking ────────────────────────────────────────────────────────────


def _chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    merged: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and (
            len(buf) + len(para) + 1 > MAX_CHUNK_CHARS
            or (len(buf) >= MERGE_THRESHOLD and len(para) >= MERGE_THRESHOLD)
        ):
            merged.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}".strip() if buf else para
    if buf:
        merged.append(buf)

    chunks: list[str] = []
    for para in merged:
        if len(para) <= MAX_CHUNK_CHARS:
            chunks.append(para)
            continue
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if buf and len(buf) + len(sent) > MAX_CHUNK_CHARS:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip() if buf else sent
        if buf:
            chunks.append(buf)
    return chunks


# ── Audio helpers ────────────────────────────────────────────────────────────


def _silence(sr: int, seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _join(parts: list[np.ndarray], sr: int) -> np.ndarray:
    if not parts:
        return np.array([], dtype=np.float32)
    pieces = [parts[0], _silence(sr, SILENCE_TITLE)]
    for p in parts[1:]:
        pieces.extend([p, _silence(sr, SILENCE_BODY)])
    pieces.append(_silence(sr, SILENCE_CHAPTER_END))
    return np.concatenate(pieces)


# ── Synthesis ────────────────────────────────────────────────────────────────


def _load_model(turbo: bool):
    if turbo:
        from chatterbox import ChatterboxTTSTurbo

        log.info("Loading Chatterbox Turbo …")
        return ChatterboxTTSTurbo.from_pretrained(device="cuda")
    else:
        from chatterbox import ChatterboxTTS

        log.info("Loading Chatterbox …")
        return ChatterboxTTS.from_pretrained(device="cuda")


def synthesise_chapters(
    chapters,
    output_dir: Path,
    *,
    ref_audio: Path | None = None,
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
    turbo: bool = True,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sel = chapters[starting_chapter:ending_chapter]
    wav_paths = [
        output_dir / f"chapter_{starting_chapter + i:04d}.wav" for i in range(len(sel))
    ]

    model = _load_model(turbo)
    sr = model.sr
    ref = str(ref_audio) if ref_audio and ref_audio.exists() else None

    total_chunks = sum(
        1 + len(_chunk_text(ch.text))
        for i, ch in enumerate(sel)
        if not (wav_paths[i].exists() and wav_paths[i].stat().st_size > 0)
    )
    chunk_num = 0

    for i, ch in enumerate(sel):
        wp = wav_paths[i]
        ch_idx = starting_chapter + i

        if wp.exists() and wp.stat().st_size > 0:
            log.info("Skipping chapter %d (exists)", ch_idx)
            continue

        texts = [ch.title] + _chunk_text(ch.text)
        parts: list[np.ndarray] = []

        for text in texts:
            chunk_num += 1
            t0 = time.monotonic()

            kwargs = {}
            if ref:
                kwargs["audio_prompt_path"] = ref
            if not turbo:
                kwargs["exaggeration"] = exaggeration
                kwargs["cfg_weight"] = cfg_weight

            with torch.inference_mode():
                wav = model.generate(text, **kwargs)

            audio = wav.squeeze().cpu().numpy().astype(np.float32)
            dur = len(audio) / sr
            elapsed = time.monotonic() - t0

            log.info(
                "Chunk %d/%d ch%d %.1fs in %.1fs RTF:%.2f",
                chunk_num,
                total_chunks,
                ch_idx,
                dur,
                elapsed,
                dur / elapsed if elapsed else 0,
            )
            parts.append(audio)

        # Write chapter to disk immediately — don't accumulate in RAM
        joined = _join(parts, sr)
        sf.write(str(wp), joined, sr, format="WAV", subtype="FLOAT")
        log.info("Wrote ch%d '%s' %.1fs", ch_idx + 1, ch.title[:40], len(joined) / sr)
        del parts, joined

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
