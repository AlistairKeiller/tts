"""Synthesise chapters to WAV files using Chatterbox TTS."""

import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker

log = logging.getLogger(__name__)

MAX_RETRIES = 3
SILENCE_BODY = 0.25
SILENCE_TITLE = 1.0
SILENCE_CHAPTER_END = 0.5
MIN_WORDS = 6

chunker = SentenceChunker(chunk_size=500)


# ── Text cleanup ─────────────────────────────────────────────────────────────


def _fix_caps(sent: str) -> str:
    """Lowercase sentences with 3+ consecutive ALL-CAPS words."""
    words = sent.split()
    for i in range(len(words) - 2):
        if words[i].isupper() and words[i + 1].isupper() and words[i + 2].isupper():
            return sent.lower().capitalize()
    return sent


def _merge_short(sentences: list[str]) -> list[str]:
    """Merge sentences shorter than MIN_WORDS with neighbors."""
    if not sentences:
        return []
    result: list[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
        elif len(buf.split()) < MIN_WORDS:
            buf += " " + sent
        else:
            result.append(buf)
            buf = sent
    if buf:
        if result and len(buf.split()) < MIN_WORDS:
            result[-1] += " " + buf
        else:
            result.append(buf)
    return result


def _chunk_chapter(text: str) -> list[str]:
    """Split chapter text into sentence-level chunks, merging short ones."""
    sentences = [c.text for c in chunker.chunk(text)]
    return _merge_short(sentences)


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


# ── Generation with retries ──────────────────────────────────────────────────


def _generate(model, text: str, ref: str | None, turbo: bool, **kwargs) -> np.ndarray:
    """Generate audio for a single chunk, retrying up to MAX_RETRIES times."""
    text = _fix_caps(text.strip())
    gen_kwargs = {}
    if ref:
        gen_kwargs["audio_prompt_path"] = ref
    if not turbo:
        gen_kwargs.update(kwargs)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            wav = model.generate(text, **gen_kwargs)
            return wav.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning(
                "Attempt %d failed for '%s…': %s — retrying", attempt, text[:50], e
            )


# ── Main entry point ─────────────────────────────────────────────────────────


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
        1 + len(_chunk_chapter(ch.text))
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

        texts = [ch.title] + _chunk_chapter(ch.text)
        parts: list[np.ndarray] = []

        for text in texts:
            chunk_num += 1
            t0 = time.monotonic()

            with torch.inference_mode():
                audio = _generate(
                    model,
                    text,
                    ref,
                    turbo,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

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

        # Write immediately — don't accumulate across chapters
        joined = _join(parts, sr)
        sf.write(str(wp), joined, sr, format="WAV", subtype="FLOAT")
        log.info("Wrote ch%d '%s' %.1fs", ch_idx + 1, ch.title[:40], len(joined) / sr)
        del parts, joined

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
