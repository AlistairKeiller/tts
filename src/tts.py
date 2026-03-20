import gc
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=500)

INITIAL_BATCH = 16
MIN_BATCH = 1
GROW_AFTER = 3  # consecutive successes before trying a larger batch
PAUSE_SECONDS = 0.3
VOICE_SEED = 42
ANCHOR_TEXT = (
    "This is a calm and steady narration voice, reading at a comfortable pace "
    "with clear enunciation and a warm, natural tone."
)


def _silence(sr: int, seconds: float = PAUSE_SECONDS) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_voice_prompt(dev: str, speaker: str, output_dir: Path):
    attn = "flash_attention_2" if dev == "cuda" else "eager"
    kw = dict(device_map=dev, dtype=torch.bfloat16, attn_implementation=attn)

    log.info("Generating anchor clip (speaker=%s)", speaker)
    cv = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", **kw)
    _set_seed(VOICE_SEED)
    ref_wavs, sr = cv.generate_custom_voice(
        text=ANCHOR_TEXT,
        language="English",
        speaker=speaker,
        do_sample=False,
    )
    del cv
    _free_vram()

    ref_path = output_dir / "reference.wav"
    sf.write(str(ref_path), ref_wavs[0], sr, format="WAV", subtype="FLOAT")
    log.info("Saved reference clip to %s", ref_path)

    log.info("Building voice-clone prompt")
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **kw)
    prompt = base.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr), ref_text=ANCHOR_TEXT
    )
    return base, prompt


def _generate_batch(model, texts, voice_prompt):
    """Try to generate; on OOM return None so caller can shrink and retry."""
    try:
        _set_seed(VOICE_SEED)
        wavs, sr = model.generate_voice_clone(
            text=texts,
            language=["English"] * len(texts),
            voice_clone_prompt=voice_prompt,
        )
        return wavs, sr
    except torch.cuda.OutOfMemoryError:
        log.warning("OOM on batch of %d — clearing VRAM", len(texts))
        _free_vram()
        return None


def _flush_chapter(wav_path: Path, parts: list[np.ndarray], sr: int):
    pause = _silence(sr, PAUSE_SECONDS)
    pieces = [x for p in parts for x in (p.astype(np.float32), pause)][:-1]
    audio = np.concatenate(pieces)
    sf.write(str(wav_path), audio, sr, format="WAV", subtype="FLOAT")
    log.info("Wrote '%s'  %.1fs", wav_path.name, len(audio) / sr)


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    speaker: str = "Aiden",
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", dev)

    model, voice_prompt = _build_voice_prompt(dev, speaker, output_dir)

    sel = chapters[starting_chapter:ending_chapter]
    wav_paths = [
        output_dir / f"chapter_{starting_chapter + i:04d}.wav" for i in range(len(sel))
    ]

    pending: list[tuple[int, str]] = []
    for i, ch in enumerate(sel):
        if wav_paths[i].exists() and wav_paths[i].stat().st_size > 0:
            log.info("Skipping chapter %d (exists)", starting_chapter + i)
            continue
        pending.extend((i, c.text) for c in chunker.chunk(ch.text))

    if not pending:
        return wav_paths

    chunks_expected: dict[int, int] = {}
    for idx, _ in pending:
        chunks_expected[idx] = chunks_expected.get(idx, 0) + 1
    chunks_received: dict[int, int] = {}
    ch_wavs: dict[int, list[np.ndarray]] = {}
    sr = 0

    batch_size = INITIAL_BATCH
    successes = 0
    pos = 0

    log.info("Processing %d chunks, starting batch size %d", len(pending), batch_size)

    while pos < len(pending):
        batch = pending[pos : pos + batch_size]
        texts = [t for _, t in batch]
        t0 = time.monotonic()

        result = _generate_batch(model, texts, voice_prompt)

        if result is None:
            batch_size = max(MIN_BATCH, batch_size // 2)
            successes = 0
            log.info("Reduced batch size to %d", batch_size)
            if batch_size < MIN_BATCH:
                raise RuntimeError("OOM even with batch size 1")
            continue

        wavs, sr = result
        el = time.monotonic() - t0
        dur = sum(len(w) for w in wavs) / sr
        log.info(
            "Batch @%d  %d chunks  %.1fs audio  in %.1fs  (RTF: %.2f)  [bs=%d]",
            pos,
            len(texts),
            dur,
            el,
            dur / el if el else 0,
            batch_size,
        )

        for (idx, _), w in zip(batch, wavs):
            ch_wavs.setdefault(idx, []).append(w)
            chunks_received[idx] = chunks_received.get(idx, 0) + 1

        pos += len(batch)
        del wavs, texts, result
        _free_vram()

        for idx in list(ch_wavs):
            if chunks_received.get(idx, 0) >= chunks_expected[idx]:
                _flush_chapter(wav_paths[idx], ch_wavs.pop(idx), sr)

        successes += 1
        if successes >= GROW_AFTER:
            batch_size += 1
            successes = 0
            log.info("Grew batch size to %d", batch_size)

    for idx, parts in ch_wavs.items():
        _flush_chapter(wav_paths[idx], parts, sr)

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
