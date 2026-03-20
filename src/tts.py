import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel
from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=500)
VOICE_SEED = 42
PAUSE_SECONDS = 0.3
ANCHOR_TEXT = (
    "This is a calm and steady narration voice, reading at a comfortable pace "
    "with clear enunciation and a warm, natural tone."
)


def _set_seed():
    torch.manual_seed(VOICE_SEED)
    np.random.seed(VOICE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(VOICE_SEED)


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_voice_prompt(dev, speaker, output_dir):
    kw = dict(
        device_map=dev,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if dev == "cuda" else "eager",
    )

    cv = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", **kw)
    _set_seed()
    ref_wavs, sr = cv.generate_custom_voice(
        text=ANCHOR_TEXT, language="English", speaker=speaker, do_sample=False
    )
    del cv
    _free()

    sf.write(
        str(output_dir / "reference.wav"),
        ref_wavs[0],
        sr,
        format="WAV",
        subtype="FLOAT",
    )
    log.info("Saved reference.wav")

    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **kw)
    return base, base.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr), ref_text=ANCHOR_TEXT
    )


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
    model, prompt = _build_voice_prompt(dev, speaker, output_dir)

    sel = chapters[starting_chapter:ending_chapter]
    wav_paths = [
        output_dir / f"chapter_{starting_chapter + i:04d}.wav" for i in range(len(sel))
    ]

    pending: list[tuple[int, str]] = []
    for i, ch in enumerate(sel):
        if wav_paths[i].exists() and wav_paths[i].stat().st_size > 0:
            continue
        pending.extend((i, c.text) for c in chunker.chunk(ch.text))

    if not pending:
        return wav_paths

    expected = {}
    for idx, _ in pending:
        expected[idx] = expected.get(idx, 0) + 1
    received = {}
    ch_wavs: dict[int, list[np.ndarray]] = {}
    sr = 0
    bs = 1
    best = 1
    pos = 0

    log.info("Processing %d chunks", len(pending))
    while pos < len(pending):
        batch = pending[pos : pos + bs]
        texts = [t for _, t in batch]
        _set_seed()
        t0 = time.monotonic()

        try:
            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=["English"] * len(texts),
                voice_clone_prompt=prompt,
                max_new_tokens=2048,
            )
        except torch.cuda.OutOfMemoryError:
            bs = max(1, bs // 2)
            _free()
            log.warning("OOM → bs=%d", bs)
            continue

        el = time.monotonic() - t0
        dur = sum(len(w) for w in wavs) / sr
        log.info(
            "@%d  %d chunks  %.0fs audio  %.0fs wall  RTF=%.1f  bs=%d",
            pos,
            len(batch),
            dur,
            el,
            dur / el if el else 0,
            bs,
        )

        for (idx, _), w in zip(batch, wavs):
            ch_wavs.setdefault(idx, []).append(w)
            received[idx] = received.get(idx, 0) + 1
        pos += len(batch)
        del wavs
        _free()

        for idx in list(ch_wavs):
            if received.get(idx, 0) >= expected[idx]:
                pause = np.zeros(int(sr * PAUSE_SECONDS), dtype=np.float32)
                pieces = [
                    x for p in ch_wavs.pop(idx) for x in (p.astype(np.float32), pause)
                ][:-1]
                sf.write(
                    str(wav_paths[idx]),
                    np.concatenate(pieces),
                    sr,
                    format="WAV",
                    subtype="FLOAT",
                )

        if bs < best:
            bs = best
        else:
            best = bs
            bs += 1

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
