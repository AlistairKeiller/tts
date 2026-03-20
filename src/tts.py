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

BATCH_CHARS = 31_250
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


def _build_voice_prompt(dev: str, speaker: str, output_dir: Path):
    """Generate an anchor clip with CustomVoice, then build a Base-model clone prompt from it."""
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
    if dev == "cuda":
        torch.cuda.empty_cache()

    ref_path = output_dir / "reference.wav"
    sf.write(str(ref_path), ref_wavs[0], sr, format="WAV", subtype="FLOAT")
    log.info("Saved reference clip to %s", ref_path)

    log.info("Building voice-clone prompt")
    base = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", **kw)
    prompt = base.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr), ref_text=ANCHOR_TEXT
    )
    return base, prompt


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

    batches: list[list[tuple[int, str]]] = [[]]
    cur_chars = 0
    for item in pending:
        if cur_chars and cur_chars + len(item[1]) > BATCH_CHARS:
            batches.append([])
            cur_chars = 0
        batches[-1].append(item)
        cur_chars += len(item[1])
    if not batches[0]:
        return wav_paths

    log.info("Processing %d chunks in %d batches", len(pending), len(batches))
    ch_wavs: dict[int, list[np.ndarray]] = {}
    sr = 0

    with torch.inference_mode():
        for bi, batch in enumerate(batches):
            texts = [t for _, t in batch]
            _set_seed(VOICE_SEED)
            t0 = time.monotonic()

            wavs, sr = model.generate_voice_clone(
                text=texts,
                language=["English"] * len(texts),
                voice_clone_prompt=voice_prompt,
            )

            el = time.monotonic() - t0
            dur = sum(len(w) for w in wavs) / sr
            log.info(
                "Batch %d/%d  %d chunks  %.1fs audio  in %.1fs  (RTF: %.2f)",
                bi + 1,
                len(batches),
                len(texts),
                dur,
                el,
                dur / el if el else 0,
            )

            for (idx, _), w in zip(batch, wavs):
                ch_wavs.setdefault(idx, []).append(w)
            del wavs
            if dev == "cuda":
                torch.cuda.empty_cache()

    # stitch chunks into per-chapter WAVs
    pause = _silence(sr, PAUSE_SECONDS)
    for idx, parts in ch_wavs.items():
        pieces = [x for p in parts for x in (p.astype(np.float32), pause)][:-1]
        audio = np.concatenate(pieces)
        sf.write(str(wav_paths[idx]), audio, sr, format="WAV", subtype="FLOAT")
        log.info(
            "Wrote ch %d '%s'  %.1fs",
            starting_chapter + idx + 1,
            sel[idx].title[:40],
            len(audio) / sr,
        )

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
