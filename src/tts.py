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
chunker = SentenceChunker(chunk_size=1500)
BATCH_CHARS = 125_000


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
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=dev,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if dev == "cuda" else "eager",
    )
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
            t0 = time.monotonic()
            wavs, sr = model.generate_custom_voice(
                text=texts,
                language=["Auto"] * len(texts),
                speaker=[speaker] * len(texts),
            )
            el = time.monotonic() - t0
            dur = sum(len(w) for w in wavs) / sr
            log.info(
                "Batch %d/%d  %d chunks  %.1fs audio in %.1fs  (RTF: %.2f)",
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

    for idx, parts in ch_wavs.items():
        audio = np.concatenate(parts).astype(np.float32)
        sf.write(str(wav_paths[idx]), audio, sr, format="WAV", subtype="FLOAT")
        log.info(
            "Wrote ch %d '%s' %.1fs",
            starting_chapter + idx + 1,
            sel[idx].title[:40],
            len(audio) / sr,
        )

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
