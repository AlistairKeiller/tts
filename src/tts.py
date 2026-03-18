import io
import logging
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from chonkie import SentenceChunker
from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=1500)

DEFAULT_BASE = "http://127.0.0.1:8080"


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    base_url: str = DEFAULT_BASE,
    reference_id: str | None = None,
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

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

    log.info("Processing %d chunks", len(pending))

    client = httpx.Client(base_url=base_url, timeout=300)
    ch_wavs: dict[int, list[np.ndarray]] = {}
    sr = 0

    for ci, (idx, text) in enumerate(pending):
        payload: dict = {"text": text, "format": "wav", "normalize": True}
        if reference_id:
            payload["reference_id"] = reference_id

        t0 = time.monotonic()
        resp = client.post("/v1/tts", json=payload)
        resp.raise_for_status()
        data, sr = sf.read(io.BytesIO(resp.content))
        el = time.monotonic() - t0
        dur = len(data) / sr
        log.info(
            "Chunk %d/%d  ch %d  %.1fs audio in %.1fs  (RTF: %.2f)",
            ci + 1,
            len(pending),
            starting_chapter + idx,
            dur,
            el,
            dur / el if el else 0,
        )
        ch_wavs.setdefault(idx, []).append(data)

    client.close()

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
