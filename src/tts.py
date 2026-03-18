import base64, io, logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx, numpy as np, soundfile as sf
from chonkie import SentenceChunker
from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=600)
CROSSFADE = 0.03
SILENCE = 0.5
MAX_WORKERS = 4

NARRATOR_SAMPLE_TEXT = (
    "In the quiet hours before dawn, the world seems to hold its breath. "
    "Every story begins with a single moment, a choice that sets everything in motion. "
    "The pages ahead are filled with wonder and possibility, "
    "and it is my pleasure to guide you through each one."
)


def _create_narrator_reference(client: httpx.Client, temperature: float) -> dict:
    """Generate a calm narrator sample and return an inline reference dict."""
    log.info("Creating narrator voice reference...")
    resp = client.post(
        "/v1/tts",
        json={
            "text": f"[calm, professional narration] {NARRATOR_SAMPLE_TEXT}",
            "format": "wav",
            "normalize": True,
            "temperature": max(temperature - 0.2, 0.1),
            "repetition_penalty": 1.3,
        },
    )
    resp.raise_for_status()
    audio_b64 = base64.b64encode(resp.content).decode()
    log.info("Narrator reference generated (%.1f KB)", len(resp.content) / 1024)
    return {"audio": audio_b64, "text": NARRATOR_SAMPLE_TEXT}


def _synth_one(
    base_url: str, payload: dict, ci: int, total: int, ch_idx: int
) -> tuple[int, np.ndarray, int]:
    """Synthesise a single chunk (runs in a worker thread)."""
    with httpx.Client(base_url=base_url, timeout=300) as c:
        t0 = time.monotonic()
        resp = c.post("/v1/tts", json=payload)
        resp.raise_for_status()
        data, sr = sf.read(io.BytesIO(resp.content))
        el = time.monotonic() - t0
        log.info(
            "Chunk %d/%d  ch %d  %.1fs audio in %.1fs  (RTF: %.2f)",
            ci + 1,
            total,
            ch_idx,
            len(data) / sr,
            el,
            len(data) / sr / el if el else 0,
        )
    return ci, data, sr


def _crossfade(parts: list[np.ndarray], sr: int) -> np.ndarray:
    if len(parts) <= 1:
        return parts[0] if parts else np.array([], dtype=np.float32)
    n = int(CROSSFADE * sr)
    out = parts[0].copy()
    for p in parts[1:]:
        if n and len(out) >= n and len(p) >= n:
            fade = np.linspace(0, 1, n, dtype=np.float32)
            if out.ndim > 1:
                fade = fade[:, None]
            out = np.concatenate(
                [out[:-n], out[-n:] * (1 - fade) + p[:n] * fade, p[n:]]
            )
        else:
            out = np.concatenate([out, p])
    return out


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8080",
    temperature: float = 0.5,
    repetition_penalty: float = 1.3,
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
    max_workers: int = MAX_WORKERS,
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

    log.info("Processing %d chunks (%d workers)", len(pending), max_workers)

    with httpx.Client(base_url=base_url, timeout=300) as c:
        narrator_ref = _create_narrator_reference(c, temperature)

    jobs: list[tuple[int, int, dict]] = []
    for ci, (idx, text) in enumerate(pending):
        payload: dict = {
            "text": text,
            "format": "wav",
            "normalize": True,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "references": [narrator_ref],
        }
        jobs.append((ci, idx, payload))

    ch_wavs: dict[int, list[tuple[int, np.ndarray]]] = {}
    sr = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _synth_one, base_url, payload, ci, len(pending), starting_chapter + idx
            ): idx
            for ci, idx, payload in jobs
        }
        for fut in as_completed(futures):
            ci, data, sr = fut.result()
            ch_wavs.setdefault(futures[fut], []).append((ci, data))

    for idx, parts in ch_wavs.items():
        parts.sort(key=lambda x: x[0])
        audio = np.concatenate(
            [
                _crossfade([p for _, p in parts], sr).astype(np.float32),
                np.zeros(int(SILENCE * sr), dtype=np.float32),
            ]
        )
        sf.write(str(wav_paths[idx]), audio, sr, format="WAV", subtype="FLOAT")
        log.info(
            "Wrote ch %d '%s' %.1fs",
            starting_chapter + idx + 1,
            sel[idx].title[:40],
            len(audio) / sr,
        )

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]
