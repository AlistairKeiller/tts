import io, logging, time
from pathlib import Path

import httpx, numpy as np, soundfile as sf
from chonkie import SentenceChunker
from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=600)
CROSSFADE = 0.03
SILENCE = 0.5

NARRATOR_SAMPLE_TEXT = (
    "In the quiet hours before dawn, the world seems to hold its breath. "
    "Every story begins with a single moment, a choice that sets everything in motion. "
    "The pages ahead are filled with wonder and possibility, "
    "and it is my pleasure to guide you through each one."
)
NARRATOR_PROMPT = f"[calm, professional narration] {NARRATOR_SAMPLE_TEXT}"


def _create_narrator_reference(client: httpx.Client, temperature: float) -> str:
    """Generate a calm narrator sample and register it as a voice reference."""
    log.info("Creating narrator voice reference from default voice...")

    resp = client.post(
        "/v1/tts",
        json={
            "text": NARRATOR_PROMPT,
            "format": "wav",
            "normalize": True,
            "temperature": max(temperature - 0.2, 0.1),
            "repetition_penalty": 1.3,
        },
    )
    resp.raise_for_status()
    wav_bytes = resp.content

    resp2 = client.post(
        "/v1/references",
        files={"audio": ("narrator.wav", wav_bytes, "audio/wav")},
        data={"text": NARRATOR_SAMPLE_TEXT},
    )
    resp2.raise_for_status()
    body = resp2.json()
    ref_id = body.get("reference_id") or body.get("id")
    log.info("Created narrator reference: %s", ref_id)
    return ref_id


def _crossfade(parts: list[np.ndarray], sr: int) -> np.ndarray:
    if len(parts) <= 1:
        return parts[0] if parts else np.array([], dtype=np.float32)
    n = int(CROSSFADE * sr)
    out = parts[0].copy()
    for p in parts[1:]:
        if n and len(out) >= n and len(p) >= n:
            fo = np.linspace(1, 0, n, dtype=np.float32)
            fi = np.linspace(0, 1, n, dtype=np.float32)
            mix = (
                out[-n:] * fo + p[:n] * fi
                if out.ndim == 1
                else out[-n:] * fo[:, None] + p[:n] * fi[:, None]
            )
            out = np.concatenate([out[:-n], mix, p[n:]])
        else:
            out = np.concatenate([out, p])
    return out


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8080",
    reference_id: str | None = None,
    temperature: float = 0.5,
    repetition_penalty: float = 1.3,
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
        payload: dict = {
            "text": text,
            "format": "wav",
            "normalize": True,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "reference_id": reference_id,
        }
        if reference_id:
            payload["reference_id"] = reference_id
        t0 = time.monotonic()
        resp = client.post("/v1/tts", json=payload)
        resp.raise_for_status()
        data, sr = sf.read(io.BytesIO(resp.content))
        el = time.monotonic() - t0
        log.info(
            "Chunk %d/%d  ch %d  %.1fs audio in %.1fs  (RTF: %.2f)",
            ci + 1,
            len(pending),
            starting_chapter + idx,
            len(data) / sr,
            el,
            len(data) / sr / el if el else 0,
        )
        ch_wavs.setdefault(idx, []).append(data)
    client.close()

    for idx, parts in ch_wavs.items():
        audio = np.concatenate(
            [
                _crossfade(parts, sr).astype(np.float32),
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
