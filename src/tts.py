import asyncio, io, logging, tempfile, time
from pathlib import Path

import httpx, numpy as np, soundfile as sf
from chonkie import SentenceChunker
from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=1200)
CROSSFADE, SILENCE, MAX_CONCURRENT = 0.03, 0.5, 24

NARRATOR_TEXT = (
    "In the quiet hours before dawn, the world seems to hold its breath. "
    "Every story begins with a single moment, a choice that sets everything in motion. "
    "The pages ahead are filled with wonder and possibility, "
    "and it is my pleasure to guide you through each one."
)


async def _make_ref(client: httpx.AsyncClient, path: Path) -> dict:
    resp = await client.post(
        "/v1/audio/speech",
        json={
            "input": f"[calm, professional, articulate male narration] {NARRATOR_TEXT}",
        },
    )
    resp.raise_for_status()
    path.write_bytes(resp.content)
    log.info("Narrator ref saved (%.1f KB)", len(resp.content) / 1024)
    return {"audio_path": str(path), "text": NARRATOR_TEXT}


async def _synth(client, sem, payload, ci, total, ch_idx):
    async with sem:
        t0 = time.monotonic()
        resp = await client.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()
        data, sr = sf.read(io.BytesIO(resp.content))
        el = time.monotonic() - t0
        log.info(
            "Chunk %d/%d ch%d %.1fs in %.1fs RTF:%.2f",
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


async def _run(
    chapters,
    output_dir,
    *,
    base_url,
    starting_chapter,
    ending_chapter,
    max_concurrent,
    **_kw,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sel = chapters[starting_chapter:ending_chapter]
    wav_paths = [
        output_dir / f"chapter_{starting_chapter + i:04d}.wav" for i in range(len(sel))
    ]

    pending: dict[int, list[tuple[int, str]]] = {}
    ci = 0
    for i, ch in enumerate(sel):
        if wav_paths[i].exists() and wav_paths[i].stat().st_size > 0:
            log.info("Skipping chapter %d (exists)", starting_chapter + i)
            continue
        for c in chunker.chunk(ch.text):
            pending.setdefault(i, []).append((ci, c.text))
            ci += 1

    if not pending:
        return wav_paths

    log.info(
        "Processing %d chunks, %d chapters, %d concurrent",
        ci,
        len(pending),
        max_concurrent,
    )
    limits = httpx.Limits(
        max_connections=max_concurrent + 4, max_keepalive_connections=max_concurrent
    )

    async with httpx.AsyncClient(
        base_url=base_url, timeout=600, limits=limits
    ) as client:
        ref_path = Path(tempfile.mkdtemp(prefix="ref_")) / "narrator.wav"
        ref = await _make_ref(client, ref_path)
        sem = asyncio.Semaphore(max_concurrent)

        for idx in sorted(pending):
            tasks = [
                _synth(
                    client,
                    sem,
                    {"input": text, "references": [ref]},
                    c,
                    ci,
                    starting_chapter + idx,
                )
                for c, text in pending[idx]
            ]
            results = await asyncio.gather(*tasks)
            sr = results[0][2]
            parts = sorted([(c, d) for c, d, _ in results])
            audio = np.concatenate(
                [
                    _crossfade([p for _, p in parts], sr).astype(np.float32),
                    np.zeros(int(SILENCE * sr), dtype=np.float32),
                ]
            )
            sf.write(str(wav_paths[idx]), audio, sr, format="WAV", subtype="FLOAT")
            log.info(
                "Wrote ch%d '%s' %.1fs",
                starting_chapter + idx + 1,
                sel[idx].title[:40],
                len(audio) / sr,
            )

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]


def synthesise_chapters(
    chapters,
    output_dir,
    *,
    base_url="http://127.0.0.1:8000",
    temperature=0.5,
    repetition_penalty=1.3,
    starting_chapter=0,
    ending_chapter=None,
    max_workers=MAX_CONCURRENT,
):
    return asyncio.run(
        _run(
            chapters,
            output_dir,
            base_url=base_url,
            starting_chapter=starting_chapter,
            ending_chapter=ending_chapter,
            max_concurrent=max_workers,
        )
    )
