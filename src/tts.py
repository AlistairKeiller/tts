import asyncio, io, logging, re, shutil, tempfile, time
from pathlib import Path

import httpx, numpy as np, soundfile as sf
from epub_parser import Chapter

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 700
MERGE_THRESHOLD = 200  # merge consecutive paragraphs shorter than this
MAX_CONCURRENT = 24
MAX_RETRIES = 3
SILENCE_BODY = 0.25
SILENCE_TITLE = 1.0
SILENCE_CHAPTER_END = 0.5

NARRATOR_TEXT = (
    "In the quiet hours before dawn, the world seems to hold its breath. "
    "Every story begins with a single moment, a choice that sets everything in motion. "
    "The pages ahead are filled with wonder and possibility, "
    "and it is my pleasure to guide you through each one."
)


def _chunk_text(text: str) -> list[str]:
    """Split on paragraphs, merge short ones, subdivide long ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge consecutive short paragraphs (dialogue, etc.)
    merged = []
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

    # Subdivide any that are still too long
    chunks = []
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


def _silence(sr: int, seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _join(parts: list[np.ndarray], sr: int) -> np.ndarray:
    """Join chunks: longer pause after title (index 0), shorter between body."""
    if not parts:
        return np.array([], dtype=np.float32)
    pieces = [parts[0], _silence(sr, SILENCE_TITLE)]
    for p in parts[1:]:
        pieces.extend([p, _silence(sr, SILENCE_BODY)])
    pieces.append(_silence(sr, SILENCE_CHAPTER_END))
    return np.concatenate(pieces)


async def _get_ref(
    client: httpx.AsyncClient, ref_audio: Path | None, ref_dir: Path
) -> dict:
    if ref_audio and ref_audio.exists():
        dst = ref_dir / "narrator.wav"
        shutil.copy2(ref_audio, dst)
        log.info("Using supplied reference: %s", ref_audio)
        return {"audio_path": str(dst), "text": NARRATOR_TEXT}

    resp = await client.post(
        "/v1/audio/speech",
        json={
            "input": f"[calm, professional, articulate male narration] {NARRATOR_TEXT}",
        },
    )
    resp.raise_for_status()
    dst = ref_dir / "narrator.wav"
    dst.write_bytes(resp.content)
    log.info("Narrator ref generated (%.1f KB)", len(resp.content) / 1024)
    return {"audio_path": str(dst), "text": NARRATOR_TEXT}


async def _synth(client, sem, payload, ch_i, ci, total):
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                resp = await client.post("/v1/audio/speech", json=payload)
                resp.raise_for_status()
                data, sr = sf.read(io.BytesIO(resp.content))
                el = time.monotonic() - t0
                log.info(
                    "Chunk %d/%d ch%d %.1fs in %.1fs RTF:%.2f",
                    ci + 1,
                    total,
                    ch_i,
                    len(data) / sr,
                    el,
                    len(data) / sr / el if el else 0,
                )
                return ch_i, ci, data, sr
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = 2**attempt
                log.warning(
                    "Chunk %d/%d failed (attempt %d/%d): %s — retrying in %ds",
                    ci + 1,
                    total,
                    attempt,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)


async def _run(
    chapters,
    output_dir,
    *,
    base_url,
    starting_chapter,
    ending_chapter,
    max_concurrent,
    ref_audio,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sel = chapters[starting_chapter:ending_chapter]
    wav_paths = [
        output_dir / f"chapter_{starting_chapter + i:04d}.wav" for i in range(len(sel))
    ]

    jobs = []
    ci = 0
    for i, ch in enumerate(sel):
        if wav_paths[i].exists() and wav_paths[i].stat().st_size > 0:
            log.info("Skipping chapter %d (exists)", starting_chapter + i)
            continue
        for text in [ch.title] + _chunk_text(ch.text):
            jobs.append((i, ci, text))
            ci += 1

    if not jobs:
        return wav_paths

    log.info("Processing %d chunks, %d concurrent", len(jobs), max_concurrent)
    limits = httpx.Limits(
        max_connections=max_concurrent + 4, max_keepalive_connections=max_concurrent
    )
    ref_dir = Path(tempfile.mkdtemp(prefix="ref_"))

    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=600, limits=limits
        ) as client:
            ref = await _get_ref(client, ref_audio, ref_dir)
            sem = asyncio.Semaphore(max_concurrent)
            tasks = [
                _synth(
                    client,
                    sem,
                    {"input": txt, "references": [ref]},
                    starting_chapter + ch_i,
                    ci,
                    len(jobs),
                )
                for ch_i, ci, txt in jobs
            ]
            results = await asyncio.gather(*tasks)
    finally:
        shutil.rmtree(ref_dir, ignore_errors=True)

    by_ch: dict[int, list[tuple[int, np.ndarray]]] = {}
    sr = results[0][3]
    for ch_idx, ci, data, _ in results:
        by_ch.setdefault(ch_idx, []).append((ci, data))

    for ch_idx in sorted(by_ch):
        i = ch_idx - starting_chapter
        parts = [d for _, d in sorted(by_ch[ch_idx])]
        audio = _join(parts, sr)
        sf.write(str(wav_paths[i]), audio, sr, format="WAV", subtype="FLOAT")
        log.info(
            "Wrote ch%d '%s' %.1fs", ch_idx + 1, sel[i].title[:40], len(audio) / sr
        )

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]


def synthesise_chapters(
    chapters,
    output_dir,
    *,
    base_url="http://127.0.0.1:8000",
    ref_audio=None,
    starting_chapter=0,
    ending_chapter=None,
    max_workers=MAX_CONCURRENT,
):
    return asyncio.run(
        _run(
            chapters,
            output_dir,
            base_url=base_url,
            ref_audio=ref_audio,
            starting_chapter=starting_chapter,
            ending_chapter=ending_chapter,
            max_concurrent=max_workers,
        )
    )
