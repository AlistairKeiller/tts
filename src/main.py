"""epub2audiobook: Convert an EPUB to an M4B audiobook via Fish Speech S2 Pro."""

import asyncio
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

import ebooklib
import ffmpeg
import httpx
import numpy as np
import soundfile as sf
import typer
from bs4 import BeautifulSoup
from chonkie import SentenceChunker
from ebooklib import epub
from lxml import etree

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=500)
PAUSE = 0.3
MAX_RETRIES = 5
MIN_CHAPTER_DURATION = 1.0  # seconds

NARRATOR_TEXT = """
"...And so, I cleaved the Snow Tyrant's head clean off." He finished the description of the battle with a thoughtful expression, looking at the stars burning in the cold black sky beyond the window. "I think that it was mostly lying through its teeth to confuse me — well, actually, I don't think that it had a mouth, so it was lying through whatever it had instead of teeth. Puppeteer was definitely not as immune to the madness of Corruption as it presented itself, at least. As for the rest…" He frowned. "...Maybe there was some truth to what it said, after all. Even if it twisted the meaning of it all entirely to sow seeds of doubt into my mind." Speaking of which, Sunny had felt a few of those still growing in his heart after returning from Ariel's Game — soon to give birth to larval Worm's of Doubt, certainly. He poisoned them with Death Will and obliterated them completely, shivering in fear and disgust, then asked Nephis to purify him with her radiant flames just in case. Kai underwent the same cleansing.
"""

# ── EPUB ─────────────────────────────────────────────────────────────────────

_AD_RE = re.compile(
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}[^◈✪★☆●■▶►⊛⊕✦✧☛➤⌘\n]{1,60}"
    r"[◈✪★☆●■▶►⊛⊕✦✧☛➤⌘]{1,2}(?:\s*\([^)]{0,30}\))?",
    re.IGNORECASE,
)
_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "u": "urn:oasis:names:tc:opendocument:xmlns:container",
}


def _clean(text: str) -> str:
    text = _AD_RE.sub("", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_footnotes(soup: BeautifulSoup) -> None:
    for a in soup.find_all("a", href=True):
        if not any(c.isalpha() for c in a.get_text()):
            a.decompose()
    for sup in soup.find_all("sup"):
        if sup.get_text(strip=True).isdigit():
            sup.decompose()


def _toc_map(book) -> dict[str, str]:
    m: dict[str, str] = {}

    def walk(items):
        for item in items:
            if hasattr(item, "href") and hasattr(item, "title"):
                m[item.href.split("#")[0]] = item.title
            elif hasattr(item, "__iter__"):
                walk(item)

    walk(getattr(book, "toc", []))
    return m


def _cover(path: str) -> str | None:
    try:
        with zipfile.ZipFile(path) as z:
            t = etree.fromstring(z.read("META-INF/container.xml"))
            rf = t.xpath("/u:container/u:rootfiles/u:rootfile", namespaces=_NS)[0].get(
                "full-path"
            )
            t = etree.fromstring(z.read(rf))
            meta = t.xpath("//opf:metadata/opf:meta[@name='cover']", namespaces=_NS)
            if not meta:
                return None
            cid = meta[0].get("content")
            item = t.xpath(f"//opf:manifest/opf:item[@id='{cid}']", namespaces=_NS)
            if not item:
                return None
            href = item[0].get("href")
            zp = os.path.join(os.path.dirname(rf), href)
            out = path.rsplit(".", 1)[0] + ".png"
            from PIL import Image

            Image.open(z.open(zp)).save(out)
            return out
    except Exception:
        return None


@dataclass
class Chapter:
    title: str
    text: str


@dataclass
class BookMeta:
    title: str
    author: str
    cover_path: str | None


def parse_epub(path: str) -> tuple[list[Chapter], BookMeta]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = epub.read_epub(path)

    t = book.get_metadata("DC", "title")
    a = book.get_metadata("DC", "creator")
    meta = BookMeta(
        title=t[0][0] if t else "Unknown",
        author=a[0][0] if a else "Unknown",
        cover_path=_cover(path),
    )
    toc = _toc_map(book)
    items = {i.get_id(): i for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    chapters: list[Chapter] = []
    for idx, (sid, _) in enumerate(book.spine):
        item = items.get(sid)
        if not item or not item.get_body_content():
            continue
        soup = BeautifulSoup(
            item.get_body_content().decode("utf-8", errors="replace"), "html.parser"
        )
        _strip_footnotes(soup)
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 20:
            continue
        title = toc.get(item.get_name())
        if not title:
            h = soup.find(["h1", "h2", "h3"])
            title = h.get_text(strip=True) if h else f"Chapter {idx + 1}"
        chapters.append(Chapter(title=title, text=_clean(text)))
    return chapters, meta


# ── Text chunking ────────────────────────────────────────────────────────────


def _fix_caps(s: str) -> str:
    words = s.split()
    for i in range(len(words) - 2):
        if words[i].isupper() and words[i + 1].isupper() and words[i + 2].isupper():
            return s.lower().capitalize()
    return s


def _merge_short(sents: list[str], min_words: int = 6) -> list[str]:
    """Merge sentences shorter than min_words into their neighbours."""
    if not sents:
        return []
    out: list[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf.split()) < min_words:
            buf += " " + s
        else:
            out.append(buf)
            buf = s
    if buf:
        if out and len(buf.split()) < min_words:
            out[-1] += " " + buf
        else:
            out.append(buf)
    return out


def _chunk(text: str) -> list[str]:
    """Split text into TTS-friendly chunks via chonkie + short-sentence merging."""
    return _merge_short([c.text for c in chunker.chunk(text)])


# ── TTS via Fish S2 server ───────────────────────────────────────────────────


async def _get_ref(
    client: httpx.AsyncClient, ref_audio: Path | None, ref_dir: Path
) -> dict:
    """Get or generate a reference audio for voice cloning."""
    if ref_audio and ref_audio.exists():
        dst = ref_dir / "narrator.wav"
        shutil.copy2(ref_audio, dst)
        log.info("Using supplied reference: %s", ref_audio)
        return {"audio_path": str(dst), "text": NARRATOR_TEXT}

    resp = await client.post(
        "/v1/audio/speech",
        json={
            "input": f"[calm, professional, articulate male narration] {NARRATOR_TEXT}",
            "seed": 42,
            "temperature": 0.3,
            "top_p": 0.7,
        },
    )
    resp.raise_for_status()
    dst = ref_dir / "narrator.wav"
    dst.write_bytes(resp.content)
    log.info("Narrator ref generated (%.1f KB)", len(resp.content) / 1024)
    return {"audio_path": str(dst), "text": NARRATOR_TEXT}


async def _synth_chunk(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    payload: dict,
    chunk_id: int,
    total: int,
) -> tuple[np.ndarray, int]:
    """Synthesise one chunk with retries. Returns (audio_array, sample_rate)."""
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                resp = await client.post("/v1/audio/speech", json=payload)
                resp.raise_for_status()
                data, sr = sf.read(io.BytesIO(resp.content))
                if len(data) == 0:
                    raise ValueError("Server returned empty audio")
                el = time.monotonic() - t0
                log.info(
                    "Chunk %d/%d %.1fs in %.1fs RTF:%.2f",
                    chunk_id,
                    total,
                    len(data) / sr,
                    el,
                    len(data) / sr / el if el else 0,
                )
                return data.astype(np.float32), sr
            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                sf.LibsndfileError,
                ValueError,
            ) as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = 2**attempt
                log.warning(
                    "Chunk %d attempt %d failed: %s — retry in %ds",
                    chunk_id,
                    attempt,
                    e,
                    wait,
                )
                await asyncio.sleep(wait)


def _atomic_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a WAV file atomically via rename to avoid corrupt partial files."""
    tmp_path = path.with_suffix(".tmp.wav")
    sf.write(str(tmp_path), audio, sr, format="WAV", subtype="FLOAT")
    tmp_path.rename(path)


async def _run(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    base_url: str,
    start: int,
    end: int | None,
    max_concurrent: int,
    ref_audio: Path | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sel = chapters[start:end]
    wav_paths = [output_dir / f"ch_{start + i:04d}.wav" for i in range(len(sel))]

    # Build per-chapter job lists: list of (chapter_index, [texts])
    chapter_jobs: list[tuple[int, list[str]]] = []
    for i, ch in enumerate(sel):
        if wav_paths[i].exists() and wav_paths[i].stat().st_size > 0:
            try:
                info = sf.info(str(wav_paths[i]))
                if info.duration >= MIN_CHAPTER_DURATION:
                    log.info("Skip ch%d (exists, %.1fs)", start + i, info.duration)
                    continue
                log.warning(
                    "ch%d exists but too short (%.2fs) — regenerating",
                    start + i,
                    info.duration,
                )
            except Exception:
                log.warning("ch%d exists but unreadable — regenerating", start + i)
        texts = [_fix_caps(ch.title)] + [_fix_caps(t) for t in _chunk(ch.text)]
        chapter_jobs.append((i, texts))

    if not chapter_jobs:
        return wav_paths

    total_chunks = sum(len(texts) for _, texts in chapter_jobs)
    log.info(
        "Processing %d chunks across %d chapters, %d concurrent",
        total_chunks,
        len(chapter_jobs),
        max_concurrent,
    )

    limits = httpx.Limits(
        max_connections=max_concurrent + 4, max_keepalive_connections=max_concurrent
    )
    timeout = httpx.Timeout(connect=15, read=300, write=30, pool=30)
    ref_dir = Path(tempfile.mkdtemp(prefix="ref_"))
    failed_chapters: list[int] = []

    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=timeout, limits=limits
        ) as client:
            ref = await _get_ref(client, ref_audio, ref_dir)
            sem = asyncio.Semaphore(max_concurrent)
            chunk_num = 0

            for i, texts in chapter_jobs:
                tasks = []
                for text in texts:
                    chunk_num += 1
                    payload = {
                        "input": text,
                        "references": [ref],
                        "seed": 42,
                        "temperature": 0.3,
                        "top_p": 0.7,
                    }
                    tasks.append(
                        _synth_chunk(client, sem, payload, chunk_num, total_chunks)
                    )

                try:
                    results = await asyncio.gather(*tasks)
                except Exception:
                    log.exception(
                        "ch%d '%s' failed — skipping",
                        start + i + 1,
                        sel[i].title[:40],
                    )
                    failed_chapters.append(start + i + 1)
                    continue

                if not results:
                    log.warning("ch%d produced no audio — skipping", start + i + 1)
                    failed_chapters.append(start + i + 1)
                    continue

                sr = results[0][1]
                gap = np.zeros(int(PAUSE * sr), dtype=np.float32)
                parts: list[np.ndarray] = []
                for audio, _ in results:
                    parts.extend([audio, gap])

                joined = np.concatenate(parts)
                _atomic_write_wav(wav_paths[i], joined, sr)
                log.info(
                    "Wrote ch%d '%s' %.1fs",
                    start + i + 1,
                    sel[i].title[:40],
                    len(joined) / sr,
                )
                del parts, joined
    finally:
        shutil.rmtree(ref_dir, ignore_errors=True)

    if failed_chapters:
        log.warning("Failed chapters: %s", failed_chapters)

    return [p for p in wav_paths if p.exists() and p.stat().st_size > 0]


def synthesise(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    ref_audio: Path | None = None,
    start: int = 0,
    end: int | None = None,
    max_workers: int = 1,
) -> list[Path]:
    coro = _run(
        chapters,
        output_dir,
        base_url=base_url,
        start=start,
        end=end,
        max_concurrent=max_workers,
        ref_audio=ref_audio,
    )

    try:
        asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply()
    except RuntimeError:
        pass

    return asyncio.run(coro)


# ── M4B ──────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _aac_codec() -> str:
    """Detect libfdk_aac once and cache the result."""
    return (
        "libfdk_aac"
        if "libfdk_aac" in subprocess.getoutput("ffmpeg -encoders")
        else "aac"
    )


def build_m4b(
    wavs: list[Path],
    titles: list[str],
    output: Path,
    *,
    book_title: str = "Audiobook",
    book_author: str = "Unknown",
    cover_path: str | None = None,
    bitrate: str = "128k",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    cursor = 0
    spans: list[tuple[str, int, int]] = []
    for wp, t in zip(wavs, titles):
        dur = int(sf.info(str(wp)).duration * 1000)
        spans.append((t, cursor, cursor + dur))
        cursor += dur

    meta = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    meta.write(
        f";FFMETADATA1\ntitle={book_title}\nartist={book_author}\nalbum={book_title}\n\n"
    )
    for t, s, e in spans:
        safe_title = (
            t.replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
        )
        meta.write(
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={s}\nEND={e}\ntitle={safe_title}\n\n"
        )
    meta.close()

    cat = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for wp in wavs:
        safe_path = str(wp.resolve()).replace("'", "'\\''")
        cat.write(f"file '{safe_path}'\n")
    cat.close()

    try:
        codec = _aac_codec()
        log.info("Using AAC codec: %s", codec)
        (
            ffmpeg.input(cat.name, f="concat", safe=0)
            .audio.filter("loudnorm", I=-16, TP=-1.5, LRA=11)
            .output(
                str(output),
                map_metadata=1,
                acodec=codec,
                audio_bitrate=bitrate,
                ac=1,
                movflags="+faststart",
            )
            .global_args("-i", meta.name)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        log.error(
            "ffmpeg stderr:\n%s",
            e.stderr.decode(errors="replace") if e.stderr else "(none)",
        )
        raise
    finally:
        Path(meta.name).unlink(missing_ok=True)
        Path(cat.name).unlink(missing_ok=True)

    if cover_path and Path(cover_path).exists():
        try:
            from mutagen import mp4

            m = mp4.MP4(str(output))
            m["covr"] = [mp4.MP4Cover(Path(cover_path).read_bytes())]
            m.save()
        except Exception as e:
            log.warning("Cover embed failed: %s", e)

    log.info("M4B written: %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)


# ── CLI ──────────────────────────────────────────────────────────────────────

app = typer.Typer()


@app.command()
def main(
    epub: Annotated[Path, typer.Argument(help="Input .epub file.")],
    output: Annotated[Optional[Path], typer.Option("-o")] = None,
    url: Annotated[str, typer.Option("--url")] = "http://127.0.0.1:8000",
    ref: Annotated[Optional[Path], typer.Option("--ref", "-r")] = None,
    bitrate: Annotated[str, typer.Option()] = "128k",
    start: Annotated[int, typer.Option()] = 0,
    end: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option("-j")] = 2,
    list_chapters: Annotated[bool, typer.Option("--list")] = False,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify", help="Spot-check WAVs for silence before building M4B."
        ),
    ] = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    assert shutil.which("ffmpeg"), "ffmpeg not found"

    chapters, meta = parse_epub(str(epub))
    assert chapters, "No chapters found"
    log.info(
        "Parsed '%s' by %s — %d chapters",
        meta.title,
        meta.author,
        len(chapters),
    )

    if list_chapters:
        for i, ch in enumerate(chapters):
            print(f"{i}. {ch.title}")
        return

    tmp = Path(tempfile.mkdtemp(prefix="epub2ab_"))
    try:
        wavs = synthesise(
            chapters,
            tmp,
            base_url=url,
            ref_audio=ref,
            start=start,
            end=end,
            max_workers=workers,
        )
        assert wavs, "Nothing synthesised"

        if verify:
            _verify_wavs(wavs)

        titles = [c.title for c in chapters[start:end]][: len(wavs)]
        out = output or epub.with_suffix(".m4b")
        build_m4b(
            wavs,
            titles,
            out,
            book_title=meta.title,
            book_author=meta.author,
            cover_path=meta.cover_path,
            bitrate=bitrate,
        )
        print(f"✅ {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verify_wavs(wavs: list[Path]) -> None:
    """Spot-check WAVs for silence or corruption."""
    import random

    sample = random.sample(wavs, min(5, len(wavs)))
    for wp in sample:
        try:
            data, sr = sf.read(str(wp))
            rms = np.sqrt(np.mean(data**2))
            duration = len(data) / sr
            if rms < 1e-5:
                log.warning("⚠️  %s appears to be silence (RMS=%.2e)", wp.name, rms)
            elif duration < MIN_CHAPTER_DURATION:
                log.warning("⚠️  %s suspiciously short (%.2fs)", wp.name, duration)
            else:
                log.info("✓ %s OK (%.1fs, RMS=%.4f)", wp.name, duration, rms)
        except Exception as e:
            log.warning("⚠️  %s unreadable: %s", wp.name, e)


if __name__ == "__main__":
    app()
