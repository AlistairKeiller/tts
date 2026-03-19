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
import zipfile
from dataclasses import dataclass
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
MAX_RETRIES = 3

NARRATOR_TEXT = (
    "In the quiet hours before dawn, the world seems to hold its breath. "
    "Every story begins with a single moment, a choice that sets everything in motion. "
    "The pages ahead are filled with wonder and possibility, "
    "and it is my pleasure to guide you through each one."
)

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
            "input": f"[calm, professional, articulate male narration] {NARRATOR_TEXT}"
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
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
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
            log.info("Skip ch%d (exists)", start + i)
            continue
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
    ref_dir = Path(tempfile.mkdtemp(prefix="ref_"))

    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=600, limits=limits
        ) as client:
            ref = await _get_ref(client, ref_audio, ref_dir)
            sem = asyncio.Semaphore(max_concurrent)
            chunk_num = 0

            # Process one chapter at a time, but chunks within a chapter run concurrently
            for i, texts in chapter_jobs:
                tasks = []
                for text in texts:
                    chunk_num += 1
                    payload = {"input": text, "references": [ref]}
                    tasks.append(
                        _synth_chunk(client, sem, payload, chunk_num, total_chunks)
                    )

                results = await asyncio.gather(*tasks)

                # Interleave silence between chunks
                sr = results[0][1]
                gap = np.zeros(int(PAUSE * sr), dtype=np.float32)
                parts: list[np.ndarray] = []
                for audio, _ in results:
                    parts.extend([audio, gap])

                joined = np.concatenate(parts)
                sf.write(str(wav_paths[i]), joined, sr, format="WAV", subtype="FLOAT")
                log.info(
                    "Wrote ch%d '%s' %.1fs",
                    start + i + 1,
                    sel[i].title[:40],
                    len(joined) / sr,
                )
                del parts, joined
    finally:
        shutil.rmtree(ref_dir, ignore_errors=True)

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
    return asyncio.run(
        _run(
            chapters,
            output_dir,
            base_url=base_url,
            start=start,
            end=end,
            max_concurrent=max_workers,
            ref_audio=ref_audio,
        )
    )


# ── M4B ──────────────────────────────────────────────────────────────────────


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
        meta.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={s}\nEND={e}\ntitle={t}\n\n")
    meta.close()

    cat = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for wp in wavs:
        cat.write(f"file '{wp.resolve()}'\n")
    cat.close()

    try:
        codec = (
            "libfdk_aac"
            if "libfdk_aac" in subprocess.getoutput("ffmpeg -encoders")
            else "aac"
        )
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
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s"
    )
    assert shutil.which("ffmpeg"), "ffmpeg not found"

    chapters, meta = parse_epub(str(epub))
    assert chapters, "No chapters found"

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


if __name__ == "__main__":
    app()
