import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from faster_qwen3_tts import FasterQwen3TTS

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=1500)

SAMPLE_RATE = 24_000  # fallback if model doesn't report one


def _to_numpy(wav) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        return wav.cpu().float().numpy()
    return np.asarray(wav, dtype=np.float32)


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    speaker: str = "Aiden",
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
) -> list[Path]:
    """Generate one WAV per chapter, return list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    model = FasterQwen3TTS.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    )

    wav_paths: list[Path] = []
    with torch.inference_mode():
        for i, ch in enumerate(
            chapters[starting_chapter:ending_chapter], start=starting_chapter
        ):
            wav_path = output_dir / f"chapter_{i:04d}.wav"
            if wav_path.exists() and wav_path.stat().st_size > 0:
                log.info("Skipping chapter %d (already exists)", i)
                wav_paths.append(wav_path)
                continue

            log.info("Chapter %d/%d  '%s'", i + 1, len(chapters), ch.title[:40])
            chunks = [c.text for c in chunker.chunk(ch.text)]
            if not chunks:
                log.warning("Chapter %d has no text chunks, skipping", i)
                continue

            wavs: list[np.ndarray] = []
            sr = SAMPLE_RATE
            for chunk in chunks:
                wav, chunk_sr = model.generate_custom_voice(
                    text=chunk,
                    language="english",
                    speaker=speaker,
                )
                log.debug("chunk_sr=%r  type=%s", chunk_sr, type(chunk_sr))
                if chunk_sr is not None and int(chunk_sr) > 0:
                    sr = int(chunk_sr)
                wavs.append(_to_numpy(wav))

            log.info("Writing %s  (sr=%d, chunks=%d)", wav_path.name, sr, len(wavs))
            sf.write(str(wav_path), np.concatenate(wavs).astype(np.float32), sr)
            del wavs
            torch.cuda.empty_cache()
            wav_paths.append(wav_path)
    return wav_paths
