import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=1500)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    device = _detect_device()
    log.info("Using device: %s", device)

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    attn = "flash_attention_2" if device.startswith("cuda") else "eager"
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    if device.startswith("cuda"):
        model = torch.compile(model, mode="reduce-overhead")

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
            wavs, sr = model.generate_custom_voice(
                text=chunks,
                language=["Auto"] * len(chunks),
                speaker=[speaker] * len(chunks),
            )
            sf.write(str(wav_path), np.concatenate(wavs).astype(np.float32), sr)
            del wavs
            torch.cuda.empty_cache()
            wav_paths.append(wav_path)

    return wav_paths
