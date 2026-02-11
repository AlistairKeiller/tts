import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=500)


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
    attn = "flash_attention_2" if device.startswith("cuda") else "eager"
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )

    wav_paths: list[Path] = []
    with torch.inference_mode():
        for i, ch in enumerate(
            chapters[starting_chapter:ending_chapter], start=starting_chapter
        ):
            log.info("Chapter %d/%d  '%s'", i + 1, len(chapters), ch.title[:40])
            chunks = [c.text for c in chunker.chunk(ch.text)]
            wavs, sr = model.generate_custom_voice(
                text=chunks,
                language=["Auto"] * len(chunks),
                speaker=[speaker] * len(chunks),
            )
            wav_path = output_dir / f"chapter_{i:04d}.wav"
            sf.write(str(wav_path), np.concatenate(wavs), sr)
            wav_paths.append(wav_path)

    return wav_paths
