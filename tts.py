import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(tokenizer="character", chunk_size=500, chunk_overlap=0)


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    speaker: str = "Aiden",
    device: str = "cuda:0",
) -> list[Path]:
    """Generate one WAV per chapter, return list of paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    attn = "flash_attention_2" if torch.cuda.is_available() else "eager"
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map=device if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        attn_implementation=attn,
    )

    wav_paths: list[Path] = []
    for i, ch in enumerate(chapters):
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
