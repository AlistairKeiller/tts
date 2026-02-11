import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from chonkie import SentenceChunker
from openai import OpenAI

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=500)


def synthesise_chapters(
    chapters: list[Chapter],
    output_dir: Path,
    *,
    speaker: str = "Aiden",
    starting_chapter: int = 0,
    ending_chapter: int | None = None,
    vllm_base_url: str = "http://localhost:8091/v1",
) -> list[Path]:
    """Generate one WAV per chapter using vLLM-Omni."""
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=vllm_base_url, api_key="none")
    wav_paths = []
    sr = 48000

    for i, ch in enumerate(
        chapters[starting_chapter:ending_chapter], start=starting_chapter
    ):
        log.info("Chapter %d/%d  '%s'", i + 1, len(chapters), ch.title[:40])
        chunks = [c.text for c in chunker.chunk(ch.text)]
        all_wavs = []

        for chunk in chunks:
            with client.audio.speech.with_streaming_response.create(
                model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                voice=speaker,
                input=chunk,
            ) as response:
                temp_path = output_dir / f"temp_{i}.wav"
                response.stream_to_file(str(temp_path))
                wav_data, sr = sf.read(str(temp_path))
                all_wavs.append(wav_data)
                temp_path.unlink()

        wav_path = output_dir / f"chapter_{i:04d}.wav"
        sf.write(str(wav_path), np.concatenate(all_wavs), sr)
        wav_paths.append(wav_path)

    return wav_paths
