import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chonkie import SentenceChunker
from qwen_tts import Qwen3TTSModel

from epub_parser import Chapter

log = logging.getLogger(__name__)
chunker = SentenceChunker(chunk_size=1500)


def _patch_fa4() -> bool:
    """Replace F.scaled_dot_product_attention with FlashAttention-4 on Blackwell/Hopper."""
    try:
        from flash_attn.cute import flash_attn_func as fa4_func
    except ImportError:
        log.warning("flash_attn.cute (FA4) not available, using default attention")
        return False

    import torch.nn.functional as F

    _original = F.scaled_dot_product_attention

    def _fa4_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        # FA4 requires bf16/fp16, no mask, 4-dim tensors
        # SDPA layout: (B, H, S, D) -> FA4 layout: (B, S, H, D)
        if (
            q.dtype in (torch.float16, torch.bfloat16)
            and attn_mask is None
            and q.ndim == 4
        ):
            out = fa4_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                causal=is_causal,
            )
            return out.transpose(1, 2)
        return _original(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw
        )

    F.scaled_dot_product_attention = _fa4_sdpa
    log.info("Patched SDPA with FlashAttention-4 (CuTe DSL)")
    return True


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

    _patch_fa4()

    attn = "sdpa" if device.startswith("cuda") else "eager"
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
            wav_path = output_dir / f"chapter_{i:04d}.wav"
            if wav_path.exists() and wav_path.stat().st_size > 0:
                log.info("Skipping chapter %d (already exists)", i)
                wav_paths.append(wav_path)
                continue

            chunks = [c.text for c in chunker.chunk(ch.text)]

            t0 = time.monotonic()
            wavs, sr = model.generate_custom_voice(
                text=chunks,
                language=["Auto"] * len(chunks),
                speaker=[speaker] * len(chunks),
            )
            elapsed = time.monotonic() - t0

            audio = np.concatenate(wavs).astype(np.float32)
            audio_dur = len(audio) / sr
            rtf = audio_dur / elapsed if elapsed > 0 else float("inf")

            log.info(
                "Chapter %d/%d  '%s'  %.1fs audio in %.1fs  (RTF: %.2f)",
                i + 1,
                len(chapters),
                ch.title[:40],
                audio_dur,
                elapsed,
                rtf,
            )

            sf.write(str(wav_path), audio, sr, format="WAV", subtype="FLOAT")
            del wavs, audio
            wav_paths.append(wav_path)
    return wav_paths
