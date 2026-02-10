# epub2audiobook

Convert EPUB e-books into M4B audiobooks with chapter markers using **Qwen3-TTS**.

## Prerequisites

- **Python ≥ 3.10**
- **CUDA GPU** (recommended — Qwen3-TTS runs on GPU; CPU fallback is slow)
- **ffmpeg** installed and on your `$PATH` (used to encode AAC and embed chapters)

## Install

```bash
# create a clean environment (recommended)
conda create -n epub2audiobook python=3.12 -y && conda activate epub2audiobook

# install the package
pip install -e .

# (optional) flash-attention for lower VRAM usage
pip install -U flash-attn --no-build-isolation
```

## Usage

```bash
# basic — uses default 0.6B model + Aiden voice
epub2audiobook mybook.epub

# specify output, model, voice, and language
epub2audiobook mybook.epub -o mybook.m4b \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --speaker Vivian \
    --language Chinese

# add a style instruction
epub2audiobook mybook.epub --instruct "Read calmly like a professional audiobook narrator."

# keep intermediate WAV files for inspection
epub2audiobook mybook.epub --keep-wav -v
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `epub` | *(required)* | Path to the `.epub` file |
| `-o / --output` | `<epub>.m4b` | Output M4B path |
| `--model` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | HuggingFace model id or local path |
| `--speaker` | `Aiden` | Speaker voice name |
| `--language` | `Auto` | Target language (`Auto`, `English`, `Chinese`, …) |
| `--instruct` | *(empty)* | Style instruction for the TTS model |
| `--device` | `cuda:0` | Torch device |
| `--dtype` | `bfloat16` | Model precision |
| `--bitrate` | `64k` | AAC bitrate for M4B |
| `--title` | EPUB filename | Book title in M4B metadata |
| `--author` | *(empty)* | Author in M4B metadata |
| `--keep-wav` | off | Keep intermediate chapter WAV files |
| `-v / --verbose` | off | Debug logging |

### Available Speakers

| Speaker | Description | Native Language |
|---|---|---|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, slightly husky | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

## How It Works

1. **Parse EPUB** — extracts chapter structure and plain text using `ebooklib`
2. **TTS** — feeds each chapter (split into manageable chunks) through Qwen3-TTS
3. **M4B assembly** — concatenates chapter WAVs, encodes to AAC, and embeds ffmpeg chapter metadata into a single `.m4b` file
