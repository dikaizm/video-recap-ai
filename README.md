# Video Scene Analyzer — Movie Recap Pipeline

Turns a local movie file into a narrated recap video. The pipeline:

1. **Analyze** — extracts frames and sends them to a local VLM (Ollama) to describe each scene
2. **Transform** — builds a storyboard from scene descriptions, selects the recap subset
3. **Narrate** *(optional)* — ranks scenes by importance and writes voiceover lines via DeepSeek
4. **Evaluate** *(optional)* — runs a coherence pass on the narration script via DeepSeek
5. **TTS** — generates speech with Qwen3-TTS (Apple Silicon) and chops into per-scene MP3s
6. **Render** — composes the final video with Remotion

---

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3) — required for mlx_audio TTS
- **Python 3.11+**
- **Node.js 18+** and `npx`
- **ffmpeg** and **ffprobe** on PATH
- **Ollama** running locally with a vision model pulled

## Installation

```bash
# 1. Clone and set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install Node dependencies (Remotion)
cd remotion && npm install && cd ..

# 3. Pull the VLM model into Ollama
ollama pull gemma4:e2b

# 4. Download Qwen3 TTS model (requires HuggingFace token)
huggingface-cli download \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit

# 5. Set up environment variables
cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN=<your huggingface token>
#   DEEPSEEK_API_KEY=<your deepseek api key>  ← required for --narrate
```

---

## Input Folder Structure

Create a folder under `input/` for each movie:

```
input/
└── my-movie/
    ├── video.mp4          # required — any format ffmpeg can read
    └── story.md           # optional — synopsis to guide narration accuracy
```

The pipeline auto-discovers `video.{mp4,mkv,avi,...}` and `story.{md,txt}` inside the folder.

---

## Basic Usage

```bash
source venv/bin/activate

# Minimal run — VLM analysis + TTS, no DeepSeek narration
python recap_pipeline/main.py --input input/my-movie/

# Full run — with DeepSeek narration + coherence evaluation
python recap_pipeline/main.py --input input/my-movie/ --narrate

# Full run with 480p output (smaller file)
python recap_pipeline/main.py --input input/my-movie/ --narrate --render-height 480
```

Output lands in `output/my-movie_<timestamp>/`:

```
output/my-movie_20260429_133655/
├── recap.mp4                  # full-resolution render
├── recap_480p.mp4             # downscaled render (if --render-height used)
├── storyboard.json            # scene storyboard (timestamps, frames)
├── storyboard_narrated.json   # storyboard + DeepSeek narration
├── storyboard_evaluated.json  # storyboard after coherence pass
├── storyboard_tts.json        # storyboard + ttsText field per scene
├── pipeline.log               # full pipeline log
├── run_log.json               # timing and metadata summary
└── voiceover/
    ├── _full.mp3              # concatenated full TTS audio
    ├── _chunk_00.mp3          # individual TTS generation chunks
    ├── scene_01.mp3           # per-scene trimmed audio
    ├── scene_02.mp3
    ├── ...
    └── tts_manifest.json      # TTS review: chunk texts, scene timestamps
```

---

## Reusing Analysis Across Runs

VLM analysis is the slowest step (~10–30 min for a feature film). Reuse an existing result:

```bash
python recap_pipeline/main.py \
  --input input/my-movie/ \
  --skip-analysis \
  --analysis-json output/my-movie_<timestamp>/analysis.json \
  --narrate
```

---

## All Options

| Flag | Default | Description |
|---|---|---|
| `--input FOLDER` | — | Input folder with `video.*` and optional `story.*` |
| `--skip-analysis` | off | Skip VLM frame analysis |
| `--analysis-json PATH` | — | Reuse existing analysis JSON (implies `--skip-analysis`) |
| `--narrate` | off | Generate narration via DeepSeek before TTS |
| `--narrate-force` | off | Regenerate narration even if cached |
| `--story-context TEXT\|FILE` | — | Synopsis text or `.md/.txt` file to guide narration |
| `--no-evaluate` | off | Skip story coherence evaluation after narration |
| `--no-tts` | off | Skip TTS — render with no voiceover |
| `--qwen3-speed FLOAT` | `1.3` | TTS speech speed (1.0 = normal, 1.5 = faster) |
| `--qwen3-speaker NAME` | `Ryan` | Speaker voice name |
| `--qwen3-instruct TEXT` | `warm` | Voice style instruction |
| `--qwen3-mode` | `custom` | `custom` (named speaker) or `design` (style-only) |
| `--qwen3-model-path PATH` | `models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Local TTS model directory |
| `--recap-ratio FLOAT` | `0.15` | Recap length as fraction of original (0.15 = 15%) |
| `--render-height INT` | — | Downscale output to this height (e.g. `480`) |
| `--fps INT` | `30` | Output frame rate |
| `--decode-height INT` | `640` | Frame decode height for VLM input (0 = full resolution) |
| `--ollama-model` | `gemma4:e2b` | Ollama VLM model name |
| `--ollama-host` | `http://localhost:11434` | Ollama API base URL |
| `--concurrency INT` | half CPU cores | Remotion render concurrency |
| `--gl` | `angle` | Remotion WebGL backend (`angle`, `swiftshader`, `egl`) |
| `--deepseek-key KEY` | `$DEEPSEEK_API_KEY` | DeepSeek API key for narration and TTS alignment |
| `--output PATH` | auto | Override output MP4 path |

---

## Reviewing TTS Output

Before rendering you can inspect what the TTS generated:

**`voiceover/tts_manifest.json`** — structured review of the TTS run:
```json
{
  "totalWords": 906,
  "audioDurationSec": 458.8,
  "chunks": [
    { "index": 0, "scenes": [1,2,...,10], "words": 147, "text": "A man counts..." }
  ],
  "scenes": [
    { "window": 1, "ttsText": "A man counts...", "audioPath": "scene_01.mp3", "start": 0.0, "end": 7.2 }
  ]
}
```

**`storyboard_tts.json`** — full storyboard with `ttsText` added to each scene for easy diffing against `narratedText`.

---

## Pipeline Internals

### Why chunked TTS?
Qwen3-TTS silently truncates input beyond ~200 words. The pipeline splits narration into ≤150-word chunks, generates each separately, then concatenates with ffmpeg before STT alignment.

### Audio chopping
After TTS, the pipeline runs `faster-whisper` (`base` model) on the full audio to get word-level timestamps, then uses DeepSeek to align the STT transcript against known narration text, yielding exact sentence-boundary cut points. Falls back to proportional word-count indexing if LLM alignment fails.

### Scene display duration
Each scene displays for exactly as long as its voiceover audio plays (audio duration + 0.4s padding). There is no fixed clip length — the video follows the narration.
