# Claude Instructions for video-recap-ai Pipeline

## Overview
This is a video-to-recap pipeline that analyzes videos, generates narration, adds text-to-speech, and renders the final output as a video with overlays.

**Pipeline stages:**
1. **Analysis** — Extract frames with PyAV, send to Ollama VLM (vision model) for scene descriptions
2. **Transform** — Select key scenes (storyboard), extract keyframes
3. **Narrate** (optional) — Generate narration via DeepSeek API using scene descriptions
4. **TTS** (optional) — Convert narration to speech with ElevenLabs or local TTS
5. **Align** — Sync voiceover timing to scene durations
6. **Render** — Use Remotion to create final video with overlays (logos, captions, credits, greeting card)
7. **Metadata** — Generate YouTube title/description via DeepSeek

The pipeline also runs an **agent self-review loop** by default — it renders an initial output, then iteratively evaluates and refines it based on VLM feedback.

---

## Prerequisites

### System Requirements
- **Python 3.10+** (code uses PEP 604 union syntax `str | None`)
- **Ollama** running locally (default: `http://localhost:11434`)
  - Model: `smolvlm:500m` (or specify `--ollama-model <name>`)
  - Run: `ollama serve` before starting the pipeline

### Environment Variables
Create a `.env` file in the project root:
```env
DEEPSEEK_API_KEY=sk_...          # For narration (if --narrate used)
ELEVENLABS_API_KEY=sk_...        # For TTS (if using ElevenLabs)
```

### Dependencies
```bash
pip install pyav faster-whisper torch torchaudio moviepy pillow pydantic requests
```

### Activate Local Virtual Environment
**Before running any commands**, activate the local venv:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```
All `python` and `python3` commands in this document assume the venv is active.

---

## Running the Pipeline

**⚠️ Always activate the venv first** (see "Activate Local Virtual Environment" above).

### Basic Usage

**Folder-based input (recommended):**
```bash
python recap_pipeline/main.py --input input/movie-name/
```

Expects:
- `input/movie-name/video.{mp4,mkv,mov,...}` — required
- `input/movie-name/story.{md,txt}` — optional (provides character/genre context)

**Direct video path:**
```bash
python recap_pipeline/main.py --video /path/to/video.mp4
```

---

## Key Flags

### Analysis
- `--ollama-model <name>` — VLM model name (default: `smolvlm:500m`)
- `--ollama-host <url>` — Ollama endpoint (default: `http://localhost:11434`)
- `--decode-height <px>` — Downscale frames before VLM for speed (e.g., 640)
- `--skip-analysis` — Use existing `analysis.json`

### Narration & Audio
- `--narrate` — Generate narration via DeepSeek (requires `DEEPSEEK_API_KEY`)
- `--no-tts` — Skip text-to-speech (render without voiceover)
- `--tts-backend <backend>` — TTS provider: `elevenlabs` (default), `gtts`, `pyttsx3`
- `--voice-name <name>` — ElevenLabs voice (default: `aria`)

### Rendering & Output
- `--no-agent` — Skip agent self-review loop (runs by default)
- `--skip-render` — Stop after narration/TTS, don't render video
- `--resume` — Resume from last completed step (checks `run.state.json`)
- `--output-dir <path>` — Custom output directory (default: `output/<run_id>/`)

### VLM Quality Control
- `--vlm-min-coverage <ratio>` — Min fraction of scenes with descriptions (default: 0.25)
- `--context <text>` — Background info for VLM (e.g., "movie about espionage, main character is a spy")

---

## Understanding the Agent Loop

When run with the default (no `--no-agent`):

1. **Initial render** — runs narration + TTS + Remotion render
2. **Agent review** — VLM evaluates the render against the original scene descriptions
3. **Evaluation** — agent scoring: scene match, dialogue quality, pacing
4. **Self-correct** — agent auto-fixes detected issues:
   - Hallucinated scenes? Regenerates narration
   - Dialogue/beat mismatch? Re-aligns timing
   - Weak scene match? Updates storyboard
5. **Re-render** — applies corrections and renders again
6. **Repeat** — iterates until quality threshold met or max iterations reached

Check `output/<run_id>/agent_review.json` for iteration details.

---

## Scene Analysis: Resume & Incremental Save

The analysis step now supports **resumable analysis**:
- After each VLM window is processed, results are saved to `*_analysis.json` incrementally
- If analysis crashes/stops mid-way, re-running with the same `--video` and settings will:
  1. Detect the partial output
  2. Validate settings (interval, window size, overlap, model)
  3. Seek directly to the next unprocessed window
  4. Continue from there

**Important:** The sliding-window frame overlap is preserved during resume — the seek lands at exactly the right timestamp to continue the window sequence.

**To force a fresh start:** delete the `*_analysis.json` file.

---

## Output Structure

```
output/<run_id>/
├── analysis.json              # VLM scene descriptions
├── transcript.json            # Speech transcription
├── storyboard.json           # Key scenes selected for recap
├── voiceover.mp3             # Generated speech
├── beats.json                # Scene timing/alignment
├── <run_id>_render.mp4       # Final video output
├── <run_id>_metadata.md      # YouTube title/description
├── run.state.json            # Pipeline state for resume
└── agent_review.json         # Agent iteration history (if --agent used)
```

---

## Common Issues

### Ollama Not Found
```
Error: could not connect to Ollama at http://localhost:11434
```
→ Run `ollama serve` in another terminal before starting the pipeline.

### VLM Description Coverage Too Low
```
[error] VLM description coverage too low (10% < 25%).
```
→ Ollama model is not producing good descriptions. Try:
- Ensuring frames are not black/corrupt: `--decode-height 640` (check first few frames)
- Providing context: `--context "heist movie with main character named John"`
- Lowering threshold: `--vlm-min-coverage 0.1`

### Python 3.9 Not Supported
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```
→ Code requires Python 3.10+. Check `python3 --version` and upgrade if needed.

### Out of Memory During Analysis
→ Use `--decode-height 480` to reduce frame resolution before VLM inference.

---

## Performance Tips

1. **Speed up analysis:**
   - `--decode-height 640` — process lower-res frames
   - `--interval 10` — sample frames every 10s instead of 5s
   - `--window 2` — use 2-frame windows instead of 4

2. **Speed up narration:**
   - `--no-agent` — skip the agent review loop (1-2 iterations)
   - Narration is serial; TTS is the next bottleneck

3. **Reduce final output size:**
   - Video is rendered at 1080p by default; check `remotion/src/config.ts`

---

## Development & Debugging

### Check Pipeline State
```bash
cat output/<run_id>/run.state.json
```

### Re-run Single Step
```bash
# Just narration on existing analysis
python recap_pipeline/main.py --input input/movie/ \
    --skip-analysis --analysis-json output/<run_id>/analysis.json \
    --narrate --no-agent --skip-render

# Just render (no narration changes)
python recap_pipeline/main.py --input input/movie/ \
    --skip-analysis --skip-narrate --skip-render=false
```

### Incremental Debugging
Use `--resume` to pick up from where the pipeline last failed:
```bash
python recap_pipeline/main.py --input input/movie/ --resume
```

---

## When to NOT Use Flags

- **Don't use `--analysis-json` and `--skip-analysis` together** — skip-analysis takes precedence
- **Don't use `--no-agent` if you want metadata** — agent loop ends with Step 9 (metadata generation)
- **Don't use `--narrate` without `DEEPSEEK_API_KEY`** — pipeline will fail at narration
