
#🎬
> Software video player & encoder — H.264 / HEVC / AV1

Two deployable versions:

| Version | File | Use when |
|---|---|---|
| Desktop | `player.py` | Local use, full playback control |
| Web | `streamlit_app.py` | Deploy free to the cloud |

---

## Prerequisites

### FFmpeg (required by both versions)
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html, add bin/ to PATH
```

### Python packages
```bash
pip install -r requirements.txt
```

---

## Run locally

### Desktop (PyQt6)
```bash
python player.py
```
**Keyboard shortcuts:**
- `Space`  — Play / Pause
- `←` `→`  — Seek ±5 seconds
- `Ctrl+O` — Open file
- `Escape` — Stop

### Web (Streamlit)
```bash
streamlit run streamlit_app.py
```

---

## Deploy for free

### Option A — Streamlit Community Cloud (easiest)
1. Push this folder to a **public** GitHub repo
2. Go to https://share.streamlit.io → "New app"
3. Point to `streamlit_app.py`
4. Add a `packages.txt` file with content: `ffmpeg`  ← makes Streamlit install it for you

**`packages.txt`** (create in same folder):
```
ffmpeg
```

### Option B — Hugging Face Spaces
1. Create a new Space → SDK: **Streamlit**
2. Upload files, add `packages.txt` with `ffmpeg`
3. Free GPU is optional (encoder runs on CPU)

### Option C — Railway / Render
- Add start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- Set env var: `PORT=8501`
- Install FFmpeg via a build command or Dockerfile

---

## Architecture notes

| Component | Desktop | Web |
|---|---|---|
| Playback | OpenCV frame-by-frame → PyQt6 | Browser native HTML5 `<video>` |
| Encoding | FFmpeg CLI via subprocess | FFmpeg CLI via subprocess |
| Threading | QThread workers | Streamlit runs synchronously (blocking) |
| Seek | Frame-accurate (OpenCV) | Browser-native |

**Why OpenCV for desktop playback?**  
OpenCV + FFmpeg backend handles AVI/HEVC/AV1 containers that PyQt6's built-in multimedia
can't decode without platform-specific codecs.

**Why HTML5 for web playback?**  
Serving raw frames via Streamlit would be ~30× slower than letting the browser decode natively.

---

## CRF Reference
| CRF | Quality |
|---|---|
| 0 | Lossless (huge files) |
| 18 | Visually lossless |
| 23 | Default balanced (H.264) |
| 28 | Compact (noticeable loss) |
| 51 | Worst quality |
