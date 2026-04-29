"""
VideoForge Web – Encoder + VMAF Analytics + AI Video Enhancement
Run:  streamlit run app.py

Deploy:
  Streamlit Community Cloud → packages.txt: ffmpeg
  HF Spaces (Streamlit SDK) → same packages.txt
"""

import os, json, subprocess, tempfile, time, re
import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoForge · Pro Encoder + AI Enhancement",
    page_icon="🎬✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Light Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.stApp { background-color: #f8fafc; }

.vf-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #3b82f6 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
}
.vf-header h1 { 
    color: white; font-size: 1.8rem; font-weight: 600; margin: 0; 
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px;
}
.vf-header p { color: #dbeafe; margin: 6px 0 0; font-size: 0.92rem; opacity: 0.95; }

.vf-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 3px 12px;
    font-size: 0.72rem; color: #e0f2fe; margin: 4px 4px 0 0; font-weight: 500;
}
.vf-badge::before { content: "•"; margin-right: 4px; opacity: 0.7; }
.vf-badge.ai { background: linear-gradient(135deg, #7c3aed, #a855f7); border-color: #c4b5fd; }

.mode-toggle {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 12px 16px; margin: 16px 0 24px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.mode-toggle .toggle-label { font-weight: 600; color: #334155; font-size: 0.9rem; }
.mode-toggle .toggle-desc { color: #64748b; font-size: 0.82rem; margin-left: auto; }
.mode-active {
    background: #dbeafe; border-color: #93c5fd; color: #1e40af;
    padding: 2px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;
}

.vf-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; 
    text-transform: uppercase; color: #64748b; margin: 20px 0 10px;
    display: flex; align-items: center; gap: 8px;
}
.vf-label::before {
    content: ""; width: 20px; height: 2px; background: #3b82f6; border-radius: 2px;
}
.vf-label.ai::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }

[data-testid="metric-container"] {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6); color: white; border: none;
    border-radius: 10px; padding: 10px 24px; font-weight: 500; font-size: 0.9rem;
    transition: all 0.2s; box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
.stButton > button:disabled { background: #cbd5e1; box-shadow: none; cursor: not-allowed; }
.stButton > button.ai-btn {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    box-shadow: 0 2px 4px rgba(124, 58, 237, 0.25);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 6px;
}
.stProgress.ai > div > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7, #c4b5fd);
}

.cmp-table { 
    width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:8px; 
    background: white; border-radius: 10px; overflow: hidden; 
}
.cmp-table th {
    background: #f1f5f9; color:#475569; font-weight:600; padding:12px 14px; text-align:left;
    border-bottom:2px solid #e2e8f0; white-space: nowrap; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.cmp-table td {
    padding:11px 14px; border-bottom:1px solid #f1f5f9; color:#1e293b;
    font-family:'IBM Plex Mono', monospace; white-space: nowrap; font-size: 0.82rem;
}
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr:hover td { background:#f8fafc; }
.best-val { color:#166534; font-weight:600; }
.w-badge { 
    background:#dcfce7; color:#166534; border-radius:4px; padding:2px 8px; 
    font-size:0.68rem; font-weight:700; margin-left:6px; text-transform: uppercase;
}

.chip-avc  { background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }
.chip-hevc { background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }
.chip-av1  { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }
.chip-enh { background:#faf5ff; color:#6d28d9; border:1px solid #ddd6fe; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }

.q-exc { color:#166534; font-weight:600; }
.q-gd  { color:#1e40af; font-weight:600; }
.q-ok  { color:#b45309; font-weight:600; }
.q-bad { color:#b91c1c; font-weight:600; }

.src-bar {
    background:#eff6ff; border-radius:10px; padding:12px 18px; font-size:0.84rem;
    color:#1e40af; margin-top:14px; border-left:4px solid #3b82f6;
    display: flex; flex-wrap: wrap; gap: 12px 20px;
}
.src-bar b { color: #1e293b; }

.insight-note {
    background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:0.86rem; color:#854d0e; margin-top:14px;
    display: flex; gap: 10px; align-items: flex-start;
}
.insight-note::before { content: "💡"; font-size: 1.1rem; }
.insight-note.ai {
    background:#faf5ff; border-color:#c4b5fd; color:#5b21b6;
}
.insight-note.ai::before { content: "✨"; }

.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#f1f5f9; border-radius:10px; padding:4px; margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] { 
    border-radius:8px; padding:8px 20px; font-size:0.88rem; font-weight: 500; color: #64748b;
}
.stTabs [aria-selected="true"] { 
    background:white !important; box-shadow:0 2px 8px rgba(0,0,0,0.1); 
    color: #1e293b !important; font-weight: 600;
}

label { font-weight:500; color:#334155; font-size:0.88rem; }
.stAlert { border-radius:10px; border-width: 1px; }

.audio-metric {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px;
    background: #f8fafc; border-radius: 8px; font-size: 0.82rem; border: 1px solid #e2e8f0;
}
.audio-metric .icon {
    width: 24px; height: 24px; background: #3b82f6; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.7rem; font-weight: 600;
}

.proc-estimate {
    background: #f1f5f9; border-radius: 8px; padding: 10px 14px;
    font-size: 0.82rem; color: #475569; margin-top: 8px;
    border-left: 3px solid #64748b;
}
.proc-estimate.ai { border-left-color: #7c3aed; background: #faf5ff; color: #5b21b6; }

.enh-preview {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #7c3aed20, #a855f720);
    border: 1px solid #c4b5fd; border-radius: 20px;
    padding: 4px 12px; font-size: 0.75rem; color: #6d28d9; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Backend Utilities
# ══════════════════════════════════════════════════════════════════════════════

def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except:
        return False

def vmaf_ok() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-filters"], stderr=subprocess.STDOUT, text=True, timeout=10)
        return "libvmaf" in out
    except:
        return False

def probe(path: str) -> dict:
    """Extract comprehensive video AND audio metadata."""
    r = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "vcodec": "unknown", "vbitrate_kbps": 0,
        "acodec": "unknown", "abitrate_kbps": 0,
        "sample_rate": 0, "channels": 0, "audio_duration": 0.0,
        "has_audio": False, "color_space": "unknown", "bit_depth": 8
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            text=True, stderr=subprocess.DEVNULL)
        d = json.loads(out)
        fmt = d.get("format", {})
        r["duration"] = float(fmt.get("duration", 0))
        r["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000
        
        for s in d.get("streams", []):
            if s.get("codec_type") == "video":
                r["width"] = s.get("width", 0)
                r["height"] = s.get("height", 0)
                r["vcodec"] = s.get("codec_name", "unknown")
                r["color_space"] = s.get("color_space", "unknown")
                r["bit_depth"] = int(s.get("bits_per_raw_sample", 8) or 8)
                try:
                    n, dn = map(int, s.get("r_frame_rate","0/1").split("/"))
                    r["fps"] = round(n/dn, 2) if dn else 0.0
                except:
                    pass
            elif s.get("codec_type") == "audio":
                r["has_audio"] = True
                r["acodec"] = s.get("codec_name", "unknown")
                r["abitrate_kbps"] = int(s.get("bit_rate", 0) or 0) // 1000
                r["sample_rate"] = int(s.get("sample_rate", 0) or 0)
                r["channels"] = int(s.get("channels", 0) or 0)
                r["audio_duration"] = float(s.get("duration", 0) or 0)
    except:
        pass
    return r


def build_enhance_filters(settings: dict, src_meta: dict) -> list:
    """Build FFmpeg filter chain for AI enhancements (Vnova-style)."""
    filters = []
    
    if settings.get("denoise"):
        strength = settings.get("denoise_strength", 5)
        filters.append(f"hqdn3d={strength}:{strength}:{strength/2}:{strength/2}")
    
    if settings.get("sharpen"):
        amount = settings.get("sharpen_amount", 0.5)
        threshold = settings.get("sharpen_threshold", 5)
        filters.append(f"unsharp=5:5:{amount}:{threshold}")
    
    if settings.get("upscale"):
        target_w = settings.get("upscale_width", src_meta["width"] * 2)
        target_h = settings.get("upscale_height", src_meta["height"] * 2)
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(f"scale={target_w}:{target_h}:flags={algo}+accurate_rnd+full_chroma_int")
    
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.append("zscale=transfer=linear,format=gbrpf32le")
        filters.append(f"tonemap={settings.get('tonemap_algo', 'hable')}:desat=0.2")
        filters.append("zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le")
    
    if settings.get("color_enhance"):
        vibrance = settings.get("vibrance", 0.15)
        contrast = settings.get("contrast", 1.0)
        filters.append(f"eq=contrast={contrast}:saturation={1.0 + vibrance}")
    
    if settings.get("deblock"):
        strength = settings.get("deblock_strength", 5)
        filters.append(f"deblock=filter=strong:alpha={strength}:beta={strength//2}")
    
    if settings.get("frame_interp"):
        target_fps = settings.get("target_fps", 60)
        filters.append(f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    
    return filters


def estimate_processing_time(src_meta: dict, settings: dict) -> str:
    """Estimate processing time impact based on enhancements."""
    base_factor = 1.0
    if settings.get("denoise"): base_factor *= 1.3
    if settings.get("sharpen"): base_factor *= 1.1
    if settings.get("upscale"): base_factor *= 2.5 if settings.get("upscale_algo") == "lanczos" else 1.8
    if settings.get("hdr_convert"): base_factor *= 1.6
    if settings.get("color_enhance"): base_factor *= 1.15
    if settings.get("deblock"): base_factor *= 1.4
    if settings.get("frame_interp"): base_factor *= 3.0
    
    estimated_minutes = (src_meta["duration"] / 60) * base_factor
    if estimated_minutes < 1:
        return f"~{int(estimated_minutes * 60)}s"
    elif estimated_minutes < 10:
        return f"~{estimated_minutes:.1f} min"
    else:
        return f"~{estimated_minutes:.0f} min"


def encode(input_path, output_path, codec, crf, enhance_settings: dict, src_meta: dict,
           progress_cb=None, duration=0.0):
    """Encode with optional enhancement filters."""
    cmap = {
        "AVC (H.264)":  ("libx264",    ["-preset", "fast"]),
        "HEVC (H.265)": ("libx265",    ["-preset", "fast"]),
        "AV1":          ("libaom-av1", ["-cpu-used","8","-tile-columns","2", "-threads","4","-usage","realtime"]),
    }
    lib, extra = cmap.get(codec, ("libx264",["-preset","fast"]))
    filters = build_enhance_filters(enhance_settings, src_meta)
    
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", lib, "-crf", str(crf)] + extra
    if filters:
        filter_complex = ",".join(filters)
        cmd += ["-vf", filter_complex]
    cmd += ["-c:a", "copy", "-movflags", "+faststart", output_path]
    
    lines = []; t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1)
        for line in proc.stderr:
            lines.append(line.rstrip())
            if progress_cb and "time=" in line and duration > 0:
                try:
                    ts = line.split("time=")[1].split(" ")[0]
                    h, m, s = map(float, ts.split(":"))
                    progress_cb(min((h*3600+m*60+s)/duration, 1.0))
                except:
                    pass
        proc.wait()
        elapsed = time.time() - t0
        log = "\n".join(lines[-40:])
        if proc.returncode == 0:
            return True, "Done!", log, elapsed
        hints = {-6:"OOM — AI enhancements need extra RAM. Try disabling upscaling/frame interpolation.",
                 -9:"OOM (SIGKILL) — reduce enhancement complexity or file size.",
                 -11:"Segfault — check filter compatibility.",
                 1:"FFmpeg error — see log."}
        return False, hints.get(proc.returncode, f"FFmpeg exit {proc.returncode}"), log, elapsed
    except FileNotFoundError:
        return False, "FFmpeg not found.", "", 0.0
    except Exception as e:
        return False, str(e), "\n".join(lines), time.time()-t0


def quality_metrics(ref: str, dist: str, do_vmaf: bool) -> dict:
    """Compute PSNR, SSIM and optionally VMAF via FFmpeg."""
    res = {"psnr": None, "ssim": None, "vmaf": None}
    
    try:
        cmd = ["ffmpeg","-y","-i",dist,"-i",ref,
               "-filter_complex","[0:v][1:v]psnr[po];[0:v][1:v]ssim[so]",
               "-map","[po]","-map","[so]","-f","null","-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"PSNR", line, re.I):
                m = re.search(r"average[:\s]+([0-9.]+|inf)", line, re.I)
                if m:
                    v = m.group(1)
                    res["psnr"] = 100.0 if v == "inf" else round(float(v), 3)
            if re.search(r"SSIM", line, re.I):
                m = re.search(r"All[:\s]+([0-9.]+)", line, re.I)
                if m:
                    res["ssim"] = round(float(m.group(1)), 5)
    except:
        pass
    
    if do_vmaf:
        try:
            vf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            vf.close()
            cmd = ["ffmpeg","-y","-i",dist,"-i",ref,
                   "-filter_complex", f"[0:v][1:v]libvmaf=log_fmt=json:log_path={vf.name}",
                   "-f","null","-"]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            with open(vf.name) as f:
                vdata = json.load(f)
            score = (vdata.get("pooled_metrics",{}).get("vmaf",{}).get("mean")
                     or vdata.get("VMAF score") or vdata.get("aggregate",{}).get("VMAF_score"))
            if score is not None:
                res["vmaf"] = round(float(score), 2)
            os.unlink(vf.name)
        except:
            pass
    return res


def vmaf_display(v):
    if v is None: return "—", ""
    if v >= 93: return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80: return f"{v:.1f} · Good", "q-gd"
    if v >= 60: return f"{v:.1f} · Fair", "q-ok"
    return f"{v:.1f} · Poor", "q-bad"

def psnr_display(v):
    if v is None: return "—"
    tag = "Excellent" if v>=50 else "Good" if v>=40 else "Acceptable" if v>=30 else "Poor"
    return f"{v:.2f} dB · {tag}"

def format_audio_codec(codec: str) -> str:
    mapping = {"aac":"AAC","mp3":"MP3","opus":"Opus","vorbis":"Vorbis",
               "ac3":"AC-3","eac3":"E-AC-3","flac":"FLAC","pcm_s16le":"PCM 16-bit","alac":"ALAC"}
    return mapping.get(codec, codec.upper())

def format_sample_rate(sr: int) -> str:
    return f"{sr//1000} kHz" if sr >= 1000 else f"{sr} Hz"

def format_channels(ch: int) -> str:
    return {1:"Mono",2:"Stereo",6:"5.1",8:"7.1"}.get(ch, f"{ch} ch")


# ══════════════════════════════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "results": [], "inp": None, "meta": None, "sz": 0.0, "name": "",
    "enable_encoding": True, "enhance_settings": {
        "denoise": False, "denoise_strength": 5,
        "sharpen": False, "sharpen_amount": 0.5, "sharpen_threshold": 5,
        "upscale": False, "upscale_algo": "lanczos",
        "hdr_convert": False, "tonemap_algo": "hable",
        "color_enhance": False, "vibrance": 0.15, "contrast": 1.0,
        "deblock": False, "deblock_strength": 5,
        "frame_interp": False, "target_fps": 60
    }
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  Main Page
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="vf-header">
  <h1>🎬✨ VideoForge <span style="font-size:0.9em;opacity:0.9;font-weight:400">AI Pro</span></h1>
  <p>Professional encoding with AI enhancement · Vnova-style quality upscaling</p>
  <span class="vf-badge">H.264</span><span class="vf-badge">HEVC</span>
  <span class="vf-badge">AV1</span><span class="vf-badge">VMAF</span>
  <span class="vf-badge ai">✨ AI Enhance</span>
</div>""", unsafe_allow_html=True)

if not ffmpeg_ok():
    st.error("🔧 FFmpeg not found. Add `ffmpeg` to packages.txt and redeploy.")
    st.stop()

HAS_VMAF = vmaf_ok()

# Mode Toggle
st.markdown(f"""
<div class="mode-toggle">
  <span class="toggle-label">🎛️ Mode:</span>
  <span class="mode-active">{'⚙️ Encoder' if st.session_state.enable_encoding else '🎬 Test Player'}</span>
  <span class="toggle-desc">
    {'Full workflow with AI enhancement & encoding' if st.session_state.enable_encoding 
     else 'Playback & analytics only — no processing'}
  </span>
</div>""", unsafe_allow_html=True)

enable_encoding = st.toggle("⚙️ Enable Encoding Mode", value=st.session_state.enable_encoding,
    help="Toggle between encoder mode (with AI enhancements) and test player mode (analytics only).")
st.session_state.enable_encoding = enable_encoding

# File Upload
st.markdown('<div class="vf-label">📁 Source Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a video or click to browse", 
    type=["avi","mp4","mkv","mov","webm","flv","ts","m4v","mxf","prores"],
    label_visibility="collapsed")

if not uploaded:
    st.info("👆 Upload a video to begin" + (" analysis & enhancement" if enable_encoding else " analysis"))
    st.stop()

suf = os.path.splitext(uploaded.name)[-1].lower()
if st.session_state.name != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded.read())
        st.session_state.inp = tmp.name
    st.session_state.meta = probe(st.session_state.inp)
    st.session_state.sz = os.path.getsize(st.session_state.inp) / (1024*1024)
    st.session_state.name = uploaded.name
    if enable_encoding:
        st.session_state.results = []

meta = st.session_state.meta
sz_mb = st.session_state.sz
inp = st.session_state.inp

# Source Preview & Metadata
col_v, col_m = st.columns([3, 2], gap="large")
with col_v:
    st.markdown("### ▶️ Preview")
    st.video(inp)

with col_m:
    st.markdown("**📊 Source Media Info**")
    st.markdown('<div style="margin:14px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🎥 Video</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    r1.metric("Duration", f"{meta['duration']:.1f}s")
    r2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    r1.metric("Frame Rate", f"{meta['fps']} fps")
    r2.metric("Codec", meta["vcodec"].upper())
    r1.metric("Bitrate", f"{meta['vbitrate_kbps']} kbps")
    r2.metric("File Size", f"{sz_mb:.2f} MB")
    
    if meta["has_audio"]:
        st.markdown('<div style="margin:18px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🔊 Audio</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        a1.metric("Codec", format_audio_codec(meta["acodec"]))
        a2.metric("Channels", format_channels(meta["channels"]))
        a1.metric("Sample Rate", format_sample_rate(meta["sample_rate"]))
        a2.metric("Bitrate", f"{meta['abitrate_kbps']} kbps" if meta["abitrate_kbps"] > 0 else "Variable")
        if meta["audio_duration"] > 0 and meta["duration"] > 0:
            sync_diff = abs(meta["audio_duration"] - meta["duration"])
            sync_status = "✓ Synced" if sync_diff < 0.1 else f"⚠ {sync_diff:.2f}s"
            st.markdown(f'<div class="audio-metric"><span class="icon">🔗</span> A/V Sync: <b>{sync_status}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin:18px 0 8px;font-weight:600;color:#94a3b8;font-size:0.85rem">🔇 No audio</div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  ✨ AI ENHANCEMENT PANEL (Vnova-Style) - FIXED VERSION
# ══════════════════════════════════════════════════════════════════════════════
if enable_encoding:
    st.markdown('<div class="vf-label ai">✨ AI Video Enhancement</div>', unsafe_allow_html=True)
    st.markdown("Professional-grade enhancements similar to Vnova's pipeline. Toggle features below:")
    
    # Enhancement Presets
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("🎬 Standard Clean", use_container_width=True):
            st.session_state.enhance_settings.update({
                "denoise": True, "denoise_strength": 4,
                "sharpen": True, "sharpen_amount": 0.3,
                "deblock": True, "deblock_strength": 4,
                "upscale": False, "hdr_convert": False, "frame_interp": False
            })
            st.rerun()
    with preset_col2:
        if st.button("🔍 Detail Boost", use_container_width=True):
            st.session_state.enhance_settings.update({
                "sharpen": True, "sharpen_amount": 0.7, "sharpen_threshold": 3,
                "color_enhance": True, "vibrance": 0.2, "contrast": 1.1,
                "denoise": False, "upscale": False
            })
            st.rerun()
    with preset_col3:
        if st.button("🚀 AI Upscale 2x", use_container_width=True):
            st.session_state.enhance_settings.update({
                "upscale": True, "upscale_width": meta["width"]*2, "upscale_height": meta["height"]*2,
                "upscale_algo": "lanczos", "denoise": True, "denoise_strength": 5,
                "sharpen": True, "sharpen_amount": 0.4
            })
            st.rerun()
    
    # Enhancement Controls Grid - CLEAN STREAMLIT WIDGETS (NO RAW HTML)
    enh_col1, enh_col2 = st.columns(2)
    
    with enh_col1:
        # Denoise
        with st.container(border=True):
            st.markdown('**🧹 Denoise**')
            denoise = st.checkbox("Enable temporal noise reduction", 
                                 key="chk_denoise",
                                 value=st.session_state.enhance_settings["denoise"],
                                 help="Reduces grain and compression artifacts using 3D filtering")
            st.session_state.enhance_settings["denoise"] = denoise
            if denoise:
                st.session_state.enhance_settings["denoise_strength"] = st.slider(
                    "Strength", 1, 10, st.session_state.enhance_settings["denoise_strength"],
                    help="Higher = more aggressive noise removal")
            st.caption("Reduces grain, compression artifacts, and sensor noise.")
        
        # Sharpening
        with st.container(border=True):
            st.markdown('**🔍 Detail Enhancement**')
            sharpen = st.checkbox("Enable adaptive sharpening", 
                                key="chk_sharpen",
                                value=st.session_state.enhance_settings["sharpen"])
            st.session_state.enhance_settings["sharpen"] = sharpen
            if sharpen:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.enhance_settings["sharpen_amount"] = st.slider(
                        "Amount", -1.5, 1.5, st.session_state.enhance_settings["sharpen_amount"], 0.1)
                with c2:
                    st.session_state.enhance_settings["sharpen_threshold"] = st.slider(
                        "Threshold", 0, 50, st.session_state.enhance_settings["sharpen_threshold"])
            st.caption("Enhances edge definition using adaptive unsharp masking.")
        
        # AI Upscaling
        with st.container(border=True):
            st.markdown('**🔬 Resolution Upscaling**')
            upscale = st.checkbox("Enable high-quality upscaling", 
                                key="chk_upscale",
                                value=st.session_state.enhance_settings["upscale"])
            st.session_state.enhance_settings["upscale"] = upscale
            if upscale:
                st.session_state.enhance_settings["upscale_algo"] = st.selectbox(
                    "Algorithm", ["lanczos", "spline", "bicubic"], 
                    index=["lanczos","spline","bicubic"].index(
                        st.session_state.enhance_settings["upscale_algo"]))
                st.session_state.enhance_settings["upscale_width"] = st.number_input(
                    "Target Width", min_value=meta["width"], max_value=7680, 
                    value=st.session_state.enhance_settings.get("upscale_width", meta["width"]*2),
                    step=meta["width"])
                st.session_state.enhance_settings["upscale_height"] = st.number_input(
                    "Target Height", min_value=meta["height"], max_value=4320,
                    value=st.session_state.enhance_settings.get("upscale_height", meta["height"]*2),
                    step=meta["height"])
            st.caption("High-quality interpolation. For true AI super-resolution, integrate Real-ESRGAN externally.")
    
    with enh_col2:
        # HDR Conversion
        with st.container(border=True):
            st.markdown('**🌈 HDR → SDR Tonemapping**')
            hdr = st.checkbox("Enable HDR conversion", 
                            key="chk_hdr",
                            value=st.session_state.enhance_settings["hdr_convert"],
                            disabled=meta.get("bit_depth", 8) < 10)
            st.session_state.enhance_settings["hdr_convert"] = hdr
            if hdr and meta.get("bit_depth", 8) >= 10:
                st.session_state.enhance_settings["tonemap_algo"] = st.selectbox(
                    "Tonemap Algorithm", ["hable", "reinhard", "mobius", "linear"],
                    index=["hable","reinhard","mobius","linear"].index(
                        st.session_state.enhance_settings["tonemap_algo"]))
            elif meta.get("bit_depth", 8) < 10:
                st.caption("💡 Source is SDR (8-bit). HDR conversion requires 10-bit+ HDR10/HLG source.")
            st.caption("Converts HDR10/HLG to SDR with perceptual tonemapping.")
        
        # Color Enhancement
        with st.container(border=True):
            st.markdown('**🎨 Color Enhancement**')
            color = st.checkbox("Enable color boost", 
                              key="chk_color",
                              value=st.session_state.enhance_settings["color_enhance"])
            st.session_state.enhance_settings["color_enhance"] = color
            if color:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.enhance_settings["vibrance"] = st.slider(
                        "Vibrance", -0.5, 0.5, st.session_state.enhance_settings["vibrance"], 0.05)
                with c2:
                    st.session_state.enhance_settings["contrast"] = st.slider(
                        "Contrast", 0.5, 2.0, st.session_state.enhance_settings["contrast"], 0.05)
            st.caption("Enhances vibrancy while preserving natural skin tones.")
        
        # Artifact Reduction
        with st.container(border=True):
            st.markdown('**🧩 Artifact Reduction**')
            deblock = st.checkbox("Enable deblocking filter", 
                                 key="chk_deblock",
                                 value=st.session_state.enhance_settings["deblock"])
            st.session_state.enhance_settings["deblock"] = deblock
            if deblock:
                st.session_state.enhance_settings["deblock_strength"] = st.slider(
                    "Strength", 1, 10, st.session_state.enhance_settings["deblock_strength"])
            st.caption("Reduces macroblocking and ringing from heavy compression.")
    
    # Frame Interpolation (Advanced)
    with st.expander("🎞️ Advanced: Frame Interpolation (Motion Smoothing)"):
        frame_interp = st.checkbox("Enable frame interpolation", 
                                  value=st.session_state.enhance_settings["frame_interp"])
        st.session_state.enhance_settings["frame_interp"] = frame_interp
        if frame_interp:
            st.session_state.enhance_settings["target_fps"] = st.selectbox(
                "Target Frame Rate", [30, 48, 60, 120], 
                index=[30,48,60,120].index(st.session_state.enhance_settings["target_fps"]))
            st.warning("⚠️ Frame interpolation is CPU-intensive and may increase processing time 3-5x.")
        st.caption("Creates intermediate frames for smoother playback (e.g., 24fps → 60fps).")
    
    # Processing Estimate & Preview
    active_enhancements = sum([
        st.session_state.enhance_settings["denoise"],
        st.session_state.enhance_settings["sharpen"],
        st.session_state.enhance_settings["upscale"],
        st.session_state.enhance_settings["hdr_convert"],
        st.session_state.enhance_settings["color_enhance"],
        st.session_state.enhance_settings["deblock"],
        st.session_state.enhance_settings["frame_interp"]
    ])
    
    if active_enhancements > 0:
        est_time = estimate_processing_time(meta, st.session_state.enhance_settings)
        st.info(f"⏱️ **Estimated processing time**: {est_time} ({active_enhancements} enhancement{'s' if active_enhancements>1 else ''} active)")
        
        enh_names = []
        if st.session_state.enhance_settings["denoise"]: enh_names.append("🧹 Denoise")
        if st.session_state.enhance_settings["sharpen"]: enh_names.append("🔍 Sharpen")
        if st.session_state.enhance_settings["upscale"]: enh_names.append("🔬 Upscale")
        if st.session_state.enhance_settings["hdr_convert"]: enh_names.append("🌈 HDR")
        if st.session_state.enhance_settings["color_enhance"]: enh_names.append("🎨 Color")
        if st.session_state.enhance_settings["deblock"]: enh_names.append("🧩 Deblock")
        if st.session_state.enhance_settings["frame_interp"]: enh_names.append("🎞️ Interp")
        
        if enh_names:
            st.markdown(f"✨ **Active**: {' + '.join(enh_names)}")
    
    st.divider()
    
    # Encoder Settings
    st.markdown('<div class="vf-label">⚙️ Encoder Settings</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([2, 2, 1, 2])
    with s1:
        codec = st.selectbox("Video Codec", ["AVC (H.264)", "HEVC (H.265)", "AV1"],
            help="H.264 = fastest · HEVC = ~40% smaller · AV1 = best compression")
    with s2:
        crf = st.slider("CRF Quality", 0, 51, 23,
            help="Lower = better quality · 0=lossless · 18=visually lossless · 23=balanced")
    with s3:
        do_vmaf = st.checkbox("VMAF", value=HAS_VMAF, disabled=not HAS_VMAF)
        do_psnr = st.checkbox("PSNR/SSIM", value=True)
    with s4:
        if crf < 19: ql, qc = "🟢 High Quality", "#166534"
        elif crf < 29: ql, qc = "🟡 Balanced", "#b45309"
        elif crf < 40: ql, qc = "🟠 Compact", "#ea580c"
        else: ql, qc = "🔴 Low Quality", "#b91c1c"
        st.markdown(f"""
        <div style="text-align:center;padding:8px 0">
            <div style="font-weight:600;color:{qc}">{ql}</div>
            <div style="font-size:0.75rem;color:#64748b">CRF {crf}</div>
        </div>""", unsafe_allow_html=True)
    
    if codec == "AV1" and active_enhancements > 0:
        st.warning("⚠️ **AV1 + Enhancements**: Very resource intensive. May crash on free tiers. Use H.264/HEVC for cloud deployment.")
    
    # Encode Controls
    st.markdown('<div class="vf-label" style="margin-top:10px">🚀 Run Process</div>', unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1.2, 1.2, 1, 5])
    
    # Preview button
    with b1:
        if st.button("🔍 Preview Impact", use_container_width=True):
            est_meta = meta.copy()
            if st.session_state.enhance_settings["upscale"]:
                est_meta["width"] = st.session_state.enhance_settings["upscale_width"]
                est_meta["height"] = st.session_state.enhance_settings["upscale_height"]
            if st.session_state.enhance_settings["frame_interp"]:
                est_meta["fps"] = st.session_state.enhance_settings["target_fps"]
            
            st.markdown("### 📊 Estimated Output")
            bc1, bc2 = st.columns(2)
            with bc1:
                st.metric("Source", f"{meta['width']}×{meta['height']}", f"@ {meta['fps']} fps")
            with bc2:
                st.metric("Enhanced", f"{est_meta['width']}×{est_meta['height']}", f"@ {est_meta['fps']} fps")
                if est_meta["width"] > meta["width"]:
                    up_pct = ((est_meta["width"]/meta["width"])**2 - 1)*100
                    st.caption(f"+{up_pct:.0f}% more pixels")
    
    go = b2.button("✨ Enhance + Encode", type="primary", use_container_width=True, 
                   help="Apply selected enhancements then encode with chosen codec/CRF")
    clear = b3.button("🗑 Clear", use_container_width=True)
    
    if clear:
        st.session_state.results = []
        st.rerun()
    
    if go:
        out_path = inp.replace(suf, f"_enh_{codec.split()[0].lower()}_crf{crf}.mp4")
        bar = st.progress(0.0, text=f"Initializing enhancements + {codec}…", key="prog_encode")

        with st.spinner(f"✨ Applying enhancements + encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(
                inp, out_path, codec, crf, 
                st.session_state.enhance_settings, meta,
                progress_cb=lambda p: bar.progress(p, text=f"Processing… {p*100:.0f}%"),
                duration=meta["duration"],
            )

        if not ok:
            bar.empty()
            st.error(f"❌ {msg}")
            if fflog:
                with st.expander("📋 FFmpeg Log"):
                    st.code(fflog, language="bash")
        else:
            bar.progress(1.0, text="✅ Complete — analyzing quality…")
            out_meta = probe(out_path)
            out_sz = os.path.getsize(out_path) / (1024*1024)
            saved_pct = (1 - out_sz/sz_mb)*100 if sz_mb else 0

            qual = {"psnr":None,"ssim":None,"vmaf":None}
            if do_psnr or (do_vmaf and HAS_VMAF):
                with st.spinner("🔍 Computing quality metrics…"):
                    qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

            st.session_state.results.append({
                "codec": codec, "crf": crf,
                "size_mb": out_sz, "bitrate": out_meta["vbitrate_kbps"],
                "enc_time": enc_t, "saved": saved_pct,
                "cr": sz_mb/out_sz if out_sz else 0,
                "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"],
                "path": out_path,
                "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"],
                "sample_rate": meta["sample_rate"], "channels": meta["channels"],
                "enhancements": {k:v for k,v in st.session_state.enhance_settings.items() 
                               if k in ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"] and v},
                "out_res": f"{out_meta['width']}×{out_meta['height']}",
                "out_fps": out_meta["fps"]
            })
            bar.empty()
            
            q_parts = []
            if qual["vmaf"]: q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"]: q_parts.append(f"PSNR {qual['psnr']:.2f}dB")
            q_str = " · ".join(q_parts) if q_parts else "Analysis complete"
            
            enh_summary = " + ".join([k.capitalize() for k,v in st.session_state.enhance_settings.items() 
                                     if k in ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"] and v])
            enh_str = f" · ✨ {enh_summary}" if enh_summary else ""
            
            st.success(f"✅ {codec} CRF {crf}{enh_str} · {out_sz:.2f} MB · saved {saved_pct:.1f}% · {enc_t:.1f}s · {q_str}")

else:
    # Test Player Mode
    st.markdown('<div class="vf-label">🎬 Test Player Mode</div>', unsafe_allow_html=True)
    st.info("🎧 **Playback & Analytics**: Preview your source video. All metrics displayed. No processing performed.")
    if meta["has_audio"]:
        st.markdown(f"""
        <div style="background:#f1f5f9;border-radius:10px;padding:14px;margin:12px 0">
            <div style="font-weight:600;margin-bottom:8px;color:#334155">🔊 Audio Stream</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">{format_audio_codec(meta['acodec'])}</span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">{format_channels(meta['channels'])}</span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">{format_sample_rate(meta['sample_rate'])}</span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">{meta['abitrate_kbps']} kbps</span>
            </div>
        </div>""", unsafe_allow_html=True)
    st.caption("💡 Enable encoding mode above to access AI enhancement features.")

# ══════════════════════════════════════════════════════════════════════════════
#  Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════
results = st.session_state.results
if not results or not enable_encoding:
    if enable_encoding:
        st.caption("📊 Results appear here after encoding completes.")
    st.stop()

st.divider()
st.markdown('<div class="vf-label">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

tab_tbl, tab_chart, tab_dl = st.tabs(["📋 Comparison", "📊 Charts", "⬇ Downloads"])

# Comparison Table
with tab_tbl:
    best_sz = min(r["size_mb"] for r in results)
    best_cr = max(r["cr"] for r in results)
    best_spd = min(r["enc_time"] for r in results)
    best_vm = max((r["vmaf"] or 0) for r in results) if any(r["vmaf"] for r in results) else None
    
    def best_mark(val, best, fmt="{}", higher_better=False):
        if val is None or best is None: return "—"
        is_best = (abs(val - best) < 0.01) if isinstance(val, float) else (val == best)
        s = fmt.format(val)
        if is_best and best is not None:
            return f'<span class="best-val">{s} <span class="w-badge">Best</span></span>'
        return s

    rows_html = ""
    for r in results:
        cs = r["codec"].split()[0]
        chip_cls = {"AVC":"chip-avc","HEVC":"chip-hevc","AV1":"chip-av1"}.get(cs,"")
        tag = f'<span class="{chip_cls}">{cs}</span>'
        
        if r.get("enhancements"):
            enh_count = len([v for v in r["enhancements"].values() if v])
            tag += f' <span class="chip-enh">✨ {enh_count}</span>'

        vmaf_txt, vmaf_cls = vmaf_display(r["vmaf"])
        vmaf_cell = f'<span class="{vmaf_cls}">{vmaf_txt}</span>'
        if r["vmaf"] and best_vm and abs(r["vmaf"] - best_vm) < 0.01 and len(results)>1:
            vmaf_cell += ' <span class="w-badge">Best</span>'

        audio_info = f"{format_audio_codec(r['acodec'])} · {format_channels(r['channels'])}" if r.get('acodec') else "—"
        res_info = r.get("out_res", f"{meta['width']}×{meta['height']}")
        fps_info = r.get("out_fps", meta["fps"])

        rows_html += f"""<tr>
          <td>{tag}</td>
          <td style="font-family:'IBM Plex Mono'">{r['crf']}</td>
          <td>{best_mark(r['size_mb'], best_sz, "{:.2f} MB")}</td>
          <td>{r['bitrate']} kbps</td>
          <td>{best_mark(r['cr'], best_cr, "{:.2f}×", True)}</td>
          <td>{r['saved']:.1f}%</td>
          <td>{best_mark(r['enc_time'], best_spd, "{:.1f}s")}</td>
          <td>{vmaf_cell}</td>
          <td>{psnr_display(r['psnr'])}</td>
          <td>{"%.4f" % r['ssim'] if r['ssim'] else "—"}</td>
          <td style="font-size:0.78rem;color:#64748b">{res_info}@{fps_info}fps</td>
          <td style="font-size:0.78rem;color:#64748b">{audio_info}</td>
        </tr>"""

    st.markdown(f"""
    <table class="cmp-table">
      <thead><tr>
        <th>Codec</th><th>CRF</th><th>Size</th><th>Bitrate</th><th>Ratio</th>
        <th>Saved</th><th>Time</th><th>VMAF ↑</th><th>PSNR ↑</th><th>SSIM ↑</th>
        <th>Resolution</th><th>Audio</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="src-bar">
      <b>Source:</b> {meta['vcodec'].upper()} · {sz_mb:.2f} MB · {meta['vbitrate_kbps']} kbps ·
      {meta['width']}×{meta['height']} @ {meta['fps']} fps
      {f" · 🔊 {format_audio_codec(meta['acodec'])} {format_channels(meta['channels'])}" if meta['has_audio'] else ""}
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("📏 VMAF: 93+ Excellent · 80-93 Good | PSNR: 40+ dB Good | ✨ = enhancements applied | 🏆 Best = winner")

# Charts & Insights
with tab_chart:
    df = pd.DataFrame([{
        "Codec": r["codec"] + (f" ✨" if r.get("enhancements") else ""),
        "File Size (MB)": round(r["size_mb"], 3),
        "Bitrate (kbps)": r["bitrate"],
        "Encode Time (s)": round(r["enc_time"], 2),
        "Space Saved (%)": round(r["saved"], 1),
        "VMAF": r["vmaf"],
        "PSNR (dB)": round(r["psnr"], 2) if r["psnr"] else None,
        "Resolution": r.get("out_res", f"{meta['width']}×{meta['height']}"),
    } for r in results])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**📦 File Size**")
        size_df = pd.DataFrame(
            [{"Codec":"🎬 Original","File Size (MB)":round(sz_mb,3)}]
            + [{"Codec":r["codec"]+("✨" if r.get("enhancements") else ""),"File Size (MB)":round(r["size_mb"],3)} for r in results]
        ).set_index("Codec")
        st.bar_chart(size_df, color="#3b82f6", use_container_width=True)
        st.markdown("**⏱️ Process Time**")
        st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], color="#f97316", use_container_width=True)

    with c2:
        st.markdown("**📡 Bitrate**")
        brate_df = pd.DataFrame(
            [{"Codec":"🎬 Original","Bitrate (kbps)":meta['vbitrate_kbps']}]
            + [{"Codec":r["codec"]+("✨" if r.get("enhancements") else ""),"Bitrate (kbps)":r["bitrate"]} for r in results]
        ).set_index("Codec")
        st.bar_chart(brate_df, color="#8b5cf6", use_container_width=True)
        st.markdown("**💾 Space Saved**")
        st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], color="#10b981", use_container_width=True)

    q_cols = [c for c in ["VMAF","PSNR (dB)"] if df[c].notna().any()]
    if q_cols:
        st.markdown("---")
        st.markdown("**🎯 Quality Comparison**")
        qdf = df.set_index("Codec")[q_cols].dropna(how="all")
        st.bar_chart(qdf, use_container_width=True)
        st.caption("Higher = better quality. ✨ indicates AI enhancements applied.")

    if len(results) > 1:
        st.markdown("---")
        st.markdown("**🔍 Smart Insights**")
        
        enhanced_results = [r for r in results if r.get("enhancements")]
        if enhanced_results:
            best_enhanced = max(enhanced_results, key=lambda r: (r["vmaf"] or 0) + (r["psnr"] or 0)/5)
            st.markdown(f"""
            <div class="insight-note ai">
                <b>✨ Enhancement Winner:</b> <span style="font-weight:600">{best_enhanced['codec']}</span> 
                with {len([v for v in best_enhanced['enhancements'].values() if v])} enhancements 
                achieved VMAF {best_enhanced['vmaf']:.1f} at {best_enhanced['size_mb']:.2f} MB.
            </div>""", unsafe_allow_html=True)
        
        eff_candidates = [r for r in results if r["vmaf"] and r["size_mb"] > 0]
        if eff_candidates:
            best_eff = max(eff_candidates, key=lambda r: r["vmaf"] / r["size_mb"])
            enh_tag = " ✨" if best_eff.get("enhancements") else ""
            st.markdown(
                f'<div class="insight-note"><b>Efficiency Pick:</b> <span style="font-weight:600">{best_eff["codec"]}{enh_tag}</span> delivers best quality-per-MB (VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB).</div>',
                unsafe_allow_html=True
            )

# Downloads
with tab_dl:
    st.markdown("### ⬇️ Download Processed Files")
    st.caption("Files stored temporarily. Download immediately.")
    
    for r in results:
        cs = r["codec"].split()[0]
        enh_tag = "✨" if r.get("enhancements") else ""
        col_dl, col_info = st.columns([1, 3])
        
        with col_dl:
            try:
                with open(r["path"], "rb") as f:
                    st.download_button(
                        label=f"⬇ {cs}{enh_tag} CRF{r['crf']}",
                        data=f,
                        file_name=f"videoforge_{cs.lower()}_enh_crf{r['crf']}.mp4",
                        mime="video/mp4", use_container_width=True,
                        key=f"dl_{cs}_{r['crf']}_{id(r)}",
                    )
            except FileNotFoundError:
                st.caption("⚠️ Temp file expired — re-process to download")
        
        with col_info:
            metrics = [f"{r['size_mb']:.2f} MB", f"{r['bitrate']} kbps", f"{r.get('out_res','N/A')}"]
            if r.get("vmaf"): metrics.append(f"VMAF {r['vmaf']:.1f}")
            st.caption(" · ".join(metrics))
            if r.get("enhancements"):
                enh_list = [k.capitalize() for k,v in r["enhancements"].items() if v]
                st.caption(f"✨ Enhancements: {', '.join(enh_list)}")
            if r.get("acodec"):
                st.caption(f"🔊 {format_audio_codec(r['acodec'])} · {format_channels(r['channels'])}")

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;font-size:0.8rem;padding:16px 0">VideoForge AI Pro · FFmpeg + Streamlit · Light Theme · ✨ Vnova-style enhancements</div>', unsafe_allow_html=True)
