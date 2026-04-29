"""
VideoForge Web – Encoder + VMAF Quality Analytics + Audio Metrics
Run:  streamlit run streamlit_app.py

Deploy:
  Streamlit Community Cloud → add packages.txt containing just:  ffmpeg
  HF Spaces (Streamlit SDK) → same packages.txt trick
"""

import os, json, subprocess, tempfile, time, re
import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VideoForge · Encoder & Quality Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Light theme CSS (refined) ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.stApp { background-color: #f8fafc; }

/* Header */
.vf-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #3b82f6 100%);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
}
.vf-header h1 { 
    color: white; 
    font-size: 1.8rem; 
    font-weight: 600; 
    margin: 0; 
    letter-spacing: -0.02em; 
    display: flex;
    align-items: center;
    gap: 10px;
}
.vf-header p  { 
    color: #dbeafe; 
    margin: 6px 0 0; 
    font-size: 0.92rem; 
    opacity: 0.95;
}
.vf-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; 
    padding: 3px 12px;
    font-size: 0.72rem; 
    color: #e0f2fe;
    margin: 4px 4px 0 0;
    font-weight: 500;
}
.vf-badge::before { content: "•"; margin-right: 4px; opacity: 0.7; }

/* Mode toggle */
.mode-toggle {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 16px 0 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.mode-toggle .toggle-label {
    font-weight: 600;
    color: #334155;
    font-size: 0.9rem;
}
.mode-toggle .toggle-desc {
    color: #64748b;
    font-size: 0.82rem;
    margin-left: auto;
}
.mode-active {
    background: #dbeafe;
    border-color: #93c5fd;
    color: #1e40af;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Section labels */
.vf-label {
    font-size: 0.7rem; 
    font-weight: 700;
    letter-spacing: 0.14em; 
    text-transform: uppercase;
    color: #64748b; 
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vf-label::before {
    content: "";
    width: 20px;
    height: 2px;
    background: #3b82f6;
    border-radius: 2px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white; 
    border: 1px solid #e2e8f0;
    border-radius: 12px; 
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6); 
    color: white; 
    border: none;
    border-radius: 10px; 
    padding: 10px 24px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500; 
    font-size: 0.9rem;
    transition: all 0.2s;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
}
.stButton > button:hover { 
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}
.stButton > button:disabled {
    background: #cbd5e1;
    box-shadow: none;
    cursor: not-allowed;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    border-radius: 6px;
}

/* Comparison table */
.cmp-table { width:100%; border-collapse:collapse; font-size:0.85rem; margin-top:8px; background: white; border-radius: 10px; overflow: hidden; }
.cmp-table th {
    background: #f1f5f9; color:#475569; font-weight:600;
    padding:12px 14px; text-align:left;
    border-bottom:2px solid #e2e8f0;
    white-space: nowrap;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.cmp-table td {
    padding:11px 14px; border-bottom:1px solid #f1f5f9;
    color:#1e293b; font-family:'IBM Plex Mono', monospace;
    white-space: nowrap;
    font-size: 0.82rem;
}
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr:hover td { background:#f8fafc; }
.best-val { color:#166534; font-weight:600; }
.w-badge { 
    background:#dcfce7; color:#166534; 
    border-radius:4px; padding:2px 8px; 
    font-size:0.68rem; font-weight:700; margin-left:6px;
    text-transform: uppercase;
}

/* Codec chips */
.chip-avc  { background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }
.chip-hevc { background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }
.chip-av1  { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; border-radius:6px; padding:2px 10px; font-size:0.76rem; font-weight:600; }

/* VMAF colour */
.q-exc { color:#166534; font-weight:600; }
.q-gd  { color:#1e40af; font-weight:600; }
.q-ok  { color:#b45309; font-weight:600; }
.q-bad { color:#b91c1c; font-weight:600; }

/* Source ref bar */
.src-bar {
    background:#eff6ff; border-radius:10px;
    padding:12px 18px; font-size:0.84rem;
    color:#1e40af; margin-top:14px;
    border-left:4px solid #3b82f6;
    display: flex;
    flex-wrap: wrap;
    gap: 12px 20px;
}
.src-bar b { color: #1e293b; }

/* Insight card */
.insight-note {
    background:#fffbeb; border:1px solid #fcd34d;
    border-radius:10px; padding:14px 18px;
    font-size:0.86rem; color:#854d0e; margin-top:14px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.insight-note::before { content: "💡"; font-size: 1.1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#f1f5f9;
    border-radius:10px; padding:4px;
    margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] { 
    border-radius:8px; padding:8px 20px; 
    font-size:0.88rem; font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] { 
    background:white !important; 
    box-shadow:0 2px 8px rgba(0,0,0,0.1); 
    color: #1e293b !important;
    font-weight: 600;
}

label { font-weight:500; color:#334155; font-size:0.88rem; }
.stAlert { border-radius:10px; border-width: 1px; }

/* Audio metrics */
.audio-metric {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f8fafc;
    border-radius: 8px;
    font-size: 0.82rem;
    border: 1px solid #e2e8f0;
}
.audio-metric .icon {
    width: 24px;
    height: 24px;
    background: #3b82f6;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Toggle switch styling */
.stToggle > label {
    font-weight: 600;
    color: #1e293b;
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
    except Exception:
        return False


def vmaf_ok() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-filters"], stderr=subprocess.STDOUT,
                                      text=True, timeout=10)
        return "libvmaf" in out
    except Exception:
        return False


def probe(path: str) -> dict:
    """Extract comprehensive video AND audio metadata."""
    r = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "vcodec": "unknown", "vbitrate_kbps": 0,
        "acodec": "unknown", "abitrate_kbps": 0,
        "sample_rate": 0, "channels": 0, "audio_duration": 0.0,
        "has_audio": False
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            text=True, stderr=subprocess.DEVNULL)
        d = json.loads(out)
        fmt = d.get("format", {})
        r["duration"] = float(fmt.get("duration", 0))
        r["vbitrate_kbps"] = int(fmt.get("bit_rate", 0)) // 1000
        
        for s in d.get("streams", []):
            if s.get("codec_type") == "video":
                r["width"] = s.get("width", 0)
                r["height"] = s.get("height", 0)
                r["vcodec"] = s.get("codec_name", "unknown")
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
    except Exception:
        pass
    return r


def encode(input_path, output_path, codec, crf, progress_cb=None, duration=0.0):
    """Returns (ok, msg, fflog, encode_seconds)."""
    cmap = {
        "AVC (H.264)":  ("libx264",    ["-preset", "fast"]),
        "HEVC (H.265)": ("libx265",    ["-preset", "fast"]),
        "AV1":          ("libaom-av1", ["-cpu-used","8","-tile-columns","2",
                                        "-threads","4","-usage","realtime"]),
    }
    lib, extra = cmap.get(codec, ("libx264",["-preset","fast"]))
    cmd = (["ffmpeg","-y","-i",input_path,"-c:v",lib,"-crf",str(crf)]
           + extra + ["-c:a","copy","-movflags","+faststart",output_path])
    lines = []; t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, text=True, bufsize=1)
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
        hints = {-6:"OOM (SIGABRT) — AV1 needs 1-4 GB RAM. Use H.264 on free tiers.",
                 -9:"OOM (SIGKILL) — use H.264 or a smaller file.",
                 -11:"Segfault — corrupted input or codec bug.",
                 1:"FFmpeg error — see log."}
        return False, hints.get(proc.returncode, f"FFmpeg exit {proc.returncode}"), log, elapsed
    except FileNotFoundError:
        return False, "FFmpeg not found. Add to PATH.", "", 0.0
    except Exception as e:
        return False, str(e), "\n".join(lines), time.time()-t0


def quality_metrics(ref: str, dist: str, do_vmaf: bool) -> dict:
    """Compute PSNR, SSIM and optionally VMAF via FFmpeg."""
    res = {"psnr": None, "ssim": None, "vmaf": None}

    # ── PSNR + SSIM ──
    try:
        cmd = ["ffmpeg","-y","-i",dist,"-i",ref,
               "-filter_complex","[0:v][1:v]psnr[po];[0:v][1:v]ssim[so]",
               "-map","[po]","-map","[so]","-f","null","-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                      text=True, timeout=180)
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

    # ── VMAF ──
    if do_vmaf:
        try:
            vf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            vf.close()
            cmd = ["ffmpeg","-y","-i",dist,"-i",ref,
                   "-filter_complex",
                   f"[0:v][1:v]libvmaf=log_fmt=json:log_path={vf.name}",
                   "-f","null","-"]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            with open(vf.name) as f:
                vdata = json.load(f)
            score = (vdata.get("pooled_metrics",{}).get("vmaf",{}).get("mean")
                     or vdata.get("VMAF score")
                     or vdata.get("aggregate",{}).get("VMAF_score"))
            if score is not None:
                res["vmaf"] = round(float(score), 2)
            os.unlink(vf.name)
        except:
            pass
    return res


def vmaf_display(v):
    if v is None: return "—", ""
    if v >= 93:   return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80:   return f"{v:.1f} · Good",      "q-gd"
    if v >= 60:   return f"{v:.1f} · Fair",       "q-ok"
    return f"{v:.1f} · Poor", "q-bad"


def psnr_display(v):
    if v is None: return "—"
    tag = "Excellent" if v>=50 else "Good" if v>=40 else "Acceptable" if v>=30 else "Poor"
    return f"{v:.2f} dB · {tag}"


def format_audio_codec(codec: str) -> str:
    """Human-readable audio codec names."""
    mapping = {
        "aac": "AAC", "mp3": "MP3", "opus": "Opus",
        "vorbis": "Vorbis", "ac3": "AC-3", "eac3": "E-AC-3",
        "flac": "FLAC", "pcm_s16le": "PCM 16-bit", "alac": "ALAC"
    }
    return mapping.get(codec, codec.upper())


def format_sample_rate(sr: int) -> str:
    """Format sample rate with unit."""
    if sr >= 1000:
        return f"{sr//1000} kHz"
    return f"{sr} Hz"


def format_channels(ch: int) -> str:
    """Human-readable channel count."""
    mapping = {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}
    return mapping.get(ch, f"{ch} ch")


# ══════════════════════════════════════════════════════════════════════════════
#  Session State
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "results": [], "inp": None, "meta": None, 
    "sz": 0.0, "name": "", "enable_encoding": True
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  Main Page
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="vf-header">
  <h1>🎬 VideoForge <span style="font-size:0.9em;opacity:0.9;font-weight:400">Pro</span></h1>
  <p>Professional video encoding with VMAF quality analytics & audio metrics</p>
  <span class="vf-badge">H.264</span><span class="vf-badge">HEVC</span>
  <span class="vf-badge">AV1</span><span class="vf-badge">VMAF</span>
  <span class="vf-badge">PSNR</span><span class="vf-badge">SSIM</span>
  <span class="vf-badge">🔊 Audio</span>
</div>""", unsafe_allow_html=True)

# ── FFmpeg Check ─────────────────────────────────────────────────────────────
if not ffmpeg_ok():
    st.error("🔧 FFmpeg not found. Add `ffmpeg` to packages.txt and redeploy.")
    st.stop()

HAS_VMAF = vmaf_ok()

# ── Encoding Mode Toggle (NEW FEATURE) ───────────────────────────────────────
st.markdown(f"""
<div class="mode-toggle">
  <span class="toggle-label">🎛️ Operation Mode:</span>
  <span class="mode-active">{'⚙️ Encoder' if st.session_state.enable_encoding else '🎬 Test Player'}</span>
  <span class="toggle-desc">
    {'Full encoding workflow with quality comparison' if st.session_state.enable_encoding 
     else 'Playback & analytics only — no encoding performed'}
  </span>
</div>""", unsafe_allow_html=True)

enable_encoding = st.toggle(
    "⚙️ Enable Encoding Mode", 
    value=st.session_state.enable_encoding,
    help="Toggle between full encoder mode and test player mode. In test player mode, upload a video to analyze and playback without encoding."
)
st.session_state.enable_encoding = enable_encoding

# ── File Upload ──────────────────────────────────────────────────────────────
st.markdown('<div class="vf-label">📁 Source Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a video file or click to browse", 
    type=["avi","mp4","mkv","mov","webm","flv","ts","m4v","mxf","prores"],
    label_visibility="collapsed"
)

if not uploaded:
    st.info("👆 Upload a video file to begin analysis" + (" and encoding" if enable_encoding else ""))
    st.stop()

# Handle new file upload
suf = os.path.splitext(uploaded.name)[-1].lower()
if st.session_state.name != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded.read())
        st.session_state.inp = tmp.name
    st.session_state.meta = probe(st.session_state.inp)
    st.session_state.sz = os.path.getsize(st.session_state.inp) / (1024*1024)
    st.session_state.name = uploaded.name
    if enable_encoding:
        st.session_state.results = []  # Clear results when switching to encoding mode with new file

meta = st.session_state.meta
sz_mb = st.session_state.sz
inp = st.session_state.inp

# ── Source Preview & Metadata ────────────────────────────────────────────────
col_v, col_m = st.columns([3, 2], gap="large")

with col_v:
    st.markdown("### ▶️ Preview")
    st.video(inp)

with col_m:
    st.markdown("**📊 Source Media Info**")
    
    # Video metrics
    st.markdown('<div style="margin:14px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🎥 Video</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    r1.metric("Duration", f"{meta['duration']:.1f}s")
    r2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    r1.metric("Frame Rate", f"{meta['fps']} fps")
    r2.metric("Codec", meta["vcodec"].upper())
    r1.metric("Bitrate", f"{meta['vbitrate_kbps']} kbps")
    r2.metric("File Size", f"{sz_mb:.2f} MB")
    
    # Audio metrics (NEW)
    if meta["has_audio"]:
        st.markdown('<div style="margin:18px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🔊 Audio</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        a1.metric("Codec", format_audio_codec(meta["acodec"]))
        a2.metric("Channels", format_channels(meta["channels"]))
        a1.metric("Sample Rate", format_sample_rate(meta["sample_rate"]))
        a2.metric("Bitrate", f"{meta['abitrate_kbps']} kbps" if meta["abitrate_kbps"] > 0 else "Variable")
        
        # Audio sync indicator
        if meta["audio_duration"] > 0 and meta["duration"] > 0:
            sync_diff = abs(meta["audio_duration"] - meta["duration"])
            sync_status = "✓ Synced" if sync_diff < 0.1 else f"⚠ {sync_diff:.2f}s diff"
            st.markdown(f'<div class="audio-metric"><span class="icon">🔗</span> A/V Sync: <b>{sync_status}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin:18px 0 8px;font-weight:600;color:#94a3b8;font-size:0.85rem">🔇 No audio track detected</div>', unsafe_allow_html=True)

st.divider()

# ── Encoder Settings (only shown when encoding enabled) ─────────────────────
if enable_encoding:
    st.markdown('<div class="vf-label">⚙️ Encoder Settings</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([2, 2, 1, 2])
    
    with s1:
        codec = st.selectbox(
            "Video Codec", 
            ["AVC (H.264)", "HEVC (H.265)", "AV1"],
            help="H.264 = fastest compatibility · HEVC = ~40% smaller · AV1 = best compression, higher RAM"
        )
    with s2:
        crf = st.slider(
            "CRF Quality", 0, 51, 23,
            help="Lower = better quality · 0=lossless · 18=visually lossless · 23=balanced · 28+=compact"
        )
    with s3:
        do_vmaf = st.checkbox("VMAF", value=HAS_VMAF, disabled=not HAS_VMAF,
                             help="Perceptual quality score 0–100 (requires libvmaf)")
        do_psnr = st.checkbox("PSNR/SSIM", value=True, help="Traditional objective metrics")
    with s4:
        # CRF quality indicator
        if crf < 19:
            ql, qc = "🟢 High Quality", "#166534"
        elif crf < 29:
            ql, qc = "🟡 Balanced", "#b45309"
        elif crf < 40:
            ql, qc = "🟠 Compact", "#ea580c"
        else:
            ql, qc = "🔴 Low Quality", "#b91c1c"
        st.markdown(f"""
        <div style="text-align:center;padding:8px 0">
            <div style="font-weight:600;color:{qc}">{ql}</div>
            <div style="font-size:0.75rem;color:#64748b">CRF {crf}</div>
        </div>""", unsafe_allow_html=True)

    # AV1 warning
    if codec == "AV1":
        st.warning("⚠️ **AV1 Encoding**: Requires 1–4 GB RAM. May crash on free cloud tiers (exit -6). Works reliably locally. Use H.264/HEVC for cloud deployment.")

    # ── Encode Controls ─────────────────────────────────────────────────────
    st.markdown('<div class="vf-label" style="margin-top:10px">🚀 Run Encode</div>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.4, 1, 5])
    go = b1.button("⚙ Encode", type="primary", use_container_width=True)
    clear = b2.button("🗑 Clear", use_container_width=True)
    
    if clear:
        st.session_state.results = []
        st.rerun()

    if go:
        out_path = inp.replace(suf, f"_{codec.split()[0].lower()}_crf{crf}.mp4")
        bar = st.progress(0.0, text=f"Initializing {codec}…")

        with st.spinner(f"Encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(
                inp, out_path, codec, crf,
                progress_cb=lambda p: bar.progress(p, text=f"Encoding {codec}… {p*100:.0f}%"),
                duration=meta["duration"],
            )

        if not ok:
            bar.empty()
            st.error(f"❌ {msg}")
            if fflog:
                with st.expander("📋 FFmpeg Log"):
                    st.code(fflog, language="bash")
        else:
            bar.progress(1.0, text="✅ Encode complete — analyzing quality…")
            out_meta = probe(out_path)
            out_sz = os.path.getsize(out_path) / (1024*1024)
            saved_pct = (1 - out_sz/sz_mb)*100 if sz_mb else 0

            qual = {"psnr":None,"ssim":None,"vmaf":None}
            if do_psnr or (do_vmaf and HAS_VMAF):
                with st.spinner("🔍 Computing quality metrics (PSNR/SSIM/VMAF)…"):
                    qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

            st.session_state.results.append({
                "codec": codec, "crf": crf,
                "size_mb": out_sz, "bitrate": out_meta["vbitrate_kbps"],
                "enc_time": enc_t, "saved": saved_pct,
                "cr": sz_mb/out_sz if out_sz else 0,
                "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"],
                "path": out_path,
                # Audio metrics preserved (copied from source since we use -c:a copy)
                "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"],
                "sample_rate": meta["sample_rate"], "channels": meta["channels"]
            })
            bar.empty()
            
            # Success message with key metrics
            q_parts = []
            if qual["vmaf"]: q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"]: q_parts.append(f"PSNR {qual['psnr']:.2f}dB")
            q_str = " · ".join(q_parts) if q_parts else "Quality analysis complete"
            st.success(f"✅ {codec} CRF {crf} · {out_sz:.2f} MB · saved {saved_pct:.1f}% · {enc_t:.1f}s · {q_str}")

# ── Test Player Mode (when encoding disabled) ────────────────────────────────
else:
    st.markdown('<div class="vf-label">🎬 Test Player Mode</div>', unsafe_allow_html=True)
    st.info("🎧 **Playback & Analytics Mode**: Use the video player above to preview your source. All audio/video metrics are displayed in the sidebar. No encoding is performed in this mode.")
    
    # Quick audio visualization hint
    if meta["has_audio"]:
        st.markdown(f"""
        <div style="background:#f1f5f9;border-radius:10px;padding:14px;margin:12px 0">
            <div style="font-weight:600;margin-bottom:8px;color:#334155">🔊 Audio Stream</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">
                    {format_audio_codec(meta['acodec'])}
                </span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">
                    {format_channels(meta['channels'])}
                </span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">
                    {format_sample_rate(meta['sample_rate'])}
                </span>
                <span style="background:white;padding:4px 10px;border-radius:6px;font-size:0.8rem;border:1px solid #e2e8f0">
                    {meta['abitrate_kbps']} kbps
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
    
    st.caption("💡 Tip: Enable encoding mode above to compare different codecs and quality settings.")

# ══════════════════════════════════════════════════════════════════════════════
#  Results Dashboard (only when encoding enabled AND results exist)
# ══════════════════════════════════════════════════════════════════════════════
results = st.session_state.results
if not results or not enable_encoding:
    if enable_encoding:
        st.caption("📊 Results and analytics will appear here after encoding completes.")
    st.stop()

st.divider()
st.markdown('<div class="vf-label">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

tab_tbl, tab_chart, tab_dl = st.tabs(["📋 Comparison", "📊 Charts", "⬇ Downloads"])

# ── Comparison Table ─────────────────────────────────────────────────────────
with tab_tbl:
    # Calculate best values for highlighting
    best_sz = min(r["size_mb"] for r in results)
    best_cr = max(r["cr"] for r in results)
    best_spd = min(r["enc_time"] for r in results)
    best_vm = max((r["vmaf"] or 0) for r in results) if any(r["vmaf"] for r in results) else None
    best_pn = max((r["psnr"] or 0) for r in results) if any(r["psnr"] for r in results) else None
    best_ss = max((r["ssim"] or 0) for r in results) if any(r["ssim"] for r in results) else None

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

        vmaf_txt, vmaf_cls = vmaf_display(r["vmaf"])
        vmaf_cell = f'<span class="{vmaf_cls}">{vmaf_txt}</span>'
        if r["vmaf"] and best_vm and abs(r["vmaf"] - best_vm) < 0.01 and len(results)>1:
            vmaf_cell += ' <span class="w-badge">Best</span>'

        # Audio info for encoded files (copied from source)
        audio_info = f"{format_audio_codec(r['acodec'])} · {format_channels(r['channels'])}" if r.get('acodec') else "—"

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
          <td style="font-size:0.78rem;color:#64748b">{audio_info}</td>
        </tr>"""

    st.markdown(f"""
    <table class="cmp-table">
      <thead><tr>
        <th>Codec</th><th>CRF</th><th>Size</th><th>Bitrate</th>
        <th>Ratio</th><th>Saved</th><th>Time</th>
        <th>VMAF ↑</th><th>PSNR ↑</th><th>SSIM ↑</th><th>Audio</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="src-bar">
      <b>Source:</b> {meta['vcodec'].upper()} · {sz_mb:.2f} MB · {meta['vbitrate_kbps']} kbps ·
      {meta['width']}×{meta['height']} @ {meta['fps']} fps
      {f" · 🔊 {format_audio_codec(meta['acodec'])} {format_channels(meta['channels'])}" if meta['has_audio'] else ""}
    </div>
    """, unsafe_allow_html=True)

    st.caption("📏 VMAF: 93+ Excellent · 80-93 Good · 60-80 Fair · <60 Poor | PSNR: 40+ dB Good | SSIM: Closer to 1.0 = better | 🏆 Best = winner in column")

# ── Charts & Insights ────────────────────────────────────────────────────────
with tab_chart:
    df = pd.DataFrame([{
        "Codec": r["codec"],
        "File Size (MB)": round(r["size_mb"], 3),
        "Bitrate (kbps)": r["bitrate"],
        "Encode Time (s)": round(r["enc_time"], 2),
        "Space Saved (%)": round(r["saved"], 1),
        "Comp. Ratio": round(r["cr"], 2),
        "VMAF": r["vmaf"],
        "PSNR (dB)": round(r["psnr"], 2) if r["psnr"] else None,
        "SSIM": round(r["ssim"], 5) if r["ssim"] else None,
    } for r in results])

    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown("**📦 File Size Comparison**")
        size_df = pd.DataFrame(
            [{"Codec":"🎬 Original","File Size (MB)":round(sz_mb,3)}]
            + [{"Codec":r["codec"],"File Size (MB)":round(r["size_mb"],3)} for r in results]
        ).set_index("Codec")
        st.bar_chart(size_df, color="#3b82f6", use_container_width=True)

        st.markdown("**⏱️ Encode Time**")
        st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], color="#f97316", use_container_width=True)

    with c2:
        st.markdown("**📡 Bitrate Comparison**")
        brate_df = pd.DataFrame(
            [{"Codec":"🎬 Original","Bitrate (kbps)":meta['vbitrate_kbps']}]
            + [{"Codec":r["codec"],"Bitrate (kbps)":r["bitrate"]} for r in results]
        ).set_index("Codec")
        st.bar_chart(brate_df, color="#8b5cf6", use_container_width=True)

        st.markdown("**💾 Space Saved vs Original**")
        st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], color="#10b981", use_container_width=True)

    # Quality metrics chart
    q_cols = [c for c in ["VMAF","PSNR (dB)","SSIM"] if df[c].notna().any()]
    if q_cols:
        st.markdown("---")
        st.markdown("**🎯 Quality Metrics**")
        qdf = df.set_index("Codec")[q_cols].dropna(how="all")
        st.bar_chart(qdf, use_container_width=True)
        st.caption("⚠️ VMAF/PSNR/SSIM use different scales. Higher values = better quality in all cases.")

    # Smart insights
    if len(results) > 1:
        st.markdown("---")
        st.markdown("**🔍 Smart Insights**")
        
        smallest = min(results, key=lambda r: r["size_mb"])
        fastest = min(results, key=lambda r: r["enc_time"])
        most_saved = max(results, key=lambda r: r["saved"])
        best_qual = max(results, key=lambda r: (r["vmaf"] or 0) + (r["psnr"] or 0)/5 + (r["ssim"] or 0)*20)

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("🗜️ Smallest", smallest["codec"].split()[0], f"{smallest['size_mb']:.2f} MB")
        i2.metric("⚡ Fastest", fastest["codec"].split()[0], f"{fastest['enc_time']:.1f}s")
        i3.metric("💾 Most Saved", most_saved["codec"].split()[0], f"{most_saved['saved']:.1f}%")
        
        bq_val = ""
        if best_qual["vmaf"]: bq_val = f"VMAF {best_qual['vmaf']:.1f}"
        elif best_qual["psnr"]: bq_val = f"PSNR {best_qual['psnr']:.1f}dB"
        elif best_qual["ssim"]: bq_val = f"SSIM {best_qual['ssim']:.4f}"
        i4.metric("🎨 Best Quality", best_qual["codec"].split()[0], bq_val or "—")

        # Efficiency recommendation
        eff_candidates = [r for r in results if r["vmaf"] and r["size_mb"] > 0]
        if eff_candidates:
            best_eff = max(eff_candidates, key=lambda r: r["vmaf"] / r["size_mb"])
            st.markdown(
                f'<div class="insight-note"><b>Efficiency Pick:</b> <span style="font-weight:600">{best_eff["codec"]}</span> delivers the best quality-per-MB (VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB). Ideal for streaming or storage-constrained scenarios.</div>',
                unsafe_allow_html=True
            )

# ── Downloads ────────────────────────────────────────────────────────────────
with tab_dl:
    st.markdown("### ⬇️ Download Encoded Files")
    st.caption("Files are stored temporarily on the server. Download immediately after encoding.")
    
    for r in results:
        cs = r["codec"].split()[0]
        col_dl, col_info = st.columns([1, 3])
        
        with col_dl:
            try:
                with open(r["path"], "rb") as f:
                    st.download_button(
                        label=f"⬇ {cs} CRF{r['crf']}",
                        data=f,
                        file_name=f"videoforge_{cs.lower()}_crf{r['crf']}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_{cs}_{r['crf']}_{id(r)}",
                    )
            except FileNotFoundError:
                st.caption("⚠️ Temp file expired — re-encode to download")
        
        with col_info:
            metrics = [f"{r['size_mb']:.2f} MB", f"{r['bitrate']} kbps", f"saved {r['saved']:.1f}%"]
            if r.get("vmaf"): metrics.append(f"VMAF {r['vmaf']:.1f}")
            if r.get("psnr"): metrics.append(f"PSNR {r['psnr']:.2f}dB")
            st.caption(" · ".join(metrics))
            # Audio info
            if r.get("acodec"):
                st.caption(f"🔊 {format_audio_codec(r['acodec'])} · {format_channels(r['channels'])} · {format_sample_rate(r['sample_rate'])}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;color:#64748b;font-size:0.8rem;padding:16px 0">VideoForge Pro · Built with FFmpeg + Streamlit · Light Theme Optimized</div>', unsafe_allow_html=True)
