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
    page_title="AI Encoder",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Play Button SVG Icon + CSS ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f0f4f8;
    color: #1a202c;
}
.stApp { background-color: #f0f4f8; }

/* ── Header ── */
.vf-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #1d4ed8 100%);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(15, 23, 42, 0.25);
    position: relative;
    overflow: hidden;
}
.vf-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.vf-header::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 30%;
    width: 140px; height: 140px;
    border-radius: 50%;
    background: rgba(59,130,246,0.12);
}
.vf-header-inner {
    display: flex; align-items: center; gap: 18px; position: relative; z-index: 1;
}
.vf-play-icon {
    width: 56px; height: 56px;
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(255,255,255,0.25);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.vf-play-icon svg { filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
.vf-header h1 { 
    color: white; font-size: 1.9rem; font-weight: 700; margin: 0; 
    letter-spacing: -0.03em;
}
.vf-header h1 span { font-weight: 300; opacity: 0.75; font-size: 0.85em; }
.vf-header p { color: #bfdbfe; margin: 5px 0 0; font-size: 0.88rem; }
.vf-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; position: relative; z-index: 1; }
.vf-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: #e0f2fe; font-weight: 600; letter-spacing: 0.04em;
}
.vf-badge.ai { 
    background: linear-gradient(135deg, rgba(124,58,237,0.4), rgba(168,85,247,0.3)); 
    border-color: rgba(196,181,253,0.4); 
}

/* ── Mode Toggle ── */
.mode-toggle {
    background: white; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 14px 18px; margin: 0 0 24px;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.mode-toggle .toggle-label { font-weight: 600; color: #1e293b; font-size: 0.9rem; }
.mode-toggle .toggle-desc { color: #64748b; font-size: 0.82rem; margin-left: auto; }
.mode-active {
    background: #dbeafe; border: 1px solid #93c5fd; color: #1e40af;
    padding: 3px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 700;
}

/* ── Section Labels ── */
.vf-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em; 
    text-transform: uppercase; color: #64748b; margin: 22px 0 12px;
    display: flex; align-items: center; gap: 8px;
}
.vf-label::before {
    content: ""; width: 18px; height: 2.5px; background: #3b82f6; border-radius: 2px;
}
.vf-label.ai::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: white; border: 1px solid #e8edf2; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
[data-testid="stMetricValue"] { font-size: 1.1rem !important; font-weight: 600 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6); color: white; border: none;
    border-radius: 10px; padding: 10px 22px; font-weight: 600; font-size: 0.88rem;
    transition: all 0.2s; box-shadow: 0 2px 6px rgba(29, 78, 216, 0.25);
    font-family: 'DM Sans', sans-serif;
}
.stButton > button:hover { 
    transform: translateY(-1px); 
    box-shadow: 0 6px 16px rgba(29, 78, 216, 0.35); 
}
.stButton > button:disabled { 
    background: #cbd5e1 !important; 
    box-shadow: none !important; 
    cursor: not-allowed !important; 
    transform: none !important;
}

/* ── Progress ─ */
.stProgress > div > div {
    background: linear-gradient(90deg, #1d4ed8, #60a5fa); border-radius: 6px;
}

/* ── Comparison Table ── */
.cmp-table { 
    width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:8px; 
    background: white; border-radius: 12px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.cmp-table th {
    background: #f8fafc; color:#475569; font-weight:700; padding:12px 14px; text-align:left;
    border-bottom:2px solid #e8edf2; white-space: nowrap; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.cmp-table td {
    padding:11px 14px; border-bottom:1px solid #f1f5f9; color:#1e293b;
    font-family:'JetBrains Mono', monospace; white-space: nowrap; font-size: 0.8rem;
}
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr:hover td { background:#f8fafc; }
.best-val { color:#15803d; font-weight:700; }
.w-badge { 
    background:#dcfce7; color:#15803d; border-radius:4px; padding:2px 7px; 
    font-size:0.65rem; font-weight:700; margin-left:6px; text-transform: uppercase;
    font-family: 'DM Sans', sans-serif; letter-spacing: 0.05em;
}

/* ── Codec Chips ── */
.chip-avc  { background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-hevc { background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-av1  { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-enh { background:#faf5ff; color:#6d28d9; border:1px solid #ddd6fe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }

/* ── Quality Colors ── */
.q-exc { color:#15803d; font-weight:700; }
.q-gd  { color:#1d4ed8; font-weight:700; }
.q-ok  { color:#b45309; font-weight:700; }
.q-bad { color:#b91c1c; font-weight:700; }

/* ── Source Bar ── */
.src-bar {
    background:#eff6ff; border-radius:10px; padding:12px 18px; font-size:0.83rem;
    color:#1e40af; margin-top:12px; border-left:4px solid #3b82f6;
    display: flex; flex-wrap: wrap; gap: 10px 20px; font-family:'DM Sans',sans-serif;
}
.src-bar b { color: #1e293b; }

/* ── Insight Notes ── */
.insight-note {
    background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:0.85rem; color:#854d0e; margin-top:12px;
    display: flex; gap: 10px; align-items: flex-start;
}
.insight-note::before { content: "💡"; font-size: 1.1rem; flex-shrink:0; }
.insight-note.ai {
    background:#faf5ff; border-color:#c4b5fd; color:#5b21b6;
}
.insight-note.ai::before { content: "✨"; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:#e8edf2; border-radius:12px; padding:4px; margin-bottom: 18px;
}
.stTabs [data-baseweb="tab"] { 
    border-radius:8px; padding:8px 22px; font-size:0.87rem; font-weight:500; color:#64748b;
    font-family:'DM Sans',sans-serif;
}
.stTabs [aria-selected="true"] { 
    background:white !important; box-shadow:0 2px 8px rgba(0,0,0,0.1); 
    color: #1e293b !important; font-weight: 700;
}

/* ── Audio Metric ── */
.audio-metric {
    display: flex; align-items: center; gap: 8px; padding: 9px 14px;
    background: #f8fafc; border-radius: 9px; font-size: 0.82rem; 
    border: 1px solid #e2e8f0; margin-top: 6px;
}
.audio-metric .icon {
    width: 26px; height: 26px; background: #3b82f6; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.7rem; font-weight: 700;
}

/* ── General ── */
label { font-weight:500; color:#334155; font-size:0.87rem; }
.stAlert { border-radius:10px; border-width: 1px; }
.stDivider { margin: 20px 0; }

/* ── Container borders ── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 12px !important;
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
        out = subprocess.check_output(
            ["ffmpeg", "-filters"], stderr=subprocess.STDOUT, text=True, timeout=10
        )
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
        "has_audio": False, "color_space": "unknown", "bit_depth": 8
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        d = json.loads(out)
        fmt = d.get("format", {})
        r["duration"] = float(fmt.get("duration", 0) or 0)
        r["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000

        for s in d.get("streams", []):
            if s.get("codec_type") == "video" and r["width"] == 0:
                r["width"] = s.get("width", 0)
                r["height"] = s.get("height", 0)
                r["vcodec"] = s.get("codec_name", "unknown")
                r["color_space"] = s.get("color_space", "unknown")
                r["bit_depth"] = int(s.get("bits_per_raw_sample", 8) or 8)
                try:
                    n, dn = map(int, s.get("r_frame_rate", "0/1").split("/"))
                    r["fps"] = round(n / dn, 3) if dn else 0.0
                except Exception:
                    pass
            elif s.get("codec_type") == "audio" and not r["has_audio"]:
                r["has_audio"] = True
                r["acodec"] = s.get("codec_name", "unknown")
                r["abitrate_kbps"] = int(s.get("bit_rate", 0) or 0) // 1000
                r["sample_rate"] = int(s.get("sample_rate", 0) or 0)
                r["channels"] = int(s.get("channels", 0) or 0)
                r["audio_duration"] = float(s.get("duration", 0) or 0)
    except Exception:
        pass
    return r


def build_enhance_filters(settings: dict, src_meta: dict) -> list:
    """Build FFmpeg filter chain for AI enhancements."""
    filters = []

    # ── Denoise ─────────────────────────────────────────────────────
    if settings.get("denoise"):
        strength = float(settings.get("denoise_strength", 5))
        s2 = strength / 2
        filters.append(f"hqdn3d={strength}:{strength}:{s2}:{s2}")

    # ── Deblock (FIXED: alpha/beta must be 0.0-1.0) ────────────────
    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5))
        # Map slider 1-10 → FFmpeg range 0.0-1.0
        alpha = round(strength / 10.0, 2)   # 1→0.1, 5→0.5, 10→1.0
        beta = round(alpha * 0.5, 2)         # Beta typically half of alpha
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")

    # ── Sharpening ─────────────────────────────────────────────────
    if settings.get("sharpen"):
        amount = float(settings.get("sharpen_amount", 0.5))
        threshold = int(settings.get("sharpen_threshold", 5))
        filters.append(f"unsharp=5:5:{amount}:{threshold}")

    # ── Color Enhancement ──────────────────────────────────────────
    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15))
        contrast = float(settings.get("contrast", 1.0))
        sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")

    # ── HDR Conversion ─────────────────────────────────────────────
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.append("zscale=transfer=linear,format=gbrpf32le")
        filters.append(f"tonemap={settings.get('tonemap_algo', 'hable')}:desat=0.2")
        filters.append("zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le")

    # ── Upscaling ──────────────────────────────────────────────────
    if settings.get("upscale"):
        target_w = int(settings.get("upscale_width", src_meta["width"] * 2))
        target_h = int(settings.get("upscale_height", src_meta["height"] * 2))
        # Ensure even dimensions (required by most codecs)
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        target_h = target_h if target_h % 2 == 0 else target_h - 1
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(f"scale={target_w}:{target_h}:flags={algo}+accurate_rnd+full_chroma_int")

    # ── Frame Interpolation ────────────────────────────────────────
    if settings.get("frame_interp"):
        target_fps = int(settings.get("target_fps", 60))
        filters.append(
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        )

    return filters


def estimate_processing_time(src_meta: dict, settings: dict) -> str:
    """Estimate processing time impact based on enhancements."""
    base_factor = 1.0
    if settings.get("denoise"): base_factor *= 1.3
    if settings.get("sharpen"): base_factor *= 1.1
    if settings.get("upscale"):
        base_factor *= 2.5 if settings.get("upscale_algo") == "lanczos" else 1.8
    if settings.get("hdr_convert"): base_factor *= 1.6
    if settings.get("color_enhance"): base_factor *= 1.15
    if settings.get("deblock"): base_factor *= 1.4
    if settings.get("frame_interp"): base_factor *= 3.0

    duration_min = (src_meta.get("duration", 0) / 60) * base_factor
    if duration_min < 1:
        return f"~{max(1, int(duration_min * 60))}s"
    elif duration_min < 10:
        return f"~{duration_min:.1f} min"
    else:
        return f"~{duration_min:.0f} min"


def encode(input_path, output_path, codec, crf, enhance_settings: dict, src_meta: dict,
           progress_cb=None, duration=0.0):
    """Encode with optional enhancement filters."""
    cmap = {
        "AVC (H.264)":  ("libx264",    ["-preset", "fast"]),
        "HEVC (H.265)": ("libx265",    ["-preset", "fast"]),
        "AV1":          ("libaom-av1", ["-cpu-used", "8", "-tile-columns", "2",
                                        "-threads", "4", "-usage", "realtime"]),
    }
    lib, extra = cmap.get(codec, ("libx264", ["-preset", "fast"]))
    filters = build_enhance_filters(enhance_settings, src_meta)

    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", lib, "-crf", str(crf)] + extra
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:a", "copy", "-movflags", "+faststart", output_path]

    lines = []
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1
        )
        for line in proc.stderr:
            lines.append(line.rstrip())
            if progress_cb and "time=" in line and duration > 0:
                try:
                    ts = line.split("time=")[1].split(" ")[0]
                    parts = ts.split(":")
                    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                    progress_cb(min((h * 3600 + m * 60 + s) / duration, 0.99))
                except Exception:
                    pass
        proc.wait()
        elapsed = time.time() - t0
        log = "\n".join(lines[-50:])

        if proc.returncode == 0:
            return True, "Done!", log, elapsed

        hints = {
            -6:  "OOM — AI enhancements need extra RAM. Try disabling upscaling/frame interpolation.",
            -9:  "OOM (SIGKILL) — reduce enhancement complexity or file size.",
            -11: "Segfault — check filter compatibility with your FFmpeg build.",
            1:   "FFmpeg error — see log below for details.",
        }
        return False, hints.get(proc.returncode, f"FFmpeg exited with code {proc.returncode}"), log, elapsed

    except FileNotFoundError:
        return False, "FFmpeg not found. Add `ffmpeg` to packages.txt.", "", 0.0
    except Exception as e:
        return False, str(e), "\n".join(lines), time.time() - t0


def quality_metrics(ref: str, dist: str, do_vmaf: bool) -> dict:
    """Compute PSNR, SSIM and optionally VMAF via FFmpeg."""
    res = {"psnr": None, "ssim": None, "vmaf": None}

    # PSNR + SSIM
    try:
        cmd = [
            "ffmpeg", "-y", "-i", dist, "-i", ref,
            "-filter_complex", "[0:v][1:v]psnr[po];[0:v][1:v]ssim[so]",
            "-map", "[po]", "-map", "[so]", "-f", "null", "-"
        ]
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
    except Exception:
        pass

    # VMAF (optional)
    if do_vmaf:
        try:
            vf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            vf.close()
            cmd = [
                "ffmpeg", "-y", "-i", dist, "-i", ref,
                "-filter_complex",
                f"[0:v][1:v]libvmaf=log_fmt=json:log_path={vf.name}",
                "-f", "null", "-"
            ]
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
            with open(vf.name) as f:
                vdata = json.load(f)
            score = (
                vdata.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
                or vdata.get("VMAF score")
                or vdata.get("aggregate", {}).get("VMAF_score")
            )
            if score is not None:
                res["vmaf"] = round(float(score), 2)
            try:
                os.unlink(vf.name)
            except Exception:
                pass
        except Exception:
            pass

    return res


# ── Display helpers ────────────────────────────────────────────────────────────

def vmaf_display(v):
    if v is None:
        return "—", ""
    if v >= 93:
        return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80:
        return f"{v:.1f} · Good", "q-gd"
    if v >= 60:
        return f"{v:.1f} · Fair", "q-ok"
    return f"{v:.1f} · Poor", "q-bad"

def psnr_display(v):
    if v is None:
        return "—"
    tag = "Excellent" if v >= 50 else "Good" if v >= 40 else "Acceptable" if v >= 30 else "Poor"
    return f"{v:.2f} dB · {tag}"

def format_audio_codec(codec: str) -> str:
    mapping = {
        "aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis",
        "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC",
        "pcm_s16le": "PCM 16-bit", "alac": "ALAC"
    }
    return mapping.get(codec, codec.upper())

def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"

def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")


# ══════════════════════════════════════════════════════════════════════════════
#  Session State Init
# ══════════════════════════════════════════════════════════════════════════════

_default_enhance = {
    "denoise": False, "denoise_strength": 5,
    "sharpen": False, "sharpen_amount": 0.5, "sharpen_threshold": 5,
    "upscale": False, "upscale_algo": "lanczos",
    "upscale_width": 0, "upscale_height": 0,
    "hdr_convert": False, "tonemap_algo": "hable",
    "color_enhance": False, "vibrance": 0.15, "contrast": 1.0,
    "deblock": False, "deblock_strength": 5,
    "frame_interp": False, "target_fps": 60,
}

defaults = {
    "results": [], "inp": None, "meta": None, "sz": 0.0, "name": "",
    "enable_encoding": True, "enhance_settings": _default_enhance.copy(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="vf-header">
  <div class="vf-header-inner">
    <div class="vf-play-icon">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="32" height="32">
        <polygon points="28,20 28,80 82,50" fill="white"/>
      </svg>
    </div>
    <div>
      <h1>VideoForge <span>AI Pro</span></h1>
      <p>Professional encoding · AI enhancement · Vnova-style quality upscaling</p>
    </div>
  </div>
  <div class="vf-badges">
    <span class="vf-badge">H.264</span>
    <span class="vf-badge">HEVC</span>
    <span class="vf-badge">AV1</span>
    <span class="vf-badge">VMAF</span>
    <span class="vf-badge">PSNR/SSIM</span>
    <span class="vf-badge ai">✨ AI Enhance</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not ffmpeg_ok():
    st.error("🔧 **FFmpeg not found.** Add `ffmpeg` to `packages.txt` and redeploy.")
    st.stop()

HAS_VMAF = vmaf_ok()


# ══════════════════════════════════════════════════════════════════════════════
#  Mode Toggle
# ══════════════════════════════════════════════════════════════════════════════

mode_label = "⚙️ Encoder" if st.session_state.enable_encoding else "🎬 Test Player"
mode_desc = (
    "Full workflow with AI enhancement & encoding"
    if st.session_state.enable_encoding
    else "Playback & analytics only — no processing"
)
st.markdown(f"""
<div class="mode-toggle">
  <span class="toggle-label">🎛️ Mode:</span>
  <span class="mode-active">{mode_label}</span>
  <span class="toggle-desc">{mode_desc}</span>
</div>
""", unsafe_allow_html=True)

enable_encoding = st.toggle(
    "⚙️ Enable Encoding Mode",
    value=st.session_state.enable_encoding,
    help="Toggle between encoder mode (with AI enhancements) and test player mode (analytics only).",
)
st.session_state.enable_encoding = enable_encoding


# ══════════════════════════════════════════════════════════════════════════════
#  File Upload
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="vf-label">📁 Source Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a video or click to browse",
    type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"],
    label_visibility="collapsed",
)

if not uploaded:
    st.info("👆 Upload a video to begin" + (" analysis & enhancement" if enable_encoding else " analysis"))
    st.stop()

suf = os.path.splitext(uploaded.name)[-1].lower()

# Only re-probe if file changed
if st.session_state.name != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded.read())
        st.session_state.inp = tmp.name
    st.session_state.meta = probe(st.session_state.inp)
    st.session_state.sz = os.path.getsize(st.session_state.inp) / (1024 * 1024)
    st.session_state.name = uploaded.name
    # Reset upscale defaults to match new source
    m = st.session_state.meta
    st.session_state.enhance_settings["upscale_width"] = m["width"] * 2
    st.session_state.enhance_settings["upscale_height"] = m["height"] * 2
    if enable_encoding:
        st.session_state.results = []

meta = st.session_state.meta
sz_mb = st.session_state.sz
inp = st.session_state.inp


# ══════════════════════════════════════════════════════════════════════════════
#  Source Preview + Metadata
# ══════════════════════════════════════════════════════════════════════════════

col_v, col_m = st.columns([3, 2], gap="large")

with col_v:
    st.markdown("### ▶ Preview")
    st.video(inp)

with col_m:
    st.markdown("**📊 Source Media Info**")
    st.markdown(
        '<div style="margin:12px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🎥 Video Stream</div>',
        unsafe_allow_html=True
    )
    r1, r2 = st.columns(2)
    r1.metric("Duration", f"{meta['duration']:.1f}s")
    r2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    r1.metric("Frame Rate", f"{meta['fps']} fps")
    r2.metric("Codec", meta["vcodec"].upper())
    r1.metric("Bitrate", f"{meta['vbitrate_kbps']} kbps" if meta["vbitrate_kbps"] else "—")
    r2.metric("File Size", f"{sz_mb:.2f} MB")

    if meta["has_audio"]:
        st.markdown(
            '<div style="margin:16px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🔊 Audio Stream</div>',
            unsafe_allow_html=True
        )
        a1, a2 = st.columns(2)
        a1.metric("Codec", format_audio_codec(meta["acodec"]))
        a2.metric("Channels", format_channels(meta["channels"]))
        a1.metric("Sample Rate", format_sample_rate(meta["sample_rate"]))
        a2.metric(
            "Bitrate",
            f"{meta['abitrate_kbps']} kbps" if meta["abitrate_kbps"] > 0 else "Variable"
        )
        if meta["audio_duration"] > 0 and meta["duration"] > 0:
            sync_diff = abs(meta["audio_duration"] - meta["duration"])
            sync_status = "✓ Synced" if sync_diff < 0.1 else f"⚠ {sync_diff:.2f}s off"
            st.markdown(
                f'<div class="audio-metric"><span class="icon">🔗</span>'
                f' A/V Sync: <b style="margin-left:4px">{sync_status}</b></div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="margin:16px 0 4px;font-weight:500;color:#94a3b8;font-size:0.84rem">🔇 No audio track detected</div>',
            unsafe_allow_html=True
        )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  AI ENHANCEMENT PANEL  (Encoder mode only)
# ══════════════════════════════════════════════════════════════════════════════

if enable_encoding:
    st.markdown('<div class="vf-label ai">✨ AI Video Enhancement</div>', unsafe_allow_html=True)
    st.markdown(
        "Professional-grade enhancements. Choose a preset or customize individually:"
    )

    # ── DISTINCT PRESETS with Clear Use Cases ─────────────────────────────────
    st.markdown("### 🎯 Quick Presets")
    st.caption("One-click setups for common scenarios:")
    
    p1, p2, p3 = st.columns(3)
    
    with p1:
        st.markdown("#### 🎬 Standard Clean")
        st.caption("**Best for:** Noisy, grainy, or compressed footage")
        st.markdown("• Reduces noise & grain")
        st.markdown("• Removes blocking artifacts")
        st.markdown("• Mild sharpening")
        if st.button("Apply Standard Clean", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({
                "denoise": True, "denoise_strength": 5,
                "deblock": True, "deblock_strength": 5,
                "sharpen": True, "sharpen_amount": 0.3, "sharpen_threshold": 8,
                "upscale": False, "hdr_convert": False, "frame_interp": False,
                "color_enhance": False,
            })
            st.rerun()
    
    with p2:
        st.markdown("#### 🎨 Detail & Color Boost")
        st.caption("**Best for:** Dull, flat, or low-contrast footage")
        st.markdown("• Enhances sharpness")
        st.markdown("• Boosts color vibrance")
        st.markdown("• Increases contrast")
        if st.button("Apply Detail Boost", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({
                "sharpen": True, "sharpen_amount": 0.8, "sharpen_threshold": 3,
                "color_enhance": True, "vibrance": 0.25, "contrast": 1.15,
                "denoise": False, "deblock": False,
                "upscale": False, "hdr_convert": False, "frame_interp": False,
            })
            st.rerun()
    
    with p3:
        st.markdown("#### 🚀 AI Upscale 2×")
        st.caption("**Best for:** Low-resolution content (480p→1080p, 1080p→4K)")
        st.markdown("• 2× resolution upscale")
        st.markdown("• Pre-denoise for quality")
        st.markdown("• Post-sharpen details")
        if st.button("Apply AI Upscale 2×", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({
                "upscale": True,
                "upscale_width": meta["width"] * 2,
                "upscale_height": meta["height"] * 2,
                "upscale_algo": "lanczos",
                "denoise": True, "denoise_strength": 6,
                "sharpen": True, "sharpen_amount": 0.5, "sharpen_threshold": 5,
                "hdr_convert": False, "frame_interp": False,
                "deblock": False, "color_enhance": False,
            })
            st.rerun()

    st.divider()

    # Helper: read from session state (widgets write back on change)
    es = st.session_state.enhance_settings

    enh_col1, enh_col2 = st.columns(2)

    # ── Left Column ──────────────────────────────────────────────────────────
    with enh_col1:

        # Denoise
        with st.container(border=True):
            st.markdown("**🧹 Denoise**")
            es["denoise"] = st.checkbox(
                "Enable temporal noise reduction",
                value=es["denoise"],
                key="chk_denoise",
                help="Reduces grain and compression artifacts using 3D filtering",
            )
            if es["denoise"]:
                es["denoise_strength"] = st.slider(
                    "Strength", 1, 10, es["denoise_strength"],
                    key="sl_denoise_str",
                    help="Higher = more aggressive noise removal",
                )
            st.caption("Reduces grain, compression artifacts, and sensor noise.")

        # Sharpening
        with st.container(border=True):
            st.markdown("**🔍 Detail Enhancement**")
            es["sharpen"] = st.checkbox(
                "Enable adaptive sharpening",
                value=es["sharpen"],
                key="chk_sharpen",
            )
            if es["sharpen"]:
                cs1, cs2 = st.columns(2)
                with cs1:
                    es["sharpen_amount"] = st.slider(
                        "Amount", -1.5, 1.5, es["sharpen_amount"], 0.1,
                        key="sl_sharpen_amt",
                    )
                with cs2:
                    es["sharpen_threshold"] = st.slider(
                        "Threshold", 0, 50, es["sharpen_threshold"],
                        key="sl_sharpen_thr",
                    )
            st.caption("Enhances edge definition using adaptive unsharp masking.")

        # Upscaling
        with st.container(border=True):
            st.markdown("**🔬 Resolution Upscaling**")
            es["upscale"] = st.checkbox(
                "Enable high-quality upscaling",
                value=es["upscale"],
                key="chk_upscale",
            )
            if es["upscale"]:
                algo_opts = ["lanczos", "spline", "bicubic"]
                es["upscale_algo"] = st.selectbox(
                    "Algorithm",
                    algo_opts,
                    index=algo_opts.index(es["upscale_algo"]) if es["upscale_algo"] in algo_opts else 0,
                    key="sel_upscale_algo",
                )
                es["upscale_width"] = st.number_input(
                    "Target Width (px)",
                    min_value=meta["width"],
                    max_value=7680,
                    value=max(meta["width"], es.get("upscale_width") or meta["width"] * 2),
                    step=max(2, meta["width"] // 2),
                    key="ni_upscale_w",
                )
                es["upscale_height"] = st.number_input(
                    "Target Height (px)",
                    min_value=meta["height"],
                    max_value=4320,
                    value=max(meta["height"], es.get("upscale_height") or meta["height"] * 2),
                    step=max(2, meta["height"] // 2),
                    key="ni_upscale_h",
                )
            st.caption(
                "High-quality interpolation. For true AI super-resolution, "
                "integrate Real-ESRGAN externally."
            )

    # ── Right Column ─────────────────────────────────────────────────────────
    with enh_col2:

        # HDR Conversion
        is_hdr_source = meta.get("bit_depth", 8) >= 10
        with st.container(border=True):
            st.markdown("**🌈 HDR → SDR Tonemapping**")
            es["hdr_convert"] = st.checkbox(
                "Enable HDR conversion",
                value=es["hdr_convert"] and is_hdr_source,
                key="chk_hdr",
                disabled=not is_hdr_source,
                help="Requires 10-bit+ HDR10/HLG source",
            )
            if es["hdr_convert"] and is_hdr_source:
                tmap_opts = ["hable", "reinhard", "mobius", "linear"]
                es["tonemap_algo"] = st.selectbox(
                    "Tonemap Algorithm",
                    tmap_opts,
                    index=tmap_opts.index(es["tonemap_algo"]) if es["tonemap_algo"] in tmap_opts else 0,
                    key="sel_tonemap",
                )
            elif not is_hdr_source:
                st.caption(
                    "💡 Source is SDR (8-bit). "
                    "HDR conversion requires a 10-bit+ HDR10/HLG source."
                )
            st.caption("Converts HDR10/HLG to SDR with perceptual tonemapping.")

        # Color Enhancement
        with st.container(border=True):
            st.markdown("**🎨 Color Enhancement**")
            es["color_enhance"] = st.checkbox(
                "Enable color boost",
                value=es["color_enhance"],
                key="chk_color",
            )
            if es["color_enhance"]:
                cc1, cc2 = st.columns(2)
                with cc1:
                    es["vibrance"] = st.slider(
                        "Vibrance", -0.5, 0.5, es["vibrance"], 0.05, key="sl_vibrance"
                    )
                with cc2:
                    es["contrast"] = st.slider(
                        "Contrast", 0.5, 2.0, es["contrast"], 0.05, key="sl_contrast"
                    )
            st.caption("Enhances vibrancy while preserving natural skin tones.")

        # Artifact Reduction
        with st.container(border=True):
            st.markdown("**🧩 Artifact Reduction**")
            es["deblock"] = st.checkbox(
                "Enable deblocking filter",
                value=es["deblock"],
                key="chk_deblock",
            )
            if es["deblock"]:
                es["deblock_strength"] = st.slider(
                    "Strength", 1, 10, es["deblock_strength"], key="sl_deblock_str"
                )
            st.caption("Reduces macroblocking and ringing from heavy compression.")

    # ── Frame Interpolation (Advanced) ───────────────────────────────────────
    with st.expander("🎞️ Advanced: Frame Interpolation (Motion Smoothing)"):
        es["frame_interp"] = st.checkbox(
            "Enable frame interpolation",
            value=es["frame_interp"],
            key="chk_frame_interp",
        )
        if es["frame_interp"]:
            fps_opts = [30, 48, 60, 120]
            cur_fps = es.get("target_fps", 60)
            fps_idx = fps_opts.index(cur_fps) if cur_fps in fps_opts else 2
            es["target_fps"] = st.selectbox(
                "Target Frame Rate", fps_opts, index=fps_idx, key="sel_target_fps"
            )
            st.warning(
                "⚠️ Frame interpolation is CPU-intensive and may increase processing time 3–5×."
            )
        st.caption(
            "Creates intermediate frames for smoother playback (e.g., 24 fps → 60 fps)."
        )

    # ── Enhancement Summary ───────────────────────────────────────────────────
    _enh_keys = ["denoise", "sharpen", "upscale", "hdr_convert",
                 "color_enhance", "deblock", "frame_interp"]
    active_enhancements = sum(bool(es.get(k)) for k in _enh_keys)

    if active_enhancements > 0:
        est_time = estimate_processing_time(meta, es)
        st.info(
            f"⏱️ **Estimated processing time**: {est_time} "
            f"({active_enhancements} enhancement{'s' if active_enhancements > 1 else ''} active)"
        )
        enh_labels = {
            "denoise": "🧹 Denoise", "sharpen": "🔍 Sharpen",
            "upscale": "🔬 Upscale", "hdr_convert": "🌈 HDR",
            "color_enhance": "🎨 Color", "deblock": "🧩 Deblock",
            "frame_interp": "🎞️ Interp",
        }
        active_names = [enh_labels[k] for k in _enh_keys if es.get(k)]
        st.markdown(f"✨ **Active**: {' + '.join(active_names)}")

    st.divider()

    # ── Encoder Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="vf-label">⚙️ Encoder Settings</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([2, 2, 1, 2])

    with s1:
        codec = st.selectbox(
            "Video Codec", ["AVC (H.264)", "HEVC (H.265)", "AV1"],
            help="H.264 = fastest · HEVC = ~40% smaller · AV1 = best compression",
        )
    with s2:
        crf = st.slider(
            "CRF Quality", 0, 51, 23,
            help="Lower = better quality · 0 = lossless · 18 = visually lossless · 23 = balanced",
        )
    with s3:
        do_vmaf = st.checkbox("VMAF", value=HAS_VMAF, disabled=not HAS_VMAF)
        do_psnr = st.checkbox("PSNR/SSIM", value=True)
    with s4:
        if crf < 19:
            ql, qc = "🟢 High Quality", "#15803d"
        elif crf < 29:
            ql, qc = "🟡 Balanced", "#b45309"
        elif crf < 40:
            ql, qc = "🟠 Compact", "#ea580c"
        else:
            ql, qc = "🔴 Low Quality", "#b91c1c"
        st.markdown(
            f'<div style="text-align:center;padding:10px 0">'
            f'<div style="font-weight:700;color:{qc};font-size:1rem">{ql}</div>'
            f'<div style="font-size:0.74rem;color:#64748b;margin-top:2px">CRF {crf}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if codec == "AV1" and active_enhancements > 0:
        st.warning(
            "⚠️ **AV1 + Enhancements**: Very resource intensive. "
            "May crash on free-tier cloud. Use H.264/HEVC for cloud deployment."
        )

    # ── Run Controls ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="vf-label" style="margin-top:8px">🚀 Run</div>',
        unsafe_allow_html=True
    )
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 4])

    with b1:
        if st.button("🔍 Preview Impact", use_container_width=True):
            est_meta = meta.copy()
            if es.get("upscale"):
                est_meta["width"] = int(es.get("upscale_width") or meta["width"] * 2)
                est_meta["height"] = int(es.get("upscale_height") or meta["height"] * 2)
            if es.get("frame_interp"):
                est_meta["fps"] = es.get("target_fps", 60)
            st.markdown("#### 📊 Estimated Output")
            bc1, bc2 = st.columns(2)
            bc1.metric("Source", f"{meta['width']}×{meta['height']}", f"@ {meta['fps']} fps")
            bc2.metric(
                "Enhanced",
                f"{est_meta['width']}×{est_meta['height']}",
                f"@ {est_meta['fps']} fps",
            )
            if est_meta["width"] > meta["width"] and meta["width"] > 0:
                ratio = (est_meta["width"] / meta["width"]) ** 2
                st.caption(f"+{(ratio - 1)*100:.0f}% more pixels ({ratio:.1f}× resolution)")

    go = b2.button(
        "✨ Enhance + Encode",
        type="primary",
        use_container_width=True,
        help="Apply selected enhancements then encode with chosen codec/CRF",
    )
    clear = b3.button("🗑 Clear", use_container_width=True)

    if clear:
        st.session_state.results = []
        st.rerun()

    if go:
        codec_short = codec.split()[0].lower()
        enh_tag = "enh_" if active_enhancements > 0 else ""
        out_path = inp.replace(suf, f"_{enh_tag}{codec_short}_crf{crf}.mp4")

        bar = st.progress(0.0, text=f"⏳ Initializing {codec}…")

        with st.spinner(f"✨ Processing ({active_enhancements} enhancements) + encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(
                inp, out_path, codec, crf,
                es, meta,
                progress_cb=lambda p: bar.progress(p, text=f"⚙️ Processing… {p*100:.0f}%"),
                duration=meta["duration"],
            )

        if not ok:
            bar.empty()
            st.error(f"❌ {msg}")
            if fflog:
                with st.expander("📋 FFmpeg Log"):
                    st.code(fflog, language="bash")
        else:
            bar.progress(1.0, text="✅ Encoding complete — computing quality metrics…")
            out_meta = probe(out_path)
            out_sz = os.path.getsize(out_path) / (1024 * 1024)
            saved_pct = (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0

            qual = {"psnr": None, "ssim": None, "vmaf": None}
            if do_psnr or (do_vmaf and HAS_VMAF):
                with st.spinner("🔍 Computing quality metrics (PSNR/SSIM/VMAF)…"):
                    qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

            active_enh_dict = {k: v for k, v in es.items() if k in _enh_keys and v}
            st.session_state.results.append({
                "codec": codec, "crf": crf,
                "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0),
                "enc_time": enc_t, "saved": saved_pct,
                "cr": sz_mb / out_sz if out_sz > 0 else 0.0,
                "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"],
                "path": out_path,
                "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"],
                "sample_rate": meta["sample_rate"], "channels": meta["channels"],
                "enhancements": active_enh_dict,
                "out_res": f"{out_meta.get('width', meta['width'])}×{out_meta.get('height', meta['height'])}",
                "out_fps": out_meta.get("fps", meta["fps"]),
            })
            bar.empty()

            q_parts = []
            if qual["vmaf"] is not None:
                q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"] is not None:
                q_parts.append(f"PSNR {qual['psnr']:.2f} dB")
            q_str = " · ".join(q_parts) if q_parts else "Analysis complete"

            enh_summary = " + ".join(
                k.capitalize() for k in _enh_keys if es.get(k)
            )
            enh_str = f" · ✨ {enh_summary}" if enh_summary else ""

            st.success(
                f"✅ {codec} CRF {crf}{enh_str} · "
                f"{out_sz:.2f} MB · saved {saved_pct:.1f}% · "
                f"{enc_t:.1f}s · {q_str}"
            )

else:
    # ── Test Player Mode ──────────────────────────────────────────────────────
    st.markdown('<div class="vf-label">🎬 Test Player Mode</div>', unsafe_allow_html=True)
    st.info(
        "🎧 **Playback & Analytics**: Preview your source video. "
        "All metadata displayed. No processing performed."
    )
    if meta["has_audio"]:
        st.markdown(
            f"""<div style="background:#f1f5f9;border-radius:12px;padding:14px 18px;margin:12px 0">
            <div style="font-weight:700;margin-bottom:10px;color:#334155">🔊 Audio Stream Details</div>
            <div style="display:flex;gap:8px;flex-wrap:wrap">
                <span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_audio_codec(meta['acodec'])}</span>
                <span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_channels(meta['channels'])}</span>
                <span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_sample_rate(meta['sample_rate'])}</span>
                <span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{meta['abitrate_kbps']} kbps</span>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.caption("💡 Enable Encoding Mode above to access AI enhancement features.")


# ══════════════════════════════════════════════════════════════════════════════
#  Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state.results
if not results or not enable_encoding:
    if enable_encoding:
        st.caption("📊 Results will appear here after encoding completes.")
    st.stop()

st.divider()
st.markdown('<div class="vf-label">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

tab_tbl, tab_chart, tab_dl = st.tabs(["📋 Comparison Table", "📊 Charts", "⬇ Downloads"])


# ── Comparison Table ──────────────────────────────────────────────────────────
with tab_tbl:
    best_sz  = min(r["size_mb"] for r in results)
    best_cr  = max(r["cr"] for r in results)
    best_spd = min(r["enc_time"] for r in results)
    vmaf_vals = [r["vmaf"] for r in results if r["vmaf"] is not None]
    best_vm  = max(vmaf_vals) if vmaf_vals else None

    def best_mark(val, best, fmt="{}", higher_better=False):
        if val is None or best is None:
            return "—"
        if higher_better:
            is_best = (abs(val - best) < 0.01)
        else:
            is_best = (abs(val - best) < 0.01)
        s = fmt.format(val)
        if is_best and len(results) > 1:
            return f'<span class="best-val">{s} <span class="w-badge">Best</span></span>'
        return s

    rows_html = ""
    for r in results:
        cs = r["codec"].split()[0]
        chip_cls = {"AVC": "chip-avc", "HEVC": "chip-hevc", "AV1": "chip-av1"}.get(cs, "")
        tag = f'<span class="{chip_cls}">{cs}</span>'

        enh_count = len([v for v in r.get("enhancements", {}).values() if v])
        if enh_count > 0:
            tag += f' <span class="chip-enh">✨ ×{enh_count}</span>'

        vmaf_txt, vmaf_cls = vmaf_display(r["vmaf"])
        vmaf_cell = f'<span class="{vmaf_cls}">{vmaf_txt}</span>'
        if r["vmaf"] is not None and best_vm is not None and abs(r["vmaf"] - best_vm) < 0.01 and len(results) > 1:
            vmaf_cell += ' <span class="w-badge">Best</span>'

        audio_info = (
            f"{format_audio_codec(r.get('acodec',''))} · {format_channels(r.get('channels',0))}"
            if r.get("acodec") and r["acodec"] != "unknown"
            else "—"
        )
        res_info = r.get("out_res", f"{meta['width']}×{meta['height']}")
        fps_info = r.get("out_fps", meta["fps"])

        rows_html += f"""<tr>
          <td>{tag}</td>
          <td>{r['crf']}</td>
          <td>{best_mark(r['size_mb'], best_sz, '{:.2f} MB')}</td>
          <td>{r['bitrate']} kbps</td>
          <td>{best_mark(r['cr'], best_cr, '{:.2f}×', True)}</td>
          <td>{r['saved']:.1f}%</td>
          <td>{best_mark(r['enc_time'], best_spd, '{:.1f}s')}</td>
          <td>{vmaf_cell}</td>
          <td>{psnr_display(r['psnr'])}</td>
          <td>{'%.5f' % r['ssim'] if r['ssim'] else '—'}</td>
          <td style="font-size:0.77rem;color:#64748b">{res_info} @ {fps_info} fps</td>
          <td style="font-size:0.77rem;color:#64748b">{audio_info}</td>
        </tr>"""

    st.markdown(
        f"""<table class="cmp-table">
          <thead><tr>
            <th>Codec</th><th>CRF</th><th>Size</th><th>Bitrate</th><th>Ratio</th>
            <th>Saved</th><th>Time</th><th>VMAF ↑</th><th>PSNR ↑</th><th>SSIM ↑</th>
            <th>Resolution</th><th>Audio</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        <div class="src-bar">
          <b>Source:</b> {meta['vcodec'].upper()} · {sz_mb:.2f} MB · 
          {meta['vbitrate_kbps']} kbps · {meta['width']}×{meta['height']} @ {meta['fps']} fps
          {f" · 🔊 {format_audio_codec(meta['acodec'])} {format_channels(meta['channels'])}" if meta['has_audio'] else ""}
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "📏 VMAF: 93+ Excellent · 80–93 Good · 60–80 Fair | "
        "PSNR: 40+ dB Good | ✨ = AI enhancements | 🏆 Best = winner across runs"
    )


# ── Charts ───────────────────────────────────────────────────────────────────
with tab_chart:
    df = pd.DataFrame([{
        "Codec": r["codec"] + (" ✨" if r.get("enhancements") else ""),
        "File Size (MB)": round(r["size_mb"], 3),
        "Bitrate (kbps)": r["bitrate"],
        "Encode Time (s)": round(r["enc_time"], 2),
        "Space Saved (%)": round(r["saved"], 1),
        "VMAF": r["vmaf"],
        "PSNR (dB)": round(r["psnr"], 2) if r["psnr"] else None,
    } for r in results])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**📦 File Size Comparison**")
        size_df = pd.DataFrame(
            [{"Codec": "🎬 Original", "File Size (MB)": round(sz_mb, 3)}]
            + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else ""), "File Size (MB)": round(r["size_mb"], 3)} for r in results]
        ).set_index("Codec")
        st.bar_chart(size_df, color="#3b82f6", use_container_width=True)

        st.markdown("**⏱️ Encoding Time**")
        st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], color="#f97316", use_container_width=True)

    with c2:
        st.markdown("**📡 Bitrate Comparison**")
        brate_df = pd.DataFrame(
            [{"Codec": "🎬 Original", "Bitrate (kbps)": meta["vbitrate_kbps"]}]
            + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else ""), "Bitrate (kbps)": r["bitrate"]} for r in results]
        ).set_index("Codec")
        st.bar_chart(brate_df, color="#8b5cf6", use_container_width=True)

        st.markdown("**💾 Space Saved**")
        st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], color="#10b981", use_container_width=True)

    q_cols = [c for c in ["VMAF", "PSNR (dB)"] if df[c].notna().any()]
    if q_cols:
        st.divider()
        st.markdown("**🎯 Quality Metrics**")
        qdf = df.set_index("Codec")[q_cols].dropna(how="all")
        st.bar_chart(qdf, use_container_width=True)
        st.caption("Higher = better quality. ✨ = AI enhancements applied.")

    if len(results) > 1:
        st.divider()
        st.markdown("**🔍 Smart Insights**")

        enhanced_results = [r for r in results if r.get("enhancements")]
        if enhanced_results:
            # BUG FIX: only pick enhancement winner if VMAF or PSNR available
            scored = [r for r in enhanced_results if r.get("vmaf") is not None or r.get("psnr") is not None]
            if scored:
                best_enh = max(
                    scored,
                    key=lambda r: (r.get("vmaf") or 0) + (r.get("psnr") or 0) / 5,
                )
                n_enh = len([v for v in best_enh["enhancements"].values() if v])
                vmaf_str = f"VMAF {best_enh['vmaf']:.1f}" if best_enh.get("vmaf") else f"PSNR {best_enh.get('psnr', 0):.2f} dB"
                st.markdown(
                    f'<div class="insight-note ai">'
                    f'<b>Enhancement Winner:</b> <span style="font-weight:700">{best_enh["codec"]}</span> '
                    f'with {n_enh} enhancement{"s" if n_enh!=1 else ""} '
                    f'achieved {vmaf_str} at {best_enh["size_mb"]:.2f} MB.</div>',
                    unsafe_allow_html=True,
                )

        eff_candidates = [r for r in results if r.get("vmaf") and r["size_mb"] > 0]
        if eff_candidates:
            best_eff = max(eff_candidates, key=lambda r: r["vmaf"] / r["size_mb"])
            enh_tag = " ✨" if best_eff.get("enhancements") else ""
            st.markdown(
                f'<div class="insight-note"><b>Efficiency Pick:</b> '
                f'<span style="font-weight:700">{best_eff["codec"]}{enh_tag}</span> '
                f'delivers best quality-per-MB '
                f'(VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB).</div>',
                unsafe_allow_html=True,
            )


# ── Downloads ─────────────────────────────────────────────────────────────────
with tab_dl:
    st.markdown("### ⬇ Download Processed Files")
    st.caption("Files are stored temporarily in this session. Download immediately after encoding.")

    for i, r in enumerate(results):
        cs = r["codec"].split()[0]
        enh_tag = " ✨" if r.get("enhancements") else ""
        col_dl, col_info = st.columns([1, 3])

        with col_dl:
            fname = f"videoforge_{cs.lower()}{'_enh' if r.get('enhancements') else ''}_crf{r['crf']}.mp4"
            try:
                with open(r["path"], "rb") as f:
                    st.download_button(
                        label=f"⬇ {cs}{enh_tag} CRF {r['crf']}",
                        data=f,
                        file_name=fname,
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_{cs}_{r['crf']}_{i}",
                    )
            except FileNotFoundError:
                st.caption("⚠️ Temp file expired — re-process to download.")

        with col_info:
            metrics = [
                f"{r['size_mb']:.2f} MB",
                f"{r['bitrate']} kbps",
                r.get("out_res", "N/A"),
            ]
            if r.get("vmaf") is not None:
                metrics.append(f"VMAF {r['vmaf']:.1f}")
            if r.get("psnr") is not None:
                metrics.append(f"PSNR {r['psnr']:.2f} dB")
            st.caption(" · ".join(metrics))
            if r.get("enhancements"):
                enh_list = [k.capitalize() for k, v in r["enhancements"].items() if v]
                st.caption(f"✨ Enhancements: {', '.join(enh_list)}")
            if r.get("acodec") and r["acodec"] not in ("unknown", ""):
                st.caption(
                    f"🔊 {format_audio_codec(r['acodec'])} · {format_channels(r.get('channels', 0))}"
                )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:12px 0">'
    "VideoForge AI Pro · FFmpeg + Streamlit · ✨ Vnova-style enhancements · DM Sans"
    "</div>",
    unsafe_allow_html=True,
)
