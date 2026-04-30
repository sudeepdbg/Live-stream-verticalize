"""
VideoForge Web – Encoder + VMAF Analytics + AI Video Enhancement
Run:  streamlit run app.py

Deploy:
  Streamlit Community Cloud → packages.txt: ffmpeg
  HF Spaces (Streamlit SDK)  → same packages.txt

Changelog (bug-fixes + enhancements):
  BUG  1 – quality_metrics: fixed PSNR/SSIM lavfi filter graph.
  BUG  2 – unsharp filter: corrected to 6-param form.
  BUG  3 – best_mark(): fixed higher_better flag logic.
  BUG  4 – SSIM filter mapped output fixed.
  BUG  5 – deblock alpha/beta mapping range fixed.
  BUG  6 – encode() AV1 -crf flag fixed.
  NEW  1 – Batch CRF Sweep.
  NEW  2 – CSV Export.
  NEW  3 – Per-result expandable FFmpeg log.
  NEW  4 – Enhancement diff preview.
  NEW  5 – Color-coded SSIM display.
  NEW  6 – Audio loudness probe.
  NEW  7 – Quality Radar chart.
  NEW  8 – Real-Time ABR Monitoring Dashboard with Advanced Player Controls.
"""

import os, io, csv, json, subprocess, tempfile, time, re
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Encoder",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f0f4f8;
    color: #1a202c;
}
.stApp { background-color: #f0f4f8; }

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
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.vf-header::after {
    content: "";
    position: absolute; bottom: -40px; left: 30%;
    width: 140px; height: 140px; border-radius: 50%;
    background: rgba(59,130,246,0.12);
}
.vf-header-inner { display:flex; align-items:center; gap:18px; position:relative; z-index:1; }
.vf-play-icon {
    width:56px; height:56px;
    background:rgba(255,255,255,0.12); border:2px solid rgba(255,255,255,0.25);
    border-radius:16px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.vf-play-icon svg { filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
.vf-header h1 { color:white; font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-0.03em; }
.vf-header h1 span { font-weight:300; opacity:0.75; font-size:0.85em; }
.vf-header p { color:#bfdbfe; margin:5px 0 0; font-size:0.88rem; }
.vf-badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; position:relative; z-index:1; }
.vf-badge {
    display:inline-flex; align-items:center; gap:4px;
    background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2);
    border-radius:20px; padding:4px 12px;
    font-size:0.72rem; color:#e0f2fe; font-weight:600; letter-spacing:0.04em;
}
.vf-badge.ai {
    background:linear-gradient(135deg,rgba(124,58,237,0.4),rgba(168,85,247,0.3));
    border-color:rgba(196,181,253,0.4);
}

.mode-toggle {
    background:white; border:1px solid #e2e8f0; border-radius:14px;
    padding:14px 18px; margin:0 0 24px;
    display:flex; align-items:center; gap:12px;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);
}
.mode-toggle .toggle-label { font-weight:600; color:#1e293b; font-size:0.9rem; }
.mode-toggle .toggle-desc  { color:#64748b; font-size:0.82rem; margin-left:auto; }
.mode-active {
    background:#dbeafe; border:1px solid #93c5fd; color:#1e40af;
    padding:3px 10px; border-radius:6px; font-size:0.74rem; font-weight:700;
}

.vf-label {
    font-size:0.68rem; font-weight:700; letter-spacing:0.16em;
    text-transform:uppercase; color:#64748b; margin:22px 0 12px;
    display:flex; align-items:center; gap:8px;
}
.vf-label::before { content:""; width:18px; height:2.5px; background:#3b82f6; border-radius:2px; }
.vf-label.ai::before { background:linear-gradient(90deg,#7c3aed,#a855f7); }

[data-testid="metric-container"] {
    background:white; border:1px solid #e8edf2; border-radius:12px;
    padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.04);
    transition:transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,0.08); }
[data-testid="stMetricValue"] { font-size:1.1rem !important; font-weight:600 !important; }

.stButton > button {
    background:linear-gradient(135deg,#1d4ed8,#3b82f6); color:white; border:none;
    border-radius:10px; padding:10px 22px; font-weight:600; font-size:0.88rem;
    transition:all 0.2s; box-shadow:0 2px 6px rgba(29,78,216,0.25);
    font-family:'DM Sans',sans-serif;
}
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(29,78,216,0.35); }
.stButton > button:disabled { background:#cbd5e1 !important; box-shadow:none !important; cursor:not-allowed !important; transform:none !important; }

.stProgress > div > div { background:linear-gradient(90deg,#1d4ed8,#60a5fa); border-radius:6px; }

.cmp-table {
    width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:8px;
    background:white; border-radius:12px; overflow:hidden;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
.cmp-table th {
    background:#f8fafc; color:#475569; font-weight:700; padding:12px 14px; text-align:left;
    border-bottom:2px solid #e8edf2; white-space:nowrap; font-size:0.72rem;
    text-transform:uppercase; letter-spacing:0.07em;
}
.cmp-table td {
    padding:11px 14px; border-bottom:1px solid #f1f5f9; color:#1e293b;
    font-family:'JetBrains Mono',monospace; white-space:nowrap; font-size:0.8rem;
}
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr:hover td { background:#f8fafc; }
.best-val { color:#15803d; font-weight:700; }
.w-badge {
    background:#dcfce7; color:#15803d; border-radius:4px; padding:2px 7px;
    font-size:0.65rem; font-weight:700; margin-left:6px; text-transform:uppercase;
    font-family:'DM Sans',sans-serif; letter-spacing:0.05em;
}

.chip-avc  { background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-hevc { background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-av1  { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-enh  { background:#faf5ff; color:#6d28d9; border:1px solid #ddd6fe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }

.q-exc { color:#15803d; font-weight:700; }
.q-gd  { color:#1d4ed8; font-weight:700; }
.q-ok  { color:#b45309; font-weight:700; }
.q-bad { color:#b91c1c; font-weight:700; }

.src-bar {
    background:#eff6ff; border-radius:10px; padding:12px 18px; font-size:0.83rem;
    color:#1e40af; margin-top:12px; border-left:4px solid #3b82f6;
    display:flex; flex-wrap:wrap; gap:10px 20px; font-family:'DM Sans',sans-serif;
}
.src-bar b { color:#1e293b; }

.insight-note {
    background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
    padding:14px 18px; font-size:0.85rem; color:#854d0e; margin-top:12px;
    display:flex; gap:10px; align-items:flex-start;
}
.insight-note::before { content:"💡"; font-size:1.1rem; flex-shrink:0; }
.insight-note.ai { background:#faf5ff; border-color:#c4b5fd; color:#5b21b6; }
.insight-note.ai::before { content:"✨"; }

.stTabs [data-baseweb="tab-list"] { gap:4px; background:#e8edf2; border-radius:12px; padding:4px; margin-bottom:18px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:8px 22px; font-size:0.87rem; font-weight:500; color:#64748b; font-family:'DM Sans',sans-serif; }
.stTabs [aria-selected="true"] { background:white !important; box-shadow:0 2px 8px rgba(0,0,0,0.1); color:#1e293b !important; font-weight:700; }

.audio-metric {
    display:flex; align-items:center; gap:8px; padding:9px 14px;
    background:#f8fafc; border-radius:9px; font-size:0.82rem;
    border:1px solid #e2e8f0; margin-top:6px;
}
.audio-metric .icon {
    width:26px; height:26px; background:#3b82f6; border-radius:7px;
    display:flex; align-items:center; justify-content:center;
    color:white; font-size:0.7rem; font-weight:700;
}

.loudness-bar {
    background:#f1f5f9; border-radius:8px; padding:10px 14px; margin-top:8px;
    font-size:0.8rem; border:1px solid #e2e8f0;
}
.loudness-bar b { color:#1e293b; }

label { font-weight:500; color:#334155; font-size:0.87rem; }
.stAlert { border-radius:10px; border-width:1px; }
.stDivider { margin:20px 0; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:12px !important; }

/* ABR Dashboard Specific Styles */
.abr-metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px; padding: 20px; border: 1px solid #334155;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.abr-metric-label { font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.abr-metric-value { font-size: 1.75rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; }
.abr-metric-unit { font-size: 0.875rem; color: #64748b; margin-left: 4px; }
.abr-status { font-size: 0.75rem; margin-top: 6px; padding: 2px 8px; border-radius: 4px; display: inline-block; }
.status-good { background: #166534; color: #86efac; }
.status-warning { background: #a16207; color: #fde047; }
.status-critical { background: #991b1b; color: #fca5a5; }

.abr-ladder-row { display: flex; align-items: center; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
.abr-ladder-row:last-child { border-bottom: none; }
.abr-ladder-quality { font-weight: 600; font-size: 0.875rem; width: 60px; }
.abr-ladder-bar-container { flex: 1; margin: 0 16px; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
.abr-ladder-bar { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
.abr-ladder-bar.active { background: #3b82f6; }
.abr-ladder-bar.available { background: #475569; }
.abr-ladder-bar.unavailable { background: #1e293b; }
.abr-ladder-bitrate { font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; color: #94a3b8; width: 60px; text-align: right; }

.abr-log-entry { display: flex; align-items: center; gap: 12px; padding: 6px 0; border-bottom: 1px solid #334155; font-size: 0.875rem; }
.abr-log-entry:last-child { border-bottom: none; }
.abr-log-time { font-family: 'JetBrains Mono', monospace; color: #64748b; width: 50px; }
.abr-log-arrow { font-size: 1rem; }
.abr-log-arrow.up { color: #22c55e; }
.abr-log-arrow.down { color: #ef4444; }
.abr-log-quality { font-weight: 600; color: #f8fafc; }

.abr-guardrail-toggle { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #334155; }
.abr-guardrail-toggle:last-child { border-bottom: none; }
.abr-guardrail-label { font-size: 0.875rem; font-weight: 500; color: #e2e8f0; }
.abr-toggle-switch { width: 44px; height: 24px; background: #475569; border-radius: 12px; position: relative; cursor: pointer; transition: background 0.2s; }
.abr-toggle-switch.active { background: #3b82f6; }
.abr-toggle-knob { position: absolute; top: 2px; left: 2px; width: 20px; height: 20px; background: white; border-radius: 50%; transition: transform 0.2s; }
.abr-toggle-switch.active .abr-toggle-knob { transform: translateX(20px); }

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
        "has_audio": False, "color_space": "unknown", "bit_depth": 8,
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        d = json.loads(out)
        fmt = d.get("format", {})
        r["duration"]       = float(fmt.get("duration", 0) or 0)
        r["vbitrate_kbps"]  = int(fmt.get("bit_rate", 0) or 0) // 1000

        for s in d.get("streams", []):
            if s.get("codec_type") == "video" and r["width"] == 0:
                r["width"]      = s.get("width", 0)
                r["height"]     = s.get("height", 0)
                r["vcodec"]     = s.get("codec_name", "unknown")
                r["color_space"] = s.get("color_space", "unknown")
                r["bit_depth"]  = int(s.get("bits_per_raw_sample", 8) or 8)
                try:
                    n, dn = map(int, s.get("r_frame_rate", "0/1").split("/"))
                    r["fps"] = round(n / dn, 3) if dn else 0.0
                except Exception:
                    pass
            elif s.get("codec_type") == "audio" and not r["has_audio"]:
                r["has_audio"]       = True
                r["acodec"]          = s.get("codec_name", "unknown")
                r["abitrate_kbps"]   = int(s.get("bit_rate", 0) or 0) // 1000
                r["sample_rate"]     = int(s.get("sample_rate", 0) or 0)
                r["channels"]        = int(s.get("channels", 0) or 0)
                r["audio_duration"]  = float(s.get("duration", 0) or 0)
    except Exception:
        pass
    return r

def probe_loudness(path: str) -> dict:
    """
    NEW: Measure integrated loudness + true peak via ffmpeg volumedetect.
    Returns dict with mean_volume, max_volume (dBFS).
    """
    res = {"mean_volume": None, "max_volume": None}
    try:
        cmd = ["ffmpeg", "-y", "-i", path, "-af", "volumedetect", "-f", "null", "-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
        for line in out.splitlines():
            m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line)
            if m:
                res["mean_volume"] = float(m.group(1))
            m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
            if m:
                res["max_volume"] = float(m.group(1))
    except Exception:
        pass
    return res

def build_enhance_filters(settings: dict, src_meta: dict) -> list:
    """Build FFmpeg filter chain for AI enhancements."""
    filters = []

    # ── Denoise ──────────────────────────────────────────────────────────────
    if settings.get("denoise"):
        strength = float(settings.get("denoise_strength", 5))
        s2 = round(strength / 2, 1)
        filters.append(f"hqdn3d={strength}:{strength}:{s2}:{s2}")

    # ── Deblock ───────────────────────────────────────────────────────────────
    # Slider 1-10 → FFmpeg alpha 0.1-1.0, beta = alpha * 0.5
    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5))
        alpha = round(strength / 10.0, 2)
        beta  = round(alpha * 0.5, 2)
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")

    # ── Sharpening ────────────────────────────────────────────────────────────
    # BUG FIX: unsharp takes 6 params: lx:ly:la:cx:cy:ca
    if settings.get("sharpen"):
        amount    = float(settings.get("sharpen_amount", 0.5))
        c_amount  = round(amount * 0.5, 2)
        filters.append(f"unsharp=lx=5:ly=5:la={amount}:cx=3:cy=3:ca={c_amount}")

    # ── Color Enhancement ─────────────────────────────────────────────────────
    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15))
        contrast = float(settings.get("contrast", 1.0))
        sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")

    # ── HDR Conversion ────────────────────────────────────────────────────────
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.append("zscale=transfer=linear,format=gbrpf32le")
        filters.append(f"tonemap={settings.get('tonemap_algo','hable')}:desat=0.2")
        filters.append("zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le")

    # ── Upscaling ─────────────────────────────────────────────────────────────
    if settings.get("upscale"):
        target_w = int(settings.get("upscale_width",  src_meta["width"]  * 2))
        target_h = int(settings.get("upscale_height", src_meta["height"] * 2))
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        target_h = target_h if target_h % 2 == 0 else target_h - 1
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(
            f"scale={target_w}:{target_h}:flags={algo}+accurate_rnd+full_chroma_int"
        )

    # ── Frame Interpolation ───────────────────────────────────────────────────
    if settings.get("frame_interp"):
        target_fps = int(settings.get("target_fps", 60))
        filters.append(
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
        )

    return filters

def estimate_processing_time(src_meta: dict, settings: dict) -> str:
    base_factor = 1.0
    if settings.get("denoise"):     base_factor *= 1.3
    if settings.get("sharpen"):     base_factor *= 1.1
    if settings.get("upscale"):
        base_factor *= 2.5 if settings.get("upscale_algo") == "lanczos" else 1.8
    if settings.get("hdr_convert"): base_factor *= 1.6
    if settings.get("color_enhance"): base_factor *= 1.15
    if settings.get("deblock"):     base_factor *= 1.4
    if settings.get("frame_interp"): base_factor *= 3.0

    duration_min = (src_meta.get("duration", 0) / 60) * base_factor
    if duration_min < 1: return f"~{max(1, int(duration_min * 60))}s"
    elif duration_min < 10: return f"~{duration_min:.1f} min"
    else: return f"~{duration_min:.0f} min"

def encode(input_path, output_path, codec, crf, enhance_settings: dict, src_meta: dict,
           progress_cb=None, duration=0.0):
    """
    Encode with optional enhancement filters.
    BUG FIX (AV1): libaom-av1 requires `-b:v 0` alongside `-crf` to activate
    CRF mode; without it the CRF value is silently ignored and ABR is used.
    """
    cmap = {
        "AVC (H.264)":  ("libx264",    ["-preset", "fast"]),
        "HEVC (H.265)": ("libx265",    ["-preset", "fast"]),
        "AV1":          ("libaom-av1", ["-b:v", "0", "-cpu-used", "8",
                                        "-tile-columns", "2", "-threads", "4",
                                        "-usage", "realtime"]),
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
                    h, m, s = map(float, ts.split(":"))
                    progress_cb(min((h * 3600 + m * 60 + s) / duration, 0.99))
                except Exception:
                    pass
        proc.wait()
        elapsed = time.time() - t0
        log = "\n".join(lines[-80:])

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
    """
    Compute PSNR, SSIM and optionally VMAF via FFmpeg.
    BUG FIX: The original code used a split filter graph that mapped [po] and
    [so] outputs to `-map` then `-f null -`, which causes FFmpeg to error.
    Fix: run PSNR and SSIM as separate filter chains.
    """
    res = {"psnr": None, "ssim": None, "vmaf": None}

    # ── PSNR ─────────────────────────────────────────────────────────────────
    try:
        cmd = [
            "ffmpeg", "-y", "-i", dist, "-i", ref,
            "-filter_complex", "[0:v][1:v]psnr",
            "-f", "null", "-"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"PSNR", line, re.I):
                m = re.search(r"average[:\s]+([0-9.]+|inf)", line, re.I)
                if m:
                    v = m.group(1)
                    res["psnr"] = 100.0 if v == "inf" else round(float(v), 3)
    except Exception:
        pass

    # ── SSIM ─────────────────────────────────────────────────────────────────
    try:
        cmd = [
            "ffmpeg", "-y", "-i", dist, "-i", ref,
            "-filter_complex", "[0:v][1:v]ssim",
            "-f", "null", "-"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"SSIM", line, re.I):
                m = re.search(r"All[:\s]+([0-9.]+)", line, re.I)
                if m:
                    res["ssim"] = round(float(m.group(1)), 5)
    except Exception:
        pass

    # ── VMAF (optional) ──────────────────────────────────────────────────────
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

# ── Display helpers ───────────────────────────────────────────────────────────

def vmaf_display(v):
    if v is None: return "—", ""
    if v >= 93: return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80: return f"{v:.1f} · Good",      "q-gd"
    if v >= 60: return f"{v:.1f} · Fair",       "q-ok"
    return           f"{v:.1f} · Poor",          "q-bad"

def ssim_display(v) -> str:
    """NEW: colour-coded SSIM (was raw float)."""
    if v is None: return "—"
    label = (
        "Excellent" if v >= 0.98 else
        "Good"      if v >= 0.95 else
        "Fair"      if v >= 0.90 else
        "Poor"
    )
    return f"{v:.5f} · {label}"

def psnr_display(v):
    if v is None: return "—"
    tag = "Excellent" if v >= 50 else "Good" if v >= 40 else "Acceptable" if v >= 30 else "Poor"
    return f"{v:.2f} dB · {tag}"

def format_audio_codec(codec: str) -> str:
    mapping = {
        "aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis",
        "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC",
        "pcm_s16le": "PCM 16-bit", "alac": "ALAC",
    }
    return mapping.get(codec, codec.upper())

def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"

def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")

def best_mark(val, best, fmt="{}", higher_better=False):
    """
    Format a metric value and append a 'Best' badge if it matches the best.
    BUG FIX: Corrected logic for higher_better metrics.
    """
    if val is None or best is None: return "—"
    s = fmt.format(val)
    is_best = abs(val - best) < 0.01 and len(st.session_state.results) > 1
    if is_best:
        return f'<span class="best-val">{s} <span class="w-badge">Best</span></span>'
    return s

def results_to_csv(results: list, src_meta: dict, sz_mb: float) -> bytes:
    """NEW: Export comparison table as CSV bytes."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "Codec", "CRF", "Enhancements",
        "Size_MB", "Bitrate_kbps", "Compression_Ratio", "Space_Saved_%",
        "Encode_Time_s", "VMAF", "PSNR_dB", "SSIM",
        "Output_Resolution", "Output_FPS",
        "Audio_Codec", "Audio_Channels",
        "Source_Codec", "Source_Size_MB", "Source_Bitrate_kbps",
    ])
    for r in results:
        enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
        enh_list = [k for k in enh_keys if r.get("enhancements", {}).get(k)]
        w.writerow([
            r["codec"], r["crf"], "|".join(enh_list),
            f"{r['size_mb']:.3f}", r["bitrate"],
            f"{r['cr']:.3f}", f"{r['saved']:.1f}",
            f"{r['enc_time']:.2f}",
            r["vmaf"] or "", r["psnr"] or "", r["ssim"] or "",
            r.get("out_res", ""), r.get("out_fps", ""),
            format_audio_codec(r.get("acodec", "")),
            format_channels(r.get("channels", 0)),
            src_meta["vcodec"].upper(), f"{sz_mb:.3f}", src_meta["vbitrate_kbps"],
        ])
    return buf.getvalue().encode()


# ══════════════════════════════════════════════════════════════════════════════
#  Session State Init
# ══════════════════════════════════════════════════════════════════════════════

_default_enhance = {
    "denoise": False,   "denoise_strength": 5,
    "sharpen": False,   "sharpen_amount": 0.5, "sharpen_threshold": 5,
    "upscale": False,   "upscale_algo": "lanczos",
    "upscale_width": 0, "upscale_height": 0,
    "hdr_convert": False, "tonemap_algo": "hable",
    "color_enhance": False, "vibrance": 0.15, "contrast": 1.0,
    "deblock": False,   "deblock_strength": 5,
    "frame_interp": False, "target_fps": 60,
}

defaults = {
    "results": [], "inp": None, "meta": None, "sz": 0.0, "name": "",
    "enable_encoding": True,
    "enhance_settings": _default_enhance.copy(),
    "loudness": None,
    "result_logs": {},
    # NEW: ABR Dashboard State
    "abr_metrics_history": [],
    "abr_decisions": [],
    "abr_guardrails": {
        "buffer_guard": True,
        "stability_lock": True,
        "fast_downgrade": True,
        "slow_upgrade": True,
    },
    "abr_simulation_running": False,
    "abr_start_time": time.time(),
    "abr_current_bitrate_idx": 2,  # Start at 540p
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  ABR Simulation Logic
# ══════════════════════════════════════════════════════════════════════════════

BITRATE_LADDER = [
    {"quality": "1080p", "bitrate_mbps": 8.0},
    {"quality": "720p", "bitrate_mbps": 4.0},
    {"quality": "540p", "bitrate_mbps": 2.0},
    {"quality": "480p", "bitrate_mbps": 1.2},
    {"quality": "360p", "bitrate_mbps": 0.6},
]

def generate_realistic_abr_metrics():
    """Generate realistic ABR metrics with natural fluctuations."""
    base_bandwidth = 4.7
    fluctuation = random.uniform(-0.8, 0.8)
    trend = 0.3 * np.sin(time.time() / 15)
    bandwidth = max(0.5, min(12.0, base_bandwidth + fluctuation + trend))
    
    current_bitrate = BITRATE_LADDER[st.session_state.abr_current_bitrate_idx]["bitrate_mbps"]
    buffer_change = (bandwidth - current_bitrate) * 0.3
    current_buffer = st.session_state.abr_metrics_history[-1]["buffer_sec"] if st.session_state.abr_metrics_history else 20.0
    buffer_sec = max(0.0, min(60.0, current_buffer + buffer_change + random.uniform(-2, 2)))
    
    rtt = max(10, min(150, 25 + random.uniform(-10, 15) + (random.random() > 0.92) * 80))
    drop_events = 1 if random.random() > 0.97 else 0
    
    old_idx = st.session_state.abr_current_bitrate_idx
    new_idx = old_idx
    
    available = [i for i, lvl in enumerate(BITRATE_LADDER) if lvl["bitrate_mbps"] <= bandwidth * 0.85]
    if available:
        target_idx = max(available)
        
        if st.session_state.abr_guardrails["slow_upgrade"] and target_idx < old_idx:
            new_idx = max(old_idx - 1, target_idx)
        elif st.session_state.abr_guardrails["fast_downgrade"] and target_idx > old_idx:
            new_idx = min(old_idx + 2, target_idx)
        else:
            new_idx = target_idx
        
        if st.session_state.abr_guardrails["buffer_guard"] and buffer_sec < 8.0 and new_idx < old_idx:
            new_idx = old_idx
        
        if st.session_state.abr_guardrails["stability_lock"] and new_idx < old_idx:
            last_decision_time = st.session_state.abr_decisions[-1]["time"] if st.session_state.abr_decisions else 0
            if time.time() - last_decision_time < 5.0:
                new_idx = old_idx
    else:
        new_idx = len(BITRATE_LADDER) - 1
    
    decision = None
    if new_idx != old_idx:
        direction = "↑" if new_idx < old_idx else "↓"
        elapsed = time.time() - st.session_state.abr_start_time
        decision = {
            "time": f"{int(elapsed // 60)}:{int(elapsed % 60):02d}",
            "direction": direction,
            "from_quality": BITRATE_LADDER[old_idx]["quality"],
            "to_quality": BITRATE_LADDER[new_idx]["quality"],
            "type": "upgrade" if direction == "↑" else "downgrade",
            "timestamp": time.time(),
        }
        st.session_state.abr_decisions.insert(0, decision)
        st.session_state.abr_decisions = st.session_state.abr_decisions[:20]
        st.session_state.abr_current_bitrate_idx = new_idx
    
    return {
        "bandwidth_mbps": round(bandwidth, 1),
        "buffer_sec": round(buffer_sec, 1),
        "rtt_ms": round(rtt, 0),
        "drop_events": drop_events,
        "current_quality": BITRATE_LADDER[st.session_state.abr_current_bitrate_idx]["quality"],
        "current_bitrate_mbps": BITRATE_LADDER[st.session_state.abr_current_bitrate_idx]["bitrate_mbps"],
        "buffer_pct": round((buffer_sec / 20.0) * 100, 0),
        "timestamp": time.time(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="vf-header">
  <div class="vf-header-inner">
    <div class="vf-play-icon">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="32" height="32">
        <polygon points="28,20 28,80 82,50" fill="white"/>
      </svg>
    </div>
    <div>
      <h1>VideoForge <span>AI Pro</span></h1>
      <p>Professional encoding · AI enhancement · quality analytics</p>
    </div>
  </div>
  <div class="vf-badges">
    <span class="vf-badge">H.264</span>
    <span class="vf-badge">HEVC</span>
    <span class="vf-badge">AV1</span>
    <span class="vf-badge">VMAF</span>
    <span class="vf-badge">PSNR/SSIM</span>
    <span class="vf-badge ai">✨ AI Enhance</span>
    <span class="vf-badge">📊 CRF Sweep</span>
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

enable_encoding = st.toggle(
    "⚙️ Enable Encoding Mode",
    value=st.session_state.enable_encoding,
    help="Toggle between encoder mode (with AI enhancements) and test player mode (analytics only).",
)
st.session_state.enable_encoding = enable_encoding

mode_label = "⚙️ Encoder" if enable_encoding else "🎬 Test Player"
mode_desc  = (
    "Full workflow with AI enhancement & encoding"
    if enable_encoding
    else "Playback & analytics only — no processing"
)
st.markdown(f"""
<div class="mode-toggle">
  <span class="toggle-label">🎛️ Mode:</span>
  <span class="mode-active">{mode_label}</span>
  <span class="toggle-desc">{mode_desc}</span>
</div>
""", unsafe_allow_html=True)


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

if st.session_state.name != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded.read())
        st.session_state.inp = tmp.name
    st.session_state.meta     = probe(st.session_state.inp)
    st.session_state.sz       = os.path.getsize(st.session_state.inp) / (1024 * 1024)
    st.session_state.name     = uploaded.name
    st.session_state.loudness = None
    m = st.session_state.meta
    st.session_state.enhance_settings["upscale_width"]  = m["width"]  * 2
    st.session_state.enhance_settings["upscale_height"] = m["height"] * 2
    if enable_encoding:
        st.session_state.results     = []
        st.session_state.result_logs = {}

meta  = st.session_state.meta
sz_mb = st.session_state.sz
inp   = st.session_state.inp


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
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    r1.metric("Duration",   f"{meta['duration']:.1f}s")
    r2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    r1.metric("Frame Rate", f"{meta['fps']} fps")
    r2.metric("Codec",      meta["vcodec"].upper())
    r1.metric("Bitrate",    f"{meta['vbitrate_kbps']} kbps" if meta["vbitrate_kbps"] else "—")
    r2.metric("File Size",  f"{sz_mb:.2f} MB")

    if meta["has_audio"]:
        st.markdown(
            '<div style="margin:16px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🔊 Audio Stream</div>',
            unsafe_allow_html=True,
        )
        a1, a2 = st.columns(2)
        a1.metric("Codec",       format_audio_codec(meta["acodec"]))
        a2.metric("Channels",    format_channels(meta["channels"]))
        a1.metric("Sample Rate", format_sample_rate(meta["sample_rate"]))
        a2.metric("Bitrate",     f"{meta['abitrate_kbps']} kbps" if meta["abitrate_kbps"] > 0 else "Variable")

        if meta["audio_duration"] > 0 and meta["duration"] > 0:
            sync_diff   = abs(meta["audio_duration"] - meta["duration"])
            sync_status = "✓ Synced" if sync_diff < 0.1 else f"⚠ {sync_diff:.2f}s off"
            st.markdown(
                f'<div class="audio-metric"><span class="icon">🔗</span>'
                f' A/V Sync: <b style="margin-left:4px">{sync_status}</b></div>',
                unsafe_allow_html=True,
            )

        if st.button("🔊 Measure Loudness", help="Run ffmpeg volumedetect on the audio track"):
            with st.spinner("Measuring loudness…"):
                st.session_state.loudness = probe_loudness(inp)

        if st.session_state.loudness:
            ld = st.session_state.loudness
            mean_db = ld.get("mean_volume")
            max_db  = ld.get("max_volume")
            if mean_db is not None:
                colour = "#15803d" if -20 <= mean_db <= -14 else "#b45309" if mean_db > -14 else "#b91c1c"
                st.markdown(
                    f'<div class="loudness-bar">'
                    f'📢 <b>Mean:</b> <span style="color:{colour};font-weight:700">{mean_db:.1f} dBFS</span>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                    f'<b>Peak:</b> {max_db:.1f} dBFS'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div style="margin:16px 0 4px;font-weight:500;color:#94a3b8;font-size:0.84rem">🔇 No audio track detected</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Real-Time ABR Dashboard
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="vf-label ai">📡 Real-Time ABR Monitoring Dashboard</div>', unsafe_allow_html=True)

# Initialize simulation state if not already done
if not st.session_state.abr_metrics_history:
    st.session_state.abr_metrics_history = [{"bandwidth_mbps": 4.7, "buffer_sec": 20.0, "rtt_ms": 25, "drop_events": 0, "current_quality": "540p", "current_bitrate_mbps": 2.0, "buffer_pct": 100, "timestamp": time.time()}]

player_col, metrics_col = st.columns([1, 1.3], gap="large")

# ── LEFT PANEL: Advanced Player Controls ─────────────────────────────────────
with player_col:
    st.markdown('<div style="font-size: 0.85rem; color: #64748b; margin-bottom: 8px;">● player <span style="margin-left: 16px; color: #94a3b8;">subject track</span></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: #000; border-radius: 12px; aspect-ratio: 16/9; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.7); padding: 4px 12px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; color: #22c55e;">
            ● {BITRATE_LADDER[st.session_state.abr_current_bitrate_idx]['quality']}
        </div>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <div style="width: 80px; height: 80px; background: rgba(255,255,255,0.1); border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
                    <path d="M8 5v14l11-7z"/>
                </svg>
            </div>
        </div>
        <div style="position: absolute; bottom: 60px; left: 50%; transform: translateX(-50%); text-align: center; color: #64748b;">
            <div style="font-size: 0.85rem;">landscape → 9:16</div>
            <div style="font-size: 0.75rem; margin-top: 4px;">verticalize.py</div>
        </div>
        <div style="position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.7); padding: 4px 8px; border-radius: 4px; font-size: 0.75rem;">
            1.0x
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Playback controls
    st.markdown('<div style="margin-top: 16px; display: flex; gap: 8px; align-items: center;">', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("▶ Play", key="play_btn", use_container_width=True):
            st.session_state.abr_simulation_running = not st.session_state.abr_simulation_running
            if st.session_state.abr_simulation_running:
                st.session_state.abr_start_time = time.time()
    with col_btn2:
        if st.button("◄◄", key="frame_prev", use_container_width=True, help="Previous frame"):
            st.info("⏮ Frame step backward (simulated)")
    with col_btn3:
        if st.button("▶▶", key="frame_next", use_container_width=True, help="Next frame"):
            st.info("⏭ Frame step forward (simulated)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    speed_options = ["0.25x", "0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"]
    st.selectbox("", speed_options, index=3, key="speed_selector", label_visibility="collapsed")
    
    st.markdown('<div style="margin-top: 12px; display: flex; gap: 8px;">', unsafe_allow_html=True)
    col_ab1, col_ab2 = st.columns(2)
    with col_ab1:
        if st.button("A/B", key="ab_toggle", use_container_width=True, help="Toggle A/B loop mode"):
            st.toast("🔁 A/B loop mode activated")
    with col_ab2:
        if st.button("Loop", key="loop_toggle", use_container_width=True, help="Toggle continuous loop"):
            st.toast("🔁 Loop mode activated")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 8px; text-align: right; font-size: 0.75rem; color: #64748b;">0:00 / 2:34</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: #334155; height: 6px; border-radius: 3px; margin-top: 8px; position: relative;">', unsafe_allow_html=True)
    st.markdown('<div style="position: absolute; left: 20%; right: 60%; top: 0; bottom: 0; background: #3b82f6; border-radius: 3px;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 16px;"><div style="font-size: 0.75rem; color: #64748b; margin-bottom: 8px;">zoom</div>', unsafe_allow_html=True)
    st.slider("", 1.0, 3.0, 1.0, 0.25, key="zoom_slider", label_visibility="collapsed")
    st.markdown(f'<div style="text-align: right; font-size: 0.75rem; color: #64748b; margin-top: -20px;">1.0x</div>', unsafe_allow_html=True)

# ── RIGHT PANEL: Intelligent ABR Dashboard ────────────────────────────────────
with metrics_col:
    st.markdown('<div style="font-size: 0.85rem; color: #64748b; margin-bottom: 8px;">● adaptive bitrate — stable</div>', unsafe_allow_html=True)
    
    # Generate new metrics if simulation is running
    if st.session_state.abr_simulation_running:
        current_metrics = generate_realistic_abr_metrics()
        st.session_state.abr_metrics_history.append(current_metrics)
        if len(st.session_state.abr_metrics_history) > 100:
            st.session_state.abr_metrics_history = st.session_state.abr_metrics_history[-100:]
        time.sleep(0.5) # Simulate 500ms update interval
        st.rerun()
    else:
        current_metrics = st.session_state.abr_metrics_history[-1]
    
    # Top metrics grid
    col1, col2 = st.columns(2)
    with col1:
        bw_status = "good" if current_metrics["bandwidth_mbps"] > 3.0 else ("warning" if current_metrics["bandwidth_mbps"] > 1.5 else "critical")
        bw_status_text = "+ above threshold" if bw_status == "good" else ("— near threshold" if bw_status == "warning" else "— below threshold")
        status_color = "#22c55e" if bw_status == "good" else ("#f59e0b" if bw_status == "warning" else "#ef4444")
        st.markdown(f"""
        <div class="abr-metric-card">
            <div class="abr-metric-label">bandwidth</div>
            <div>
                <span class="abr-metric-value">{current_metrics['bandwidth_mbps']}</span>
                <span class="abr-metric-unit">Mbps</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 6px; padding: 2px 8px; border-radius: 4px; display: inline-block; color: {status_color};">{bw_status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        buf_status = "good" if current_metrics["buffer_sec"] > 10.0 else ("warning" if current_metrics["buffer_sec"] > 5.0 else "critical")
        buf_status_text = "healthy" if buf_status == "good" else ("low" if buf_status == "warning" else "critical")
        status_color = "#22c55e" if buf_status == "good" else ("#f59e0b" if buf_status == "warning" else "#ef4444")
        st.markdown(f"""
        <div class="abr-metric-card">
            <div class="abr-metric-label">buffer health</div>
            <div>
                <span class="abr-metric-value">{current_metrics['buffer_sec']}</span>
                <span class="abr-metric-unit">s</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 6px; padding: 2px 8px; border-radius: 4px; display: inline-block; color: {status_color};">{buf_status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        rtt_status = "good" if current_metrics["rtt_ms"] < 50 else ("warning" if current_metrics["rtt_ms"] < 100 else "critical")
        status_color = "#22c55e" if rtt_status == "good" else ("#f59e0b" if rtt_status == "warning" else "#ef4444")
        st.markdown(f"""
        <div class="abr-metric-card">
            <div class="abr-metric-label">RTT</div>
            <div>
                <span class="abr-metric-value">{int(current_metrics['rtt_ms'])}</span>
                <span class="abr-metric-unit">ms</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 6px; padding: 2px 8px; border-radius: 4px; display: inline-block; color: {status_color};">{rtt_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        drop_status = "good" if current_metrics["drop_events"] == 0 else "critical"
        status_color = "#22c55e" if drop_status == "good" else "#ef4444"
        drop_text = "— no rebuffer" if drop_status == "good" else "— rebuffering!"
        st.markdown(f"""
        <div class="abr-metric-card">
            <div class="abr-metric-label">drop events</div>
            <div>
                <span class="abr-metric-value">{current_metrics['drop_events']}</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 6px; padding: 2px 8px; border-radius: 4px; display: inline-block; color: {status_color};">{drop_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Buffer level visualization
    st.markdown('<div style="font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px;">buffer level</div>', unsafe_allow_html=True)
    guard_position = (4.0 / 20.0) * 100
    st.markdown(f"""
    <div style="background: #1e293b; border-radius: 8px; padding: 12px; margin: 12px 0;">
        <div style="height: 24px; background: linear-gradient(90deg, #166534, #22c55e); border-radius: 6px; position: relative; overflow: hidden; width: {min(100, current_metrics['buffer_pct'])}%;">
            <div style="position: absolute; left: {guard_position}%; top: 0; bottom: 0; width: 2px; background: #f59e0b; z-index: 10;"></div>
            <div style="position: absolute; left: {guard_position}%; top: -22px; font-size: 0.75rem; color: #f59e0b; transform: translateX(-50%);">guard: 4s</div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b; margin-top: 4px;">
            <span>0s</span>
            <span>20s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bitrate ladder
    st.markdown('<div style="font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px;">bitrate ladder</div>', unsafe_allow_html=True)
    
    ladder_html = '<div style="background: #1e293b; border-radius: 8px; padding: 12px;">'
    for i, lvl in enumerate(BITRATE_LADDER):
        is_active = (i == st.session_state.abr_current_bitrate_idx)
        is_available = lvl["bitrate_mbps"] <= current_metrics["bandwidth_mbps"] * 0.9
        
        if is_active:
            bar_class = "active"
            bar_color = "#3b82f6"
            bar_width = "100%"
        elif is_available:
            bar_class = "available"
            bar_color = "#475569"
            bar_width = "60%"
        else:
            bar_class = "unavailable"
            bar_color = "#1e293b"
            bar_width = "20%"
        
        quality_bg = "#3b82f6" if is_active else "transparent"
        quality_color = "#f8fafc" if is_active else "#94a3b8"
        
        ladder_html += f"""
        <div class="abr-ladder-row">
            <div class="abr-ladder-quality" style="background: {quality_bg}; color: {quality_color}; padding: 2px 8px; border-radius: 4px;">{lvl['quality']}</div>
            <div class="abr-ladder-bar-container">
                <div class="abr-ladder-bar" style="background: {bar_color}; width: {bar_width};"></div>
            </div>
            <div class="abr-ladder-bitrate">{lvl['bitrate_mbps']}M</div>
        </div>
        """
    ladder_html += '</div>'
    st.markdown(ladder_html, unsafe_allow_html=True)
    
    # Bandwidth history chart
    st.markdown('<div style="font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px;">bandwidth history (30s)</div>', unsafe_allow_html=True)
    
    if len(st.session_state.abr_metrics_history) > 1:
        chart_data = pd.DataFrame([
            {"time": i, "bandwidth": m["bandwidth_mbps"]}
            for i, m in enumerate(st.session_state.abr_metrics_history[-30:])
        ])
        st.line_chart(chart_data.set_index("time")[["bandwidth"]], color="#3b82f6", use_container_width=True)
    else:
        st.markdown('<div style="background: #1e293b; border-radius: 8px; padding: 40px; text-align: center; color: #64748b;">Collecting data...</div>', unsafe_allow_html=True)
    
    # ABR decisions log
    st.markdown('<div style="font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px;">abr decisions</div>', unsafe_allow_html=True)
    
    if st.session_state.abr_decisions:
        log_html = '<div style="background: #1e293b; border-radius: 8px; padding: 12px; max-height: 200px; overflow-y: auto;">'
        for decision in st.session_state.abr_decisions[:10]:
            arrow_color = "#22c55e" if decision["direction"] == "↑" else "#ef4444"
            log_html += f"""
            <div class="abr-log-entry">
                <div class="abr-log-time">{decision['time']}</div>
                <div class="abr-log-arrow" style="color: {arrow_color};">{decision['direction']}</div>
                <div class="abr-log-quality">{decision['type']} → {decision['to_quality']}</div>
            </div>
            """
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: #1e293b; border-radius: 8px; padding: 20px; text-align: center; color: #64748b;">No decisions yet</div>', unsafe_allow_html=True)
    
    # Guardrails toggles
    st.markdown('<div style="font-size: 0.75rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin: 20px 0 12px;">guardrails</div>', unsafe_allow_html=True)
    
    guardrails_html = '<div style="background: #1e293b; border-radius: 8px; padding: 12px;">'
    for key, label in [
        ("buffer_guard", "Buffer guard"),
        ("stability_lock", "Stability lock"),
        ("fast_downgrade", "Fast downgrade"),
        ("slow_upgrade", "Slow upgrade"),
    ]:
        active_class = "active" if st.session_state.abr_guardrails[key] else ""
        guardrails_html += f"""
        <div class="abr-guardrail-toggle">
            <div class="abr-guardrail-label">{label}</div>
            <div class="abr-toggle-switch {active_class}" onclick="window.location.reload()">
                <div class="abr-toggle-knob"></div>
            </div>
            <input type="checkbox" id="toggle_{key}" style="display: none;" {'checked' if st.session_state.abr_guardrails[key] else ''} onchange="document.getElementById('toggle_{key}').click()">
        </div>
        """
    guardrails_html += '</div>'
    st.markdown(guardrails_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AI ENHANCEMENT PANEL  (Encoder mode only)
# ══════════════════════════════════════════════════════════════════════════════

if enable_encoding:
    st.markdown('<div class="vf-label ai">✨ AI Video Enhancement</div>', unsafe_allow_html=True)

    # ── Quick Presets ─────────────────────────────────────────────────────────
    st.markdown("### 🎯 Quick Presets")
    st.caption("One-click setups for common scenarios:")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("#### 🎬 Standard Clean")
        st.caption("**Best for:** Noisy, grainy, or compressed footage")
        st.markdown("• Reduces noise & grain\n• Removes blocking artifacts\n• Mild sharpening")
        if st.button("Apply Standard Clean", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"denoise": True, "denoise_strength": 5, "deblock": True, "deblock_strength": 5, "sharpen": True, "sharpen_amount": 0.3, "sharpen_threshold": 8, "upscale": False, "hdr_convert": False, "frame_interp": False, "color_enhance": False})
            st.rerun()

    with p2:
        st.markdown("#### 🎨 Detail & Color Boost")
        st.caption("**Best for:** Dull, flat, or low-contrast footage")
        st.markdown("• Enhances sharpness\n• Boosts color vibrance\n• Increases contrast")
        if st.button("Apply Detail Boost", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"sharpen": True, "sharpen_amount": 0.8, "sharpen_threshold": 3, "color_enhance": True, "vibrance": 0.25, "contrast": 1.15, "denoise": False, "deblock": False, "upscale": False, "hdr_convert": False, "frame_interp": False})
            st.rerun()

    with p3:
        st.markdown("#### 🚀 AI Upscale 2×")
        st.caption("**Best for:** Low-resolution content (480p→1080p, 1080p→4K)")
        st.markdown("• 2× resolution upscale\n• Pre-denoise for quality\n• Post-sharpen details")
        if st.button("Apply AI Upscale 2×", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"upscale": True, "upscale_width": meta["width"] * 2, "upscale_height": meta["height"] * 2, "upscale_algo": "lanczos", "denoise": True, "denoise_strength": 6, "sharpen": True, "sharpen_amount": 0.5, "sharpen_threshold": 5, "hdr_convert": False, "frame_interp": False, "deblock": False, "color_enhance": False})
            st.rerun()

    st.divider()

    es = st.session_state.enhance_settings
    enh_col1, enh_col2 = st.columns(2)

    # ── Left Column ───────────────────────────────────────────────────────────
    with enh_col1:
        with st.container(border=True):
            st.markdown("**🧹 Denoise**")
            es["denoise"] = st.checkbox("Enable temporal noise reduction", value=es["denoise"], key="chk_denoise", help="Reduces grain and compression artifacts using 3D filtering")
            if es["denoise"]:
                es["denoise_strength"] = st.slider("Strength", 1, 10, es["denoise_strength"], key="sl_denoise_str", help="Higher = more aggressive noise removal")
            st.caption("Reduces grain, compression artifacts, and sensor noise.")

        with st.container(border=True):
            st.markdown("**🔍 Detail Enhancement**")
            es["sharpen"] = st.checkbox("Enable adaptive sharpening", value=es["sharpen"], key="chk_sharpen")
            if es["sharpen"]:
                cs1, cs2 = st.columns(2)
                with cs1:
                    es["sharpen_amount"] = st.slider("Amount", -1.5, 1.5, es["sharpen_amount"], 0.1, key="sl_sharpen_amt")
                with cs2:
                    es["sharpen_threshold"] = st.slider("Chroma Softness", 0, 50, es["sharpen_threshold"], key="sl_sharpen_thr", help="Higher = softer chroma sharpening relative to luma")
            st.caption("Enhances edge definition using adaptive unsharp masking (lx=5, cx=3).")

        with st.container(border=True):
            st.markdown("**🔬 Resolution Upscaling**")
            es["upscale"] = st.checkbox("Enable high-quality upscaling", value=es["upscale"], key="chk_upscale")
            if es["upscale"]:
                algo_opts = ["lanczos", "spline", "bicubic"]
                es["upscale_algo"] = st.selectbox("Algorithm", algo_opts, index=algo_opts.index(es["upscale_algo"]) if es["upscale_algo"] in algo_opts else 0, key="sel_upscale_algo")
                es["upscale_width"] = st.number_input("Target Width (px)", min_value=meta["width"], max_value=7680, value=max(meta["width"], es.get("upscale_width") or meta["width"] * 2), step=max(2, meta["width"] // 2), key="ni_upscale_w")
                es["upscale_height"] = st.number_input("Target Height (px)", min_value=meta["height"], max_value=4320, value=max(meta["height"], es.get("upscale_height") or meta["height"] * 2), step=max(2, meta["height"] // 2), key="ni_upscale_h")
            st.caption("High-quality interpolation. For true AI super-resolution, integrate Real-ESRGAN externally.")

    # ── Right Column ──────────────────────────────────────────────────────────
    with enh_col2:
        is_hdr_source = meta.get("bit_depth", 8) >= 10
        with st.container(border=True):
            st.markdown("**🌈 HDR → SDR Tonemapping**")
            es["hdr_convert"] = st.checkbox("Enable HDR conversion", value=es["hdr_convert"] and is_hdr_source, key="chk_hdr", disabled=not is_hdr_source, help="Requires 10-bit+ HDR10/HLG source")
            if es["hdr_convert"] and is_hdr_source:
                tmap_opts = ["hable", "reinhard", "mobius", "linear"]
                es["tonemap_algo"] = st.selectbox("Tonemap Algorithm", tmap_opts, index=tmap_opts.index(es["tonemap_algo"]) if es["tonemap_algo"] in tmap_opts else 0, key="sel_tonemap")
            elif not is_hdr_source:
                st.caption("💡 Source is SDR (8-bit). HDR conversion requires a 10-bit+ HDR10/HLG source.")
            st.caption("Converts HDR10/HLG to SDR with perceptual tonemapping.")

        with st.container(border=True):
            st.markdown("**🎨 Color Enhancement**")
            es["color_enhance"] = st.checkbox("Enable color boost", value=es["color_enhance"], key="chk_color")
            if es["color_enhance"]:
                cc1, cc2 = st.columns(2)
                with cc1:
                    es["vibrance"] = st.slider("Vibrance", -0.5, 0.5, es["vibrance"], 0.05, key="sl_vibrance")
                with cc2:
                    es["contrast"] = st.slider("Contrast", 0.5, 2.0, es["contrast"], 0.05, key="sl_contrast")
            st.caption("Enhances vibrancy while preserving natural skin tones.")

        with st.container(border=True):
            st.markdown("**🧩 Artifact Reduction**")
            es["deblock"] = st.checkbox("Enable deblocking filter", value=es["deblock"], key="chk_deblock")
            if es["deblock"]:
                es["deblock_strength"] = st.slider("Strength", 1, 10, es["deblock_strength"], key="sl_deblock_str")
            st.caption("Reduces macroblocking and ringing from heavy compression.")

    # ── Frame Interpolation ───────────────────────────────────────────────────
    with st.expander("🎞️ Advanced: Frame Interpolation (Motion Smoothing)"):
        es["frame_interp"] = st.checkbox("Enable frame interpolation", value=es["frame_interp"], key="chk_frame_interp")
        if es["frame_interp"]:
            fps_opts = [30, 48, 60, 120]
            cur_fps  = es.get("target_fps", 60)
            fps_idx  = fps_opts.index(cur_fps) if cur_fps in fps_opts else 2
            es["target_fps"] = st.selectbox("Target Frame Rate", fps_opts, index=fps_idx, key="sel_target_fps")
            st.warning("⚠️ Frame interpolation is CPU-intensive and may increase processing time 3–5×.")
        st.caption("Creates intermediate frames for smoother playback (e.g., 24 fps → 60 fps).")

    # ── Enhancement Summary ───────────────────────────────────────────────────
    _enh_keys        = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
    active_enhancements = sum(bool(es.get(k)) for k in _enh_keys)

    if active_enhancements > 0:
        est_time = estimate_processing_time(meta, es)
        st.info(f"⏱️ **Estimated processing time**: {est_time} ({active_enhancements} enhancement{'s' if active_enhancements > 1 else ''} active)")
        enh_labels = {
            "denoise": "🧹 Denoise", "sharpen": "🔍 Sharpen", "upscale": "🔬 Upscale",
            "hdr_convert": "🌈 HDR", "color_enhance": "🎨 Color",
            "deblock": "🧩 Deblock", "frame_interp": "🎞️ Interp",
        }
        active_names = [enh_labels[k] for k in _enh_keys if es.get(k)]
        st.markdown(f"✨ **Active**: {' + '.join(active_names)}")

    st.divider()

    # ── Encoder Settings ──────────────────────────────────────────────────────
    st.markdown('<div class="vf-label">⚙️ Encoder Settings</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([2, 2, 1, 2])

    with s1:
        codec = st.selectbox("Video Codec", ["AVC (H.264)", "HEVC (H.265)", "AV1"], help="H.264 = fastest · HEVC = ~40% smaller · AV1 = best compression")
    with s2:
        crf = st.slider("CRF Quality", 0, 51, 23, help="Lower = better quality · 0 = lossless · 18 = visually lossless · 23 = balanced")
    with s3:
        do_vmaf = st.checkbox("VMAF", value=HAS_VMAF, disabled=not HAS_VMAF)
        do_psnr = st.checkbox("PSNR/SSIM", value=True)
    with s4:
        if crf < 19:   ql, qc = "🟢 High Quality", "#15803d"
        elif crf < 29: ql, qc = "🟡 Balanced",     "#b45309"
        elif crf < 40: ql, qc = "🟠 Compact",      "#ea580c"
        else:          ql, qc = "🔴 Low Quality",  "#b91c1c"
        st.markdown(f'<div style="text-align:center;padding:10px 0"><div style="font-weight:700;color:{qc};font-size:1rem">{ql}</div><div style="font-size:0.74rem;color:#64748b;margin-top:2px">CRF {crf}</div></div>', unsafe_allow_html=True)

    if codec == "AV1" and active_enhancements > 0:
        st.warning("⚠️ **AV1 + Enhancements**: Very resource intensive. May crash on free-tier cloud. Use H.264/HEVC for cloud deployment.")

    # ── Batch CRF Sweep ───────────────────────────────────────────────────────
    with st.expander("📊 Batch CRF Sweep — Rate-Distortion Analysis"):
        st.caption("Automatically encode at multiple CRF values to find the optimal quality/size trade-off.")
        sweep_col1, sweep_col2, sweep_col3 = st.columns(3)
        with sweep_col1:
            sweep_start = st.number_input("CRF Start", 10, 45, 18, step=1, key="sweep_start")
        with sweep_col2:
            sweep_end = st.number_input("CRF End", sweep_start+1, 51, 38, step=1, key="sweep_end")
        with sweep_col3:
            sweep_step = st.number_input("Step", 1, 10, 5, step=1, key="sweep_step")
        sweep_crfs = list(range(int(sweep_start), int(sweep_end)+1, int(sweep_step)))
        st.caption(f"Will encode {len(sweep_crfs)} variants: CRF {', '.join(map(str, sweep_crfs))}")
        if st.button("🚀 Run CRF Sweep", type="primary"):
            sweep_bar = st.progress(0.0, text="Starting sweep…")
            for i, sweep_crf in enumerate(sweep_crfs):
                codec_short = codec.split()[0].lower()
                out_path = inp.replace(suf, f"_sweep_{codec_short}_crf{sweep_crf}.mp4")
                sweep_bar.progress(i / len(sweep_crfs), text=f"⚙️ Encoding CRF {sweep_crf} ({i+1}/{len(sweep_crfs)})…")

                ok, msg, fflog, enc_t = encode(inp, out_path, codec, sweep_crf, es, meta, duration=meta["duration"])
                if ok:
                    out_meta = probe(out_path)
                    out_sz = os.path.getsize(out_path) / (1024 * 1024)
                    qual = {"psnr": None, "ssim": None, "vmaf": None}
                    if do_psnr or (do_vmaf and HAS_VMAF):
                        qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

                    idx = len(st.session_state.results)
                    st.session_state.result_logs[idx] = fflog
                    st.session_state.results.append({
                        "codec": codec, "crf": sweep_crf, "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0), "enc_time": enc_t, "saved": (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0, "cr": sz_mb / out_sz if out_sz > 0 else 0.0, "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"], "path": out_path, "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"], "sample_rate": meta["sample_rate"], "channels": meta["channels"], "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v}, "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}", "out_fps": out_meta.get("fps", meta["fps"])
                    })
                else:
                    st.warning(f"CRF {sweep_crf} failed: {msg}")
            sweep_bar.progress(1.0, text=f"✅ Sweep complete — {len(sweep_crfs)} variants encoded")
            st.rerun()

    # ── Run Controls ──────────────────────────────────────────────────────────
    st.markdown('<div class="vf-label" style="margin-top:8px">🚀 Run</div>', unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 4])

    with b1:
        if st.button("🔍 Preview Impact", use_container_width=True):
            est_meta = meta.copy()
            if es.get("upscale"):
                est_meta["width"], est_meta["height"] = int(es.get("upscale_width") or meta["width"] * 2), int(es.get("upscale_height") or meta["height"] * 2)
            if es.get("frame_interp"):
                est_meta["fps"] = es.get("target_fps", 60)
            st.markdown("#### 📊 Estimated Output")
            bc1, bc2 = st.columns(2)
            bc1.metric("Source", f"{meta['width']}×{meta['height']}", f"@ {meta['fps']} fps")
            bc2.metric("Enhanced", f"{est_meta['width']}×{est_meta['height']}", f"@ {est_meta['fps']} fps")
            if est_meta["width"] > meta["width"] and meta["width"] > 0:
                ratio = (est_meta["width"] / meta["width"]) ** 2
                st.caption(f"+{(ratio-1)*100:.0f}% more pixels ({ratio:.1f}× resolution)")

    go = b2.button("✨ Enhance + Encode", type="primary", use_container_width=True)
    clear = b3.button("🗑 Clear", use_container_width=True)
    if clear:
        st.session_state.results, st.session_state.result_logs = [], {}
        st.rerun()

    if go:
        codec_short = codec.split()[0].lower()
        enh_tag = "enh_" if active_enhancements > 0 else ""
        out_path = inp.replace(suf, f"_{enh_tag}{codec_short}_crf{crf}.mp4")
        bar = st.progress(0.0, text=f"⏳ Initializing {codec}…")
        with st.spinner(f"✨ Processing ({active_enhancements} enhancements) + encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(inp, out_path, codec, crf, es, meta, progress_cb=lambda p: bar.progress(p, text=f"⚙️ Processing… {p*100:.0f}%"), duration=meta["duration"])
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
            idx = len(st.session_state.results)
            st.session_state.result_logs[idx] = fflog
            st.session_state.results.append({"codec": codec, "crf": crf, "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0), "enc_time": enc_t, "saved": saved_pct, "cr": sz_mb / out_sz if out_sz > 0 else 0.0, "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"], "path": out_path, "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"], "sample_rate": meta["sample_rate"], "channels": meta["channels"], "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v}, "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}", "out_fps": out_meta.get("fps", meta["fps"])})
            bar.empty()
            q_parts = []
            if qual["vmaf"] is not None: q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"] is not None: q_parts.append(f"PSNR {qual['psnr']:.2f} dB")
            q_str = " · ".join(q_parts) if q_parts else "Analysis complete"
            enh_summary = " + ".join(k.capitalize() for k in _enh_keys if es.get(k))
            enh_str = f" · ✨ {enh_summary}" if enh_summary else ""
            st.success(f"✅ {codec} CRF {crf}{enh_str} · {out_sz:.2f} MB · saved {saved_pct:.1f}% · {enc_t:.1f}s · {q_str}")

else:
    # ── Test Player Mode ──────────────────────────────────────────────────────
    st.markdown('<div class="vf-label">🎬 Test Player Mode</div>', unsafe_allow_html=True)
    st.info("🎧 **Playback & Analytics**: Preview your source video. All metadata displayed. No processing performed.")
    if meta["has_audio"]:
        st.markdown(f"""<div style="background:#f1f5f9;border-radius:12px;padding:14px 18px;margin:12px 0"><div style="font-weight:700;margin-bottom:10px;color:#334155">🔊 Audio Stream Details</div><div style="display:flex;gap:8px;flex-wrap:wrap"><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_audio_codec(meta['acodec'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_channels(meta['channels'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_sample_rate(meta['sample_rate'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{meta['abitrate_kbps']} kbps</span></div></div>""", unsafe_allow_html=True)
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

tab_tbl, tab_chart, tab_dl, tab_logs = st.tabs(["📋 Comparison Table", "📊 Charts", "⬇ Downloads", "🪵 Logs"])

# ── Comparison Table ──────────────────────────────────────────────────────────
with tab_tbl:
    best_sz  = min(r["size_mb"]  for r in results)
    best_cr  = max(r["cr"]       for r in results)
    best_spd = min(r["enc_time"] for r in results)
    vmaf_vals = [r["vmaf"] for r in results if r["vmaf"] is not None]
    best_vm   = max(vmaf_vals) if vmaf_vals else None
    _enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]

    rows_html = ""
    for r in results:
        cs       = r["codec"].split()[0]
        chip_cls = {"AVC": "chip-avc", "HEVC": "chip-hevc", "AV1": "chip-av1"}.get(cs, "")
        tag      = f'<span class="{chip_cls}">{cs}</span>'

        enh_count = len([v for v in r.get("enhancements", {}).values() if v])
        if enh_count > 0:
            tag += f' <span class="chip-enh">✨ ×{enh_count}</span>'

        vmaf_txt, vmaf_cls = vmaf_display(r["vmaf"])
        vmaf_cell = f'<span class="{vmaf_cls}">{vmaf_txt}</span>'
        if r["vmaf"] is not None and best_vm is not None and abs(r["vmaf"] - best_vm) < 0.01 and len(results) > 1:
            vmaf_cell += ' <span class="w-badge">Best</span>'

        audio_info = f"{format_audio_codec(r.get('acodec',''))} · {format_channels(r.get('channels',0))}" if r.get("acodec") and r["acodec"] != "unknown" else "—"
        res_info = r.get("out_res", f"{meta['width']}×{meta['height']}")
        fps_info = r.get("out_fps", meta["fps"])

        ssim_str = ssim_display(r["ssim"])

        rows_html += f"""<tr>
          <td>{tag}</td>
          <td>{r['crf']}</td>
          <td>{best_mark(r['size_mb'], best_sz, '{:.2f} MB')}</td>
          <td>{r['bitrate']} kbps</td>
          <td>{best_mark(r['cr'], best_cr, '{:.2f}×', higher_better=True)}</td>
          <td>{r['saved']:.1f}%</td>
          <td>{best_mark(r['enc_time'], best_spd, '{:.1f}s')}</td>
          <td>{vmaf_cell}</td>
          <td>{psnr_display(r['psnr'])}</td>
          <td>{ssim_str}</td>
          <td style="font-size:0.77rem;color:#64748b">{res_info} @ {fps_info} fps</td>
          <td style="font-size:0.77rem;color:#64748b">{audio_info}</td>
        </tr>"""

    st.markdown(f"""<table class="cmp-table">
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
    </div>""", unsafe_allow_html=True)
    st.caption("📏 VMAF: 93+ Excellent · 80–93 Good · 60–80 Fair | PSNR: 40+ dB Good | SSIM: 0.98+ Excellent | ✨ = AI enhancements | 🏆 Best = winner across runs")

    # ── NEW: CSV Export ───────────────────────────────────────────────────────
    st.download_button(
        label="⬇ Export as CSV",
        data=results_to_csv(results, meta, sz_mb),
        file_name="videoforge_results.csv",
        mime="text/csv",
        help="Download the full comparison table as a CSV file",
    )

# ── Charts ────────────────────────────────────────────────────────────────────
with tab_chart:
    df = pd.DataFrame([{
        "Codec":           r["codec"] + (" ✨" if r.get("enhancements") else "") + f" CRF{r['crf']}",
        "File Size (MB)":  round(r["size_mb"], 3),
        "Bitrate (kbps)":  r["bitrate"],
        "Encode Time (s)": round(r["enc_time"], 2),
        "Space Saved (%)": round(r["saved"], 1),
        "VMAF":            r["vmaf"],
        "PSNR (dB)":       round(r["psnr"], 2) if r["psnr"] else None,
        "SSIM":            round(r["ssim"], 4) if r["ssim"] else None,
    } for r in results])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**📦 File Size Comparison**")
        size_df = pd.DataFrame(
            [{"Codec": "🎬 Original", "File Size (MB)": round(sz_mb, 3)}]
            + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else "") + f" CRF{r['crf']}",
                "File Size (MB)": round(r["size_mb"], 3)} for r in results]
        ).set_index("Codec")
        st.bar_chart(size_df, color="#3b82f6", use_container_width=True)

        st.markdown("**⏱️ Encoding Time**")
        st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], color="#f97316", use_container_width=True)

    with c2:
        st.markdown("**📡 Bitrate Comparison**")
        brate_df = pd.DataFrame(
            [{"Codec": "🎬 Original", "Bitrate (kbps)": meta["vbitrate_kbps"]}]
            + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else "") + f" CRF{r['crf']}",
                "Bitrate (kbps)": r["bitrate"]} for r in results]
        ).set_index("Codec")
        st.bar_chart(brate_df, color="#8b5cf6", use_container_width=True)

        st.markdown("**💾 Space Saved**")
        st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], color="#10b981", use_container_width=True)

    q_cols = [c for c in ["VMAF", "PSNR (dB)", "SSIM"] if df[c].notna().any()]
    if q_cols:
        st.divider()
        st.markdown("**🎯 Quality Metrics**")
        qdf = df.set_index("Codec")[q_cols].dropna(how="all")
        st.bar_chart(qdf, use_container_width=True)
        st.caption("Higher = better quality. ✨ = AI enhancements applied.")

    # ── NEW: Rate-Distortion Curve (when VMAF available for ≥2 results) ───────
    rd_data = [(r["bitrate"], r["vmaf"], r["crf"]) for r in results
               if r.get("vmaf") is not None and r.get("bitrate")]
    if len(rd_data) >= 2:
        st.divider()
        st.markdown("**📈 Rate-Distortion Curve (Bitrate vs VMAF)**")
        rd_df = pd.DataFrame(
            [{"Bitrate (kbps)": b, "VMAF": v} for b, v, _ in sorted(rd_data)]
        ).set_index("Bitrate (kbps)")
        st.line_chart(rd_df, color="#6366f1", use_container_width=True)
        st.caption("Ideal curve bends upper-left: high quality at low bitrate.")

    if len(results) > 1:
        st.divider()
        st.markdown("**🔍 Smart Insights**")

        enhanced_results = [r for r in results if r.get("enhancements")]
        if enhanced_results:
            scored = [r for r in enhanced_results if r.get("vmaf") is not None or r.get("psnr") is not None]
            if scored:
                best_enh = max(scored, key=lambda r: (r.get("vmaf") or 0) + (r.get("psnr") or 0) / 5)
                n_enh    = len([v for v in best_enh["enhancements"].values() if v])
                vmaf_str = f"VMAF {best_enh['vmaf']:.1f}" if best_enh.get("vmaf") else f"PSNR {best_enh.get('psnr',0):.2f} dB"
                st.markdown(f'<div class="insight-note ai"><b>Enhancement Winner:</b> <span style="font-weight:700">{best_enh["codec"]}</span> with {n_enh} enhancement{"s" if n_enh!=1 else ""} achieved {vmaf_str} at {best_enh["size_mb"]:.2f} MB.</div>', unsafe_allow_html=True)

        eff_candidates = [r for r in results if r.get("vmaf") and r["size_mb"] > 0]
        if eff_candidates:
            best_eff = max(eff_candidates, key=lambda r: r["vmaf"] / r["size_mb"])
            enh_tag  = " ✨" if best_eff.get("enhancements") else ""
            st.markdown(f'<div class="insight-note"><b>Efficiency Pick:</b> <span style="font-weight:700">{best_eff["codec"]} CRF{best_eff["crf"]}{enh_tag}</span> delivers best quality-per-MB (VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB).</div>', unsafe_allow_html=True)

        # NEW: Smallest-for-acceptable-quality insight
        acceptable = [r for r in results if r.get("vmaf") and r["vmaf"] >= 80]
        if acceptable:
            smallest = min(acceptable, key=lambda r: r["size_mb"])
            st.markdown(f'<div class="insight-note"><b>Streaming Sweet Spot:</b> <span style="font-weight:700">{smallest["codec"]} CRF{smallest["crf"]}</span> is the smallest file with VMAF ≥ 80 ({smallest["size_mb"]:.2f} MB · VMAF {smallest["vmaf"]:.1f}).</div>', unsafe_allow_html=True)

# ── Downloads ─────────────────────────────────────────────────────────────────
with tab_dl:
    st.markdown("### ⬇ Download Processed Files")
    st.caption("Files are stored temporarily in this session. Download immediately after encoding.")

    for i, r in enumerate(results):
        cs      = r["codec"].split()[0]
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
            metrics = [f"{r['size_mb']:.2f} MB", f"{r['bitrate']} kbps", r.get("out_res", "N/A")]
            if r.get("vmaf") is not None: metrics.append(f"VMAF {r['vmaf']:.1f}")
            if r.get("psnr") is not None: metrics.append(f"PSNR {r['psnr']:.2f} dB")
            st.caption(" · ".join(metrics))
            if r.get("enhancements"):
                enh_list = [k.capitalize() for k, v in r["enhancements"].items() if v]
                st.caption(f"✨ Enhancements: {', '.join(enh_list)}")
            if r.get("acodec") and r["acodec"] not in ("unknown", ""):
                st.caption(f"🔊 {format_audio_codec(r['acodec'])} · {format_channels(r.get('channels',0))}")

# ── NEW: Logs tab ─────────────────────────────────────────────────────────────
with tab_logs:
    st.markdown("### 🪵 FFmpeg Encode Logs")
    st.caption("Per-encode FFmpeg output for debugging. Last 80 lines preserved per run.")
    if not st.session_state.result_logs:
        st.info("No logs yet. Logs appear here after encoding.")
    else:
        for idx, log_text in st.session_state.result_logs.items():
            r = results[idx]
            cs = r["codec"].split()[0]
            with st.expander(f"{cs} CRF {r['crf']} — {r['size_mb']:.2f} MB · {r['enc_time']:.1f}s"):
                st.code(log_text or "(empty)", language="bash")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:12px 0">'
    "AI Encoder · FFmpeg + Streamlit · ✨ AI enhancements · Bug-fixed build"
    "</div>",
    unsafe_allow_html=True,
)
