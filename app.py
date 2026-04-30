"""
VideoForge Web – Encoder + VMAF Analytics + AI Video Enhancement + Real-Time Player Performance
Run:  streamlit run app.py

Deploy:
  Streamlit Community Cloud → packages.txt: ffmpeg
  HF Spaces (Streamlit SDK)  → same packages.txt

Changelog (bug-fixes + enhancements):
  BUG  1 – quality_metrics: fixed PSNR/SSIM lavfi filter graph so both filters
            run independently with their own null sinks.
  BUG  2 – unsharp filter: corrected to 6-param form `lx:ly:la:cx:cy:ca`.
  BUG  3 – best_mark(): higher_better flag was declared but never used.
  BUG  4 – SSE SSIM filter mapped output fixed to use nullsink.
  BUG  5 – deblock alpha/beta mapping range comment clarified.
  BUG  6 – encode() AV1 `-crf` flag: added `-b:v 0` to enable CRF mode.
  NEW  1 – Batch CRF Sweep: encode at a range of CRFs and plot rate-distortion.
  NEW  2 – CSV Export: download the full comparison table as a CSV file.
  NEW  3 – Per-result expandable FFmpeg log preserved in session.
  NEW  4 – Enhancement diff preview: before/after resolution & FPS side by side.
  NEW  5 – Color-coded SSIM display.
  NEW  6 – Audio loudness probe via `ffmpeg -af volumedetect`.
  NEW  7 – Quality Radar chart (VMAF / PSNR / SSIM normalised) when ≥2 results.
  NEW  8 – Real-Time Player Performance Panel: adaptive bitrate dashboard with
            bandwidth history, buffer level, bitrate ladder, ABR decisions log,
            RTT, drop events, and guardrail toggles — all rendered as a live
            HTML/JS component inside Streamlit.
"""

import os, io, csv, json, subprocess, tempfile, time, re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

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
    """Measure integrated loudness + true peak via ffmpeg volumedetect."""
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

    if settings.get("denoise"):
        strength = float(settings.get("denoise_strength", 5))
        s2 = round(strength / 2, 1)
        filters.append(f"hqdn3d={strength}:{strength}:{s2}:{s2}")

    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5))
        alpha = round(strength / 10.0, 2)
        beta  = round(alpha * 0.5, 2)
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")

    if settings.get("sharpen"):
        amount    = float(settings.get("sharpen_amount", 0.5))
        c_amount  = round(amount * 0.5, 2)
        filters.append(f"unsharp=lx=5:ly=5:la={amount}:cx=3:cy=3:ca={c_amount}")

    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15))
        contrast = float(settings.get("contrast", 1.0))
        sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")

    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.append("zscale=transfer=linear,format=gbrpf32le")
        filters.append(f"tonemap={settings.get('tonemap_algo','hable')}:desat=0.2")
        filters.append("zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le")

    if settings.get("upscale"):
        target_w = int(settings.get("upscale_width",  src_meta["width"]  * 2))
        target_h = int(settings.get("upscale_height", src_meta["height"] * 2))
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        target_h = target_h if target_h % 2 == 0 else target_h - 1
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(
            f"scale={target_w}:{target_h}:flags={algo}+accurate_rnd+full_chroma_int"
        )

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
            -6:  "OOM — AI enhancements need extra RAM.",
            -9:  "OOM (SIGKILL) — reduce enhancement complexity.",
            -11: "Segfault — check filter compatibility.",
            1:   "FFmpeg error — see log below.",
        }
        return False, hints.get(proc.returncode, f"FFmpeg exited with code {proc.returncode}"), log, elapsed

    except FileNotFoundError:
        return False, "FFmpeg not found. Add `ffmpeg` to packages.txt.", "", 0.0
    except Exception as e:
        return False, str(e), "\n".join(lines), time.time() - t0


def quality_metrics(ref: str, dist: str, do_vmaf: bool) -> dict:
    """Compute PSNR, SSIM and optionally VMAF via FFmpeg."""
    res = {"psnr": None, "ssim": None, "vmaf": None}

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
    if v is None:
        return "—", ""
    if v >= 93: return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80: return f"{v:.1f} · Good",      "q-gd"
    if v >= 60: return f"{v:.1f} · Fair",       "q-ok"
    return           f"{v:.1f} · Poor",          "q-bad"

def ssim_display(v) -> str:
    if v is None:
        return "—"
    label = (
        "Excellent" if v >= 0.98 else
        "Good"      if v >= 0.95 else
        "Fair"      if v >= 0.90 else
        "Poor"
    )
    return f"{v:.5f} · {label}"

def psnr_display(v):
    if v is None:
        return "—"
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
    if val is None or best is None:
        return "—"
    s = fmt.format(val)
    is_best = abs(val - best) < 0.01 and len(st.session_state.results) > 1
    if is_best:
        return f'<span class="best-val">{s} <span class="w-badge">Best</span></span>'
    return s


def results_to_csv(results: list, src_meta: dict, sz_mb: float) -> bytes:
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
#  Player Performance Panel HTML Component
# ══════════════════════════════════════════════════════════════════════════════

def build_player_performance_html(meta: dict, sz_mb: float) -> str:
    """
    Build the real-time player performance dashboard as a self-contained
    HTML/JS component. Simulates live ABR (adaptive bitrate) analytics based
    on actual source metadata, with animated updates every second.
    """
    width  = meta.get("width", 1920)
    height = meta.get("height", 1080)
    fps    = meta.get("fps", 30.0)
    bitrate_kbps = meta.get("vbitrate_kbps", 4000) or 4000
    duration = meta.get("duration", 154.0)

    # Build a realistic bitrate ladder from source resolution
    def ladder_from_res(w, h, bps):
        rungs = []
        if h >= 2160 or w >= 3840:
            rungs = [("4K", 16000), ("1080p", 8000), ("720p", 4000), ("540p", 2000), ("480p", 1200), ("360p", 600)]
        elif h >= 1080 or w >= 1920:
            rungs = [("1080p", 8000), ("720p", 4000), ("540p", 2000), ("480p", 1200), ("360p", 600)]
        elif h >= 720 or w >= 1280:
            rungs = [("720p", 4000), ("540p", 2000), ("480p", 1200), ("360p", 600)]
        else:
            rungs = [("480p", 1200), ("360p", 600), ("240p", 300)]
        # Label the current/active rung as the one matching source bitrate best
        return rungs

    ladder = ladder_from_res(width, height, bitrate_kbps)
    # active rung = closest bitrate to source
    active_idx = min(range(len(ladder)), key=lambda i: abs(ladder[i][1] - bitrate_kbps))
    active_label = ladder[active_idx][0]
    active_bps   = ladder[active_idx][1]
    max_bps = ladder[0][1]

    ladder_js = json.dumps([{"label": l, "bps": b} for l, b in ladder])
    active_js = active_label
    duration_js = max(int(duration), 10)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #f7f7f5;
    color: #1a1a1a;
    font-size: 13px;
  }}
  .panel {{
    background: #f7f7f5;
    padding: 0;
    min-height: 100vh;
  }}
  .panel-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px 12px;
    border-bottom: 1px solid #e8e8e5;
    background: #ffffff;
  }}
  .panel-title {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: #1a1a1a;
  }}
  .status-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.6; transform: scale(0.85); }}
  }}
  .explain-btn {{
    font-size: 12px; color: #3b82f6; cursor: pointer;
    background: none; border: 1px solid #dbeafe;
    border-radius: 6px; padding: 4px 10px; font-weight: 500;
    display: flex; align-items: center; gap: 4px;
  }}
  .explain-btn:hover {{ background: #eff6ff; }}

  /* ── Metrics grid ─────────────────────────────────────── */
  .metrics-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    padding: 14px 16px;
  }}
  .metric-card {{
    background: #ffffff;
    border: 1px solid #e8e8e5;
    border-radius: 10px;
    padding: 13px 16px;
  }}
  .metric-label {{
    font-size: 11px;
    color: #888;
    font-weight: 500;
    margin-bottom: 4px;
    letter-spacing: 0.01em;
  }}
  .metric-value {{
    font-size: 26px;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1;
    letter-spacing: -0.03em;
  }}
  .metric-value .unit {{
    font-size: 14px;
    font-weight: 500;
    color: #888;
    margin-left: 2px;
  }}
  .metric-sub {{
    font-size: 11px;
    margin-top: 5px;
    font-weight: 500;
  }}
  .sub-good  {{ color: #16a34a; }}
  .sub-warn  {{ color: #ca8a04; }}
  .sub-bad   {{ color: #dc2626; }}
  .sub-neutral {{ color: #64748b; }}

  /* ── Buffer level ─────────────────────────────────────── */
  .section {{
    padding: 0 16px 14px;
  }}
  .section-label {{
    font-size: 12px;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 8px;
  }}
  .buffer-track {{
    background: #e8e8e5;
    border-radius: 6px;
    height: 10px;
    position: relative;
    overflow: hidden;
  }}
  .buffer-fill {{
    height: 100%;
    border-radius: 6px;
    background: #22c55e;
    transition: width 0.8s ease;
    position: relative;
  }}
  .buffer-fill::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.25) 50%, transparent 100%);
    animation: shimmer 2s ease-in-out infinite;
    background-size: 200% 100%;
  }}
  @keyframes shimmer {{
    0%   {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
  }}
  .buffer-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: #888;
    margin-top: 4px;
  }}
  .guard-marker {{
    color: #f59e0b;
    font-weight: 600;
  }}

  /* ── Bitrate ladder ───────────────────────────────────── */
  .ladder-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
  }}
  .ladder-badge {{
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 2px 8px;
    min-width: 46px;
    text-align: center;
  }}
  .ladder-badge.active {{
    background: #dbeafe;
    border-color: #93c5fd;
    color: #1e40af;
  }}
  .ladder-bar-track {{
    flex: 1;
    background: #f1f5f9;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
  }}
  .ladder-bar-fill {{
    height: 100%;
    border-radius: 4px;
    background: #cbd5e1;
    transition: width 1s ease;
  }}
  .ladder-bar-fill.active {{
    background: #22c55e;
  }}
  .ladder-bps {{
    font-size: 11px;
    color: #64748b;
    min-width: 38px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}

  /* ── Bandwidth history chart ──────────────────────────── */
  .chart-wrap {{
    position: relative;
    height: 72px;
    margin: 0 16px 14px;
  }}
  canvas {{
    width: 100%;
    height: 100%;
    border-radius: 6px;
  }}

  /* ── ABR decisions ────────────────────────────────────── */
  .decisions-list {{
    max-height: 130px;
    overflow-y: auto;
    border: 1px solid #e8e8e5;
    border-radius: 8px;
    background: #ffffff;
  }}
  .decisions-list::-webkit-scrollbar {{ width: 4px; }}
  .decisions-list::-webkit-scrollbar-thumb {{ background: #e2e8f0; border-radius: 2px; }}
  .decision-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 12px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 12px;
    animation: fadeIn 0.3s ease;
  }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(-4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .decision-row:last-child {{ border-bottom: none; }}
  .dec-time {{ color: #888; min-width: 32px; font-variant-numeric: tabular-nums; font-size: 11px; }}
  .dec-arrow {{ font-size: 11px; }}
  .dec-up   {{ color: #16a34a; }}
  .dec-down {{ color: #dc2626; }}
  .dec-label {{ font-weight: 500; }}

  /* ── Guardrails ───────────────────────────────────────── */
  .guardrails-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 0 16px 16px;
  }}
  .guardrail-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid #e8e8e5;
    border-radius: 8px;
    padding: 9px 12px;
  }}
  .guardrail-label {{ font-size: 12px; font-weight: 500; color: #1a1a1a; }}
  .toggle {{
    width: 36px; height: 20px;
    background: #3b82f6;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    transition: background 0.2s;
    flex-shrink: 0;
  }}
  .toggle.off {{ background: #e2e8f0; }}
  .toggle::after {{
    content: '';
    position: absolute;
    top: 2px; left: 2px;
    width: 16px; height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
  }}
  .toggle:not(.off)::after {{ transform: translateX(16px); }}

  /* Divider */
  hr {{ border: none; border-top: 1px solid #e8e8e5; margin: 0 0 14px; }}
</style>
</head>
<body>
<div class="panel">

  <!-- Header -->
  <div class="panel-header">
    <div class="panel-title">
      <div class="status-dot" id="statusDot"></div>
      <span id="abrStatus">adaptive bitrate — stable</span>
    </div>
    <button class="explain-btn" onclick="toggleExplain()">explain ↗</button>
  </div>

  <!-- Metrics grid -->
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">bandwidth</div>
      <div class="metric-value"><span id="bwVal">4.9</span><span class="unit">Mbps</span></div>
      <div class="metric-sub sub-good" id="bwSub">+ above threshold</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">buffer health</div>
      <div class="metric-value"><span id="bufVal">20.0</span><span class="unit">s</span></div>
      <div class="metric-sub sub-good" id="bufSub">healthy</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">RTT</div>
      <div class="metric-value"><span id="rttVal">129</span><span class="unit">ms</span></div>
      <div class="metric-sub sub-warn" id="rttSub">high</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">drop events</div>
      <div class="metric-value" id="dropVal">0</div>
      <div class="metric-sub sub-neutral" id="dropSub">— no rebuffer</div>
    </div>
  </div>

  <!-- Buffer level -->
  <div class="section">
    <div class="section-label">buffer level</div>
    <div class="buffer-track">
      <div class="buffer-fill" id="bufferBar" style="width:100%"></div>
    </div>
    <div class="buffer-labels">
      <span>0s</span>
      <span class="guard-marker">guard: 4s</span>
      <span id="bufMax">20s</span>
    </div>
  </div>

  <hr>

  <!-- Bitrate ladder -->
  <div class="section">
    <div class="section-label">bitrate ladder</div>
    <div id="ladderContainer"></div>
  </div>

  <hr>

  <!-- Bandwidth history chart -->
  <div class="section">
    <div class="section-label">bandwidth history (30s)</div>
  </div>
  <div class="chart-wrap">
    <canvas id="bwChart"></canvas>
  </div>

  <hr>

  <!-- ABR decisions -->
  <div class="section">
    <div class="section-label">abr decisions</div>
    <div class="decisions-list" id="decisionsList"></div>
  </div>

  <hr style="margin-top:8px">

  <!-- Guardrails -->
  <div class="section" style="padding-bottom:4px">
    <div class="section-label">guardrails</div>
  </div>
  <div class="guardrails-grid">
    <div class="guardrail-row">
      <span class="guardrail-label">Buffer guard</span>
      <div class="toggle" id="tgl-buffer" onclick="toggleGuardrail(this)"></div>
    </div>
    <div class="guardrail-row">
      <span class="guardrail-label">Stability lock</span>
      <div class="toggle" id="tgl-stability" onclick="toggleGuardrail(this)"></div>
    </div>
    <div class="guardrail-row">
      <span class="guardrail-label">Fast downgrade</span>
      <div class="toggle" id="tgl-fast-down" onclick="toggleGuardrail(this)"></div>
    </div>
    <div class="guardrail-row">
      <span class="guardrail-label">Slow upgrade</span>
      <div class="toggle" id="tgl-slow-up" onclick="toggleGuardrail(this)"></div>
    </div>
  </div>

</div>

<script>
// ── Config from Python ─────────────────────────────────────────────────────
const LADDER   = {ladder_js};
const ACTIVE   = "{active_js}";
const SRC_BPS  = {active_bps};
const MAX_BPS  = {max_bps};
const DURATION = {duration_js};

// ── State ──────────────────────────────────────────────────────────────────
let bwHistory  = [];      // last 30 readings
let bufHistory = [];
let currentRung = ACTIVE;
let drops = 0;
let decisions = [];
let playTime = 0;
let tickCount = 0;
let explained = false;

// Pre-fill 30s of history with realistic noise
for (let i = 0; i < 30; i++) {{
  const base = SRC_BPS / 1000 * 1.2;
  bwHistory.push(base + (Math.random() - 0.5) * base * 0.3);
  bufHistory.push(18 + (Math.random() - 0.5) * 4);
}}

// ── Helpers ────────────────────────────────────────────────────────────────
function fmtTime(s) {{
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}}

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

// ── Render bitrate ladder ──────────────────────────────────────────────────
function renderLadder() {{
  const cont = document.getElementById('ladderContainer');
  cont.innerHTML = '';
  for (const rung of LADDER) {{
    const isActive = rung.label === currentRung;
    const fillPct  = (rung.bps / LADDER[0].bps * 100).toFixed(1);
    cont.innerHTML += `
      <div class="ladder-row">
        <div class="ladder-badge ${{isActive ? 'active' : ''}}">${{rung.label}}</div>
        <div class="ladder-bar-track">
          <div class="ladder-bar-fill ${{isActive ? 'active' : ''}}" style="width:${{fillPct}}%"></div>
        </div>
        <div class="ladder-bps">${{rung.bps >= 1000 ? (rung.bps/1000).toFixed(1)+'M' : rung.bps+'K'}}</div>
      </div>`;
  }}
}}

// ── Render ABR decisions ───────────────────────────────────────────────────
function renderDecisions() {{
  const cont = document.getElementById('decisionsList');
  const visible = decisions.slice(-8).reverse();
  cont.innerHTML = visible.map(d => `
    <div class="decision-row">
      <span class="dec-time">${{fmtTime(d.t)}}</span>
      <span class="dec-arrow ${{d.up ? 'dec-up' : 'dec-down'}}">${{d.up ? '↑' : '↓'}}</span>
      <span class="dec-label">${{d.up ? 'upgrade' : 'downgrade'}} → ${{d.to}}</span>
    </div>`).join('');
}}

// ── Draw bandwidth history sparkline ──────────────────────────────────────
function drawChart() {{
  const canvas = document.getElementById('bwChart');
  const dpr    = window.devicePixelRatio || 1;
  const rect   = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  const vals   = bwHistory.slice(-30);
  const maxVal = Math.max(...vals) * 1.15;
  const minVal = Math.min(...vals) * 0.85;

  // Threshold line (dashed, amber)
  const threshY = H - ((SRC_BPS / 1000 - minVal) / (maxVal - minVal)) * H;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(0, threshY);
  ctx.lineTo(W, threshY);
  ctx.stroke();
  ctx.restore();

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(59,130,246,0.18)');
  grad.addColorStop(1, 'rgba(59,130,246,0)');

  ctx.beginPath();
  vals.forEach((v, i) => {{
    const x = (i / (vals.length - 1)) * W;
    const y = H - ((v - minVal) / (maxVal - minVal)) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  const endX = W, lastY = H - ((vals[vals.length-1] - minVal) / (maxVal - minVal)) * H;
  ctx.lineTo(endX, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  vals.forEach((v, i) => {{
    const x = (i / (vals.length - 1)) * W;
    const y = H - ((v - minVal) / (maxVal - minVal)) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  ctx.lineCap  = 'round';
  ctx.stroke();
}}

// ── ABR logic: decide rung upgrades/downgrades ─────────────────────────────
function abrDecide(bw_mbps) {{
  const bw_kbps = bw_mbps * 1000;
  const idx     = LADDER.findIndex(r => r.label === currentRung);
  let   newIdx  = idx;

  // Upgrade: if bandwidth comfortably exceeds next rung up
  if (idx > 0) {{
    const nextBps = LADDER[idx - 1].bps;
    if (bw_kbps > nextBps * 1.3) newIdx = idx - 1;
  }}
  // Downgrade: if bandwidth falls below current rung
  if (bw_kbps < LADDER[idx].bps * 0.85) {{
    newIdx = Math.min(idx + 1, LADDER.length - 1);
  }}

  if (newIdx !== idx) {{
    const up   = newIdx < idx;
    const to   = LADDER[newIdx].label;
    decisions.push({{ t: playTime, up, to }});
    currentRung = to;
    renderDecisions();
  }}
}}

// ── Main tick ──────────────────────────────────────────────────────────────
function tick() {{
  tickCount++;
  playTime += 1;

  // Simulate realistic bandwidth with occasional dips
  const prev   = bwHistory[bwHistory.length - 1];
  const base   = SRC_BPS / 1000 * 1.2;
  const jitter = (Math.random() - 0.48) * base * 0.25;
  let bw       = clamp(prev + jitter, base * 0.3, base * 2.2);
  // Occasional dip event
  if (Math.random() < 0.04) bw = base * (0.4 + Math.random() * 0.3);
  bwHistory.push(bw);
  if (bwHistory.length > 60) bwHistory.shift();

  // Buffer level (20s max, drains slowly, fills above threshold)
  const bufCur = bufHistory[bufHistory.length - 1];
  const bwRatio = bw / (SRC_BPS / 1000);
  const bufDelta = bwRatio > 1 ? 0.3 : -0.2;
  const buf = clamp(bufCur + bufDelta + (Math.random() - 0.5) * 0.15, 2, 20);
  bufHistory.push(buf);
  if (bufHistory.length > 60) bufHistory.shift();

  // RTT: 80–200ms with occasional spikes
  let rtt = 80 + Math.random() * 60;
  if (Math.random() < 0.05) rtt += 100 + Math.random() * 120;
  rtt = Math.round(rtt);

  // Drop events on very low buffer
  if (buf < 3.5 && Math.random() < 0.15) drops++;

  // ABR decision
  abrDecide(bw);

  // ── Update DOM ────────────────────────────────────────────────────────
  // Bandwidth
  document.getElementById('bwVal').textContent = bw.toFixed(1);
  const bwAbove = bw > SRC_BPS / 1000;
  document.getElementById('bwSub').textContent = bwAbove ? '+ above threshold' : '— below threshold';
  document.getElementById('bwSub').className   = 'metric-sub ' + (bwAbove ? 'sub-good' : 'sub-warn');

  // Buffer
  document.getElementById('bufVal').textContent = buf.toFixed(1);
  const bufOk = buf >= 8;
  document.getElementById('bufSub').textContent = buf < 4 ? 'critical' : buf < 8 ? 'low' : 'healthy';
  document.getElementById('bufSub').className   = 'metric-sub ' + (buf < 4 ? 'sub-bad' : buf < 8 ? 'sub-warn' : 'sub-good');
  document.getElementById('bufferBar').style.width = (buf / 20 * 100).toFixed(1) + '%';
  document.getElementById('bufferBar').style.background = buf < 4 ? '#ef4444' : buf < 8 ? '#f59e0b' : '#22c55e';

  // RTT
  document.getElementById('rttVal').textContent = rtt;
  document.getElementById('rttSub').textContent = rtt < 100 ? 'good' : rtt < 160 ? 'high' : 'very high';
  document.getElementById('rttSub').className   = 'metric-sub ' + (rtt < 100 ? 'sub-good' : rtt < 160 ? 'sub-warn' : 'sub-bad');

  // Drops
  document.getElementById('dropVal').textContent = drops;
  document.getElementById('dropSub').textContent = drops === 0 ? '— no rebuffer' : drops + ' rebuffer' + (drops > 1 ? 's' : '');
  document.getElementById('dropSub').className   = 'metric-sub ' + (drops === 0 ? 'sub-neutral' : 'sub-bad');

  // Status text
  const isStable = bwAbove && buf >= 8 && drops === 0;
  document.getElementById('abrStatus').textContent =
    isStable ? 'adaptive bitrate — stable' :
    buf < 4  ? 'adaptive bitrate — buffering' :
               'adaptive bitrate — adapting';
  document.getElementById('statusDot').style.background =
    isStable ? '#22c55e' : buf < 4 ? '#ef4444' : '#f59e0b';

  // Ladder + chart
  renderLadder();
  drawChart();
}}

// ── Toggle helpers ─────────────────────────────────────────────────────────
function toggleGuardrail(el) {{ el.classList.toggle('off'); }}
function toggleExplain() {{
  explained = !explained;
  alert(explained
    ? 'ABR (Adaptive Bitrate):\\n\\n• Bandwidth: measured download speed\\n• Buffer: seconds of video pre-loaded\\n• RTT: round-trip time to CDN\\n• Drop events: rebuffer stalls\\n• Bitrate ladder: available quality rungs\\n• ABR decisions: quality switch history'
    : '');
}}

// ── Resize handler ─────────────────────────────────────────────────────────
window.addEventListener('resize', drawChart);

// ── Boot ───────────────────────────────────────────────────────────────────
renderLadder();
renderDecisions();
drawChart();

// Seed initial decisions
[
  {{ t: 0, up: true,  to: '{active_js}' }},
  {{ t: 0, up: false, to: LADDER[Math.min(LADDER.findIndex(r=>r.label==='{active_js}')+1, LADDER.length-1)].label }},
  {{ t: 0, up: true,  to: '{active_js}' }},
].forEach(d => decisions.push(d));
renderDecisions();

setInterval(tick, 1000);
tick();
</script>
</body>
</html>"""
    return html


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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
      <h1>AI Encoder <span>· VideoForge</span></h1>
      <p>Professional encoding · AI enhancement · quality analytics · live player performance</p>
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
    <span class="vf-badge">📡 ABR Monitor</span>
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
#  Source Preview + Metadata + Player Performance Panel
# ══════════════════════════════════════════════════════════════════════════════

col_v, col_m, col_perf = st.columns([3, 2, 2], gap="large")

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

# ── NEW: Real-Time Player Performance Panel ───────────────────────────────────
with col_perf:
    st.markdown("**📡 Player Performance**")
    st.caption("Live ABR simulation based on source bitrate & resolution.")
    player_html = build_player_performance_html(meta, sz_mb)
    components.html(player_html, height=780, scrolling=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  AI ENHANCEMENT PANEL  (Encoder mode only)
# ══════════════════════════════════════════════════════════════════════════════

if enable_encoding:
    st.markdown('<div class="vf-label ai">✨ AI Video Enhancement</div>', unsafe_allow_html=True)

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
                "upscale_width":  meta["width"]  * 2,
                "upscale_height": meta["height"] * 2,
                "upscale_algo": "lanczos",
                "denoise": True, "denoise_strength": 6,
                "sharpen": True, "sharpen_amount": 0.5, "sharpen_threshold": 5,
                "hdr_convert": False, "frame_interp": False,
                "deblock": False, "color_enhance": False,
            })
            st.rerun()

    st.divider()

    es = st.session_state.enhance_settings
    enh_col1, enh_col2 = st.columns(2)

    with enh_col1:
        with st.container(border=True):
            st.markdown("**🧹 Denoise**")
            es["denoise"] = st.checkbox(
                "Enable temporal noise reduction", value=es["denoise"], key="chk_denoise",
                help="Reduces grain and compression artifacts using 3D filtering",
            )
            if es["denoise"]:
                es["denoise_strength"] = st.slider(
                    "Strength", 1, 10, es["denoise_strength"], key="sl_denoise_str",
                    help="Higher = more aggressive noise removal",
                )
            st.caption("Reduces grain, compression artifacts, and sensor noise.")

        with st.container(border=True):
            st.markdown("**🔍 Detail Enhancement**")
            es["sharpen"] = st.checkbox(
                "Enable adaptive sharpening", value=es["sharpen"], key="chk_sharpen",
            )
            if es["sharpen"]:
                cs1, cs2 = st.columns(2)
                with cs1:
                    es["sharpen_amount"] = st.slider(
                        "Amount", -1.5, 1.5, es["sharpen_amount"], 0.1, key="sl_sharpen_amt",
                    )
                with cs2:
                    es["sharpen_threshold"] = st.slider(
                        "Chroma Softness", 0, 50, es["sharpen_threshold"], key="sl_sharpen_thr",
                        help="Higher = softer chroma sharpening relative to luma",
                    )
            st.caption("Enhances edge definition using adaptive unsharp masking (lx=5, cx=3).")

        with st.container(border=True):
            st.markdown("**🔬 Resolution Upscaling**")
            es["upscale"] = st.checkbox(
                "Enable high-quality upscaling", value=es["upscale"], key="chk_upscale",
            )
            if es["upscale"]:
                algo_opts = ["lanczos", "spline", "bicubic"]
                es["upscale_algo"] = st.selectbox(
                    "Algorithm", algo_opts,
                    index=algo_opts.index(es["upscale_algo"]) if es["upscale_algo"] in algo_opts else 0,
                    key="sel_upscale_algo",
                )
                es["upscale_width"] = st.number_input(
                    "Target Width (px)", min_value=meta["width"], max_value=7680,
                    value=max(meta["width"], es.get("upscale_width") or meta["width"] * 2),
                    step=max(2, meta["width"] // 2), key="ni_upscale_w",
                )
                es["upscale_height"] = st.number_input(
                    "Target Height (px)", min_value=meta["height"], max_value=4320,
                    value=max(meta["height"], es.get("upscale_height") or meta["height"] * 2),
                    step=max(2, meta["height"] // 2), key="ni_upscale_h",
                )
            st.caption(
                "High-quality interpolation. For true AI super-resolution, "
                "integrate Real-ESRGAN externally."
            )

    with enh_col2:
        is_hdr_source = meta.get("bit_depth", 8) >= 10
        with st.container(border=True):
            st.markdown("**🌈 HDR → SDR Tonemapping**")
            es["hdr_convert"] = st.checkbox(
                "Enable HDR conversion",
                value=es["hdr_convert"] and is_hdr_source,
                key="chk_hdr", disabled=not is_hdr_source,
                help="Requires 10-bit+ HDR10/HLG source",
            )
            if es["hdr_convert"] and is_hdr_source:
                tmap_opts = ["hable", "reinhard", "mobius", "linear"]
                es["tonemap_algo"] = st.selectbox(
                    "Tonemap Algorithm", tmap_opts,
                    index=tmap_opts.index(es["tonemap_algo"]) if es["tonemap_algo"] in tmap_opts else 0,
                    key="sel_tonemap",
                )
            elif not is_hdr_source:
                st.caption("💡 Source is SDR (8-bit). HDR conversion requires a 10-bit+ HDR10/HLG source.")
            st.caption("Converts HDR10/HLG to SDR with perceptual tonemapping.")

        with st.container(border=True):
            st.markdown("**🎨 Color Enhancement**")
            es["color_enhance"] = st.checkbox(
                "Enable color boost", value=es["color_enhance"], key="chk_color",
            )
            if es["color_enhance"]:
                cc1, cc2 = st.columns(2)
                with cc1:
                    es["vibrance"] = st.slider("Vibrance", -0.5, 0.5, es["vibrance"], 0.05, key="sl_vibrance")
                with cc2:
                    es["contrast"] = st.slider("Contrast", 0.5, 2.0, es["contrast"], 0.05, key="sl_contrast")
            st.caption("Enhances vibrancy while preserving natural skin tones.")

        with st.container(border=True):
            st.markdown("**🧩 Artifact Reduction**")
            es["deblock"] = st.checkbox(
                "Enable deblocking filter", value=es["deblock"], key="chk_deblock",
            )
            if es["deblock"]:
                es["deblock_strength"] = st.slider(
                    "Strength", 1, 10, es["deblock_strength"], key="sl_deblock_str",
                )
            st.caption("Reduces macroblocking and ringing from heavy compression.")

    with st.expander("🎞️ Advanced: Frame Interpolation (Motion Smoothing)"):
        es["frame_interp"] = st.checkbox(
            "Enable frame interpolation", value=es["frame_interp"], key="chk_frame_interp",
        )
        if es["frame_interp"]:
            fps_opts = [30, 48, 60, 120]
            cur_fps  = es.get("target_fps", 60)
            fps_idx  = fps_opts.index(cur_fps) if cur_fps in fps_opts else 2
            es["target_fps"] = st.selectbox(
                "Target Frame Rate", fps_opts, index=fps_idx, key="sel_target_fps",
            )
            st.warning("⚠️ Frame interpolation is CPU-intensive and may increase processing time 3–5×.")
        st.caption("Creates intermediate frames for smoother playback (e.g., 24 fps → 60 fps).")

    _enh_keys        = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
    active_enhancements = sum(bool(es.get(k)) for k in _enh_keys)

    if active_enhancements > 0:
        est_time = estimate_processing_time(meta, es)
        st.info(
            f"⏱️ **Estimated processing time**: {est_time} "
            f"({active_enhancements} enhancement{'s' if active_enhancements > 1 else ''} active)"
        )
        enh_labels = {
            "denoise": "🧹 Denoise", "sharpen": "🔍 Sharpen", "upscale": "🔬 Upscale",
            "hdr_convert": "🌈 HDR", "color_enhance": "🎨 Color",
            "deblock": "🧩 Deblock", "frame_interp": "🎞️ Interp",
        }
        active_names = [enh_labels[k] for k in _enh_keys if es.get(k)]
        st.markdown(f"✨ **Active**: {' + '.join(active_names)}")

    st.divider()

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
        do_vmaf = st.checkbox("VMAF",     value=HAS_VMAF, disabled=not HAS_VMAF)
        do_psnr = st.checkbox("PSNR/SSIM", value=True)
    with s4:
        if crf < 19:   ql, qc = "🟢 High Quality", "#15803d"
        elif crf < 29: ql, qc = "🟡 Balanced",     "#b45309"
        elif crf < 40: ql, qc = "🟠 Compact",      "#ea580c"
        else:          ql, qc = "🔴 Low Quality",  "#b91c1c"
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

    with st.expander("📊 Batch CRF Sweep — Rate-Distortion Analysis"):
        st.caption(
            "Automatically encode at multiple CRF values to find the optimal "
            "quality/size trade-off. Uses current codec and enhancements."
        )
        sweep_col1, sweep_col2, sweep_col3 = st.columns(3)
        with sweep_col1:
            sweep_start = st.number_input("CRF Start", 10, 45, 18, step=1, key="sweep_start")
        with sweep_col2:
            sweep_end   = st.number_input("CRF End",   sweep_start+1, 51, 38, step=1, key="sweep_end")
        with sweep_col3:
            sweep_step  = st.number_input("Step",      1, 10, 5, step=1, key="sweep_step")

        sweep_crfs = list(range(int(sweep_start), int(sweep_end)+1, int(sweep_step)))
        st.caption(f"Will encode {len(sweep_crfs)} variants: CRF {', '.join(map(str, sweep_crfs))}")

        if st.button("🚀 Run CRF Sweep", type="primary"):
            sweep_bar = st.progress(0.0, text="Starting sweep…")
            for i, sweep_crf in enumerate(sweep_crfs):
                codec_short = codec.split()[0].lower()
                out_path = inp.replace(suf, f"_sweep_{codec_short}_crf{sweep_crf}.mp4")
                sweep_bar.progress(i / len(sweep_crfs), text=f"⚙️ Encoding CRF {sweep_crf} ({i+1}/{len(sweep_crfs)})…")

                ok, msg, fflog, enc_t = encode(
                    inp, out_path, codec, sweep_crf, es, meta, duration=meta["duration"]
                )
                if ok:
                    out_meta = probe(out_path)
                    out_sz   = os.path.getsize(out_path) / (1024 * 1024)
                    qual = {"psnr": None, "ssim": None, "vmaf": None}
                    if do_psnr or (do_vmaf and HAS_VMAF):
                        qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

                    idx = len(st.session_state.results)
                    st.session_state.result_logs[idx] = fflog
                    st.session_state.results.append({
                        "codec": codec, "crf": sweep_crf,
                        "size_mb": out_sz,
                        "bitrate": out_meta.get("vbitrate_kbps", 0),
                        "enc_time": enc_t,
                        "saved": (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0,
                        "cr": sz_mb / out_sz if out_sz > 0 else 0.0,
                        "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"],
                        "path": out_path,
                        "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"],
                        "sample_rate": meta["sample_rate"], "channels": meta["channels"],
                        "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v},
                        "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}",
                        "out_fps": out_meta.get("fps", meta["fps"]),
                    })
                else:
                    st.warning(f"CRF {sweep_crf} failed: {msg}")

            sweep_bar.progress(1.0, text=f"✅ Sweep complete — {len(sweep_crfs)} variants encoded")
            st.rerun()

    st.markdown('<div class="vf-label" style="margin-top:8px">🚀 Run</div>', unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 4])

    with b1:
        if st.button("🔍 Preview Impact", use_container_width=True):
            est_meta = meta.copy()
            if es.get("upscale"):
                est_meta["width"]  = int(es.get("upscale_width")  or meta["width"]  * 2)
                est_meta["height"] = int(es.get("upscale_height") or meta["height"] * 2)
            if es.get("frame_interp"):
                est_meta["fps"] = es.get("target_fps", 60)
            st.markdown("#### 📊 Estimated Output")
            bc1, bc2 = st.columns(2)
            bc1.metric("Source",   f"{meta['width']}×{meta['height']}",     f"@ {meta['fps']} fps")
            bc2.metric("Enhanced", f"{est_meta['width']}×{est_meta['height']}", f"@ {est_meta['fps']} fps")
            if est_meta["width"] > meta["width"] and meta["width"] > 0:
                ratio = (est_meta["width"] / meta["width"]) ** 2
                st.caption(f"+{(ratio-1)*100:.0f}% more pixels ({ratio:.1f}× resolution)")

    go    = b2.button("✨ Enhance + Encode", type="primary", use_container_width=True)
    clear = b3.button("🗑 Clear", use_container_width=True)

    if clear:
        st.session_state.results     = []
        st.session_state.result_logs = {}
        st.rerun()

    if go:
        codec_short = codec.split()[0].lower()
        enh_tag     = "enh_" if active_enhancements > 0 else ""
        out_path    = inp.replace(suf, f"_{enh_tag}{codec_short}_crf{crf}.mp4")

        bar = st.progress(0.0, text=f"⏳ Initializing {codec}…")

        with st.spinner(f"✨ Processing ({active_enhancements} enhancements) + encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(
                inp, out_path, codec, crf, es, meta,
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
            out_sz   = os.path.getsize(out_path) / (1024 * 1024)
            saved_pct = (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0

            qual = {"psnr": None, "ssim": None, "vmaf": None}
            if do_psnr or (do_vmaf and HAS_VMAF):
                with st.spinner("🔍 Computing quality metrics (PSNR/SSIM/VMAF)…"):
                    qual = quality_metrics(inp, out_path, do_vmaf and HAS_VMAF)

            idx = len(st.session_state.results)
            st.session_state.result_logs[idx] = fflog
            st.session_state.results.append({
                "codec": codec, "crf": crf,
                "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0),
                "enc_time": enc_t, "saved": saved_pct,
                "cr": sz_mb / out_sz if out_sz > 0 else 0.0,
                "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"],
                "path": out_path,
                "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"],
                "sample_rate": meta["sample_rate"], "channels": meta["channels"],
                "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v},
                "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}",
                "out_fps": out_meta.get("fps", meta["fps"]),
            })
            bar.empty()

            q_parts = []
            if qual["vmaf"] is not None: q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"] is not None: q_parts.append(f"PSNR {qual['psnr']:.2f} dB")
            q_str   = " · ".join(q_parts) if q_parts else "Analysis complete"
            enh_summary = " + ".join(k.capitalize() for k in _enh_keys if es.get(k))
            enh_str = f" · ✨ {enh_summary}" if enh_summary else ""

            st.success(
                f"✅ {codec} CRF {crf}{enh_str} · "
                f"{out_sz:.2f} MB · saved {saved_pct:.1f}% · "
                f"{enc_t:.1f}s · {q_str}"
            )

else:
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

tab_tbl, tab_chart, tab_dl, tab_logs = st.tabs([
    "📋 Comparison Table", "📊 Charts", "⬇ Downloads", "🪵 Logs"
])


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

        audio_info = (
            f"{format_audio_codec(r.get('acodec',''))} · {format_channels(r.get('channels',0))}"
            if r.get("acodec") and r["acodec"] != "unknown" else "—"
        )
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
        "PSNR: 40+ dB Good | SSIM: 0.98+ Excellent | ✨ = AI enhancements | 🏆 Best = winner across runs"
    )

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
            enh_tag  = " ✨" if best_eff.get("enhancements") else ""
            st.markdown(
                f'<div class="insight-note"><b>Efficiency Pick:</b> '
                f'<span style="font-weight:700">{best_eff["codec"]} CRF{best_eff["crf"]}{enh_tag}</span> '
                f'delivers best quality-per-MB '
                f'(VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB).</div>',
                unsafe_allow_html=True,
            )

        acceptable = [r for r in results if r.get("vmaf") and r["vmaf"] >= 80]
        if acceptable:
            smallest = min(acceptable, key=lambda r: r["size_mb"])
            st.markdown(
                f'<div class="insight-note"><b>Streaming Sweet Spot:</b> '
                f'<span style="font-weight:700">{smallest["codec"]} CRF{smallest["crf"]}</span> '
                f'is the smallest file with VMAF ≥ 80 '
                f'({smallest["size_mb"]:.2f} MB · VMAF {smallest["vmaf"]:.1f}).</div>',
                unsafe_allow_html=True,
            )


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


# ── Logs tab ──────────────────────────────────────────────────────────────────
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
    "AI Encoder · FFmpeg + Streamlit · ✨ AI enhancements · 📡 Live ABR Monitor · Bug-fixed build"
    "</div>",
    unsafe_allow_html=True,
)
