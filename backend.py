
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

MAX_UPLOAD_MB = 300
DEFAULT_SEGMENT_SECONDS = 2
DEFAULT_GOP_FPS = 24
ABR_RUNGS = (360, 540)

ASPECT_PRESETS = {
    "Source / Passthrough": None,
    "16:9 Landscape": (16, 9),
    "9:16 Vertical": (9, 16),
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "3:4 Portrait": (3, 4),
}

ABR_RUNG_SETTINGS = {
    360: {"video_bitrate": "800k",  "maxrate": "856k",  "bufsize": "1200k", "audio_bitrate": "96k"},
    540: {"video_bitrate": "1400k", "maxrate": "1498k", "bufsize": "2100k", "audio_bitrate": "96k"},
}

CODEC_PRESETS = {
    "AVC (H.264)": ("libx264", ["-preset", "fast"]),
    "HEVC (H.265)": ("libx265", ["-preset", "fast"]),
    "AV1": ("libaom-av1", ["-b:v", "0", "-cpu-used", "8", "-tile-columns", "2", "-threads", "4", "-usage", "realtime"]),
}


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------
def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def vmaf_ok() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-filters"], stderr=subprocess.STDOUT, text=True, timeout=10)
        return "libvmaf" in out
    except Exception:
        return False


def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "stream"


def ensure_clean_dir(path: str | os.PathLike[str]) -> None:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            else:
                os.unlink(fp)
        except Exception:
            pass


def zip_dir_bytes(dir_path: str | os.PathLike[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for file_name in files:
                fp = os.path.join(root, file_name)
                arc = os.path.relpath(fp, dir_path)
                zf.write(fp, arc)
    buf.seek(0)
    return buf.getvalue()


def _to_even(v: int) -> int:
    return v if v % 2 == 0 else v - 1


def _gop_for_fps(fps: Optional[int]) -> int:
    return max((fps or DEFAULT_GOP_FPS) * 2, 48)


def format_audio_codec(codec: str) -> str:
    mapping = {"aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis", "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC", "pcm_s16le": "PCM 16-bit", "alac": "ALAC"}
    return mapping.get(codec, (codec or "unknown").upper())


def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"


def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")


def estimate_processing_time(src_meta: dict, settings: dict) -> str:
    base = 1.0
    if settings.get("denoise"): base *= 1.3
    if settings.get("sharpen"): base *= 1.1
    if settings.get("upscale"): base *= 2.3 if settings.get("upscale_algo") == "lanczos" else 1.8
    if settings.get("hdr_convert"): base *= 1.6
    if settings.get("deblock"): base *= 1.4
    if settings.get("color_enhance"): base *= 1.15
    if settings.get("frame_interp"): base *= 3.0
    minutes = (src_meta.get("duration", 0) / 60.0) * base
    if minutes < 1:
        return f"~{max(1, int(minutes * 60))}s"
    if minutes < 10:
        return f"~{minutes:.1f} min"
    return f"~{minutes:.0f} min"


# ---------------------------------------------------------
# Source/media analysis
# ---------------------------------------------------------
def probe(path: str) -> dict:
    res = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "vcodec": "unknown",
        "vbitrate_kbps": 0,
        "acodec": "unknown",
        "abitrate_kbps": 0,
        "sample_rate": 0,
        "channels": 0,
        "audio_duration": 0.0,
        "has_audio": False,
        "color_space": "unknown",
        "bit_depth": 8,
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        data = json.loads(out)
        fmt = data.get("format", {})
        res["duration"] = float(fmt.get("duration", 0) or 0)
        res["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and res["width"] == 0:
                res["width"] = int(stream.get("width", 0) or 0)
                res["height"] = int(stream.get("height", 0) or 0)
                res["vcodec"] = stream.get("codec_name", "unknown")
                res["color_space"] = stream.get("color_space", "unknown")
                res["bit_depth"] = int(stream.get("bits_per_raw_sample", 8) or 8)
                try:
                    n, d = map(int, str(stream.get("r_frame_rate", "0/1")).split("/"))
                    res["fps"] = round(n / d, 3) if d else 0.0
                except Exception:
                    pass
            elif stream.get("codec_type") == "audio" and not res["has_audio"]:
                res["has_audio"] = True
                res["acodec"] = stream.get("codec_name", "unknown")
                res["abitrate_kbps"] = int(stream.get("bit_rate", 0) or 0) // 1000
                res["sample_rate"] = int(stream.get("sample_rate", 0) or 0)
                res["channels"] = int(stream.get("channels", 0) or 0)
                res["audio_duration"] = float(stream.get("duration", 0) or 0)
    except Exception:
        pass
    return res


def probe_loudness(path: str) -> dict:
    res = {"mean_volume": None, "max_volume": None}
    try:
        out = subprocess.check_output(["ffmpeg", "-y", "-i", path, "-af", "volumedetect", "-f", "null", "-"], stderr=subprocess.STDOUT, text=True, timeout=60)
        for line in out.splitlines():
            m1 = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line)
            m2 = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line)
            if m1:
                res["mean_volume"] = float(m1.group(1))
            if m2:
                res["max_volume"] = float(m2.group(1))
    except Exception:
        pass
    return res


# ---------------------------------------------------------
# Analytics widgets HTML
# ---------------------------------------------------------
def build_player_analytics_html(meta: dict, source_label: str = "Source") -> str:
    width = int(meta.get("width", 1920) or 1920)
    height = int(meta.get("height", 1080) or 1080)
    fps = float(meta.get("fps", 30.0) or 30.0)
    bitrate = int(meta.get("vbitrate_kbps", 4000) or 4000)
    ladder = [(f"{r}p", int(re.sub(r'[^0-9]', '', ABR_RUNG_SETTINGS[r]['video_bitrate']))) for r in ABR_RUNGS]
    active = min(ladder, key=lambda x: abs(x[1] - max(1, bitrate)))[0]
    html = """
<div style='font-family: Inter, Segoe UI, Arial, sans-serif; border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#0f172a; color:#e2e8f0;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
    <div>
      <div style='font-size:12px; color:#93c5fd; text-transform:uppercase; letter-spacing:.08em;'>Playback Analytics</div>
      <div style='font-size:20px; font-weight:700;'>__SOURCE__</div>
      <div style='font-size:12px; color:#94a3b8;'>__WIDTH__×__HEIGHT__ · __FPS__ fps · source bitrate ~__BITRATE__ kbps</div>
    </div>
    <div style='padding:6px 10px; border-radius:999px; background:#14532d; color:#dcfce7; font-size:12px;'>ABR simulator</div>
  </div>
  <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-bottom:14px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Bandwidth</div><div id='bw' style='font-size:22px; font-weight:700;'>4.9 Mbps</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Buffer</div><div id='buffer' style='font-size:22px; font-weight:700;'>20.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>RTT</div><div id='rtt' style='font-size:22px; font-weight:700;'>129 ms</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Active rung</div><div id='active' style='font-size:22px; font-weight:700;'>__ACTIVE__</div></div>
  </div>
</div>
<script>
const ladder = __LADDER__;
let current = "__ACTIVE__";
let tick = 0;
setInterval(function() {
  tick += 1;
  const bw = (1.6 + Math.sin(tick / 2) * 0.4 + Math.random() * 0.2).toFixed(2);
  const buffer = (6 + Math.sin(tick / 3) * 1.3 + Math.random()).toFixed(1);
  const rtt = Math.max(55, Math.round(95 + Math.sin(tick / 2) * 20 + Math.random() * 10));
  const ranked = [...ladder].sort((a,b) => b.bps - a.bps);
  const numericBw = Number(bw) * 1000;
  let found = ranked.find(x => x.bps < numericBw * 0.72);
  current = found ? found.label : ranked[ranked.length - 1].label;
  document.getElementById('bw').textContent = bw + ' Mbps';
  document.getElementById('buffer').textContent = buffer + ' s';
  document.getElementById('rtt').textContent = rtt + ' ms';
  document.getElementById('active').textContent = current;
}, 1200);
</script>
"""
    html = html.replace("__SOURCE__", source_label)
    html = html.replace("__WIDTH__", str(width)).replace("__HEIGHT__", str(height))
    html = html.replace("__FPS__", f"{fps:.2f}").replace("__BITRATE__", str(bitrate))
    html = html.replace("__ACTIVE__", active)
    html = html.replace("__LADDER__", json.dumps([{"label": x[0], "bps": x[1]} for x in ladder]))
    return html


def build_hlsjs_player_html(manifest_url: str, title: str = "HLS playback", autoplay: bool = True, muted: bool = True, low_latency: bool = True) -> str:
    autoplay_str = "true" if autoplay else "false"
    low_latency_str = "true" if low_latency else "false"
    muted_attr = "muted" if muted else ""
    return f"""
<div style='font-family: Inter, Segoe UI, Arial, sans-serif; border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#0f172a; color:#e2e8f0;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
    <div>
      <div style='font-size:12px; color:#93c5fd; text-transform:uppercase; letter-spacing:.08em;'>Embedded HLS.js Playback</div>
      <div style='font-size:20px; font-weight:700;'>{title}</div>
      <div style='font-size:12px; color:#94a3b8; word-break:break-all;'>{manifest_url}</div>
    </div>
    <div style='padding:6px 10px; border-radius:999px; background:#172554; color:#dbeafe; font-size:12px;'>public master.m3u8 URL</div>
  </div>
  <video id='video' controls playsinline {muted_attr} style='width:100%; max-height:560px; background:#000; border-radius:12px;'></video>
  <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:14px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>State</div><div id='state' style='font-size:22px; font-weight:700;'>booting</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Current level</div><div id='level' style='font-size:22px; font-weight:700;'>-</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Buffer ahead</div><div id='buffer' style='font-size:22px; font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Dropped frames</div><div id='drops' style='font-size:22px; font-weight:700;'>0</div></div>
  </div>
  <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:10px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Latency to live edge</div><div id='latency' style='font-size:20px; font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Playback rate</div><div id='rate' style='font-size:20px; font-weight:700;'>1.00×</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Event count</div><div id='events' style='font-size:20px; font-weight:700;'>0</div></div>
  </div>
  <div style='background:#111827; border-radius:12px; padding:12px; margin-top:14px;'>
    <div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Player event log</div>
    <div id='log' style='font-size:12px; line-height:1.6; color:#cbd5e1; max-height:160px; overflow:auto;'>Waiting for manifest…</div>
  </div>
</div>
<script src='https://cdn.jsdelivr.net/npm/hls.js@latest'></script>
<script>
(function() {{
  const manifestUrl = {json.dumps(manifest_url)};
  const video = document.getElementById('video');
  const elState = document.getElementById('state');
  const elLevel = document.getElementById('level');
  const elBuffer = document.getElementById('buffer');
  const elDrops = document.getElementById('drops');
  const elLatency = document.getElementById('latency');
  const elRate = document.getElementById('rate');
  const elEvents = document.getElementById('events');
  const elLog = document.getElementById('log');
  let eventCount = 0;
  function pushLog(msg) {{
    eventCount += 1;
    elEvents.textContent = String(eventCount);
    const now = new Date().toLocaleTimeString();
    elLog.innerHTML = '[' + now + '] ' + msg + '<br/>' + elLog.innerHTML.split('<br/>').slice(0,10).join('<br/>');
  }}
  function updateMetrics() {{
    let bufferAhead = 0;
    if (video.buffered && video.buffered.length) {{
      for (let i = 0; i < video.buffered.length; i++) {{
        if (video.buffered.start(i) <= video.currentTime && video.currentTime <= video.buffered.end(i)) {{
          bufferAhead = video.buffered.end(i) - video.currentTime;
          break;
        }}
      }}
    }}
    elBuffer.textContent = bufferAhead.toFixed(1) + ' s';
    elRate.textContent = video.playbackRate.toFixed(2) + '×';
    try {{
      const q = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
      if (q && typeof q.droppedVideoFrames === 'number') elDrops.textContent = String(q.droppedVideoFrames);
    }} catch (e) {{}}
    if (window.hls && typeof window.hls.latency === 'number') elLatency.textContent = window.hls.latency.toFixed(1) + ' s';
  }}
  if (Hls.isSupported()) {{
    const hls = new Hls({{ lowLatencyMode: {low_latency_str}, liveSyncDurationCount: 2, liveMaxLatencyDurationCount: 4, maxLiveSyncPlaybackRate: 1.2, enableWorker: true }});
    window.hls = hls;
    hls.loadSource(manifestUrl);
    hls.attachMedia(video);
    hls.on(Hls.Events.MEDIA_ATTACHED, function() {{ elState.textContent = 'media attached'; pushLog('Media attached'); }});
    hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {{ elState.textContent = 'manifest parsed'; pushLog('Manifest parsed with ' + data.levels.length + ' level(s)'); if ({autoplay_str}) video.play().catch(() => {{}}); }});
    hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{ const level = hls.levels[data.level]; const label = level ? ((level.height || '?') + 'p @ ' + Math.round((level.bitrate || 0)/1000) + ' kbps') : String(data.level); elLevel.textContent = label; pushLog('Level switched to ' + label); }});
    hls.on(Hls.Events.ERROR, function(event, data) {{ pushLog('HLS error: ' + data.type + ' | ' + data.details); elState.textContent = data.fatal ? 'fatal error' : 'recoverable error'; if (data.fatal) {{ if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad(); else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError(); }} }});
  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
    video.src = manifestUrl;
    elState.textContent = 'native HLS';
    if ({autoplay_str}) video.play().catch(() => {{}});
  }} else {{
    elState.textContent = 'unsupported';
    pushLog('Browser does not support HLS playback');
  }}
  ['play','pause','waiting','playing','seeking','stalled','ended','loadedmetadata','canplay'].forEach(function(evt) {{
    video.addEventListener(evt, function() {{ elState.textContent = evt; pushLog('Video event: ' + evt); updateMetrics(); }});
  }});
  setInterval(updateMetrics, 1000);
}})();
</script>
"""


# ---------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------
def build_enhance_filters(settings: dict, src_meta: dict) -> list[str]:
    filters = []
    if settings.get("denoise"):
        s = float(settings.get("denoise_strength", 5))
        h = round(s / 2, 1)
        filters.append(f"hqdn3d={s}:{s}:{h}:{h}")
    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5))
        alpha = round(strength / 10.0, 2)
        beta = round(alpha * 0.5, 2)
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")
    if settings.get("sharpen"):
        amount = float(settings.get("sharpen_amount", 0.5))
        c_amount = round(amount * 0.5, 2)
        filters.append(f"unsharp=lx=5:ly=5:la={amount}:cx=3:cy=3:ca={c_amount}")
    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15))
        contrast = float(settings.get("contrast", 1.0))
        sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.extend(["zscale=transfer=linear,format=gbrpf32le", f"tonemap={settings.get('tonemap_algo', 'hable')}:desat=0.2", "zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le"])
    if settings.get("upscale"):
        w = _to_even(int(settings.get("upscale_width", src_meta["width"] * 2)))
        h = _to_even(int(settings.get("upscale_height", src_meta["height"] * 2)))
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(f"scale={w}:{h}:flags={algo}+accurate_rnd+full_chroma_int")
    if settings.get("frame_interp"):
        target = int(settings.get("target_fps", 60))
        filters.append(f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    return filters


def encode(input_path: str, output_path: str, codec: str, crf: int, enhance_settings: dict, src_meta: dict, progress_cb: Optional[Callable[[float], None]] = None, duration: float = 0.0) -> tuple[bool, str, str, float]:
    lib, extra = CODEC_PRESETS.get(codec, ("libx264", ["-preset", "fast"]))
    filters = build_enhance_filters(enhance_settings, src_meta)
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", lib, "-crf", str(crf)] + list(extra)
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:a", "copy", "-movflags", "+faststart", output_path]
    lines: list[str] = []
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1)
        assert proc.stderr is not None
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
        log = "\n".join(lines[-120:])
        if proc.returncode == 0:
            return True, "Done", log, elapsed
        return False, f"FFmpeg exited with code {proc.returncode}", log, elapsed
    except FileNotFoundError:
        return False, "FFmpeg not found", "", 0.0
    except Exception as exc:
        return False, str(exc), "\n".join(lines), time.time() - t0


def quality_metrics(ref: str, dist: str, do_vmaf: bool) -> dict:
    res = {"psnr": None, "ssim": None, "vmaf": None}
    try:
        out = subprocess.check_output(["ffmpeg", "-y", "-i", dist, "-i", ref, "-filter_complex", "[0:v][1:v]psnr", "-f", "null", "-"], stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"PSNR", line, re.I):
                m = re.search(r"average[:\s]+([0-9.]+|inf)", line, re.I)
                if m:
                    v = m.group(1)
                    res["psnr"] = 100.0 if v == "inf" else round(float(v), 3)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["ffmpeg", "-y", "-i", dist, "-i", ref, "-filter_complex", "[0:v][1:v]ssim", "-f", "null", "-"], stderr=subprocess.STDOUT, text=True, timeout=180)
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
            subprocess.run(["ffmpeg", "-y", "-i", dist, "-i", ref, "-filter_complex", f"[0:v][1:v]libvmaf=log_fmt=json:log_path={vf.name}", "-f", "null", "-"], capture_output=True, timeout=300, check=True)
            with open(vf.name, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            score = vdata.get("pooled_metrics", {}).get("vmaf", {}).get("mean") or vdata.get("VMAF score") or vdata.get("aggregate", {}).get("VMAF_score")
            if score is not None:
                res["vmaf"] = round(float(score), 2)
            try:
                os.unlink(vf.name)
            except Exception:
                pass
        except Exception:
            pass
    return res


# ---------------------------------------------------------
# Public/static origin helpers
# ---------------------------------------------------------
@dataclass
class OriginConfig:
    local_origin_dir: str
    public_base_url: str


def resolve_origin_config(base_dir: str, public_base_url: str) -> OriginConfig:
    if not base_dir:
        raise ValueError("Public/static origin directory is required.")
    if not public_base_url:
        raise ValueError("Public/static origin base URL is required.")
    os.makedirs(base_dir, exist_ok=True)
    return OriginConfig(local_origin_dir=os.path.abspath(base_dir), public_base_url=public_base_url.rstrip("/"))


def build_public_asset_paths(origin: OriginConfig, asset_slug: str) -> tuple[str, str]:
    asset_slug = safe_token(asset_slug)
    local_dir = os.path.join(origin.local_origin_dir, asset_slug)
    ensure_clean_dir(local_dir)
    public_url = f"{origin.public_base_url}/{asset_slug}"
    return local_dir, public_url


def create_placeholder_hls_structure(origin_dir: str, public_base_url: str, ladder: list[dict], segment_seconds: int, single_manifest_name: Optional[str] = None) -> tuple[str, str]:
    os.makedirs(origin_dir, exist_ok=True)
    if ladder:
        master = os.path.join(origin_dir, "master.m3u8")
        lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
        for variant in ladder:
            lines.append(f"#EXT-X-STREAM-INF:BANDWIDTH={variant['bandwidth']},AVERAGE-BANDWIDTH={variant['avg_bandwidth']},RESOLUTION={variant['width']}x{variant['height']}")
            lines.append(f"{variant['name']}.m3u8")
            media = os.path.join(origin_dir, f"{variant['name']}.m3u8")
            Path(media).write_text("\n".join(["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{segment_seconds}", "#EXT-X-MEDIA-SEQUENCE:0"]) + "\n", encoding="utf-8")
        Path(master).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return master, f"{public_base_url}/master.m3u8"
    token = single_manifest_name or "stream.m3u8"
    manifest = os.path.join(origin_dir, token)
    Path(manifest).write_text("\n".join(["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{segment_seconds}", "#EXT-X-MEDIA-SEQUENCE:0"]) + "\n", encoding="utf-8")
    return manifest, f"{public_base_url}/{token}"


def build_single_preview_hls_cmd(input_source: str, manifest_path: str, segment_pattern: str, aspect_label: str = "Source / Passthrough", video_bitrate: str = "1400k", audio_bitrate: str = "96k", preset: str = "superfast", fps: Optional[int] = None, segment_seconds: int = DEFAULT_SEGMENT_SECONDS) -> list[str]:
    vf = None
    if aspect_label != "Source / Passthrough":
        w, h = ladder_dimensions(aspect_label, 540)
        vf = variant_video_filter(w, h)
    cmd = ["ffmpeg", "-y", "-i", input_source]
    if vf:
        cmd += ["-vf", vf]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-profile:v", "main",
        "-b:v", video_bitrate, "-maxrate", video_bitrate, "-bufsize", f"{max(int(re.sub(r'[^0-9]', '', video_bitrate) or '1400') * 2, 1000)}k",
        "-g", str(_gop_for_fps(fps)), "-keyint_min", str(_gop_for_fps(fps)), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2",
        "-f", "hls", "-hls_time", str(segment_seconds), "-hls_list_size", "0",
        "-hls_flags", "append_list+independent_segments+program_date_time",
        "-hls_segment_filename", segment_pattern, manifest_path,
    ]
    return cmd


def build_multi_variant_preview_hls_cmd(input_source: str, origin_dir: str, aspect_label: str = "16:9 Landscape", preset: str = "superfast", fps: Optional[int] = None, segment_seconds: int = DEFAULT_SEGMENT_SECONDS, src_meta: Optional[dict] = None) -> tuple[list[str], list[dict], str]:
    os.makedirs(origin_dir, exist_ok=True)
    ladder = build_abr_ladder(aspect_label=aspect_label, src_meta=src_meta)
    split_labels = [f"v{i}" for i in range(len(ladder))]
    out_labels = [f"v{i}out" for i in range(len(ladder))]
    filter_parts = [f"[0:v]split={len(ladder)}" + "".join(f"[{x}]" for x in split_labels)]
    for idx, variant in enumerate(ladder):
        vf = variant_video_filter(variant["width"], variant["height"])
        if fps:
            vf += f",fps={fps}"
        filter_parts.append(f"[{split_labels[idx]}]{vf}[{out_labels[idx]}]")
    cmd = ["ffmpeg", "-y", "-i", input_source, "-filter_complex", ";".join(filter_parts)]
    for idx in range(len(ladder)):
        cmd += ["-map", f"[{out_labels[idx]}]", "-map", "0:a:0?"]
    for idx, variant in enumerate(ladder):
        cmd += [
            f"-c:v:{idx}", "libx264", f"-preset:v:{idx}", preset, f"-pix_fmt:v:{idx}", "yuv420p", f"-profile:v:{idx}", "main",
            f"-b:v:{idx}", variant["video_bitrate"], f"-maxrate:v:{idx}", variant["maxrate"], f"-bufsize:v:{idx}", variant["bufsize"],
            f"-g:v:{idx}", str(_gop_for_fps(fps)), f"-keyint_min:v:{idx}", str(_gop_for_fps(fps)), f"-sc_threshold:v:{idx}", "0",
            f"-c:a:{idx}", "aac", f"-b:a:{idx}", variant["audio_bitrate"], f"-ar:a:{idx}", "48000", f"-ac:a:{idx}", "2",
        ]
    master_name = "master.m3u8"
    segment_pattern = os.path.join(origin_dir, "%v_%06d.ts")
    media_playlist_pattern = os.path.join(origin_dir, "%v.m3u8")
    var_stream_map = " ".join(f"v:{idx},a:{idx},name:{variant['name']}" for idx, variant in enumerate(ladder))
    cmd += ["-master_pl_name", master_name, "-f", "hls", "-hls_time", str(segment_seconds), "-hls_segment_filename", segment_pattern, "-var_stream_map", var_stream_map, "-hls_list_size", "0", "-hls_flags", "append_list+independent_segments+program_date_time", media_playlist_pattern]
    return cmd, ladder, os.path.join(origin_dir, master_name)


def start_background_ffmpeg_job(cmd: list[str], output_dir: str, manifest_path: str, manifest_url: str, ladder: list[dict], input_source: str, log_name: str) -> dict:
    log_path = os.path.join(output_dir, log_name)
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    return {
        "proc": proc,
        "output_dir": output_dir,
        "master_manifest": manifest_path,
        "manifest_url": manifest_url,
        "log_path": log_path,
        "input_source": input_source,
        "ladder": ladder,
        "started_at": time.time(),
    }


def collect_job_outputs(job: dict) -> dict:
    out_dir = job.get("output_dir")
    zbytes = zip_dir_bytes(out_dir) if out_dir and os.path.isdir(out_dir) else None
    return {"zip_bytes": zbytes}


@dataclass
class PreviewHlsJobStartResult:
    output_dir: str
    master_manifest: str
    manifest_url: str
    public_asset_url: str
    ladder: list[dict]
    job: dict


def start_vod_public_preview_job(input_source: str, asset_name: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, abr_enabled: bool, origin: OriginConfig, src_meta: Optional[dict] = None) -> PreviewHlsJobStartResult:
    asset_slug = f"{safe_token(Path(asset_name).stem)}_{int(time.time())}"
    output_dir, public_asset_url = build_public_asset_paths(origin, asset_slug)
    if abr_enabled:
        ladder = build_abr_ladder(aspect_label=aspect_label, src_meta=src_meta)
        master_manifest, manifest_url = create_placeholder_hls_structure(output_dir, public_asset_url, ladder, segment_seconds)
        cmd, ladder, master_manifest = build_multi_variant_preview_hls_cmd(input_source=input_source, origin_dir=output_dir, aspect_label=aspect_label, preset=preset, fps=fps, segment_seconds=segment_seconds, src_meta=src_meta)
    else:
        ladder = []
        token = safe_token(Path(asset_name).stem) + ".m3u8"
        master_manifest, manifest_url = create_placeholder_hls_structure(output_dir, public_asset_url, [], segment_seconds, single_manifest_name=token)
        segment_pattern = os.path.join(output_dir, safe_token(Path(asset_name).stem) + "_%05d.ts")
        cmd = build_single_preview_hls_cmd(input_source=input_source, manifest_path=master_manifest, segment_pattern=segment_pattern, aspect_label=aspect_label, preset=preset, fps=fps, segment_seconds=segment_seconds)
    job = start_background_ffmpeg_job(cmd=cmd, output_dir=output_dir, manifest_path=master_manifest, manifest_url=manifest_url, ladder=ladder, input_source=input_source, log_name="ffmpeg_preview.log")
    return PreviewHlsJobStartResult(output_dir=output_dir, master_manifest=master_manifest, manifest_url=manifest_url, public_asset_url=public_asset_url, ladder=ladder, job=job)


def start_live_hls_job(input_source: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, abr_enabled: bool, origin: OriginConfig, src_meta: Optional[dict] = None) -> tuple[dict, list[dict], str]:
    asset_slug = f"live_{safe_token(aspect_label.lower())}_{int(time.time())}"
    output_dir, public_asset_url = build_public_asset_paths(origin, asset_slug)
    if abr_enabled:
        ladder = build_abr_ladder(aspect_label=aspect_label, src_meta=src_meta)
        master_manifest, manifest_url = create_placeholder_hls_structure(output_dir, public_asset_url, ladder, segment_seconds)
        cmd, ladder, master_manifest = build_multi_variant_preview_hls_cmd(input_source=input_source, origin_dir=output_dir, aspect_label=aspect_label, preset=preset, fps=fps, segment_seconds=segment_seconds, src_meta=src_meta)
    else:
        ladder = []
        master_manifest, manifest_url = create_placeholder_hls_structure(output_dir, public_asset_url, [], segment_seconds, single_manifest_name="live.m3u8")
        segment_pattern = os.path.join(output_dir, "live_%05d.ts")
        cmd = build_single_preview_hls_cmd(input_source=input_source, manifest_path=master_manifest, segment_pattern=segment_pattern, aspect_label=aspect_label, preset=preset, fps=fps, segment_seconds=segment_seconds)
    job = start_background_ffmpeg_job(cmd=cmd, output_dir=output_dir, manifest_path=master_manifest, manifest_url=manifest_url, ladder=ladder, input_source=input_source, log_name="ffmpeg_live.log")
    return job, ladder, public_asset_url
