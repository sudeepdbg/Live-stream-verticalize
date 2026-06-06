
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

MAX_UPLOAD_MB = 300
DEFAULT_SEGMENT_SECONDS = 4
DEFAULT_LIVE_LIST_SIZE = 6
DEFAULT_GOP_FPS = 24

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
    720: {"video_bitrate": "2800k", "maxrate": "2996k", "bufsize": "4200k", "audio_bitrate": "128k"},
    1080: {"video_bitrate": "5000k", "maxrate": "5350k", "bufsize": "7500k", "audio_bitrate": "128k"},
}

CODEC_PRESETS = {
    "AVC (H.264)": ("libx264", ["-preset", "fast"]),
    "HEVC (H.265)": ("libx265", ["-preset", "fast"]),
    "AV1": ("libaom-av1", ["-b:v", "0", "-cpu-used", "8", "-tile-columns", "2", "-threads", "4", "-usage", "realtime"]),
}

_HTTP_SERVERS: dict[str, dict] = {}


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


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def start_static_file_server(directory: str, host: str = "127.0.0.1", port: Optional[int] = None) -> dict:
    key = os.path.abspath(directory)
    existing = _HTTP_SERVERS.get(key)
    if existing:
        return existing
    port = port or _find_free_port(host)
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    server_info = {"directory": key, "host": host, "port": port, "base_url": f"http://{host}:{port}", "server": httpd, "thread": thread}
    _HTTP_SERVERS[key] = server_info
    return server_info


def stop_static_file_server(directory: str) -> None:
    key = os.path.abspath(directory)
    info = _HTTP_SERVERS.pop(key, None)
    if not info:
        return
    try:
        info["server"].shutdown()
        info["server"].server_close()
    except Exception:
        pass


def serve_manifest_url(directory: str, manifest_path: str) -> str:
    server = start_static_file_server(directory)
    manifest_name = os.path.basename(manifest_path)
    return f"{server['base_url']}/{manifest_name}"


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


def build_player_analytics_html(meta: dict, source_label: str = "Source") -> str:
    width = int(meta.get("width", 1920) or 1920)
    height = int(meta.get("height", 1080) or 1080)
    fps = float(meta.get("fps", 30.0) or 30.0)
    bitrate = int(meta.get("vbitrate_kbps", 4000) or 4000)
    if height >= 1080 or width >= 1920:
        ladder = [("1080p", 5000), ("720p", 2800), ("540p", 1400), ("360p", 800)]
    elif height >= 720 or width >= 1280:
        ladder = [("720p", 2800), ("540p", 1400), ("360p", 800)]
    else:
        ladder = [("540p", 1400), ("360p", 800)]
    active = min(ladder, key=lambda x: abs(x[1] - bitrate))[0]
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
  <div style='background:#111827; border-radius:12px; padding:12px; margin-bottom:14px;'>
    <div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Bitrate ladder</div>
    <div id='ladder'></div>
  </div>
  <div style='background:#111827; border-radius:12px; padding:12px;'>
    <div style='font-size:13px; font-weight:600; margin-bottom:8px;'>ABR decisions log</div>
    <div id='decisions' style='font-size:12px; line-height:1.6; color:#cbd5e1;'>initialised</div>
  </div>
</div>
<script>
const ladder = __LADDER__;
let current = "__ACTIVE__";
let tick = 0;
function renderLadder(active) {
  let html = '';
  for (const item of ladder) {
    const isActive = item.label === active;
    const bg = isActive ? '#1d4ed8' : '#0f172a';
    const fg = isActive ? '#eff6ff' : '#cbd5e1';
    const border = isActive ? '#93c5fd' : '#1f2937';
    html += '<div style="display:flex; justify-content:space-between; align-items:center; margin:6px 0; padding:8px 10px; border-radius:10px; background:' + bg + '; color:' + fg + '; border:1px solid ' + border + ';"><span>' + item.label + '</span><span>' + item.bps + ' kbps</span></div>';
  }
  document.getElementById('ladder').innerHTML = html;
}
renderLadder(current);
setInterval(function() {
  tick += 1;
  const bw = (4.2 + Math.sin(tick / 2) * 0.8 + Math.random() * 0.4).toFixed(2);
  const buffer = (18 + Math.sin(tick / 3) * 3 + Math.random()).toFixed(1);
  const rtt = Math.max(65, Math.round(115 + Math.sin(tick / 2) * 25 + Math.random() * 15));
  const ranked = [...ladder].sort((a,b) => b.bps - a.bps);
  const numericBw = Number(bw) * 1000;
  let found = ranked.find(x => x.bps < numericBw * 0.72);
  current = found ? found.label : ranked[ranked.length - 1].label;
  document.getElementById('bw').textContent = bw + ' Mbps';
  document.getElementById('buffer').textContent = buffer + ' s';
  document.getElementById('rtt').textContent = rtt + ' ms';
  document.getElementById('active').textContent = current;
  renderLadder(current);
  const line = '[t+' + tick + 's] active=' + current + ' | bandwidth=' + bw + ' Mbps | buffer=' + buffer + ' s | rtt=' + rtt + ' ms';
  const prev = document.getElementById('decisions').innerHTML.split('<br/>').slice(0,5).join('<br/>');
  document.getElementById('decisions').innerHTML = line + '<br/>' + prev;
}, 1200);
</script>
"""
    html = html.replace("__SOURCE__", source_label)
    html = html.replace("__WIDTH__", str(width)).replace("__HEIGHT__", str(height))
    html = html.replace("__FPS__", f"{fps:.2f}").replace("__BITRATE__", str(bitrate))
    html = html.replace("__ACTIVE__", active)
    html = html.replace("__LADDER__", json.dumps([{"label": x[0], "bps": x[1]} for x in ladder]))
    return html


def build_hlsjs_player_html(manifest_url: str, title: str = "HLS playback", autoplay: bool = False, muted: bool = True, low_latency: bool = True) -> str:
    autoplay_str = "true" if autoplay else "false"
    muted_str = "true" if muted else "false"
    low_latency_str = "true" if low_latency else "false"
    return f"""
<div style='font-family: Inter, Segoe UI, Arial, sans-serif; border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#0f172a; color:#e2e8f0;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
    <div>
      <div style='font-size:12px; color:#93c5fd; text-transform:uppercase; letter-spacing:.08em;'>In-app HLS Playback</div>
      <div style='font-size:20px; font-weight:700;'>{title}</div>
      <div style='font-size:12px; color:#94a3b8; word-break:break-all;'>{manifest_url}</div>
    </div>
    <div style='padding:6px 10px; border-radius:999px; background:#172554; color:#dbeafe; font-size:12px;'>served immediately over local HTTP</div>
  </div>
  <video id='video' controls playsinline style='width:100%; max-height:560px; background:#000; border-radius:12px;' {'muted' if muted else ''}></video>
  <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:14px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>State</div><div id='state' style='font-size:22px; font-weight:700;'>initialising</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Current level</div><div id='level' style='font-size:22px; font-weight:700;'>-</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Buffer ahead</div><div id='buffer' style='font-size:22px; font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Dropped frames</div><div id='drops' style='font-size:22px; font-weight:700;'>0</div></div>
  </div>
  <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:10px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Latency to live edge</div><div id='latency' style='font-size:20px; font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Playback rate</div><div id='rate' style='font-size:20px; font-weight:700;'>1.00×</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Stream events</div><div id='events' style='font-size:20px; font-weight:700;'>0</div></div>
  </div>
  <div style='background:#111827; border-radius:12px; padding:12px; margin-top:14px;'>
    <div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Player event log</div>
    <div id='log' style='font-size:12px; line-height:1.6; color:#cbd5e1; max-height:160px; overflow:auto;'>Booting HLS.js…</div>
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
      const quality = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
      if (quality && typeof quality.droppedVideoFrames === 'number') {{
        elDrops.textContent = String(quality.droppedVideoFrames);
      }}
    }} catch (e) {{}}
    if (window.hls && typeof window.hls.latency === 'number') {{
      elLatency.textContent = window.hls.latency.toFixed(1) + ' s';
    }}
  }}

  if (Hls.isSupported()) {{
    const hls = new Hls({{
      lowLatencyMode: {low_latency_str},
      backBufferLength: 90,
      liveSyncDurationCount: 2,
      liveMaxLatencyDurationCount: 4,
      maxLiveSyncPlaybackRate: 1.2,
      enableWorker: true,
    }});
    window.hls = hls;
    hls.loadSource(manifestUrl);
    hls.attachMedia(video);
    hls.on(Hls.Events.MEDIA_ATTACHED, function() {{
      elState.textContent = 'media attached';
      pushLog('Media attached');
      if ({autoplay_str}) video.play().catch(() => {{}});
    }});
    hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {{
      elState.textContent = 'manifest parsed';
      pushLog('Manifest parsed with ' + data.levels.length + ' levels');
      if ({autoplay_str}) video.play().catch(() => {{}});
    }});
    hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{
      const level = hls.levels[data.level];
      const label = level ? ((level.height || '?') + 'p @ ' + Math.round((level.bitrate || 0)/1000) + ' kbps') : String(data.level);
      elLevel.textContent = label;
      pushLog('Level switched to ' + label);
    }});
    hls.on(Hls.Events.FRAG_BUFFERED, function() {{
      elState.textContent = video.paused ? 'buffered / paused' : 'playing';
      updateMetrics();
    }});
    hls.on(Hls.Events.ERROR, function(event, data) {{
      pushLog('HLS error: ' + data.type + ' | ' + data.details);
      elState.textContent = data.fatal ? 'fatal error' : 'recoverable error';
      if (data.fatal) {{
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad();
        else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
      }}
    }});
  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
    video.src = manifestUrl;
    elState.textContent = 'native HLS';
    pushLog('Using native HLS playback');
    if ({autoplay_str}) video.play().catch(() => {{}});
  }} else {{
    elState.textContent = 'unsupported';
    pushLog('This browser does not support HLS playback');
  }}

  ['play','pause','waiting','playing','seeking','stalled','ended','loadedmetadata'].forEach(function(evt) {{
    video.addEventListener(evt, function() {{
      elState.textContent = evt;
      pushLog('Video event: ' + evt);
      updateMetrics();
    }});
  }});
  setInterval(updateMetrics, 1000);
}})();
</script>
"""


def build_enhance_filters(settings: dict, src_meta: dict) -> list[str]:
    filters = []
    if settings.get("denoise"):
        s = float(settings.get("denoise_strength", 5)); h = round(s / 2, 1)
        filters.append(f"hqdn3d={s}:{s}:{h}:{h}")
    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5)); alpha = round(strength / 10.0, 2); beta = round(alpha * 0.5, 2)
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")
    if settings.get("sharpen"):
        amount = float(settings.get("sharpen_amount", 0.5)); c_amount = round(amount * 0.5, 2)
        filters.append(f"unsharp=lx=5:ly=5:la={amount}:cx=3:cy=3:ca={c_amount}")
    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15)); contrast = float(settings.get("contrast", 1.0)); sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.extend(["zscale=transfer=linear,format=gbrpf32le", f"tonemap={settings.get('tonemap_algo', 'hable')}:desat=0.2", "zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le"])
    if settings.get("upscale"):
        w = _to_even(int(settings.get("upscale_width", src_meta["width"] * 2))); h = _to_even(int(settings.get("upscale_height", src_meta["height"] * 2))); algo = settings.get("upscale_algo", "lanczos")
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
        hints = {-6: "OOM — filters need more memory.", -9: "OOM / SIGKILL — reduce complexity.", -11: "Segmentation fault — filter incompatibility.", 1: "FFmpeg error — inspect log."}
        return False, hints.get(proc.returncode, f"FFmpeg exited with code {proc.returncode}"), log, elapsed
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
                    v = m.group(1); res["psnr"] = 100.0 if v == "inf" else round(float(v), 3)
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
            vf = tempfile.NamedTemporaryFile(delete=False, suffix=".json"); vf.close()
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


def results_to_csv(results: list[dict], src_meta: dict, src_size_mb: float) -> bytes:
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["Codec", "CRF", "Enhancements", "Size_MB", "Bitrate_kbps", "Compression_Ratio", "Space_Saved_%", "Encode_Time_s", "VMAF", "PSNR_dB", "SSIM", "Output_Resolution", "Output_FPS", "Audio_Codec", "Audio_Channels", "Source_Codec", "Source_Size_MB", "Source_Bitrate_kbps"])
    for r in results:
        enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
        enh_list = [k for k in enh_keys if r.get("enhancements", {}).get(k)]
        w.writerow([r["codec"], r["crf"], "|".join(enh_list), f"{r['size_mb']:.3f}", r["bitrate"], f"{r['cr']:.3f}", f"{r['saved']:.1f}", f"{r['enc_time']:.2f}", r.get("vmaf") or "", r.get("psnr") or "", r.get("ssim") or "", r.get("out_res", ""), r.get("out_fps", ""), format_audio_codec(r.get("acodec", "")), format_channels(r.get("channels", 0)), src_meta["vcodec"].upper(), f"{src_size_mb:.3f}", src_meta["vbitrate_kbps"]])
    return buf.getvalue().encode("utf-8")


def ladder_dimensions(aspect_label: str, rung: int, src_meta: Optional[dict] = None) -> tuple[int, int]:
    if aspect_label == "Source / Passthrough" and src_meta and src_meta.get("width") and src_meta.get("height"):
        sw, sh = int(src_meta["width"]), int(src_meta["height"])
        if sw >= sh:
            h = rung; w = _to_even(round(h * sw / sh)); return w, _to_even(h)
        w = rung; h = _to_even(round(w * sh / sw)); return _to_even(w), h
    mapping = {
        "16:9 Landscape": {360: (640, 360), 540: (960, 540), 720: (1280, 720), 1080: (1920, 1080)},
        "9:16 Vertical": {360: (360, 640), 540: (540, 960), 720: (720, 1280), 1080: (1080, 1920)},
        "1:1 Square": {360: (360, 360), 540: (540, 540), 720: (720, 720), 1080: (1080, 1080)},
        "4:5 Portrait": {360: (288, 360), 540: (432, 540), 720: (576, 720), 1080: (864, 1080)},
        "3:4 Portrait": {360: (270, 360), 540: (406, 540), 720: (540, 720), 1080: (810, 1080)},
    }
    return mapping.get(aspect_label, mapping["16:9 Landscape"])[rung]


def variant_video_filter(width: int, height: int) -> str:
    return f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"


def build_abr_ladder(aspect_label: str, src_meta: Optional[dict] = None) -> list[dict]:
    out = []
    for rung in (360, 540, 720, 1080):
        w, h = ladder_dimensions(aspect_label, rung, src_meta=src_meta)
        s = ABR_RUNG_SETTINGS[rung]
        vbps = int(re.sub(r"[^0-9]", "", s["video_bitrate"])) * 1000
        abps = int(re.sub(r"[^0-9]", "", s["audio_bitrate"])) * 1000
        out.append({"name": f"{rung}p", "rung": rung, "width": w, "height": h, **s, "bandwidth": vbps + abps, "avg_bandwidth": vbps})
    return out


def build_single_rendition_hls_cmd(input_source: str, manifest_path: str, segment_pattern: str, aspect_label: str = "Source / Passthrough", video_bitrate: str = "4500k", audio_bitrate: str = "128k", preset: str = "veryfast", fps: Optional[int] = None, live: bool = False, segment_seconds: int = DEFAULT_SEGMENT_SECONDS, list_size: int = DEFAULT_LIVE_LIST_SIZE) -> list[str]:
    vf = None
    if aspect_label != "Source / Passthrough":
        w, h = ladder_dimensions(aspect_label, 1080)
        vf = variant_video_filter(w, h)
    cmd = ["ffmpeg", "-y"]
    if live:
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-thread_queue_size", "1024"]
    cmd += ["-i", input_source]
    if vf:
        cmd += ["-vf", vf]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.1", "-b:v", video_bitrate, "-maxrate", video_bitrate, "-bufsize", f"{max(int(re.sub(r'[^0-9]', '', video_bitrate) or '4500') * 2, 2000)}k", "-g", str(_gop_for_fps(fps)), "-keyint_min", str(_gop_for_fps(fps)), "-sc_threshold", "0", "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2"]
    if live:
        cmd += ["-f", "hls", "-hls_time", str(segment_seconds), "-hls_list_size", str(list_size), "-hls_flags", "delete_segments+append_list+independent_segments+program_date_time", "-hls_segment_filename", segment_pattern, manifest_path]
    else:
        cmd += ["-f", "hls", "-hls_time", str(segment_seconds), "-hls_playlist_type", "vod", "-hls_list_size", "0", "-hls_flags", "independent_segments", "-hls_segment_filename", segment_pattern, manifest_path]
    return cmd


def build_multi_variant_hls_cmd(input_source: str, out_dir: str, aspect_label: str = "16:9 Landscape", preset: str = "veryfast", fps: Optional[int] = None, live: bool = False, segment_seconds: int = DEFAULT_SEGMENT_SECONDS, list_size: int = DEFAULT_LIVE_LIST_SIZE, src_meta: Optional[dict] = None) -> tuple[list[str], list[dict], str]:
    os.makedirs(out_dir, exist_ok=True)
    ladder = build_abr_ladder(aspect_label=aspect_label, src_meta=src_meta)
    split_labels = [f"v{i}" for i in range(len(ladder))]
    out_labels = [f"v{i}out" for i in range(len(ladder))]
    filter_parts = [f"[0:v]split={len(ladder)}" + "".join(f"[{x}]" for x in split_labels)]
    for idx, variant in enumerate(ladder):
        vf = variant_video_filter(variant["width"], variant["height"])
        if fps:
            vf += f",fps={fps}"
        filter_parts.append(f"[{split_labels[idx]}]{vf}[{out_labels[idx]}]")
    cmd = ["ffmpeg", "-y"]
    if live:
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay", "-thread_queue_size", "1024"]
    cmd += ["-i", input_source, "-filter_complex", ";".join(filter_parts)]
    for idx in range(len(ladder)):
        cmd += ["-map", f"[{out_labels[idx]}]", "-map", "0:a:0?"]
    for idx, variant in enumerate(ladder):
        cmd += [f"-c:v:{idx}", "libx264", f"-preset:v:{idx}", preset, f"-pix_fmt:v:{idx}", "yuv420p", f"-profile:v:{idx}", "high", f"-level:v:{idx}", "4.1", f"-b:v:{idx}", variant["video_bitrate"], f"-maxrate:v:{idx}", variant["maxrate"], f"-bufsize:v:{idx}", variant["bufsize"], f"-g:v:{idx}", str(_gop_for_fps(fps)), f"-keyint_min:v:{idx}", str(_gop_for_fps(fps)), f"-sc_threshold:v:{idx}", "0", f"-c:a:{idx}", "aac", f"-b:a:{idx}", variant["audio_bitrate"], f"-ar:a:{idx}", "48000", f"-ac:a:{idx}", "2"]
    master_name = "master.m3u8"
    segment_pattern = os.path.join(out_dir, "%v_%06d.ts")
    media_playlist_pattern = os.path.join(out_dir, "%v.m3u8")
    var_stream_map = " ".join(f"v:{idx},a:{idx},name:{variant['name']}" for idx, variant in enumerate(ladder))
    cmd += ["-master_pl_name", master_name, "-f", "hls", "-hls_time", str(segment_seconds), "-hls_segment_filename", segment_pattern, "-var_stream_map", var_stream_map]
    if live:
        cmd += ["-hls_list_size", str(list_size), "-hls_flags", "delete_segments+append_list+independent_segments+program_date_time"]
    else:
        cmd += ["-hls_playlist_type", "vod", "-hls_list_size", "0", "-hls_flags", "independent_segments"]
    cmd += [media_playlist_pattern]
    return cmd, ladder, os.path.join(out_dir, master_name)


def run_ffmpeg(cmd: list[str], timeout: Optional[int] = None) -> tuple[bool, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        combined = (proc.stderr or "") + ("\n" + proc.stdout if proc.stdout else "")
        return ok, ("Done" if ok else f"FFmpeg exited with code {proc.returncode}"), combined.strip()
    except subprocess.TimeoutExpired as exc:
        return False, "Timed out", str(exc)
    except Exception as exc:
        return False, str(exc), ""


def stop_live_job(job: Optional[dict]) -> None:
    if not job:
        return
    proc = job.get("proc")
    if not proc:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    except Exception:
        pass


@dataclass
class BuildHlsResult:
    ok: bool
    message: str
    out_dir: str
    master_manifest: str
    zip_bytes: bytes | None
    ffmpeg_log: str
    ladder: list[dict]
    manifest_url: str | None = None
    server_info: dict | None = None


def build_vod_hls_package(input_source: str, asset_name: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, abr_enabled: bool, src_meta: Optional[dict] = None) -> BuildHlsResult:
    out_dir = tempfile.mkdtemp(prefix="videoforge_hls_")
    ensure_clean_dir(out_dir)
    if abr_enabled:
        cmd, ladder, master_manifest = build_multi_variant_hls_cmd(input_source=input_source, out_dir=out_dir, aspect_label=aspect_label, preset=preset, fps=fps, live=False, segment_seconds=segment_seconds, src_meta=src_meta)
    else:
        token = safe_token(Path(asset_name).stem)
        master_manifest = os.path.join(out_dir, f"{token}.m3u8")
        segment_pattern = os.path.join(out_dir, f"{token}_%05d.ts")
        cmd = build_single_rendition_hls_cmd(input_source=input_source, manifest_path=master_manifest, segment_pattern=segment_pattern, aspect_label=aspect_label, live=False, segment_seconds=segment_seconds, preset=preset, fps=fps)
        ladder = []
    ok, msg, ffmpeg_log = run_ffmpeg(cmd)
    zbytes = zip_dir_bytes(out_dir) if ok else None
    manifest_url = None
    server_info = None
    if ok:
        server_info = start_static_file_server(out_dir)
        manifest_url = f"{server_info['base_url']}/{os.path.basename(master_manifest)}"
    return BuildHlsResult(ok=ok, message=msg, out_dir=out_dir, master_manifest=master_manifest, zip_bytes=zbytes, ffmpeg_log=ffmpeg_log, ladder=ladder, manifest_url=manifest_url, server_info=server_info)


def start_live_hls_job(input_source: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, list_size: int, abr_enabled: bool, src_meta: Optional[dict] = None) -> tuple[dict, list[dict]]:
    out_dir = tempfile.mkdtemp(prefix="videoforge_live_hls_")
    ensure_clean_dir(out_dir)
    if abr_enabled:
        cmd, ladder, master_manifest = build_multi_variant_hls_cmd(input_source=input_source, out_dir=out_dir, aspect_label=aspect_label, preset=preset, fps=fps, live=True, segment_seconds=segment_seconds, list_size=list_size, src_meta=src_meta)
    else:
        master_manifest = os.path.join(out_dir, "live.m3u8")
        segment_pattern = os.path.join(out_dir, "live_%05d.ts")
        cmd = build_single_rendition_hls_cmd(input_source=input_source, manifest_path=master_manifest, segment_pattern=segment_pattern, aspect_label=aspect_label, live=True, segment_seconds=segment_seconds, list_size=list_size, preset=preset, fps=fps)
        ladder = []
    log_path = os.path.join(out_dir, "ffmpeg_live.log")
    log_fp = open(log_path, "w", encoding="utf-8")
    server_info = start_static_file_server(out_dir)
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    manifest_url = f"{server_info['base_url']}/{os.path.basename(master_manifest)}"
    return {"proc": proc, "out_dir": out_dir, "master_manifest": master_manifest, "manifest_url": manifest_url, "server_info": server_info, "log_path": log_path, "input_source": input_source, "abr_enabled": abr_enabled}, ladder
