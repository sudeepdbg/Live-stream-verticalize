
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
DEFAULT_SEGMENT_SECONDS = 2
DEFAULT_LIVE_LIST_SIZE = 4
DEFAULT_TRANSITION_SEC = 0.35
MAX_UPLOAD_MB = 400

ASPECT_PRESETS = [
    "9:16 Vertical (TikTok Live)",
]

_HTTP_SERVERS: dict[str, dict] = {}

@dataclass
class LiveJob:
    proc: subprocess.Popen
    out_dir: str
    manifest_path: str
    manifest_url: str
    server_info: dict
    log_path: str
    reframed_path: str | None


def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def ensure_clean_dir(path: str | os.PathLike[str]) -> None:
    p = str(path)
    os.makedirs(p, exist_ok=True)
    for name in os.listdir(p):
        fp = os.path.join(p, name)
        try:
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            else:
                os.unlink(fp)
        except Exception:
            pass


def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "stream"


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
    info = {"directory": key, "host": host, "port": port, "base_url": f"http://{host}:{port}", "server": httpd, "thread": thread}
    _HTTP_SERVERS[key] = info
    return info


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
        "has_audio": False,
    }
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path
        ], text=True, stderr=subprocess.DEVNULL, timeout=30)
        data = json.loads(out)
        fmt = data.get("format", {})
        res["duration"] = float(fmt.get("duration", 0) or 0)
        res["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and res["width"] == 0:
                res["width"] = int(stream.get("width", 0) or 0)
                res["height"] = int(stream.get("height", 0) or 0)
                res["vcodec"] = stream.get("codec_name", "unknown")
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
    except Exception:
        pass
    return res


def _tiktok_crop_box(src_w: int, src_h: int) -> tuple[int, int]:
    # crop window with 9:16 aspect ratio that fits inside source
    if src_w / src_h >= 9/16:
        crop_h = src_h
        crop_w = int(round(src_h * 9 / 16))
    else:
        crop_w = src_w
        crop_h = int(round(src_w * 16 / 9))
    crop_w = max(32, crop_w - (crop_w % 2))
    crop_h = max(32, crop_h - (crop_h % 2))
    return crop_w, crop_h


def _bbox_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smart_reframe_vertical(
    input_path: str,
    output_path: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    smooth_strength: float = 0.88,
    lead_room: float = 0.18,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[bool, str]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, "Could not open input video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return False, "Invalid source dimensions"

    crop_w, crop_h = _tiktok_crop_box(src_w, src_h)
    max_x = src_w - crop_w
    max_y = src_h - crop_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        return False, "Could not create output video writer"

    # detectors
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    saliency = None
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    except Exception:
        saliency = None

    prev_gray = None
    smoothed_cx = src_w / 2
    smoothed_cy = src_h / 2
    prev_target_cx = smoothed_cx
    prev_target_cy = smoothed_cy

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        candidates = []

        # face candidates (highest priority)
        try:
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
        except Exception:
            faces = []
        for (x, y, w, h) in faces:
            candidates.append((0.65, (x, y, w, h)))

        # saliency candidate
        if saliency is not None:
            try:
                success, sal_map = saliency.computeSaliency(frame)
                if success:
                    sal_map = (sal_map * 255).astype('uint8')
                    _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        if w * h > 0.02 * src_w * src_h:
                            candidates.append((0.45, (x, y, w, h)))
            except Exception:
                pass

        # motion candidate
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diff = cv2.GaussianBlur(diff, (9, 9), 0)
            _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, None, iterations=2)
            cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 0.01 * src_w * src_h:
                    candidates.append((0.35, (x, y, w, h)))

        prev_gray = gray

        # combine candidates into target center
        if candidates:
            weight_sum = 0.0
            cx_sum = 0.0
            cy_sum = 0.0
            for weight, box in candidates:
                cx, cy = _bbox_center(box)
                weight_sum += weight
                cx_sum += cx * weight
                cy_sum += cy * weight
            target_cx = cx_sum / max(weight_sum, 1e-6)
            target_cy = cy_sum / max(weight_sum, 1e-6)
        else:
            target_cx = smoothed_cx
            target_cy = smoothed_cy

        # lead-room logic based on recent movement
        vel_x = target_cx - prev_target_cx
        vel_y = target_cy - prev_target_cy
        target_cx += vel_x * lead_room
        target_cy += vel_y * lead_room * 0.35  # less aggressive vertically
        prev_target_cx = target_cx
        prev_target_cy = target_cy

        # smoothing
        alpha = 1.0 - float(smooth_strength)
        smoothed_cx = (1 - alpha) * smoothed_cx + alpha * target_cx
        smoothed_cy = (1 - alpha) * smoothed_cy + alpha * target_cy

        # build crop box
        x0 = int(round(smoothed_cx - crop_w / 2))
        y0 = int(round(smoothed_cy - crop_h / 2))
        x0 = int(_clamp(x0, 0, max_x))
        y0 = int(_clamp(y0, 0, max_y))
        crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        if crop.size == 0:
            crop = frame
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        writer.write(crop)

        idx += 1
        if progress_cb and frame_count > 0 and idx % 5 == 0:
            progress_cb(idx / frame_count, f"Smart reframe analysing frame {idx}/{frame_count}")

    cap.release()
    writer.release()

    # pass-through / re-encode audio into final mp4 with original audio track if present
    muxed = output_path.replace('.mp4', '_muxed.mp4')
    cmd = [
        'ffmpeg', '-y', '-i', output_path, '-i', input_path,
        '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-shortest', muxed
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and os.path.exists(muxed):
        shutil.move(muxed, output_path)
    return True, 'Done'


def build_live_hls_command(
    input_path: str,
    out_dir: str,
    fps: Optional[int] = None,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
    live_list_size: int = DEFAULT_LIVE_LIST_SIZE,
    loop_input: bool = True,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
) -> tuple[list[str], str]:
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, 'live.m3u8')
    segment_pattern = os.path.join(out_dir, 'live_%05d.ts')
    cmd = ['ffmpeg', '-y']
    if loop_input:
        cmd += ['-stream_loop', '-1']
    cmd += ['-re', '-i', input_path]
    if fps:
        cmd += ['-r', str(fps)]
    gop = max((fps or 30) * 2, 48)
    cmd += [
        '-c:v', 'libx264', '-preset', 'veryfast', '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-b:v', '1400k', '-maxrate', '1498k', '-bufsize', '2100k',
        '-g', str(gop), '-keyint_min', str(gop), '-sc_threshold', '0',
        '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
        '-f', 'hls',
        '-hls_time', str(segment_seconds),
        '-hls_list_size', str(live_list_size),
        '-hls_flags', 'delete_segments+append_list+independent_segments+program_date_time',
        '-hls_segment_filename', segment_pattern,
        manifest,
    ]
    return cmd, manifest


def start_live_job_from_reframed_file(
    reframed_path: str,
    asset_name: str,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
    live_list_size: int = DEFAULT_LIVE_LIST_SIZE,
    fps: Optional[int] = None,
    loop_input: bool = True,
) -> LiveJob:
    out_dir = tempfile.mkdtemp(prefix='videoforge_live_hls_origin_')
    ensure_clean_dir(out_dir)
    server = start_static_file_server(out_dir)
    cmd, manifest = build_live_hls_command(
        reframed_path, out_dir, fps=fps, segment_seconds=segment_seconds,
        live_list_size=live_list_size, loop_input=loop_input
    )
    log_path = os.path.join(out_dir, 'ffmpeg_live.log')
    log_fp = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    manifest_url = f"{server['base_url']}/{os.path.basename(manifest)}"
    return LiveJob(proc=proc, out_dir=out_dir, manifest_path=manifest, manifest_url=manifest_url, server_info=server, log_path=log_path, reframed_path=reframed_path)


def stop_live_job(job: Optional[LiveJob]) -> None:
    if not job:
        return
    try:
        if job.proc and job.proc.poll() is None:
            job.proc.terminate()
            try:
                job.proc.wait(timeout=5)
            except Exception:
                job.proc.kill()
    except Exception:
        pass


def build_live_player_html(manifest_url: str, title: str = 'TikTok-style vertical live', autoplay: bool = True, muted: bool = True) -> str:
    autoplay_str = 'true' if autoplay else 'false'
    muted_attr = 'muted' if muted else ''
    return f"""
<div style='font-family:Inter,Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;border:1px solid #e5e7eb;border-radius:16px;padding:16px;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
    <div>
      <div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em;'>Live vertical player</div>
      <div style='font-size:20px;font-weight:700;'>{title}</div>
      <div style='font-size:12px;color:#94a3b8;word-break:break-all;'>{manifest_url}</div>
    </div>
    <div style='display:flex; gap:8px; align-items:center;'>
      <div id='liveBadge' style='padding:6px 10px; border-radius:999px; background:#7f1d1d; color:#fee2e2; font-weight:700;'>LIVE</div>
      <button id='goLiveBtn' style='padding:8px 12px; border-radius:10px; border:none; background:#1d4ed8; color:#eff6ff; cursor:pointer;'>Go live</button>
    </div>
  </div>
  <div style='display:flex; justify-content:center;'>
    <video id='video' controls playsinline {muted_attr} style='width:360px; height:640px; background:#000; border-radius:16px; object-fit:cover;'></video>
  </div>
  <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:14px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px;color:#94a3b8;'>State</div><div id='state' style='font-size:22px;font-weight:700;'>booting</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Latency</div><div id='latency' style='font-size:22px;font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Buffer ahead</div><div id='buffer' style='font-size:22px;font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Current level</div><div id='level' style='font-size:22px;font-weight:700;'>-</div></div>
  </div>
  <div style='background:#111827; border-radius:12px; padding:12px; margin-top:14px;'>
    <div style='font-size:13px;font-weight:600;margin-bottom:8px;'>Live event log</div>
    <div id='log' style='font-size:12px;line-height:1.6;color:#cbd5e1;max-height:180px;overflow:auto;'>Waiting for live playlist…</div>
  </div>
</div>
<script src='https://cdn.jsdelivr.net/npm/hls.js@latest'></script>
<script>
(function() {{
  const manifestUrl = {json.dumps(manifest_url)};
  const video = document.getElementById('video');
  const elState = document.getElementById('state');
  const elLatency = document.getElementById('latency');
  const elBuffer = document.getElementById('buffer');
  const elLevel = document.getElementById('level');
  const elLog = document.getElementById('log');
  const goLiveBtn = document.getElementById('goLiveBtn');
  function pushLog(msg) {{
    const now = new Date().toLocaleTimeString();
    elLog.innerHTML = '[' + now + '] ' + msg + '<br/>' + elLog.innerHTML.split('<br/>').slice(0,12).join('<br/>');
  }}
  function updateMetrics(hls) {{
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
    if (hls && typeof hls.latency === 'number') {{
      elLatency.textContent = hls.latency.toFixed(1) + ' s';
    }}
  }}
  function goLive(hls) {{
    if (hls && typeof hls.liveSyncPosition === 'number' && !Number.isNaN(hls.liveSyncPosition)) {{
      video.currentTime = Math.max(0, hls.liveSyncPosition);
      pushLog('Jumped to live edge');
    }}
  }}
  if (Hls.isSupported()) {{
    const hls = new Hls({{
      lowLatencyMode: true,
      backBufferLength: 20,
      liveSyncDurationCount: 2,
      liveMaxLatencyDurationCount: 4,
      maxLiveSyncPlaybackRate: 1.25,
      enableWorker: true,
      fragLoadingRetryDelay: 500,
      manifestLoadingRetryDelay: 500,
      levelLoadingRetryDelay: 500,
    }});
    hls.loadSource(manifestUrl);
    hls.attachMedia(video);
    hls.on(Hls.Events.MEDIA_ATTACHED, function() {{ elState.textContent = 'media attached'; pushLog('Media attached'); }});
    hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {{ elState.textContent = 'live playlist parsed'; pushLog('Playlist parsed with ' + data.levels.length + ' level(s)'); if ({autoplay_str}) video.play().catch(() => {{}}); }});
    hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{ const level = hls.levels[data.level]; elLevel.textContent = level ? ((level.height || '?') + 'p') : String(data.level); pushLog('Level switched'); }});
    hls.on(Hls.Events.ERROR, function(event, data) {{ pushLog('HLS error: ' + data.type + ' | ' + data.details); elState.textContent = data.fatal ? 'fatal error' : 'recoverable error'; if (data.fatal) {{ if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad(); else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError(); }} }});
    goLiveBtn.addEventListener('click', function() {{ goLive(hls); }});
    setInterval(function() {{ updateMetrics(hls); }}, 1000);
  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
    video.src = manifestUrl; elState.textContent = 'native live HLS'; if ({autoplay_str}) video.play().catch(() => {{}});
    goLiveBtn.addEventListener('click', function() {{ if (video.seekable && video.seekable.length) video.currentTime = video.seekable.end(video.seekable.length - 1); }});
  }} else {{ elState.textContent = 'unsupported'; pushLog('Browser does not support HLS playback'); }}
  ['play','pause','waiting','playing','seeking','stalled','ended','loadedmetadata','canplay'].forEach(function(evt) {{
    video.addEventListener(evt, function() {{ elState.textContent = evt; pushLog('Video event: ' + evt); }});
  }});
}})();
</script>
"""
