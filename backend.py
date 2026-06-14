from __future__ import annotations

import collections
import json
import math
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Deque
import cv2
import numpy as np
import queue
import select
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 1280
WORKING_INPUT_H = 720
PLACEHOLDER_FPS = 30.0
DEFAULT_OUTPUT_FPS = 30.0
DEFAULT_VIDEO_BITRATE = "3500k"
DEFAULT_MAXRATE = "3500k"
DEFAULT_BUFSIZE = "7000k"

# New constants for real‑time robustness
INGEST_READ_TIMEOUT = 0.75
INGEST_READ_TIMEOUT_MIN = 0.25
INGEST_READ_TIMEOUT_MAX = 1.25
MAX_CONSECUTIVE_STALLS_NON_LOOP = 12
MAX_RAW_QUEUE_SIZE = 30
MAX_PROCESSED_QUEUE_SIZE = 600

# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False

def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "stream"

def is_network_source(source: str) -> bool:
    s = (source or " ").lower().strip()
    return s.startswith(("rtmp://", "rtmps://", "srt://", "udp://", "tcp://", "http://", "https://"))

def _source_input_args(source: str, pace_input: bool = False, loop_file: bool = False) -> list[str]:
    args = [
        "-fflags", "+genpts+discardcorrupt",
        "-analyzeduration", "20000000",
        "-probesize", "20000000",
    ]
    if loop_file and not is_network_source(source):
        args += ["-stream_loop", "-1"]
    if pace_input and not is_network_source(source):
        args += ["-re"]
    if is_network_source(source):
        args += ["-rw_timeout", "15000000"]
    if source.lower().startswith(("http://", "https://")):
        args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]
    args += ["-i", source]
    return args

def _safe_json_loads(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}

def _ffprobe_json(source: str, timeout: int = 30) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format",
        "-analyzeduration", "20000000", "-probesize", "20000000", source,
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    return _safe_json_loads(out)

def probe_source(source: str) -> dict:
    res = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "vcodec": "unknown"}
    try:
        data = _ffprobe_json(source, timeout=35 if is_network_source(source) else 20)
        fmt = data.get("format", {}) if isinstance(data, dict) else {}
        res["duration"] = float(fmt.get("duration", 0) or 0)
        for stream in (data.get("streams", []) if isinstance(data, dict) else []):
            if stream.get("codec_type") == "video" and res["width"] == 0:
                res["width"] = int(stream.get("width", 0) or 0)
                res["height"] = int(stream.get("height", 0) or 0)
                res["vcodec"] = str(stream.get("codec_name", "unknown"))
                try:
                    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
                    n, d = map(int, rate.split("/"))
                    res["fps"] = round(n / d, 3) if d else 0.0
                except Exception:
                    pass
                break
    except Exception:
        pass

    if (res["width"] <= 0 or res["height"] <= 0 or res["fps"] <= 0) and not is_network_source(source):
        cap = None
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                res["width"] = res["width"] or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                res["height"] = res["height"] or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                res["fps"] = res["fps"] or float(cap.get(cv2.CAP_PROP_FPS) or 0)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if not res["duration"] and fc > 0 and res["fps"] > 0:
                    res["duration"] = fc / res["fps"]
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    if res["width"] <= 0 or res["height"] <= 0:
        if is_network_source(source):
            res["width"], res["height"] = WORKING_INPUT_W, WORKING_INPUT_H

    if res["fps"] <= 0:
        res["fps"] = PLACEHOLDER_FPS
    return res

def _vertical_crop_box(src_w: int, src_h: int) -> tuple[int, int]:
    if src_w / max(src_h, 1) >= 9 / 16:
        crop_h = src_h
        crop_w = int(round(src_h * 9 / 16))
    else:
        crop_w = src_w
        crop_h = int(round(src_w * 16 / 9))
    crop_w = max(32, crop_w - (crop_w % 2))
    crop_h = max(32, crop_h - (crop_h % 2))
    return crop_w, crop_h

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0 or width <= 0 or height <= 0:
        return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = max(width / max(w, 1), height / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    x0 = max(0, (nw - width) // 2)
    y0 = max(0, (nh - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width]

# ===========================================================================
# Detection classes (unchanged from stable version)
# OverlayDetector, SceneChangeDetector, BallTracker, _TrackedFace, PanelTracker,
# SmoothReframer – exactly as in "backend -stable version old.py".
# (They are omitted here for brevity but must be copied verbatim.)
# ===========================================================================

# ---------------------------------------------------------------------------
# Offline vertical master generation (unchanged from stable version)
# ---------------------------------------------------------------------------
def create_vertical_master(
    source_path: str,
    output_path: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    smooth_strength: float = 0.975,
    analysis_stride: int = 4,
    deadzone_ratio: float = 0.05,
    max_pan_ratio: float = 0.012,
    sport_profile: str = "auto",
    ball_tracking: bool = True,
    overlay_composite: bool = True,
    preserve_bottom_overlay: bool = False,
    panel_mode: bool = False,
    panel_max_faces: int = 4,
    panel_detection_stride: int = 2,
    panel_gap: int = 4,
    progress_cb: Optional[Callable[[float, str], None]] = None,
):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False, "Could not open input source"
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return False, "Invalid source dimensions"

    reframer = SmoothReframer(
        src_w, src_h, target_w, target_h,
        smooth_strength=smooth_strength,
        analysis_stride=analysis_stride,
        deadzone_ratio=deadzone_ratio,
        max_pan_ratio=max_pan_ratio,
        sport_profile=sport_profile,
        ball_tracking=ball_tracking,
        overlay_composite=overlay_composite,
        preserve_bottom_overlay=preserve_bottom_overlay,
        panel_mode=panel_mode,
        panel_max_faces=panel_max_faces,
        panel_detection_stride=panel_detection_stride,
        panel_gap=panel_gap,
    )
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else DEFAULT_OUTPUT_FPS,
        (target_w, target_h),
    )
    if not writer.isOpened():
        cap.release()
        return False, "Could not create output file"

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(reframer.process(frame))
            idx += 1
            if progress_cb and frame_count > 0 and idx % 5 == 0:
                progress_cb(idx / frame_count, f"Creating vertical master {idx}/{frame_count}")
    finally:
        cap.release()
        writer.release()
    return True, "Done"

# ---------------------------------------------------------------------------
# Cloudflare Stream live push helpers (improved decoupled version)
# ---------------------------------------------------------------------------
@dataclass
class CFStreamConfig:
    account_id: str
    api_token: str
    customer_code: str
    prefer_low_latency: bool = False

@dataclass
class LiveSession:
    uid: str
    rtmps_url: str
    stream_key: str
    hls_url: str
    dash_url: str
    iframe_url: str
    ffmpeg_cmd: list[str]
    proc: Optional[subprocess.Popen]
    log_path: str
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker: Optional[threading.Thread] = None
    status: str = "created"
    stats: dict = field(default_factory=dict)
    error: str = ""

def cfstream_config_from_inputs(account_id: str, api_token: str, customer_code: str, prefer_low_latency: bool = False) -> CFStreamConfig:
    if not account_id:
        raise ValueError("Cloudflare account ID is required.")
    if not api_token:
        raise ValueError("Cloudflare API token is required.")
    if not customer_code:
        raise ValueError("Cloudflare customer code is required.")
    code = customer_code.strip().replace("customer-", "").replace(".cloudflarestream.com", "").strip("/")
    return CFStreamConfig(account_id.strip(), api_token.strip(), code, bool(prefer_low_latency))

def _cf_api_request(cfg: CFStreamConfig, method: str, path: str, payload: Optional[dict] = None):
    url = f"https://api.cloudflare.com/client/v4{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {cfg.api_token}",
        "Content-Type": "application/json",
        "User-Agent": "DualFlow-Vertical-Cloudflare",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (_safe_json_loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        parsed = _safe_json_loads(body) if body else {}
        if not parsed:
            parsed = {"success": False, "errors": [{"message": body}]}
        return exc.code, parsed

def create_live_input(cfg: CFStreamConfig, name: str, recording_mode: str = "automatic") -> dict:
    payload = {
        "meta": {"name": name},
        "recording": {"mode": recording_mode, "timeoutSeconds": 0},
        "preferLowLatency": bool(cfg.prefer_low_latency),
        "enabled": True,
    }
    status, parsed = _cf_api_request(cfg, "POST", f"/accounts/{cfg.account_id}/stream/live_inputs", payload)
    if status not in (200, 201) or not parsed.get("success"):
        raise RuntimeError(f"Create live input failed: {parsed}")
    return parsed["result"]

def disable_live_input(cfg: CFStreamConfig, uid: str) -> None:
    _cf_api_request(cfg, "PUT", f"/accounts/{cfg.account_id}/stream/live_inputs/{uid}", {"enabled": False})

def build_public_playback_urls(cfg: CFStreamConfig, uid: str):
    base = f"https://customer-{cfg.customer_code}.cloudflarestream.com/{uid}"
    hls = f"{base}/manifest/video.m3u8" + ("?protocol=llhls" if cfg.prefer_low_latency else "")
    dash = f"{base}/manifest/video.mpd"
    iframe = f"{base}/iframe?autoplay=true&muted=true&controls=true&preload=metadata"
    return hls, dash, iframe

def build_realtime_rtmps_push_command(target_w: int, target_h: int, fps: float, rtmps_url: str, stream_key: str):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{target_w}x{target_h}",
        "-r", str(fps_int), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * 2),
        "-keyint_min", str(fps_int * 2),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int * 2}:min-keyint={fps_int * 2}",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-flvflags", "no_duration_filesize",
        "-f", "flv", target,
    ]

def _build_ingest_command(source: str, fps: float, pace_input: bool, loop_file: bool) -> list[str]:
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf = (
        f"fps={fps_int},"
        f"scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    effective_pace = pace_input if is_network_source(source) else False
    cmd += _source_input_args(source, pace_input=effective_pace, loop_file=loop_file)
    cmd += ["-an", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd

def _make_placeholder_frame(target_w: int, target_h: int, text: str = "Starting stream...") -> np.ndarray:
    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    frame[:] = (18, 22, 36)
    cv2.rectangle(frame, (0, 0), (target_w, int(target_h * 0.18)), (35, 55, 98), -1)
    cv2.rectangle(frame, (0, int(target_h * 0.82)), (target_w, target_h), (24, 34, 60), -1)
    cv2.putText(frame, "Vertical stream", (28, max(48, target_h // 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (28, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Cloudflare live input priming", (28, target_h // 2 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 195, 230), 1, cv2.LINE_AA)
    return frame

def _read_frame_timeout(proc: subprocess.Popen, nbytes: int, timeout: float = INGEST_READ_TIMEOUT) -> Optional[bytes]:
    if proc is None or proc.stdout is None:
        return None
    timeout = max(INGEST_READ_TIMEOUT_MIN, min(float(timeout or INGEST_READ_TIMEOUT), INGEST_READ_TIMEOUT_MAX))
    if os.name == "nt":
        q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=1)
        def _reader():
            try:
                data = proc.stdout.read(nbytes)
                q.put(data, block=False)
            except Exception:
                q.put(None, block=False)
        threading.Thread(target=_reader, daemon=True).start()
        try:
            data = q.get(timeout=timeout)
        except queue.Empty:
            return None
        return data if data and len(data) == nbytes else None
    else:
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        data = bytearray()
        while len(data) < nbytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                ready, _, _ = select.select([fd], [], [], min(remaining, 0.05))
            except Exception:
                return None
            if not ready:
                if proc.poll() is not None:
                    return None
                continue
            try:
                chunk = os.read(fd, nbytes - len(data))
            except OSError:
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

def _open_ingest_process(source: str, fps: float, pace_input: bool, loop_file: bool, log_path: str) -> subprocess.Popen:
    cmd = _build_ingest_command(source, fps=fps, pace_input=pace_input, loop_file=loop_file)
    log_fp = open(log_path, "a", encoding="utf-8")
    log_fp.write("\n=== INGEST CMD ===\n" + " ".join(cmd) + "\n")
    log_fp.flush()
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fp, bufsize=0)

def _start_output_process(session: LiveSession) -> subprocess.Popen:
    log_fp = open(session.log_path, "a", encoding="utf-8")
    log_fp.write("\n=== PUSH CMD ===\n" + " ".join(session.ffmpeg_cmd) + "\n")
    log_fp.flush()
    return subprocess.Popen(session.ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_fp, stderr=subprocess.STDOUT, bufsize=0)

def _terminate_process(proc: Optional[subprocess.Popen], timeout: float = 8.0) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        pass

def _realtime_worker(
    session: LiveSession,
    source: str,
    target_w: int,
    target_h: int,
    delay_seconds: float,
    smooth_strength: float,
    analysis_stride: int,
    deadzone_ratio: float,
    max_pan_ratio: float,
    loop_file: bool,
    pace_input: bool,
    sport_profile: str,
    ball_tracking: bool,
    overlay_composite: bool,
    preserve_bottom_overlay: bool,
    panel_mode: bool = False,
    panel_max_faces: int = 4,
    panel_detection_stride: int = 2,
    panel_gap: int = 4,
) -> None:
    session.status = "probing"
    info = probe_source(source)
    fps = DEFAULT_OUTPUT_FPS
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_frames = max(1, int(round(delay_seconds * fps)))
    max_delay_buffer = delay_frames + 240

    counters = {
        "frames_in": 0, "frames_processed": 0, "frames_out": 0,
        "frames_dropped_raw_queue": 0, "frames_dropped_processing": 0,
        "source_stalls": 0, "consecutive_source_stalls": 0,
        "ingest_restarts": 0, "output_write_failures": 0,
        "placeholder_frames": 0,
    }
    read_samples: Deque[float] = collections.deque(maxlen=180)
    process_samples: Deque[float] = collections.deque(maxlen=180)
    write_samples: Deque[float] = collections.deque(maxlen=180)
    drift_samples: Deque[float] = collections.deque(maxlen=180)

    def p95(vals: Deque[float]) -> float:
        if not vals:
            return 0.0
        sorted_vals = sorted(vals)
        idx = min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * 0.95)))
        return float(sorted_vals[idx])

    raw_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=MAX_RAW_QUEUE_SIZE)
    processed_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=MAX_PROCESSED_QUEUE_SIZE)

    stop = session.stop_event
    ingest_proc_holder: dict[str, Optional[subprocess.Popen]] = {"proc": None}
    output_proc = None

    def ingest_loop():
        seq = 0
        def restart_ingest():
            nonlocal seq
            _terminate_process(ingest_proc_holder["proc"])
            counters["ingest_restarts"] += 1
            if counters["consecutive_source_stalls"] > MAX_CONSECUTIVE_STALLS_NON_LOOP and not is_network_source(source) and not loop_file:
                return None
            return _open_ingest_process(source, fps, pace_input, loop_file, session.log_path)

        ingest_proc_holder["proc"] = restart_ingest()
        first_frame = True
        while not stop.is_set():
            proc = ingest_proc_holder.get("proc")
            if proc is None or proc.poll() is not None:
                counters["source_stalls"] += 1
                counters["consecutive_source_stalls"] += 1
                ingest_proc_holder["proc"] = restart_ingest()
                time.sleep(0.02)
                continue
            read_start = time.monotonic()
            raw = _read_frame_timeout(proc, frame_bytes, INGEST_READ_TIMEOUT)
            read_ms = (time.monotonic() - read_start) * 1000
            read_samples.append(read_ms)
            if raw is None or len(raw) != frame_bytes:
                counters["source_stalls"] += 1
                counters["consecutive_source_stalls"] += 1
                if counters["consecutive_source_stalls"] >= 3:
                    ingest_proc_holder["proc"] = restart_ingest()
                continue
            counters["consecutive_source_stalls"] = 0
            seq += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
            counters["frames_in"] += 1
            if first_frame:
                first_frame = False
                session.stats["startup_ms_to_first_source_frame"] = round((time.monotonic() - start_time) * 1000, 2)
            try:
                raw_queue.put(frame, block=False)
            except queue.Full:
                counters["frames_dropped_raw_queue"] += 1
                try:
                    raw_queue.get_nowait()
                except queue.Empty:
                    pass
                raw_queue.put(frame, block=False)

    def processing_loop():
        reframer = SmoothReframer(
            src_w, src_h, target_w, target_h,
            smooth_strength=smooth_strength,
            analysis_stride=analysis_stride,
            deadzone_ratio=deadzone_ratio,
            max_pan_ratio=max_pan_ratio,
            sport_profile=sport_profile,
            ball_tracking=ball_tracking,
            overlay_composite=overlay_composite,
            preserve_bottom_overlay=preserve_bottom_overlay,
            panel_mode=panel_mode,
            panel_max_faces=panel_max_faces,
            panel_detection_stride=panel_detection_stride,
            panel_gap=panel_gap,
        )
        while not stop.is_set():
            try:
                raw_frame = raw_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if raw_frame is None:
                continue
            proc_start = time.monotonic()
            try:
                out_frame = reframer.process(raw_frame)
            except Exception as e:
                out_frame = _make_placeholder_frame(target_w, target_h, "Processing error")
                session.error = str(e)
            proc_ms = (time.monotonic() - proc_start) * 1000
            process_samples.append(proc_ms)
            counters["frames_processed"] += 1
            if counters["frames_processed"] % fps == 0:
                with session.stats_lock:
                    if reframer.ball_tracker is not None:
                        session.stats["ball_confidence"] = round(reframer.ball_tracker.conf, 3)
                    if reframer.panel_tracker is not None:
                        session.stats["panel_active_faces"] = reframer.panel_tracker.active_count
                        session.stats["panel_detector"] = getattr(reframer.panel_tracker, "detector_backend_name", "-")
            try:
                processed_queue.put(out_frame, block=False)
            except queue.Full:
                counters["frames_dropped_processing"] += 1
                try:
                    processed_queue.get_nowait()
                except queue.Empty:
                    pass
                processed_queue.put(out_frame, block=False)

    def output_loop():
        nonlocal output_proc
        try:
            output_proc = _start_output_process(session)
        except Exception as exc:
            session.status = "ffmpeg_start_failed"
            session.error = str(exc)
            stop.set()
            return

        try:
            prime_count = min(15, max(1, int(fps * 0.5)))
            placeholder = _make_placeholder_frame(target_w, target_h)
            for _ in range(prime_count):
                if stop.is_set():
                    break
                output_proc.stdin.write(placeholder.tobytes())
                counters["placeholder_frames"] += 1
        except Exception as exc:
            session.status = "ffmpeg_pipe_broken"
            session.error = f"Could not prime output: {exc}"
            stop.set()
            return

        delay_buffer: Deque[np.ndarray] = collections.deque(maxlen=max_delay_buffer)
        buffer_lock = threading.Lock()

        def feeder():
            while not stop.is_set():
                try:
                    frame = processed_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if frame is None:
                    continue
                with buffer_lock:
                    delay_buffer.append(frame)

        feeder_thread = threading.Thread(target=feeder, daemon=True, name="delay_feeder")
        feeder_thread.start()

        frame_interval = 1.0 / fps
        next_deadline = time.monotonic()
        while not stop.is_set():
            with buffer_lock:
                if len(delay_buffer) >= delay_frames:
                    frame_to_write = delay_buffer.popleft()
                    session.status = "streaming"
                elif len(delay_buffer) > 0:
                    frame_to_write = delay_buffer.popleft()
                    session.status = "draining"
                else:
                    frame_to_write = _make_placeholder_frame(target_w, target_h, "Buffering...")
                    counters["placeholder_frames"] += 1
                    session.status = "buffering"

            if output_proc is None or output_proc.stdin is None or output_proc.poll() is not None:
                session.status = "ffmpeg_pipe_broken"
                break

            write_start = time.monotonic()
            try:
                output_proc.stdin.write(frame_to_write.tobytes())
                counters["frames_out"] += 1
                write_ms = (time.monotonic() - write_start) * 1000
                write_samples.append(write_ms)
            except Exception as exc:
                counters["output_write_failures"] += 1
                session.status = "ffmpeg_pipe_broken"
                session.error = str(exc)
                break

            next_deadline += frame_interval
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                stop.wait(timeout=sleep_for)
            else:
                drift_samples.append(max(0, -sleep_for * 1000))
                next_deadline = time.monotonic()

            if counters["frames_out"] % max(1, int(fps)) == 0:
                now = time.monotonic()
                if not hasattr(session, "_last_stats_ts"):
                    session._last_stats_ts = now
                    session._last_frames_out = counters["frames_out"]
                else:
                    elapsed = now - session._last_stats_ts
                    if elapsed >= 0.9:
                        fps_out = (counters["frames_out"] - session._last_frames_out) / elapsed
                        session._last_stats_ts = now
                        session._last_frames_out = counters["frames_out"]
                        with buffer_lock:
                            blen = len(delay_buffer)
                        session.stats.update({
                            "health": "healthy" if fps_out >= fps * 0.9 else "output_fps_low",
                            "fps_in": round(counters["frames_in"] / max(1, time.monotonic() - start_time), 1),
                            "fps_processed": round(counters["frames_processed"] / max(1, time.monotonic() - start_time), 1),
                            "fps_out": round(fps_out, 1),
                            **counters,
                            "p95_ingest_read_ms": round(p95(read_samples), 2),
                            "p95_process_ms": round(p95(process_samples), 2),
                            "p95_output_write_ms": round(p95(write_samples), 2),
                            "avg_schedule_drift_ms": round(sum(drift_samples) / max(len(drift_samples), 1), 2),
                            "p95_schedule_drift_ms": round(p95(drift_samples), 2),
                            "buffer_len": blen,
                            "buffer_seconds": round(blen / fps, 3),
                            "buffer_fill_pct": round(100 * blen / max_delay_buffer, 1),
                            "ffmpeg_alive": output_proc.poll() is None,
                            "ingest_alive": ingest_proc_holder.get("proc") is not None and ingest_proc_holder["proc"].poll() is None,
                            "updated_at_ms": int(time.time() * 1000),
                        })

        feeder_thread.join(timeout=1)

    start_time = time.monotonic()
    session.stats_lock = threading.Lock()
    session.stats = {
        "fps": fps,
        "delay_frames": delay_frames,
        "working_resolution": f"{src_w}x{src_h}",
        "source_reported_resolution": f"{info.get('width', 0)}x{info.get('height', 0)}",
        "sport_profile": sport_profile,
        "panel_mode": panel_mode,
        "pipeline": "decoupled_threads_v2",
    }

    threads = [
        threading.Thread(target=ingest_loop, daemon=True, name="ingest"),
        threading.Thread(target=processing_loop, daemon=True, name="process"),
        threading.Thread(target=output_loop, daemon=True, name="output"),
    ]
    for t in threads:
        t.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
            if output_proc is not None and output_proc.poll() is not None:
                session.status = "ffmpeg_exited"
                break
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        _terminate_process(ingest_proc_holder.get("proc"))
        if output_proc:
            try:
                output_proc.stdin.close()
            except Exception:
                pass
            _terminate_process(output_proc)
        if session.status not in {"ffmpeg_pipe_broken", "ffmpeg_start_failed", "ffmpeg_exited"}:
            session.status = "stopped"

def start_realtime_delayed_vertical_push(
    cfg: CFStreamConfig,
    source: str,
    asset_name: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 5.0,
    smooth_strength: float = 0.975,
    analysis_stride: int = 4,
    deadzone_ratio: float = 0.05,
    max_pan_ratio: float = 0.012,
    loop_file: bool = False,
    pace_input: bool = True,
    sport_profile: str = "auto",
    ball_tracking: bool = True,
    overlay_composite: bool = True,
    preserve_bottom_overlay: bool = False,
    panel_mode: bool = False,
    panel_max_faces: int = 4,
    panel_detection_stride: int = 2,
    panel_gap: int = 4,
) -> LiveSession:
    live_input = create_live_input(cfg, name=safe_token(Path(asset_name).stem))
    uid = live_input["uid"]
    rtmps_url = live_input["rtmps"]["url"]
    stream_key = live_input["rtmps"]["streamKey"]
    hls_url, dash_url, iframe_url = build_public_playback_urls(cfg, uid)
    ffmpeg_cmd = build_realtime_rtmps_push_command(target_w, target_h, DEFAULT_OUTPUT_FPS, rtmps_url, stream_key)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    session = LiveSession(uid, rtmps_url, stream_key, hls_url, dash_url, iframe_url, ffmpeg_cmd, None, log_path)
    worker = threading.Thread(
        target=_realtime_worker,
        args=(
            session, source, target_w, target_h, delay_seconds,
            smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio,
            loop_file, pace_input, sport_profile, ball_tracking,
            overlay_composite, preserve_bottom_overlay, panel_mode,
            panel_max_faces, panel_detection_stride, panel_gap,
        ),
        daemon=True,
    )
    session.worker = worker
    worker.start()
    return session

# ---------------------------------------------------------------------------
# VOD → Live push (simple ffmpeg, not using the decoupled pipeline)
# ---------------------------------------------------------------------------
def build_push_file_command(reframed_mp4: str, rtmps_url: str, stream_key: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-re", "-i", reframed_mp4,
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * 2),
        "-keyint_min", str(fps_int * 2),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int * 2}:min-keyint={fps_int * 2}",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-flvflags", "no_duration_filesize",
        "-f", "flv", target,
    ]
    return cmd

def start_vod_to_live_push(cfg: CFStreamConfig, reframed_mp4: str, asset_name: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS) -> LiveSession:
    live_input = create_live_input(cfg, name=safe_token(Path(asset_name).stem))
    uid = live_input["uid"]
    rtmps_url = live_input["rtmps"]["url"]
    stream_key = live_input["rtmps"]["streamKey"]
    hls_url, dash_url, iframe_url = build_public_playback_urls(cfg, uid)
    cmd = build_push_file_command(reframed_mp4, rtmps_url, stream_key, loop_input, output_fps=output_fps)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    return LiveSession(uid, rtmps_url, stream_key, hls_url, dash_url, iframe_url, cmd, proc, log_path, status="streaming")

def stop_live_session(cfg: CFStreamConfig, session: Optional[LiveSession]) -> None:
    if not session:
        return
    session.stop_event.set()
    try:
        if session.worker and session.worker.is_alive():
            session.worker.join(timeout=3)
    except Exception:
        pass
    try:
        if session.proc and session.proc.poll() is None:
            session.proc.terminate()
            try:
                session.proc.wait(timeout=5)
            except Exception:
                session.proc.kill()
    except Exception:
        pass
    try:
        disable_live_input(cfg, session.uid)
    except Exception:
        pass

def read_log_tail(path: str, max_chars: int = 12000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            return fp.read()[-max_chars:]
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Compatibility helpers for the UI (stats snapshot)
# ---------------------------------------------------------------------------
def _stats_snapshot(session: LiveSession) -> dict:
    """Return a copy of session stats with proper locking."""
    lock = getattr(session, "stats_lock", None)
    if lock:
        with lock:
            return dict(getattr(session, "stats", {}))
    return dict(getattr(session, "stats", {}))
