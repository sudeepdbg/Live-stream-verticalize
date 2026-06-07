from __future__ import annotations

import collections
import json
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 1280
WORKING_INPUT_H = 720
PLACEHOLDER_FPS = 30.0


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
    s = (source or "").lower().strip()
    return s.startswith(("rtmp://", "rtmps://", "srt://", "udp://", "tcp://", "http://", "https://"))


def _source_input_args(source: str, pace_input: bool = False, loop_file: bool = False) -> list[str]:
    args: list[str] = ["-fflags", "+genpts+discardcorrupt", "-analyzeduration", "20000000", "-probesize", "20000000"]
    if loop_file and not is_network_source(source):
        args += ["-stream_loop", "-1"]
    if pace_input and not is_network_source(source):
        args += ["-re"]
    # Generic network resiliency. Unsupported options are usually ignored by ffmpeg for protocols where they do not apply.
    if is_network_source(source):
        args += ["-rw_timeout", "15000000"]
        if source.lower().startswith(("http://", "https://")):
            args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]
    args += ["-i", source]
    return args


def _ffprobe_json(source: str, timeout: int = 30) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-analyzeduration",
        "20000000",
        "-probesize",
        "20000000",
        source,
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    return json.loads(out)


def probe_source(source: str) -> dict:
    res = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "vcodec": "unknown"}
    # Primary path: ffprobe
    try:
        data = _ffprobe_json(source, timeout=35 if is_network_source(source) else 20)
        fmt = data.get("format", {})
        res["duration"] = float(fmt.get("duration", 0) or 0)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and res["width"] == 0:
                res["width"] = int(stream.get("width", 0) or 0)
                res["height"] = int(stream.get("height", 0) or 0)
                res["vcodec"] = stream.get("codec_name", "unknown")
                try:
                    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
                    n, d = map(int, rate.split("/"))
                    res["fps"] = round(n / d, 3) if d else 0.0
                except Exception:
                    pass
                break
    except Exception:
        pass

    # Secondary fallback for local files: OpenCV
    if (res["width"] <= 0 or res["height"] <= 0 or res["fps"] <= 0) and not is_network_source(source):
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                res["width"] = res["width"] or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                res["height"] = res["height"] or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                res["fps"] = res["fps"] or float(cap.get(cv2.CAP_PROP_FPS) or 0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if not res["duration"] and frame_count > 0 and res["fps"] > 0:
                    res["duration"] = frame_count / res["fps"]
            cap.release()
        except Exception:
            pass

    # Tertiary fallback for RTMP/SRT/HTTP style sources: try to decode a single frame and keep default runtime assumptions
    if res["width"] <= 0 or res["height"] <= 0:
        if is_network_source(source):
            res["width"], res["height"] = WORKING_INPUT_W, WORKING_INPUT_H
    if res["fps"] <= 0:
        res["fps"] = PLACEHOLDER_FPS
    return res


def _vertical_crop_box(src_w: int, src_h: int) -> tuple[int, int]:
    if src_w / src_h >= 9 / 16:
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


class SmoothReframer:
    def __init__(
        self,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        smooth_strength: float = 0.97,
        analysis_stride: int = 6,
        deadzone_ratio: float = 0.06,
        max_pan_ratio: float = 0.01,
    ):
        self.src_w, self.src_h = src_w, src_h
        self.target_w, self.target_h = target_w, target_h
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.max_x, self.max_y = max(0, src_w - self.crop_w), max(0, src_h - self.crop_h)
        self.smooth_strength = float(smooth_strength)
        self.analysis_stride = max(1, int(analysis_stride))
        self.deadzone_px = max(8.0, self.crop_w * deadzone_ratio)
        self.max_pan_px = max(2.0, self.crop_w * max_pan_ratio)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.saliency = None
        try:
            if hasattr(cv2, "saliency"):
                self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            self.saliency = None
        self.prev_gray = None
        self.smoothed_cx = src_w / 2.0
        self.smoothed_cy = src_h / 2.0
        self.target_cx = self.smoothed_cx
        self.target_cy = self.smoothed_cy
        self.frame_idx = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_idx % self.analysis_stride == 0:
            candidates: list[tuple[float, tuple[float, float]]] = []
            try:
                faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
            except Exception:
                faces = []
            for (x, y, w, h) in faces[:2]:
                candidates.append((0.55, (x + w / 2.0, y + h / 2.0)))

            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                diff = cv2.GaussianBlur(diff, (11, 11), 0)
                _, motion = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
                motion = cv2.dilate(motion, None, iterations=2)
                cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pts = []
                for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h > 0.015 * self.src_w * self.src_h:
                        pts.append((x, y, w, h))
                if pts:
                    x0 = min(p[0] for p in pts)
                    y0 = min(p[1] for p in pts)
                    x1 = max(p[0] + p[2] for p in pts)
                    y1 = max(p[1] + p[3] for p in pts)
                    candidates.append((0.35, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))

            if self.saliency is not None:
                try:
                    success, sal_map = self.saliency.computeSaliency(frame)
                    if success:
                        sal_map = (sal_map * 255).astype("uint8")
                        _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            if w * h > 0.03 * self.src_w * self.src_h:
                                candidates.append((0.20, (x + w / 2.0, y + h / 2.0)))
                except Exception:
                    pass

            if candidates:
                ws = sum(weight for weight, _ in candidates)
                self.target_cx = sum(cx * weight for weight, (cx, _cy) in candidates) / max(ws, 1e-6)
                self.target_cy = sum(cy * weight for weight, (_cx, cy) in candidates) / max(ws, 1e-6)
            else:
                self.target_cx, self.target_cy = self.src_w / 2.0, self.src_h / 2.0

        self.prev_gray = gray
        dx = self.target_cx - self.smoothed_cx
        dy = self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px:
            dx = 0.0
        if abs(dy) < self.deadzone_px * 0.5:
            dy = 0.0

        alpha = 1.0 - self.smooth_strength
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-(self.max_pan_px * 0.4), min((self.max_pan_px * 0.4), dy * alpha))

        x0 = int(round(self.smoothed_cx - self.crop_w / 2.0))
        y0 = int(round(self.smoothed_cy - self.crop_h / 2.0))
        x0 = int(_clamp(x0, 0, self.max_x))
        y0 = int(_clamp(y0, 0, self.max_y))
        crop = frame[y0 : y0 + self.crop_h, x0 : x0 + self.crop_w]
        if crop.size == 0:
            crop = frame
        self.frame_idx += 1
        return cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)


def create_vertical_master(
    source_path: str,
    output_path: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    smooth_strength: float = 0.97,
    analysis_stride: int = 6,
    deadzone_ratio: float = 0.06,
    max_pan_ratio: float = 0.01,
    progress_cb: Optional[Callable[[float, str], None]] = None,
):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False, "Could not open input source"
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return False, "Invalid source dimensions"
    reframer = SmoothReframer(src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        return False, "Could not create output file"
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(reframer.process(frame))
        idx += 1
        if progress_cb and frame_count > 0 and idx % 5 == 0:
            progress_cb(idx / frame_count, f"Creating vertical master {idx}/{frame_count}")
    cap.release()
    writer.release()
    return True, "Done"


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
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(body) if body else {}
        except Exception:
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


def build_push_file_command(reframed_mp4: str, rtmps_url: str, stream_key: str, loop_input: bool = True):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    cmd = ["ffmpeg", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-re",
        "-i",
        reframed_mp4,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-b:v",
        "1500k",
        "-maxrate",
        "1700k",
        "-bufsize",
        "2500k",
        "-g",
        "60",
        "-keyint_min",
        "60",
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "flv",
        target,
    ]
    return cmd


def start_vod_to_live_push(cfg: CFStreamConfig, reframed_mp4: str, asset_name: str, loop_input: bool = True) -> LiveSession:
    live_input = create_live_input(cfg, name=safe_token(Path(asset_name).stem))
    uid = live_input["uid"]
    rtmps_url = live_input["rtmps"]["url"]
    stream_key = live_input["rtmps"]["streamKey"]
    hls_url, dash_url, iframe_url = build_public_playback_urls(cfg, uid)
    cmd = build_push_file_command(reframed_mp4, rtmps_url, stream_key, loop_input)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, text=True)
    return LiveSession(uid, rtmps_url, stream_key, hls_url, dash_url, iframe_url, cmd, proc, log_path, status="streaming")


def build_realtime_rtmps_push_command(target_w: int, target_h: int, fps: float, rtmps_url: str, stream_key: str):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(fps or PLACEHOLDER_FPS))))
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{target_w}x{target_h}",
        "-r",
        str(fps_int),
        "-i",
        "-",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-shortest",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps_int),
        "-b:v",
        "1500k",
        "-maxrate",
        "1700k",
        "-bufsize",
        "2500k",
        "-g",
        str(fps_int * 2),
        "-keyint_min",
        str(fps_int * 2),
        "-sc_threshold",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-f",
        "flv",
        target,
    ]


def _read_exact(stream, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _build_ingest_command(source: str, fps: float, pace_input: bool, loop_file: bool) -> list[str]:
    vf = (
        f"fps={max(1, int(round(fps or PLACEHOLDER_FPS)))},"
        f"scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    cmd += _source_input_args(source, pace_input=pace_input, loop_file=loop_file)
    cmd += [
        "-an",
        "-vf",
        vf,
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    return cmd


def _make_placeholder_frame(target_w: int, target_h: int, text: str = "Starting stream...") -> np.ndarray:
    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # dark blue background gradient-ish bands
    frame[:] = (18, 22, 36)
    cv2.rectangle(frame, (0, 0), (target_w, int(target_h * 0.18)), (35, 55, 98), -1)
    cv2.rectangle(frame, (0, int(target_h * 0.82)), (target_w, target_h), (24, 34, 60), -1)
    cv2.putText(frame, "Vertical stream", (28, max(48, target_h // 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (28, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Cloudflare live input priming", (28, target_h // 2 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 195, 230), 1, cv2.LINE_AA)
    return frame


def _open_ingest_process(source: str, fps: float, pace_input: bool, loop_file: bool, log_path: str) -> subprocess.Popen:
    cmd = _build_ingest_command(source, fps=fps, pace_input=pace_input, loop_file=loop_file)
    log_fp = open(log_path, "a", encoding="utf-8")
    log_fp.write("\n=== INGEST CMD ===\n" + " ".join(cmd) + "\n")
    log_fp.flush()
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fp)


def _start_output_process(session: LiveSession) -> subprocess.Popen:
    log_fp = open(session.log_path, "a", encoding="utf-8")
    log_fp.write("\n=== PUSH CMD ===\n" + " ".join(session.ffmpeg_cmd) + "\n")
    log_fp.flush()
    return subprocess.Popen(session.ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_fp, stderr=subprocess.STDOUT)


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
) -> None:
    session.status = "probing"
    info = probe_source(source)
    fps = float(info.get("fps") or PLACEHOLDER_FPS)
    src_w = WORKING_INPUT_W
    src_h = WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_frames = max(1, int(delay_seconds * fps))
    session.stats = {
        "fps": round(fps, 3),
        "delay_frames": delay_frames,
        "working_resolution": f"{src_w}x{src_h}",
        "source_reported_resolution": f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)}",
    }

    reframer = SmoothReframer(src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio)
    buffer: collections.deque[np.ndarray] = collections.deque(maxlen=max(delay_frames + 240, 480))
    placeholder = _make_placeholder_frame(target_w, target_h)
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        session.proc = _start_output_process(session)
    except Exception as exc:
        session.status = "ffmpeg_start_failed"
        session.error = str(exc)
        return

    # Prime Cloudflare immediately with placeholder frames so playback can start while the delay buffer fills.
    session.status = "priming_output"
    try:
        if session.proc.stdin:
            for _ in range(int(max(1.0, min(delay_seconds / 2.0, 3.0)) * fps)):
                if session.stop_event.is_set():
                    break
                session.proc.stdin.write(placeholder.tobytes())
    except Exception as exc:
        session.status = "ffmpeg_pipe_broken"
        session.error = f"Could not prime Cloudflare output: {exc}"
        return

    ingest = None
    try:
        session.status = "connecting_source"
        ingest = _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=session.log_path)
        next_deadline = time.time()
        frames_in = 0
        frames_out = 0
        source_ended = False

        while not session.stop_event.is_set():
            if not source_ended:
                raw = _read_exact(ingest.stdout, frame_bytes) if ingest and ingest.stdout else b""
                if len(raw) < frame_bytes:
                    source_ended = True
                else:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))
                    processed = reframer.process(frame)
                    buffer.append(processed)
                    frames_in += 1

            frame_to_write = None
            if len(buffer) >= delay_frames:
                session.status = "streaming"
                frame_to_write = buffer.popleft()
            elif not source_ended:
                session.status = "buffering"
                frame_to_write = placeholder
            elif buffer:
                session.status = "draining"
                frame_to_write = buffer.popleft()
            else:
                session.status = "source_ended"
                break

            if session.proc and session.proc.stdin and frame_to_write is not None:
                try:
                    session.proc.stdin.write(frame_to_write.tobytes())
                    frames_out += 1
                except Exception as exc:
                    session.status = "ffmpeg_pipe_broken"
                    session.error = str(exc)
                    break

            next_deadline += frame_interval
            sleep_for = next_deadline - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

            if frames_out % int(max(1.0, fps)) == 0:
                session.stats.update(
                    {
                        "frames_in": frames_in,
                        "frames_out": frames_out,
                        "buffer_len": len(buffer),
                        "delay_seconds": round(delay_frames / max(fps, 1.0), 2),
                    }
                )

    except Exception as exc:
        session.status = "worker_error"
        session.error = str(exc)
    finally:
        try:
            if ingest and ingest.poll() is None:
                ingest.terminate()
                try:
                    ingest.wait(timeout=3)
                except Exception:
                    ingest.kill()
        except Exception:
            pass
        try:
            if session.proc and session.proc.stdin:
                session.proc.stdin.close()
        except Exception:
            pass
        if session.status not in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed"}:
            session.status = "stopped"


def start_realtime_delayed_vertical_push(
    cfg: CFStreamConfig,
    source: str,
    asset_name: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 20.0,
    smooth_strength: float = 0.97,
    analysis_stride: int = 6,
    deadzone_ratio: float = 0.06,
    max_pan_ratio: float = 0.01,
    loop_file: bool = False,
    pace_input: bool = True,
) -> LiveSession:
    live_input = create_live_input(cfg, name=safe_token(Path(asset_name).stem))
    uid = live_input["uid"]
    rtmps_url = live_input["rtmps"]["url"]
    stream_key = live_input["rtmps"]["streamKey"]
    hls_url, dash_url, iframe_url = build_public_playback_urls(cfg, uid)
    fps = probe_source(source).get("fps") or PLACEHOLDER_FPS
    ffmpeg_cmd = build_realtime_rtmps_push_command(target_w, target_h, fps, rtmps_url, stream_key)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    session = LiveSession(uid, rtmps_url, stream_key, hls_url, dash_url, iframe_url, ffmpeg_cmd, None, log_path)
    worker = threading.Thread(
        target=_realtime_worker,
        args=(session, source, target_w, target_h, delay_seconds, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, loop_file, pace_input),
        daemon=True,
    )
    session.worker = worker
    worker.start()
    return session


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
