
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
from typing import Callable, Optional

import cv2
import numpy as np

DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 1280
WORKING_INPUT_H = 720
PLACEHOLDER_FPS = 30.0
DEFAULT_OUTPUT_FPS = 30.0
DEFAULT_VIDEO_BITRATE = "2800k"
DEFAULT_MAXRATE = "3200k"
DEFAULT_BUFSIZE = "6400k"


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
    s = (source or "").lower().strip()
    return s.startswith(("rtmp://", "rtmps://", "srt://", "udp://", "tcp://", "http://", "https://"))


def _source_input_args(source: str, pace_input: bool = False, loop_file: bool = False) -> list[str]:
    args: list[str] = [
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
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format",
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
        for stream in data.get("streams", []) if isinstance(data, dict) else []:
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
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if not res["duration"] and frame_count > 0 and res["fps"] > 0:
                    res["duration"] = frame_count / res["fps"]
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


def _mean_hsv_patch(hsv: np.ndarray, cx: float, cy: float, radius: float) -> tuple[float, float, float]:
    h, w = hsv.shape[:2]
    r = max(3, int(radius))
    x0 = max(0, int(cx - r))
    y0 = max(0, int(cy - r))
    x1 = min(w, int(cx + r + 1))
    y1 = min(h, int(cy + r + 1))
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 0.0
    mh, ms, mv = patch.reshape(-1, 3).mean(axis=0)
    return float(mh), float(ms), float(mv)


# ---------------------------------------------------------------------------
# Sport-aware reframing
# ---------------------------------------------------------------------------

class SmoothReframer:
    """
    Stable vertical reframer with multi-cue focus detection:
      - faces (broadcast close-ups)
      - motion clusters (players / action regions)
      - saliency
      - sport-aware ball tracking for basketball / cricket / soccer

    Notes:
      - output resolution is always fixed at target_w x target_h
      - ball tracking is heuristic, not ML-based, but tuned to reduce jitter
      - final center uses weighted fusion with stronger temporal smoothing
    """

    def __init__(
        self,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        smooth_strength: float = 0.975,
        analysis_stride: int = 4,
        deadzone_ratio: float = 0.05,
        max_pan_ratio: float = 0.012,
        sport_profile: str = "auto",
        ball_tracking: bool = True,
        ball_weight: float = 0.62,
        context_bias: float = 0.18,
    ):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.target_w, self.target_h = int(target_w), int(target_h)
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.max_x, self.max_y = max(0, src_w - self.crop_w), max(0, src_h - self.crop_h)
        self.smooth_strength = float(smooth_strength)
        self.analysis_stride = max(1, int(analysis_stride))
        self.deadzone_px = max(8.0, self.crop_w * deadzone_ratio)
        self.max_pan_px = max(2.0, self.crop_w * max_pan_ratio)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.saliency = None
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_motion_mask: Optional[np.ndarray] = None

        try:
            if hasattr(cv2, "saliency"):
                self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            self.saliency = None

        self.smoothed_cx = src_w / 2.0
        self.smoothed_cy = src_h / 2.0
        self.target_cx = self.smoothed_cx
        self.target_cy = self.smoothed_cy
        self.frame_idx = 0

        self.sport_profile = (sport_profile or "auto").strip().lower()
        self.ball_tracking = bool(ball_tracking)
        self.ball_weight = float(ball_weight)
        self.context_bias = float(context_bias)

        self.ball_cx = src_w / 2.0
        self.ball_cy = src_h / 2.0
        self.ball_radius = 0.0
        self.ball_conf = 0.0
        self.ball_missing = 0
        self.prev_ball_vx = 0.0
        self.prev_ball_vy = 0.0

    def _infer_sport_profile(self) -> str:
        return self.sport_profile if self.sport_profile in {"basketball", "cricket", "soccer"} else "generic"

    def _detect_motion_regions(self, gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.prev_gray is None:
            return [], None
        diff = cv2.absdiff(gray, self.prev_gray)
        diff = cv2.GaussianBlur(diff, (9, 9), 0)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)
        cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        min_area = 0.008 * self.src_w * self.src_h
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area:
                boxes.append((x, y, w, h))
        return boxes, motion

    def _score_ball_color(self, hsv: np.ndarray, cx: float, cy: float, radius: float, sport: str) -> float:
        h, s, v = _mean_hsv_patch(hsv, cx, cy, radius)
        score = 0.0
        if sport == "basketball":
            # Orange / brown-ish ball
            if 5 <= h <= 25 and s >= 70 and v >= 60:
                score = 1.0
            elif 3 <= h <= 28 and s >= 45 and v >= 45:
                score = 0.65
        elif sport == "cricket":
            # White-ball or red-ball cricket
            white_score = 1.0 if (s <= 40 and v >= 160) else 0.0
            red_score = 1.0 if (0 <= h <= 10 and s >= 90 and v >= 45) else 0.0
            alt_red = 0.75 if (170 <= h <= 179 and s >= 90 and v >= 45) else 0.0
            score = max(white_score, red_score, alt_red)
        elif sport == "soccer":
            # White-ish ball; less dependent on color because design varies
            if s <= 55 and v >= 150:
                score = 0.85
            elif s <= 80 and v >= 120:
                score = 0.45
        else:
            # generic small bright-ish object
            if v >= 150:
                score = 0.35
        return float(score)

    def _detect_ball(self, frame: np.ndarray, gray: np.ndarray, motion_mask: Optional[np.ndarray]) -> Optional[tuple[float, float, float, float]]:
        if not self.ball_tracking:
            return None

        sport = self._infer_sport_profile()
        min_dim = min(self.src_w, self.src_h)
        min_r = max(3, int(round(min_dim * 0.007)))
        max_r = max(min_r + 3, int(round(min_dim * 0.035)))

        proc_gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
        circles = cv2.HoughCircles(
            proc_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(18, int(min_dim * 0.04)),
            param1=120,
            param2=16 if sport in {"cricket", "soccer"} else 18,
            minRadius=min_r,
            maxRadius=max_r,
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best = None
        best_score = -1.0
        candidates = circles[0] if circles is not None else []

        for c in candidates[:20]:
            cx, cy, radius = float(c[0]), float(c[1]), float(c[2])

            # motion evidence around candidate
            motion_score = 0.0
            if motion_mask is not None:
                r = max(4, int(radius * 1.8))
                x0 = max(0, int(cx - r))
                y0 = max(0, int(cy - r))
                x1 = min(self.src_w, int(cx + r + 1))
                y1 = min(self.src_h, int(cy + r + 1))
                patch = motion_mask[y0:y1, x0:x1]
                if patch.size > 0:
                    motion_score = float(np.count_nonzero(patch)) / float(patch.size)

            color_score = self._score_ball_color(hsv, cx, cy, radius, sport)

            # closeness to predicted prior
            predicted_x = self.ball_cx + self.prev_ball_vx
            predicted_y = self.ball_cy + self.prev_ball_vy
            dist = math.hypot(cx - predicted_x, cy - predicted_y)
            gating = max(self.crop_w * 0.45, 90.0)
            proximity_score = max(0.0, 1.0 - (dist / gating))

            # center preference slightly helps when scoreboard / crowd noise exists
            center_bias = max(0.0, 1.0 - abs(cx - self.src_w * 0.5) / (self.src_w * 0.65))

            # cricket ball is tiny and fast, so motion gets a little more weight
            if sport == "cricket":
                score = 0.35 * motion_score + 0.35 * color_score + 0.25 * proximity_score + 0.05 * center_bias
            elif sport == "basketball":
                score = 0.28 * motion_score + 0.42 * color_score + 0.24 * proximity_score + 0.06 * center_bias
            elif sport == "soccer":
                score = 0.36 * motion_score + 0.22 * color_score + 0.32 * proximity_score + 0.10 * center_bias
            else:
                score = 0.40 * motion_score + 0.15 * color_score + 0.35 * proximity_score + 0.10 * center_bias

            # reject highly implausible candidates with almost no motion and no prior match
            if motion_score < 0.01 and proximity_score < 0.15 and color_score < 0.35:
                continue

            if score > best_score:
                best_score = score
                best = (cx, cy, radius, score)

        if best is None or best_score < 0.18:
            self.ball_missing += 1
            self.ball_conf *= 0.90
            return None

        cx, cy, radius, score = best
        vx = cx - self.ball_cx
        vy = cy - self.ball_cy
        self.prev_ball_vx = 0.55 * self.prev_ball_vx + 0.45 * vx
        self.prev_ball_vy = 0.55 * self.prev_ball_vy + 0.45 * vy
        self.ball_cx = 0.70 * self.ball_cx + 0.30 * cx
        self.ball_cy = 0.70 * self.ball_cy + 0.30 * cy
        self.ball_radius = 0.65 * self.ball_radius + 0.35 * radius if self.ball_radius > 0 else radius
        self.ball_conf = min(1.0, 0.75 * self.ball_conf + 0.50 * score)
        self.ball_missing = 0
        return self.ball_cx, self.ball_cy, self.ball_radius, self.ball_conf

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_idx % self.analysis_stride == 0:
            candidates: list[tuple[float, tuple[float, float]]] = []

            try:
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=4,
                    minSize=(36, 36),
                )
            except Exception:
                faces = []
            for (x, y, w, h) in faces[:3]:
                candidates.append((0.38, (x + w / 2.0, y + h / 2.0)))

            motion_boxes, motion_mask = self._detect_motion_regions(gray)
            if motion_boxes:
                x0 = min(p[0] for p in motion_boxes)
                y0 = min(p[1] for p in motion_boxes)
                x1 = max(p[0] + p[2] for p in motion_boxes)
                y1 = max(p[1] + p[3] for p in motion_boxes)
                candidates.append((0.34, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))

                # strongest single motion box helps on soccer / basketball transitions
                bx, by, bw, bh = motion_boxes[0]
                candidates.append((0.18, (bx + bw / 2.0, by + bh / 2.0)))
            else:
                motion_mask = None

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
                            if w * h > 0.025 * self.src_w * self.src_h:
                                candidates.append((0.12, (x + w / 2.0, y + h / 2.0)))
                except Exception:
                    pass

            ball = self._detect_ball(frame, gray, motion_mask)
            if ball is not None:
                bx, by, _br, bconf = ball
                candidates.append((self.ball_weight * max(0.20, bconf), (bx, by)))

                # For sports, keep some context around main action instead of hard-centering only on the ball.
                if motion_boxes:
                    x0 = min(p[0] for p in motion_boxes)
                    y0 = min(p[1] for p in motion_boxes)
                    x1 = max(p[0] + p[2] for p in motion_boxes)
                    y1 = max(p[1] + p[3] for p in motion_boxes)
                    context_cx = (x0 + x1) / 2.0
                    context_cy = (y0 + y1) / 2.0
                    candidates.append((self.context_bias, (context_cx, context_cy)))

            if candidates:
                ws = sum(weight for weight, _ in candidates)
                self.target_cx = sum(cx * weight for weight, (cx, _cy) in candidates) / max(ws, 1e-6)
                self.target_cy = sum(cy * weight for weight, (_cx, cy) in candidates) / max(ws, 1e-6)
            else:
                self.target_cx, self.target_cy = self.src_w / 2.0, self.src_h / 2.0

            self.prev_motion_mask = motion_mask

        self.prev_gray = gray

        dx = self.target_cx - self.smoothed_cx
        dy = self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px:
            dx = 0.0
        if abs(dy) < self.deadzone_px * 0.45:
            dy = 0.0

        alpha = 1.0 - self.smooth_strength
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-(self.max_pan_px * 0.45), min((self.max_pan_px * 0.45), dy * alpha))

        x0 = int(round(self.smoothed_cx - self.crop_w / 2.0))
        y0 = int(round(self.smoothed_cy - self.crop_h / 2.0))
        x0 = int(_clamp(x0, 0, self.max_x))
        y0 = int(_clamp(y0, 0, self.max_y))

        crop = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if crop.size == 0:
            crop = frame

        self.frame_idx += 1
        return cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Offline vertical master generation
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
        src_w,
        src_h,
        target_w,
        target_h,
        smooth_strength=smooth_strength,
        analysis_stride=analysis_stride,
        deadzone_ratio=deadzone_ratio,
        max_pan_ratio=max_pan_ratio,
        sport_profile=sport_profile,
        ball_tracking=ball_tracking,
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
# Cloudflare Stream live push
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


def cfstream_config_from_inputs(
    account_id: str,
    api_token: str,
    customer_code: str,
    prefer_low_latency: bool = False,
) -> CFStreamConfig:
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


def build_push_file_command(
    reframed_mp4: str,
    rtmps_url: str,
    stream_key: str,
    loop_input: bool = True,
    output_fps: float = DEFAULT_OUTPUT_FPS,
):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-re", "-i", reframed_mp4,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * 2),
        "-keyint_min", str(fps_int * 2),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-ac", "2",
        "-f", "flv",
        target,
    ]
    return cmd


def start_vod_to_live_push(
    cfg: CFStreamConfig,
    reframed_mp4: str,
    asset_name: str,
    loop_input: bool = True,
    output_fps: float = DEFAULT_OUTPUT_FPS,
) -> LiveSession:
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


def build_realtime_rtmps_push_command(
    target_w: int,
    target_h: int,
    fps: float,
    rtmps_url: str,
    stream_key: str,
):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{target_w}x{target_h}",
        "-r", str(fps_int),
        "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-shortest",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * 2),
        "-keyint_min", str(fps_int * 2),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", "scenecut=0",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-ac", "2",
        "-f", "flv",
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
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf = (
        f"fps={fps_int},"
        f"scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    cmd += _source_input_args(source, pace_input=pace_input, loop_file=loop_file)
    cmd += ["-an", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
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
) -> None:
    session.status = "probing"
    info = probe_source(source)

    # Use a stable output FPS for live delivery instead of trusting every source probe.
    fps = DEFAULT_OUTPUT_FPS
    src_w = WORKING_INPUT_W
    src_h = WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_frames = max(1, int(round(delay_seconds * fps)))
    session.stats = {
        "fps": round(fps, 3),
        "delay_frames": delay_frames,
        "working_resolution": f"{src_w}x{src_h}",
        "source_reported_resolution": f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)}",
        "sport_profile": sport_profile,
    }

    reframer = SmoothReframer(
        src_w,
        src_h,
        target_w,
        target_h,
        smooth_strength=smooth_strength,
        analysis_stride=analysis_stride,
        deadzone_ratio=deadzone_ratio,
        max_pan_ratio=max_pan_ratio,
        sport_profile=sport_profile,
        ball_tracking=ball_tracking,
    )
    buffer = collections.deque(maxlen=max(delay_frames + 240, 600))
    placeholder = _make_placeholder_frame(target_w, target_h)
    frame_interval = 1.0 / fps
    placeholder_frames = 0
    source_stalls = 0

    try:
        session.proc = _start_output_process(session)
    except Exception as exc:
        session.status = "ffmpeg_start_failed"
        session.error = str(exc)
        return

    session.status = "priming_output"
    try:
        if session.proc.stdin:
            prime_frames = int(max(1.0, min(delay_seconds / 2.0, 3.0)) * fps)
            for _ in range(prime_frames):
                if session.stop_event.is_set():
                    break
                session.proc.stdin.write(placeholder.tobytes())
                placeholder_frames += 1
    except Exception as exc:
        session.status = "ffmpeg_pipe_broken"
        session.error = f"Could not prime Cloudflare output: {exc}"
        return

    ingest = None
    next_deadline = time.monotonic()
    frames_in = 0
    frames_out = 0
    source_ended = False

    try:
        session.status = "connecting_source"
        ingest = _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=session.log_path)

        while not session.stop_event.is_set():
            if not source_ended:
                raw = _read_exact(ingest.stdout, frame_bytes) if ingest and ingest.stdout else b""
                if len(raw) < frame_bytes:
                    source_ended = True
                    source_stalls += 1
                    session.error = f"Source became unavailable or timed out: {source}"
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
                placeholder_frames += 1
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
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Prevent unbounded drift if processing briefly falls behind.
                next_deadline = time.monotonic()

            if frames_out % int(max(1.0, fps)) == 0:
                session.stats.update({
                    "frames_in": frames_in,
                    "frames_out": frames_out,
                    "buffer_len": len(buffer),
                    "delay_seconds": round(delay_frames / max(fps, 1.0), 2),
                    "placeholder_frames": placeholder_frames,
                    "source_stalls": source_stalls,
                    "ball_confidence": round(reframer.ball_conf, 3),
                })
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
        if session.status not in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}:
            session.status = "stopped"


def start_realtime_delayed_vertical_push(
    cfg: CFStreamConfig,
    source: str,
    asset_name: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 20.0,
    smooth_strength: float = 0.975,
    analysis_stride: int = 4,
    deadzone_ratio: float = 0.05,
    max_pan_ratio: float = 0.012,
    loop_file: bool = False,
    pace_input: bool = True,
    sport_profile: str = "auto",
    ball_tracking: bool = True,
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
            session,
            source,
            target_w,
            target_h,
            delay_seconds,
            smooth_strength,
            analysis_stride,
            deadzone_ratio,
            max_pan_ratio,
            loop_file,
            pace_input,
            sport_profile,
            ball_tracking,
        ),
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
