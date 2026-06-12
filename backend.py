
from __future__ import annotations

"""
backend.py
Production-ready vertical live/VOD reframer backend.

What this file includes
-----------------------
- Stable FFmpeg ingest/output helpers
- Live Cloudflare Stream push helpers
- Time-based buffering with drift control
- Auto mode detection (panel vs sports vs single-subject)
- MediaPipe face detection for panel mode with Haar fallback
- Ball tracking tuned for basketball / cricket / soccer
- Scoreboard-safe cropping with top/bottom overlay preservation
- Analytics for tuning latency / drops / detector health

Notes
-----
- MediaPipe is optional. If unavailable, the code automatically falls back to Haar.
- This file is intentionally self-contained and safe to import in Streamlit / server apps.
"""

import collections
import contextlib
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("backend")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, min(os.cpu_count() or 4, 4)))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 960
WORKING_INPUT_H = 540
PLACEHOLDER_FPS = 30.0
DEFAULT_OUTPUT_FPS = 30.0
DEFAULT_VIDEO_BITRATE = "3500k"
DEFAULT_MAXRATE = "3500k"
DEFAULT_BUFSIZE = "7000k"
DEFAULT_AUDIO_BITRATE = "128k"
DEFAULT_GOP_SEC = 2
MAX_BUFFER_SECONDS = 2.0
AUTO_MODE_PROBE_FRAMES = 45
AUTO_MODE_PANEL_MIN_FACES = 2
AUTO_MODE_PANEL_RATIO = 0.55

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


def _source_input_args(source: str, pace_input: bool = False, loop_file: bool = False) -> List[str]:
    args: List[str] = [
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
    res = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "vcodec": "unknown", "has_audio": False}
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
            elif stream.get("codec_type") == "audio":
                res["has_audio"] = True
    except Exception:
        pass
    return res


def _vertical_crop_box(src_w: int, src_h: int) -> Tuple[int, int]:
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


# ---------------------------------------------------------------------------
# Overlay / scorecard logic
# ---------------------------------------------------------------------------

class OverlayDetector:
    """Detects stable top and bottom broadcast graphics / scoreboards."""

    def __init__(self, stability_frames: int = 8, score_thresh: float = 0.08):
        self.stability_frames = max(2, stability_frames)
        self.score_thresh = score_thresh
        self.top_ratio = 0.0
        self.bottom_ratio = 0.0
        self._top_scores: Deque[float] = collections.deque(maxlen=stability_frames)
        self._bottom_scores: Deque[float] = collections.deque(maxlen=stability_frames)
        self._prev_gray: Optional[np.ndarray] = None

    @staticmethod
    def _band_score(gray: np.ndarray, y1: int, y2: int) -> float:
        roi = gray[y1:y2, :]
        if roi.size == 0:
            return 0.0
        edges = cv2.Canny(roi, 60, 160)
        edge_density = float(edges.mean()) / 255.0
        row_var = float(np.std(np.mean(roi, axis=1))) / 255.0
        return 0.6 * edge_density + 0.4 * row_var

    def update(self, frame: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        top_band = max(16, int(h * 0.14))
        bottom_band = max(16, int(h * 0.18))
        top_score = self._band_score(gray, 0, top_band)
        bottom_score = self._band_score(gray, h - bottom_band, h)
        self._top_scores.append(top_score)
        self._bottom_scores.append(bottom_score)
        self.top_ratio = 0.14 if (len(self._top_scores) == self.stability_frames and np.mean(self._top_scores) >= self.score_thresh) else self.top_ratio * 0.9
        self.bottom_ratio = 0.18 if (len(self._bottom_scores) == self.stability_frames and np.mean(self._bottom_scores) >= self.score_thresh) else self.bottom_ratio * 0.9
        self._prev_gray = gray
        return self.top_ratio, self.bottom_ratio


# ---------------------------------------------------------------------------
# Scene change detector
# ---------------------------------------------------------------------------

class SceneChangeDetector:
    def __init__(self, hist_diff_thresh: float = 0.55, pixel_diff_thresh: float = 45.0, cooldown_frames: int = 8):
        self.hist_diff_thresh = hist_diff_thresh
        self.pixel_diff_thresh = pixel_diff_thresh
        self.cooldown_frames = cooldown_frames
        self.prev_hist: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.cooldown = 0

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_hist = hist
            return False
        if self.cooldown > 0:
            self.cooldown -= 1
        pix = float(cv2.absdiff(gray, self.prev_gray).mean())
        hist_corr = float(cv2.compareHist(self.prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
        cut = (pix >= self.pixel_diff_thresh or hist_corr <= self.hist_diff_thresh) and self.cooldown == 0
        if cut:
            self.cooldown = self.cooldown_frames
        self.prev_gray = gray.copy()
        self.prev_hist = hist
        return cut


# ---------------------------------------------------------------------------
# Ball tracker (basketball / cricket / soccer tuned)
# ---------------------------------------------------------------------------

@dataclass
class BallObservation:
    center: Tuple[float, float]
    radius: float
    confidence: float
    source: str


class BallTracker:
    """Hybrid tracker: Hough + morphology + color + Kalman-style smoothing."""

    SPORT_HINTS = {
        "basketball": {
            "hsv_ranges": [((5, 80, 80), (25, 255, 255)), ((0, 0, 180), (180, 50, 255))],
            "min_r": 4,
            "max_r": 18,
            "speed_gate": 170,
        },
        "cricket": {
            "hsv_ranges": [((0, 0, 180), (180, 55, 255)), ((0, 60, 60), (12, 255, 255))],
            "min_r": 2,
            "max_r": 10,
            "speed_gate": 240,
        },
        "soccer": {
            "hsv_ranges": [((0, 0, 170), (180, 60, 255)), ((25, 40, 40), (95, 255, 255))],
            "min_r": 4,
            "max_r": 22,
            "speed_gate": 220,
        },
        "auto": {
            "hsv_ranges": [((0, 0, 175), (180, 65, 255)), ((5, 80, 80), (25, 255, 255)), ((25, 40, 40), (95, 255, 255))],
            "min_r": 3,
            "max_r": 22,
            "speed_gate": 220,
        },
    }

    def __init__(self, sport_profile: str = "auto", pos_alpha: float = 0.35):
        self.sport_profile = sport_profile if sport_profile in self.SPORT_HINTS else "auto"
        self.pos_alpha = pos_alpha
        self.vel_alpha = pos_alpha * 0.8
        self.center: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.radius: float = 0.0
        self.lost_frames = 0
        self.max_lost = 18
        self.last_source = "none"
        self.last_conf = 0.0

    def reset(self) -> None:
        self.center = None
        self.velocity = (0.0, 0.0)
        self.radius = 0.0
        self.lost_frames = 0
        self.last_source = "none"
        self.last_conf = 0.0

    def _search_window(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = frame.shape[:2]
        if self.center is None:
            return frame, 0, 0
        cx, cy = self.center
        gate = max(100, int(self.SPORT_HINTS[self.sport_profile]["speed_gate"]))
        x1 = max(0, int(cx - gate))
        y1 = max(0, int(cy - gate))
        x2 = min(w, int(cx + gate))
        y2 = min(h, int(cy + gate))
        roi = frame[y1:y2, x1:x2]
        return (roi if roi.size else frame), x1, y1

    def _detect_color_blob(self, frame: np.ndarray) -> Optional[BallObservation]:
        params = self.SPORT_HINTS[self.sport_profile]
        roi, ox, oy = self._search_window(frame)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in params["hsv_ranges"]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 18:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < params["min_r"] or r > params["max_r"]:
                continue
            circularity = 0.0
            peri = cv2.arcLength(c, True)
            if peri > 0:
                circularity = (4 * math.pi * area) / (peri * peri)
            score = 0.55 * circularity + 0.45 * min(area / 250.0, 1.0)
            if self.center is not None:
                pred = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
                dist = math.hypot((ox + x) - pred[0], (oy + y) - pred[1])
                score -= min(dist / params["speed_gate"], 1.0) * 0.35
            if score > best_score:
                best_score = score
                best = BallObservation((ox + x, oy + y), r, float(_clamp(score, 0.1, 0.95)), "color")
        return best

    def _detect_hough(self, frame: np.ndarray) -> Optional[BallObservation]:
        params = self.SPORT_HINTS[self.sport_profile]
        roi, ox, oy = self._search_window(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 10, param1=120, param2=14,
                                   minRadius=params["min_r"], maxRadius=params["max_r"])
        if circles is None:
            return None
        circles = np.round(circles[0, :]).astype(int)
        best = None
        best_score = -1.0
        for x, y, r in circles:
            score = 0.75
            if self.center is not None:
                pred = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
                dist = math.hypot((ox + x) - pred[0], (oy + y) - pred[1])
                score -= min(dist / params["speed_gate"], 1.0) * 0.35
            if score > best_score:
                best_score = score
                best = BallObservation((float(ox + x), float(oy + y)), float(r), float(_clamp(score, 0.1, 0.90)), "hough")
        return best

    def _merge_candidates(self, obs1: Optional[BallObservation], obs2: Optional[BallObservation]) -> Optional[BallObservation]:
        if obs1 is None:
            return obs2
        if obs2 is None:
            return obs1
        if math.hypot(obs1.center[0] - obs2.center[0], obs1.center[1] - obs2.center[1]) <= max(12.0, obs1.radius + obs2.radius):
            cx = (obs1.center[0] * obs1.confidence + obs2.center[0] * obs2.confidence) / max(obs1.confidence + obs2.confidence, 1e-6)
            cy = (obs1.center[1] * obs1.confidence + obs2.center[1] * obs2.confidence) / max(obs1.confidence + obs2.confidence, 1e-6)
            r = (obs1.radius + obs2.radius) / 2.0
            conf = max(obs1.confidence, obs2.confidence)
            return BallObservation((cx, cy), r, conf, "hybrid")
        return obs1 if obs1.confidence >= obs2.confidence else obs2

    def update(self, frame: np.ndarray) -> Optional[BallObservation]:
        cand = self._merge_candidates(self._detect_hough(frame), self._detect_color_blob(frame))
        if cand is None:
            self.lost_frames += 1
            if self.center is not None and self.lost_frames <= self.max_lost:
                self.center = (self.center[0] + self.velocity[0], self.center[1] + self.velocity[1])
                self.velocity = (self.velocity[0] * 0.9, self.velocity[1] * 0.9)
                self.last_source = "pred"
                self.last_conf = max(0.05, self.last_conf * 0.90)
                return BallObservation(self.center, max(2.5, self.radius), self.last_conf, "pred")
            self.reset()
            return None

        self.lost_frames = 0
        if self.center is None:
            self.center = cand.center
            self.velocity = (0.0, 0.0)
            self.radius = cand.radius
        else:
            vx = cand.center[0] - self.center[0]
            vy = cand.center[1] - self.center[1]
            self.velocity = (
                self.velocity[0] * (1.0 - self.vel_alpha) + vx * self.vel_alpha,
                self.velocity[1] * (1.0 - self.vel_alpha) + vy * self.vel_alpha,
            )
            self.center = (
                self.center[0] * (1.0 - self.pos_alpha) + cand.center[0] * self.pos_alpha,
                self.center[1] * (1.0 - self.pos_alpha) + cand.center[1] * self.pos_alpha,
            )
            self.radius = self.radius * 0.7 + cand.radius * 0.3
        self.last_source = cand.source
        self.last_conf = cand.confidence
        return BallObservation(self.center, self.radius, self.last_conf, self.last_source)


# ---------------------------------------------------------------------------
# Panel discussion mode
# ---------------------------------------------------------------------------

@dataclass
class _TrackedFace:
    track_id: int
    sx: float
    sy: float
    sw: float
    sh: float
    age: int = 0
    miss: int = 0


def _face_centre(tf: _TrackedFace) -> Tuple[float, float]:
    return tf.sx, tf.sy


def _match_faces_to_detections(
    tracked: List[_TrackedFace],
    detections: List[Tuple[float, float, float, float]],
    max_dist: float,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    matched: Dict[int, int] = {}
    used_det: set[int] = set()
    if not tracked or not detections:
        return matched, list(range(len(tracked))), list(range(len(detections)))
    for ti, tf in enumerate(tracked):
        best_j = -1
        best_d = float("inf")
        for dj, det in enumerate(detections):
            if dj in used_det:
                continue
            dx = det[0] - tf.sx
            dy = det[1] - tf.sy
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d = d
                best_j = dj
        if best_j >= 0 and best_d <= max_dist:
            matched[ti] = best_j
            used_det.add(best_j)
    unmatched_tracks = [i for i in range(len(tracked)) if i not in matched]
    unmatched_dets = [i for i in range(len(detections)) if i not in used_det]
    return matched, unmatched_tracks, unmatched_dets


@dataclass
class _PanelCell:
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int


def _compute_panel_layout(n: int, canvas_w: int, canvas_h: int, gap: int = 4) -> List[_PanelCell]:
    n = max(1, min(n, 4))
    cells: List[_PanelCell] = []
    if n == 1:
        cells.append(_PanelCell(0, 0, canvas_w, canvas_h))
    elif n == 2:
        h1 = canvas_h // 2
        cells += [_PanelCell(0, 0, canvas_w, h1 - gap // 2), _PanelCell(0, h1 + gap // 2, canvas_w, canvas_h - h1 - gap // 2)]
    elif n == 3:
        top_h = int(canvas_h * 0.40)
        bottom_h = canvas_h - top_h - gap
        half_w = canvas_w // 2
        cells += [
            _PanelCell(0, 0, canvas_w, top_h),
            _PanelCell(0, top_h + gap, half_w - gap // 2, bottom_h),
            _PanelCell(half_w + gap // 2, top_h + gap, canvas_w - half_w - gap // 2, bottom_h),
        ]
    else:
        half_w = canvas_w // 2
        half_h = canvas_h // 2
        cells += [
            _PanelCell(0, 0, half_w - gap // 2, half_h - gap // 2),
            _PanelCell(half_w + gap // 2, 0, canvas_w - half_w - gap // 2, half_h - gap // 2),
            _PanelCell(0, half_h + gap // 2, half_w - gap // 2, canvas_h - half_h - gap // 2),
            _PanelCell(half_w + gap // 2, half_h + gap // 2, canvas_w - half_w - gap // 2, canvas_h - half_h - gap // 2),
        ]
    return cells


class PanelTracker:
    """
    Stable face-based panel renderer.
    Uses MediaPipe if available. Falls back to Haar.

    FIX-P1: render() always copies the blended canvas before storing _prev_output.
    """

    def __init__(self, max_faces: int = 4, detection_stride: int = 3, gap: int = 4):
        self.max_faces = max(1, min(max_faces, 4))
        self.detection_stride = max(1, detection_stride)
        self.gap = max(0, gap)
        self.frame_idx = 0
        self.tracks: List[_TrackedFace] = []
        self.next_id = 0
        self._prev_output: Optional[np.ndarray] = None
        self._canvas_buffer: Optional[np.ndarray] = None
        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._mp_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self._mp_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)
            except Exception:
                self._mp_detector = None

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        h, w = frame.shape[:2]
        detections: List[Tuple[float, float, float, float]] = []
        if self._mp_detector is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._mp_detector.process(rgb)
                if res.detections:
                    for det in res.detections[: self.max_faces * 2]:
                        box = det.location_data.relative_bounding_box
                        x = _clamp(box.xmin * w, 0, w - 1)
                        y = _clamp(box.ymin * h, 0, h - 1)
                        bw = _clamp(box.width * w, 1, w)
                        bh = _clamp(box.height * h, 1, h)
                        detections.append((float(x + bw / 2.0), float(y + bh / 2.0), float(bw), float(bh)))
            except Exception:
                detections = []
        if not detections:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw = self._haar.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5,
                                              minSize=(max(28, w // 20), max(28, h // 20)))
            for (x, y, bw, bh) in raw[: self.max_faces * 2]:
                detections.append((float(x + bw / 2.0), float(y + bh / 2.0), float(bw), float(bh)))
        detections.sort(key=lambda d: d[2] * d[3], reverse=True)
        return detections[: self.max_faces]

    def update(self, frame: np.ndarray) -> List[_TrackedFace]:
        self.frame_idx += 1
        detections: List[Tuple[float, float, float, float]]
        if self.frame_idx % self.detection_stride == 1 or not self.tracks:
            detections = self._detect_faces(frame)
        else:
            detections = []

        max_dist = max(frame.shape[1], frame.shape[0]) * 0.15
        matched, unmatched_tracks, unmatched_dets = _match_faces_to_detections(self.tracks, detections, max_dist)

        for ti, dj in matched.items():
            tf = self.tracks[ti]
            cx, cy, bw, bh = detections[dj]
            tf.sx = tf.sx * 0.78 + cx * 0.22
            tf.sy = tf.sy * 0.78 + cy * 0.22
            tf.sw = tf.sw * 0.75 + bw * 0.25
            tf.sh = tf.sh * 0.75 + bh * 0.25
            tf.age += 1
            tf.miss = 0

        for ti in unmatched_tracks:
            self.tracks[ti].miss += 1
            self.tracks[ti].age += 1

        for dj in unmatched_dets:
            cx, cy, bw, bh = detections[dj]
            self.tracks.append(_TrackedFace(self.next_id, cx, cy, bw, bh, age=1, miss=0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.miss <= max(6, self.detection_stride * 3)]
        self.tracks.sort(key=lambda t: (t.sx, -t.sw * t.sh))
        return self.tracks[: self.max_faces]

    @staticmethod
    def _crop_for_face(frame: np.ndarray, tf: _TrackedFace, out_w: int, out_h: int, top_overlay_ratio: float, bottom_overlay_ratio: float) -> np.ndarray:
        h, w = frame.shape[:2]
        face_h = max(32.0, tf.sh)
        crop_h = min(float(h), face_h * 3.0)
        crop_w = min(float(w), crop_h * out_w / max(out_h, 1))
        cx = tf.sx
        # bias upward so faces sit above center and score ticker remains visible when bottom overlay present
        cy = tf.sy + crop_h * 0.08
        x1 = int(_clamp(cx - crop_w / 2, 0, max(0, w - crop_w)))
        y1 = int(_clamp(cy - crop_h / 2, top_overlay_ratio * h, max(top_overlay_ratio * h, h - bottom_overlay_ratio * h - crop_h)))
        x2 = int(min(w, x1 + crop_w))
        y2 = int(min(h, y1 + crop_h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def render(self, frame: np.ndarray, target_w: int, target_h: int, top_overlay_ratio: float = 0.0, bottom_overlay_ratio: float = 0.0) -> np.ndarray:
        tracks = self.update(frame)
        n = max(1, min(len(tracks), self.max_faces))
        cells = _compute_panel_layout(n, target_w, target_h, self.gap)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        if not tracks:
            canvas[:] = _resize_cover(frame, target_w, target_h)
        else:
            for tf, cell in zip(tracks[:n], cells):
                strip = self._crop_for_face(frame, tf, cell.dst_w, cell.dst_h, top_overlay_ratio, bottom_overlay_ratio)
                canvas[cell.dst_y:cell.dst_y + cell.dst_h, cell.dst_x:cell.dst_x + cell.dst_w] = strip
        if self._prev_output is not None and self._prev_output.shape == canvas.shape:
            canvas = cv2.addWeighted(canvas, 0.82, self._prev_output, 0.18, 0)
        # FIX-P1: store an independent copy
        self._prev_output = canvas.copy()
        return canvas


# ---------------------------------------------------------------------------
# Auto mode detection
# ---------------------------------------------------------------------------

class AutoModeDetector:
    def __init__(self, min_faces: int = AUTO_MODE_PANEL_MIN_FACES, threshold_ratio: float = AUTO_MODE_PANEL_RATIO):
        self.min_faces = min_faces
        self.threshold_ratio = threshold_ratio
        self.face_detector = PanelTracker(max_faces=4, detection_stride=1)

    def infer_mode(self, frames: Sequence[np.ndarray], ball_tracker: Optional[BallTracker] = None) -> str:
        if not frames:
            return "single"
        panel_hits = 0
        sport_hits = 0
        for frame in frames:
            faces = self.face_detector._detect_faces(frame)
            if len(faces) >= self.min_faces:
                xs = [f[0] for f in faces]
                spread = (max(xs) - min(xs)) / max(frame.shape[1], 1)
                if spread > 0.35:
                    panel_hits += 1
            if ball_tracker is not None:
                obs = ball_tracker.update(frame)
                if obs is not None and obs.confidence >= 0.35:
                    sport_hits += 1
        panel_ratio = panel_hits / max(len(frames), 1)
        sport_ratio = sport_hits / max(len(frames), 1)
        if panel_ratio >= self.threshold_ratio:
            return "panel"
        if sport_ratio >= 0.25:
            return "sports"
        return "single"


# ---------------------------------------------------------------------------
# Smooth reframer
# ---------------------------------------------------------------------------

class SmoothReframer:
    """
    Main realtime reframer.

    FIX-P2: panel mode calls overlay_detector.update() every frame and only
    throttles face detection inside PanelTracker.
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
        overlay_composite: bool = True,
        preserve_bottom_overlay: bool = False,
        panel_mode: bool = False,
        panel_max_faces: int = 4,
        panel_detection_stride: int = 3,
        panel_gap: int = 4,
        auto_mode: bool = False,
    ):
        self.src_w = src_w
        self.src_h = src_h
        self.target_w = target_w
        self.target_h = target_h
        self.smooth_strength = _clamp(smooth_strength, 0.75, 0.995)
        self.analysis_stride = max(1, analysis_stride)
        self.deadzone_ratio = _clamp(deadzone_ratio, 0.0, 0.25)
        self.max_pan_ratio = _clamp(max_pan_ratio, 0.002, 0.05)
        self.sport_profile = sport_profile
        self.ball_tracking = ball_tracking
        self.overlay_composite = overlay_composite
        self.preserve_bottom_overlay = preserve_bottom_overlay
        self.panel_mode = panel_mode
        self.auto_mode = auto_mode
        self.current_mode = "panel" if panel_mode else ("sports" if ball_tracking else "single")
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.cx = src_w / 2.0
        self.cy = src_h / 2.0
        self.frame_idx = 0
        self.overlay_detector = OverlayDetector()
        self.scene_detector = SceneChangeDetector()
        self.ball_tracker = BallTracker(sport_profile=sport_profile) if ball_tracking else None
        self.panel_tracker = PanelTracker(max_faces=panel_max_faces, detection_stride=panel_detection_stride, gap=panel_gap)
        self.auto_detector = AutoModeDetector() if auto_mode else None
        self._mode_probe: Deque[np.ndarray] = collections.deque(maxlen=AUTO_MODE_PROBE_FRAMES)
        self.stats = {
            "mode": self.current_mode,
            "ball_confidence": 0.0,
            "panel_faces": 0,
            "top_overlay_ratio": 0.0,
            "bottom_overlay_ratio": 0.0,
            "scene_cuts": 0,
        }

    def _safe_center(self, cx: float, cy: float, top_overlay_ratio: float, bottom_overlay_ratio: float) -> Tuple[float, float]:
        hw = self.crop_w / 2.0
        hh = self.crop_h / 2.0
        min_x = hw
        max_x = self.src_w - hw
        min_y = max(hh, top_overlay_ratio * self.src_h + hh * 0.3)
        max_y = min(self.src_h - hh, self.src_h - bottom_overlay_ratio * self.src_h - hh * 0.65)
        if max_y < min_y:
            max_y = min_y
        return (_clamp(cx, min_x, max_x), _clamp(cy, min_y, max_y))

    def _update_mode(self, frame: np.ndarray) -> None:
        if not self.auto_mode or self.auto_detector is None:
            self.stats["mode"] = self.current_mode
            return
        if self.frame_idx % max(8, self.analysis_stride) == 1:
            self._mode_probe.append(cv2.resize(frame, (min(640, frame.shape[1]), int(frame.shape[0] * min(640, frame.shape[1]) / max(frame.shape[1],1))), interpolation=cv2.INTER_AREA))
        if len(self._mode_probe) >= min(12, self._mode_probe.maxlen) and self.frame_idx % 15 == 0:
            frames = list(self._mode_probe)[-12:]
            mode = self.auto_detector.infer_mode(frames, self.ball_tracker if self.ball_tracking else None)
            self.current_mode = mode
        self.stats["mode"] = self.current_mode

    def _subject_center_from_saliency(self, frame: np.ndarray, top_overlay_ratio: float, bottom_overlay_ratio: float) -> Tuple[float, float]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        sal = cv2.GaussianBlur(lap, (31, 31), 0)
        y1 = int(top_overlay_ratio * h)
        y2 = int(h - bottom_overlay_ratio * h)
        if y2 <= y1:
            y1, y2 = 0, h
        roi = sal[y1:y2, :]
        if roi.size == 0 or float(roi.sum()) <= 1e-6:
            return w / 2.0, h / 2.0
        ys, xs = np.mgrid[y1:y2, 0:w]
        total = float(roi.sum())
        cx = float((xs * roi).sum() / total)
        cy = float((ys * roi).sum() / total)
        return cx, cy

    def _update_single_crop(self, frame: np.ndarray, top_overlay_ratio: float, bottom_overlay_ratio: float) -> np.ndarray:
        want_cx = self.src_w / 2.0
        want_cy = self.src_h / 2.0
        ball_obs = None
        if self.ball_tracker is not None and self.current_mode == "sports":
            ball_obs = self.ball_tracker.update(frame)
        if ball_obs is not None and ball_obs.confidence >= 0.28:
            want_cx, want_cy = ball_obs.center
            self.stats["ball_confidence"] = round(ball_obs.confidence, 3)
        else:
            self.stats["ball_confidence"] = float(round(self.stats.get("ball_confidence", 0.0) * 0.9, 3))
            want_cx, want_cy = self._subject_center_from_saliency(frame, top_overlay_ratio, bottom_overlay_ratio)
        # scorecard-safe cropping: keep bottom ticker and top scorebug by compressing center into safe lane
        want_cx, want_cy = self._safe_center(want_cx, want_cy, top_overlay_ratio, bottom_overlay_ratio)
        alpha = 1.0 - self.smooth_strength
        deadzone_x = self.crop_w * self.deadzone_ratio
        deadzone_y = self.crop_h * self.deadzone_ratio
        dx = want_cx - self.cx
        dy = want_cy - self.cy
        if abs(dx) > deadzone_x:
            step_x = np.sign(dx) * min(abs(dx) * alpha, self.src_w * self.max_pan_ratio)
            self.cx += step_x
        if abs(dy) > deadzone_y:
            step_y = np.sign(dy) * min(abs(dy) * alpha, self.src_h * (self.max_pan_ratio * 0.8))
            self.cy += step_y
        self.cx, self.cy = self._safe_center(self.cx, self.cy, top_overlay_ratio, bottom_overlay_ratio)
        x1 = int(_clamp(self.cx - self.crop_w / 2.0, 0, max(0, self.src_w - self.crop_w)))
        y1 = int(_clamp(self.cy - self.crop_h / 2.0, 0, max(0, self.src_h - self.crop_h)))
        x2 = min(self.src_w, x1 + self.crop_w)
        y2 = min(self.src_h, y1 + self.crop_h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        out = cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        if self.overlay_composite and (top_overlay_ratio > 1e-3 or (bottom_overlay_ratio > 1e-3 and self.preserve_bottom_overlay)):
            full_scaled = _resize_cover(frame, self.target_w, self.target_h)
            if top_overlay_ratio > 1e-3:
                top_h = max(1, int(self.target_h * top_overlay_ratio))
                out[:top_h, :] = full_scaled[:top_h, :]
            if self.preserve_bottom_overlay and bottom_overlay_ratio > 1e-3:
                bot_h = max(1, int(self.target_h * bottom_overlay_ratio))
                out[self.target_h - bot_h :, :] = full_scaled[self.target_h - bot_h :, :]
        return out

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        top_overlay_ratio, bottom_overlay_ratio = self.overlay_detector.update(frame)
        self.stats["top_overlay_ratio"] = round(top_overlay_ratio, 3)
        self.stats["bottom_overlay_ratio"] = round(bottom_overlay_ratio, 3)
        if self.scene_detector.update(frame):
            self.stats["scene_cuts"] += 1
            # faster recenter on cuts
            self.cx, self.cy = self.src_w / 2.0, self.src_h / 2.0
            if self.ball_tracker is not None:
                self.ball_tracker.reset()
        self._update_mode(frame)
        if self.current_mode == "panel":
            out = self.panel_tracker.render(frame, self.target_w, self.target_h, top_overlay_ratio, bottom_overlay_ratio if self.preserve_bottom_overlay else 0.0)
            self.stats["panel_faces"] = min(len(self.panel_tracker.tracks), 4)
            return out
        self.stats["panel_faces"] = 0
        return self._update_single_crop(frame, top_overlay_ratio, bottom_overlay_ratio)


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
    overlay_composite: bool = True,
    preserve_bottom_overlay: bool = False,
    panel_mode: bool = False,
    panel_max_faces: int = 4,
    panel_detection_stride: int = 3,
    panel_gap: int = 4,
    auto_mode: bool = False,
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
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
        auto_mode=auto_mode,
    )
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if src_w != frame.shape[1] or src_h != frame.shape[0]:
            frame = cv2.resize(frame, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        out_frame = reframer.process(frame)
        out.write(out_frame)
        fi += 1
        if progress_cb and fi % max(1, frame_count // 100 if frame_count > 0 else 30) == 0:
            progress_cb(fi / max(frame_count, 1), f"Rendering {fi}/{frame_count or '?'}")
    out.release()
    cap.release()
    return True, output_path


# ---------------------------------------------------------------------------
# Cloudflare Stream live push helpers
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
    ffmpeg_cmd: List[str]
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
    base = f"https://api.cloudflare.com/client/v4/accounts/{cfg.account_id}"
    url = base + path
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
    status, parsed = _cf_api_request(cfg, "POST", "/stream/live_inputs", payload)
    if status not in (200, 201) or not parsed.get("success"):
        raise RuntimeError(f"Create live input failed: {parsed}")
    return parsed["result"]


def disable_live_input(cfg: CFStreamConfig, uid: str) -> None:
    _cf_api_request(cfg, "PUT", f"/stream/live_inputs/{uid}", {"enabled": False})


def build_public_playback_urls(cfg: CFStreamConfig, uid: str):
    base = f"https://customer-{cfg.customer_code}.cloudflarestream.com/{uid}"
    hls = f"{base}/manifest/video.m3u8" + ("?protocol=llhls" if cfg.prefer_low_latency else "")
    dash = f"{base}/manifest/video.mpd"
    iframe = f"{base}/iframe?autoplay=true&muted=true&controls=true&preload=metadata"
    return hls, dash, iframe


def build_push_file_command(reframed_mp4: str, rtmps_url: str, stream_key: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-re", "-i", reframed_mp4,
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * DEFAULT_GOP_SEC),
        "-keyint_min", str(fps_int * DEFAULT_GOP_SEC),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int * DEFAULT_GOP_SEC}:min-keyint={fps_int * DEFAULT_GOP_SEC}",
        "-c:a", "aac", "-b:a", DEFAULT_AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
        "-flvflags", "no_duration_filesize",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
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
        "-g", str(fps_int * DEFAULT_GOP_SEC),
        "-keyint_min", str(fps_int * DEFAULT_GOP_SEC),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int * DEFAULT_GOP_SEC}:min-keyint={fps_int * DEFAULT_GOP_SEC}",
        "-c:a", "aac", "-b:a", DEFAULT_AUDIO_BITRATE, "-ar", "48000", "-ac", "2",
        "-flvflags", "no_duration_filesize",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "flv", target,
    ]


def _read_exact(stream, nbytes: int) -> bytes:
    chunks: List[bytes] = []
    remaining = nbytes
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _build_ingest_command(source: str, fps: float, pace_input: bool, loop_file: bool) -> List[str]:
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf = (
        f"fps={fps_int},"
        f"scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay"]
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


def _terminate_process(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    if proc is None:
        return
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=timeout)
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.kill()


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
    panel_detection_stride: int = 3,
    panel_gap: int = 4,
    auto_mode: bool = False,
) -> None:
    session.status = "probing"
    info = probe_source(source)
    fps = float(info.get("fps") or DEFAULT_OUTPUT_FPS)
    if fps <= 1:
        fps = DEFAULT_OUTPUT_FPS
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_seconds = max(0.0, delay_seconds)
    max_buffer_frames = max(int(round(MAX_BUFFER_SECONDS * fps)), int(round(delay_seconds * fps)) + 15)
    session.stats = {
        "fps": round(fps, 3),
        "delay_seconds": round(delay_seconds, 3),
        "buffer_frames": 0,
        "working_resolution": f"{src_w}x{src_h}",
        "source_reported_resolution": f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)}",
        "sport_profile": sport_profile,
        "panel_mode": panel_mode,
        "auto_mode": auto_mode,
        "frames_in": 0,
        "frames_out": 0,
        "frame_drops": 0,
        "source_stalls": 0,
        "mode": "panel" if panel_mode else ("sports" if ball_tracking else "single"),
        "ball_confidence": 0.0,
        "panel_faces": 0,
        "processing_ms": 0.0,
        "ingest_restarts": 0,
    }
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
        auto_mode=auto_mode,
    )
    buffer: collections.deque[Tuple[float, np.ndarray]] = collections.deque()
    placeholder = _make_placeholder_frame(target_w, target_h)
    frame_interval = 1.0 / fps
    next_output_ts = time.monotonic()
    ingest_log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ingest.log").name
    ingest_proc = _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=ingest_log_path)
    session.proc = _start_output_process(session)
    session.status = "streaming"

    def restart_ingest() -> subprocess.Popen:
        _terminate_process(ingest_proc)
        session.stats["ingest_restarts"] += 1
        return _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=ingest_log_path)

    try:
        while not session.stop_event.is_set():
            loop_started = time.monotonic()
            if ingest_proc.poll() is not None:
                session.stats["source_stalls"] += 1
                session.stats["ingest_restarts"] += 1
                ingest_proc = _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=ingest_log_path)
                time.sleep(0.15)
                continue
            raw = _read_exact(ingest_proc.stdout, frame_bytes) if ingest_proc.stdout is not None else b""
            if len(raw) != frame_bytes:
                session.stats["source_stalls"] += 1
                # keep stream alive with placeholder frame and restart ingest if repeated
                buffer.append((time.monotonic(), placeholder.copy()))
                session.stats["frames_in"] += 1
                ingest_proc = restart_ingest()
                time.sleep(0.10)
            else:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3)
                processed = reframer.process(frame)
                buffer.append((time.monotonic(), processed.copy()))
                session.stats["frames_in"] += 1
                session.stats["mode"] = reframer.stats.get("mode", session.stats["mode"])
                session.stats["ball_confidence"] = reframer.stats.get("ball_confidence", 0.0)
                session.stats["panel_faces"] = reframer.stats.get("panel_faces", 0)
            # cap backlog for realtime correctness
            while len(buffer) > max_buffer_frames:
                buffer.popleft()
                session.stats["frame_drops"] += 1
            # time-based release so delay does not drift and speed does not fast-forward
            now = time.monotonic()
            release_before = now - delay_seconds
            while buffer and buffer[0][0] <= release_before:
                out_frame = buffer.popleft()[1]
                if session.proc is None or session.proc.stdin is None or session.proc.poll() is not None:
                    raise RuntimeError("Output process died")
                try:
                    session.proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError as exc:
                    raise RuntimeError("Output pipe broken") from exc
                session.stats["frames_out"] += 1
                break  # only one output frame per loop iteration; pacing controlled below
            # if startup and nothing mature yet, feed placeholder on cadence
            if session.stats["frames_out"] == 0 and session.proc is not None and session.proc.stdin is not None:
                try:
                    session.proc.stdin.write(placeholder.tobytes())
                except Exception:
                    pass
            next_output_ts += frame_interval
            sleep = next_output_ts - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # do not accumulate negative drift
                next_output_ts = time.monotonic()
            session.stats["buffer_frames"] = len(buffer)
            session.stats["processing_ms"] = round((time.monotonic() - loop_started) * 1000.0, 2)
    except Exception as exc:
        session.error = str(exc)
        session.status = "failed"
        logger.exception("Realtime worker failed: %s", exc)
    finally:
        _terminate_process(ingest_proc)
        _terminate_process(session.proc)
        session.proc = None
        if session.status != "failed":
            session.status = "stopped"


def start_realtime_delayed_vertical_push(
    cfg: CFStreamConfig,
    source: str,
    asset_name: str,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 0.0,
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
    panel_detection_stride: int = 3,
    panel_gap: int = 4,
    auto_mode: bool = False,
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
            overlay_composite,
            preserve_bottom_overlay,
            panel_mode,
            panel_max_faces,
            panel_detection_stride,
            panel_gap,
            auto_mode,
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
            session.worker.join(timeout=4)
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
