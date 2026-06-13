
from __future__ import annotations

import collections
import contextlib
import json
import logging
import math
import os
import queue
import re
import select
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

logger = logging.getLogger("backend")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, min(os.cpu_count() or 4, 4)))
except Exception:
    pass

DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 1280
WORKING_INPUT_H = 720
PLACEHOLDER_FPS = 30.0
DEFAULT_OUTPUT_FPS = 30.0
DEFAULT_VIDEO_BITRATE = "3500k"
DEFAULT_MAXRATE = "3500k"
DEFAULT_BUFSIZE = "3500k"
MAX_BUFFER_SECONDS = 0.50
LIVE_STARTUP_PRIME_SECONDS = 0.15
INGEST_READ_TIMEOUT = 0.75
INGEST_READ_TIMEOUT_MIN = 0.25
INGEST_READ_TIMEOUT_MAX = 1.25
CLEANUP_LOG_MAX_AGE_SECONDS = 6 * 60 * 60
MAX_CONSECUTIVE_STALLS_NON_LOOP = 12

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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
    is_net = is_network_source(source)
    args = [
        "-fflags", "+genpts+discardcorrupt+nobuffer" if is_net else "+genpts+discardcorrupt",
        "-analyzeduration", "20000000",
        "-probesize", "20000000",
    ]
    if loop_file and not is_net:
        args += ["-stream_loop", "-1"]
    if pace_input and not is_net:
        args += ["-re"]
    if is_net:
        args += ["-rw_timeout", "15000000"]
    if source.lower().startswith(("http://", "https://")):
        args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]
    args += ["-i", source]
    return args


def _safe_json_loads(text: str) -> dict:
    try:
        p = json.loads(text)
        return p if isinstance(p, dict) else {}
    except Exception:
        return {}


def _ffprobe_json(source: str, timeout: int = 30) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", "-analyzeduration", "20000000", "-probesize", "20000000", source]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout)
    return _safe_json_loads(out)


def probe_source(source: str) -> dict:
    res = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "vcodec": "unknown"}
    try:
        data = _ffprobe_json(source, timeout=35 if is_network_source(source) else 20)
        fmt = data.get("format", {}) if isinstance(data, dict) else {}
        res["duration"] = float(fmt.get("duration", 0) or 0)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                res["width"] = int(s.get("width", 0) or 0)
                res["height"] = int(s.get("height", 0) or 0)
                res["vcodec"] = str(s.get("codec_name", "unknown"))
                rate = str(s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1")
                try:
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


def _source_has_audio(path: str) -> bool:
    try:
        data = _ffprobe_json(path, timeout=15)
        return any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    except Exception:
        return True


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _vertical_crop_box(src_w: int, src_h: int) -> tuple[int, int]:
    if src_w / max(src_h, 1) >= 9 / 16:
        crop_h = src_h
        crop_w = int(round(src_h * 9 / 16))
    else:
        crop_w = src_w
        crop_h = int(round(src_w * 16 / 9))
    return max(32, crop_w - crop_w % 2), max(32, crop_h - crop_h % 2)


def _resize_cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = max(width / max(w, 1), height / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = max(0, (nw - width) // 2), max(0, (nh - height) // 2)
    return resized[y0:y0 + height, x0:x0 + width]


def _overlay_heights(target_h: int, top_strip: Optional[np.ndarray], bottom_strip: Optional[np.ndarray], src_h: int, overlay_composite: bool, preserve_bottom: bool) -> tuple[int, int]:
    top_h = 0
    if overlay_composite and top_strip is not None and top_strip.size:
        base = 0.075 if top_strip.shape[0] < 0.05 * src_h else 0.082
        top_h = int(_clamp(round(target_h * base), 42, int(target_h * 0.11)))
    bottom_h = 0
    if overlay_composite and preserve_bottom and bottom_strip is not None and bottom_strip.size:
        bottom_h = int(_clamp(round(target_h * 0.045), 20, int(target_h * 0.07)))
    return top_h, bottom_h

# -----------------------------------------------------------------------------
# CV components
# -----------------------------------------------------------------------------
class OverlayDetector:
    def __init__(self, src_w: int, src_h: int, top_scan_ratio: float = 0.18, bottom_scan_ratio: float = 0.14, warmup_frames: int = 18):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.top_scan_h = max(24, int(round(src_h * top_scan_ratio)))
        self.bottom_scan_h = max(24, int(round(src_h * bottom_scan_ratio)))
        self.warmup_frames = max(4, int(warmup_frames))
        self.top_avg: Optional[np.ndarray] = None
        self.bottom_avg: Optional[np.ndarray] = None
        self.frame_count = 0
        self.top_overlay: Optional[tuple[int, int]] = None
        self.bottom_overlay: Optional[tuple[int, int]] = None
        self.top_hold = 0
        self.bottom_hold = 0
        self.exclusion_mask = np.zeros((self.src_h, self.src_w), dtype=np.uint8)

    def _detect_band(self, patch: np.ndarray, avg: np.ndarray, top: bool) -> Optional[tuple[int, int]]:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        avg_g = cv2.cvtColor(np.clip(avg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff = np.abs(gray.astype(np.float32) - avg_g.astype(np.float32)).mean(axis=1)
        edges = cv2.Canny(gray, 60, 180).mean(axis=1) / 255.0
        active = (diff < 12.0) & (edges > 0.018)
        idx = np.where(active)[0]
        if idx.size == 0:
            return None
        y0, y1 = int(idx[0]), int(idx[-1] + 1)
        if top:
            if y0 > int(0.18 * patch.shape[0]) or y1 - y0 < 10:
                return None
            return max(0, y0 - 6), min(patch.shape[0], y1 + 6)
        if y1 < int(0.50 * patch.shape[0]) or y1 - y0 < 8:
            return None
        return max(0, y0 - 4), min(patch.shape[0], y1 + 4)

    def update(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        top_patch = frame[:self.top_scan_h].astype(np.float32)
        bot_patch = frame[self.src_h - self.bottom_scan_h:].astype(np.float32)
        if self.top_avg is None:
            self.top_avg, self.bottom_avg = top_patch.copy(), bot_patch.copy()
            return
        alpha = 0.60 if self.frame_count <= self.warmup_frames else 0.93
        self.top_avg = alpha * self.top_avg + (1 - alpha) * top_patch
        self.bottom_avg = alpha * self.bottom_avg + (1 - alpha) * bot_patch
        if self.frame_count <= self.warmup_frames:
            return
        tr = self._detect_band(frame[:self.top_scan_h], self.top_avg, True)
        br_local = self._detect_band(frame[self.src_h - self.bottom_scan_h:], self.bottom_avg, False)
        br = None
        if br_local:
            off = self.src_h - self.bottom_scan_h
            br = (off + br_local[0], off + br_local[1])
        if tr:
            self.top_overlay, self.top_hold = tr, 10
        elif self.top_hold > 0:
            self.top_hold -= 1
        else:
            self.top_overlay = None
        if br:
            self.bottom_overlay, self.bottom_hold = br, 10
        elif self.bottom_hold > 0:
            self.bottom_hold -= 1
        else:
            self.bottom_overlay = None
        self.exclusion_mask[:] = 0
        if self.top_overlay:
            self.exclusion_mask[self.top_overlay[0]:self.top_overlay[1], :] = 255
        if self.bottom_overlay:
            self.exclusion_mask[self.bottom_overlay[0]:self.bottom_overlay[1], :] = 255

    def get_play_area_bounds(self) -> tuple[int, int]:
        top = (self.top_overlay[1] if self.top_overlay else 0) + 6
        bot = (self.bottom_overlay[0] if self.bottom_overlay else self.src_h) - 6
        if bot <= top + 32:
            return 0, self.src_h
        return min(self.src_h - 32, top), max(32, bot)

    def extract_top_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.top_overlay:
            return None
        y0, y1 = self.top_overlay
        return frame[y0:y1].copy()

    def extract_bottom_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.bottom_overlay:
            return None
        y0, y1 = self.bottom_overlay
        return frame[y0:y1].copy()

class SceneChangeDetector:
    def __init__(self, hist_diff_thresh: float = 0.55, pixel_diff_thresh: float = 45.0, cooldown_frames: int = 8):
        self.hist_diff_thresh, self.pixel_diff_thresh, self.cooldown_frames = hist_diff_thresh, pixel_diff_thresh, cooldown_frames
        self.prev_hist: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.cooldown = 0

    def check(self, gray: np.ndarray) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if self.prev_gray is not None and self.prev_gray.shape != gray.shape:
            self.prev_hist, self.prev_gray, self.cooldown = hist, gray.copy(), self.cooldown_frames
            return False
        cut = False
        if self.prev_hist is not None and self.cooldown <= 0:
            diff = 1.0 - float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
            pix = float(np.mean(cv2.absdiff(gray, self.prev_gray))) if self.prev_gray is not None else 0.0
            if diff > self.hist_diff_thresh and pix > self.pixel_diff_thresh:
                cut, self.cooldown = True, self.cooldown_frames
        self.prev_hist, self.prev_gray = hist, gray.copy()
        return cut

class BallTracker:
    def __init__(self, src_w: int, src_h: int, sport_profile: str = "auto"):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.sport = (sport_profile or "auto").lower()
        self.cx, self.cy = src_w / 2.0, src_h / 2.0
        self.vx = self.vy = 0.0
        self.radius = 0.0
        self.conf = 0.0
        self.missing_count = 0
        md = min(src_w, src_h)
        self.min_r = max(3, int(md * 0.006))
        self.max_r = max(self.min_r + 4, int(md * 0.038))

    def update(self, frame: np.ndarray, gray: np.ndarray, motion_mask: Optional[np.ndarray], exclusion_mask: Optional[np.ndarray] = None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.sport == "basketball":
            mask = cv2.inRange(hsv, np.array([3, 80, 80]), np.array([22, 255, 255]))
        elif self.sport == "cricket":
            mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 50, 255])), cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255])))
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array([165, 100, 60]), np.array([179, 255, 255])))
        else:
            mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 65, 255]))
        if exclusion_mask is not None:
            mask[exclusion_mask > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -1.0
        for c in cnts[:80]:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if not (self.min_r * 0.7 <= r <= self.max_r * 1.8):
                continue
            dist = math.hypot(cx - (self.cx + self.vx), cy - (self.cy + self.vy))
            score = max(0, 1 - dist / max(self.src_w * 0.18, 40))
            if score > best_score:
                best_score, best = score, (cx, cy, r)
        if best is None or best_score < 0.12:
            self.missing_count += 1
            self.conf *= 0.93
            if self.missing_count > 18:
                self.conf = 0.0
            return None
        cx, cy, r = best
        nvx, nvy = cx - self.cx, cy - self.cy
        self.vx, self.vy = 0.45 * self.vx + 0.55 * nvx, 0.45 * self.vy + 0.55 * nvy
        self.cx, self.cy = 0.60 * self.cx + 0.40 * cx, 0.60 * self.cy + 0.40 * cy
        self.radius = 0.6 * self.radius + 0.4 * r if self.radius > 0 else r
        self.conf = min(1.0, 0.70 * self.conf + 0.35 * max(0.4, best_score))
        self.missing_count = 0
        return self.cx, self.cy, self.radius, self.conf

    def reset_position(self, cx: float, cy: float) -> None:
        self.cx, self.cy, self.vx, self.vy = cx, cy, 0.0, 0.0
        self.conf *= 0.3

@dataclass
class _TrackedFace:
    face_id: int
    sx: float
    sy: float
    sw: float
    sh: float
    raw_x: float
    raw_y: float
    raw_w: float
    raw_h: float
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0
    missing_frames: int = 0
    active: bool = True
    _pos_alpha: float = 0.90
    _size_alpha: float = 0.92

    def update(self, x: float, y: float, w: float, h: float) -> None:
        psx, psy, psw, psh = self.sx, self.sy, self.sw, self.sh
        self.raw_x, self.raw_y, self.raw_w, self.raw_h = x, y, w, h
        self.sx = self._pos_alpha * self.sx + (1 - self._pos_alpha) * x
        self.sy = self._pos_alpha * self.sy + (1 - self._pos_alpha) * y
        self.sw = self._size_alpha * self.sw + (1 - self._size_alpha) * w
        self.sh = self._size_alpha * self.sh + (1 - self._size_alpha) * h
        self.vx = 0.70 * self.vx + 0.30 * (self.sx - psx)
        self.vy = 0.70 * self.vy + 0.30 * (self.sy - psy)
        self.vw = 0.70 * self.vw + 0.30 * (self.sw - psw)
        self.vh = 0.70 * self.vh + 0.30 * (self.sh - psh)
        self.missing_frames = 0
        self.active = True

    def extrapolate(self) -> None:
        self.missing_frames += 1
        if self.missing_frames > 12:
            self.active = False

    def tick_smooth(self) -> None:
        self.sx += self.vx
        self.sy += self.vy
        self.sw = max(1, self.sw + self.vw)
        self.sh = max(1, self.sh + self.vh)
        self.vx *= 0.82
        self.vy *= 0.82
        self.vw *= 0.70
        self.vh *= 0.70

@dataclass
class _PanelCell:
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int


def _compute_panel_layout(n: int, w: int, h: int, gap: int = 4) -> list[_PanelCell]:
    n = max(1, min(n, 4))
    if n == 1:
        return [_PanelCell(0, 0, w, h)]
    if n == 2:
        rh = (h - gap) // 2
        return [_PanelCell(0, 0, w, rh), _PanelCell(0, rh + gap, w, h - rh - gap)]
    if n == 3:
        th, cw = (h - gap) // 2, (w - gap) // 2
        return [_PanelCell(0, 0, w, th), _PanelCell(0, th + gap, cw, h - th - gap), _PanelCell(cw + gap, th + gap, w - cw - gap, h - th - gap)]
    rh, cw = (h - gap) // 2, (w - gap) // 2
    return [_PanelCell(0, 0, cw, rh), _PanelCell(cw + gap, 0, w - cw - gap, rh), _PanelCell(0, rh + gap, cw, h - rh - gap), _PanelCell(cw + gap, rh + gap, w - cw - gap, h - rh - gap)]

class PanelTracker:
    def __init__(self, src_w: int, src_h: int, max_faces: int = 4, max_missing_frames: int = 24, layout_hold_frames: int = 15, blend_frames: int = 10, min_face_area_ratio: float = 0.0012, pos_alpha: float = 0.90, size_alpha: float = 0.92):
        self.src_w, self.src_h = src_w, src_h
        self.max_faces, self.max_missing_frames = max_faces, max_missing_frames
        self.layout_hold_frames, self.blend_frames = layout_hold_frames, blend_frames
        self.min_face_area = min_face_area_ratio * src_w * src_h
        self.pos_alpha, self.size_alpha = pos_alpha, size_alpha
        self._tracked: list[_TrackedFace] = []
        self._next_id = 0
        self._active_count = 0
        self._candidate_count = 0
        self._candidate_hold = 0
        self._blend_remaining = 0
        self._prev_output: Optional[np.ndarray] = None
        self._match_dist = max(src_w, src_h) * 0.18
        self.detector_backend_name = "haar"
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self._mp_face = None
        self._mp_available = False
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.60)
            self._mp_available = True
            self.detector_backend_name = "mediapipe"
        except Exception:
            pass

    @property
    def active_count(self) -> int:
        return len([tf for tf in self._tracked if tf.active])

    @staticmethod
    def _nms(faces: list[tuple[int, int, int, int]], iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        keep = []
        for b in faces:
            x1, y1, w1, h1 = b
            ok = True
            for k in keep:
                x2, y2, w2, h2 = k
                inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union > iou_thresh:
                    ok = False
                    break
            if ok:
                keep.append(b)
        return keep

    def detect_faces(self, frame: np.ndarray, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._mp_available and self._mp_face is not None:
            try:
                h, w = frame.shape[:2]
                scale = 960.0 / w if w > 960 else 1.0
                small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
                res = self._mp_face.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                sh, sw = small.shape[:2]
                inv = 1.0 / scale
                faces = []
                if res and res.detections:
                    for det in res.detections:
                        score = float(det.score[0]) if getattr(det, "score", None) else 0
                        if score < 0.60:
                            continue
                        rb = det.location_data.relative_bounding_box
                        x, y, ww, hh = int(rb.xmin * sw * inv), int(rb.ymin * sh * inv), int(rb.width * sw * inv), int(rb.height * sh * inv)
                        x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
                        ww, hh = min(ww, w - x), min(hh, h - y)
                        if ww >= 48 and hh >= 48 and 0.60 <= ww / max(hh, 1) <= 1.60:
                            faces.append((x, y, ww, hh))
                if faces:
                    self.detector_backend_name = "mediapipe"
                    return self._nms(faces, 0.35)
            except Exception:
                pass
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        inv = 1.0 / scale
        faces = []
        for det in (self.face_detector, self.profile_detector):
            try:
                found = det.detectMultiScale(small, scaleFactor=1.10, minNeighbors=5, minSize=(24, 24))
                faces += [(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for x, y, w, h in found]
            except Exception:
                pass
        self.detector_backend_name = "haar"
        return self._nms([(x, y, w, h) for x, y, w, h in faces if 0.65 <= w / max(h, 1) <= 1.50], 0.4)

    def update_detections(self, raw_faces: list[tuple[int, int, int, int]]) -> None:
        faces = [(float(x), float(y), float(w), float(h)) for x, y, w, h in raw_faces if w * h >= self.min_face_area]
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:self.max_faces]
        used = set()
        for tf in self._tracked:
            best_i, best_d = -1, self._match_dist
            for i, (x, y, w, h) in enumerate(faces):
                if i in used:
                    continue
                d = math.hypot(tf.sx - (x + w / 2), tf.sy - (y + h / 2))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i >= 0:
                x, y, w, h = faces[best_i]
                tf.update(x + w / 2, y + h / 2, w, h)
                used.add(best_i)
            else:
                tf.extrapolate()
        for i, (x, y, w, h) in enumerate(faces):
            if i not in used:
                self._tracked.append(_TrackedFace(self._next_id, x + w / 2, y + h / 2, w, h, x + w / 2, y + h / 2, w, h, _pos_alpha=self.pos_alpha, _size_alpha=self.size_alpha))
                self._next_id += 1
        self._tracked = [tf for tf in self._tracked if tf.missing_frames <= self.max_missing_frames]
        current_n = max(1, min(len([tf for tf in self._tracked if tf.active]), self.max_faces))
        if current_n != self._candidate_count:
            self._candidate_count, self._candidate_hold = current_n, 0
        else:
            self._candidate_hold += 1
        if self._candidate_hold >= self.layout_hold_frames and current_n != self._active_count:
            self._active_count = current_n
            self._blend_remaining = self.blend_frames

    def tick_extrapolation(self) -> None:
        for tf in self._tracked:
            tf.tick_smooth()

    def _crop_person(self, frame: np.ndarray, face: _TrackedFace, cell_w: int, cell_h: int, zoom: float = 1.0) -> np.ndarray:
        fh, fw = frame.shape[:2]
        ar = cell_w / max(cell_h, 1)
        cw = min(max(face.sw * 3.0 * zoom, face.sw * 2.5), fw * 0.95)
        ch = min(cw / max(ar, 0.01), fh * 0.95)
        if cw / max(ch, 1) > ar:
            cw = ch * ar
        else:
            ch = cw / max(ar, 0.01)
        cx, cy = face.sx, face.sy - face.sh * 0.35
        x0 = int(_clamp(round(cx - cw / 2), 0, max(0, fw - int(cw))))
        y0 = int(_clamp(round(cy - ch * 0.42), 0, max(0, fh - int(ch))))
        x1, y1 = max(x0 + 1, min(fw, x0 + int(cw))), max(y0 + 1, min(fh, y0 + int(ch)))
        return frame[y0:y1, x0:x1]

    def render(self, source_frame: np.ndarray, canvas_w: int, canvas_h: int, gap: int = 4) -> np.ndarray:
        if self._active_count == 0:
            self._active_count = 1
        active = sorted([tf for tf in self._tracked if tf.active], key=lambda tf: tf.sx)
        if not active:
            out = _resize_cover(source_frame, canvas_w, canvas_h)
            self._prev_output = out.copy()
            return out
        n = min(self._active_count, len(active))
        cells = _compute_panel_layout(n, canvas_w, canvas_h, gap)
        active = active[:n]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        avg_size = sum(f.sw * f.sh for f in active) / max(len(active), 1)
        for i, c in enumerate(cells):
            f = active[i]
            ratio = math.sqrt((f.sw * f.sh) / max(avg_size, 1))
            crop = self._crop_person(source_frame, f, c.dst_w, c.dst_h, _clamp(1.0 / max(ratio, 0.5), 0.7, 1.4))
            canvas[c.dst_y:c.dst_y + c.dst_h, c.dst_x:c.dst_x + c.dst_w] = _resize_cover(crop, c.dst_w, c.dst_h)
        if gap >= 2:
            for c in cells:
                if c.dst_x > 0:
                    canvas[:, max(0, c.dst_x - gap // 2):min(canvas_w, c.dst_x + gap // 2)] = (8, 8, 8)
                if c.dst_y > 0:
                    canvas[max(0, c.dst_y - gap // 2):min(canvas_h, c.dst_y + gap // 2), :] = (8, 8, 8)
        out = canvas.copy()
        if self._blend_remaining > 0 and self._prev_output is not None and self._prev_output.shape == out.shape:
            a = self._blend_remaining / max(self.blend_frames, 1)
            out = cv2.addWeighted(self._prev_output, a, out, 1 - a, 0)
            self._blend_remaining -= 1
        self._prev_output = out.copy()
        return out

class AutoModeDetector:
    def __init__(self, probe_frames: int = 45, min_faces: int = 2, panel_ratio: float = 0.55):
        self.probe_frames, self.min_faces, self.panel_ratio = max(8, probe_frames), min_faces, panel_ratio
        self._face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._panel_hits = 0
        self._fed = 0

    def feed(self, frame: np.ndarray) -> None:
        if self._fed >= self.probe_frames:
            return
        self._fed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        faces = self._face_det.detectMultiScale(small, scaleFactor=1.12, minNeighbors=4, minSize=(20, 20))
        if len(faces) >= self.min_faces:
            xs = [int(x) for x, _, _, _ in faces]
            if (max(xs) - min(xs)) / max(small.shape[1], 1) > 0.30:
                self._panel_hits += 1

    def ready(self) -> bool:
        return self._fed >= self.probe_frames

    def result(self) -> str:
        return "panel" if self._fed and self._panel_hits / max(self._fed, 1) >= self.panel_ratio else "single"

class SmoothReframer:
    def __init__(self, src_w: int, src_h: int, target_w: int, target_h: int, smooth_strength: float = 0.975, analysis_stride: int = 4, deadzone_ratio: float = 0.05, max_pan_ratio: float = 0.012, sport_profile: str = "auto", ball_tracking: bool = True, ball_weight: float = 0.55, context_bias: float = 0.20, overlay_composite: bool = True, preserve_bottom_overlay: bool = False, panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, panel_max_missing_frames: int = 24, panel_layout_hold_frames: int = 15, panel_blend_frames: int = 10, panel_min_face_area_ratio: float = 0.0012, panel_pos_alpha: float = 0.90, panel_size_alpha: float = 0.92, overlay_stride: int = 2, auto_mode: bool = False):
        self.src_w, self.src_h, self.target_w, self.target_h = int(src_w), int(src_h), int(target_w), int(target_h)
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.max_x, self.max_y = max(0, src_w - self.crop_w), max(0, src_h - self.crop_h)
        self.smooth_strength = float(smooth_strength)
        self.analysis_stride = max(1, int(analysis_stride))
        self.deadzone_px = max(8.0, self.crop_w * deadzone_ratio)
        self.max_pan_px = max(2.0, self.crop_w * max_pan_ratio)
        self.ball_weight, self.context_bias = ball_weight, context_bias
        self.overlay_composite, self.preserve_bottom_overlay = overlay_composite, preserve_bottom_overlay
        self.panel_mode = bool(panel_mode)
        if self.panel_mode:
            auto_mode = False
        self.auto_mode = bool(auto_mode)
        self.panel_detection_stride, self.panel_gap = max(1, int(panel_detection_stride)), int(panel_gap)
        self.overlay_detector = OverlayDetector(src_w, src_h)
        self.scene_detector = SceneChangeDetector()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.ball_tracker = None if self.panel_mode or not ball_tracking else BallTracker(src_w, src_h, sport_profile)
        self.panel_tracker = PanelTracker(src_w, src_h, panel_max_faces, panel_max_missing_frames, panel_layout_hold_frames, panel_blend_frames, panel_min_face_area_ratio, panel_pos_alpha, panel_size_alpha) if self.panel_mode else None
        self._auto_detector = AutoModeDetector() if self.auto_mode else None
        self._auto_decided = False
        self._init = dict(panel_max_faces=panel_max_faces, panel_max_missing_frames=panel_max_missing_frames, panel_layout_hold_frames=panel_layout_hold_frames, panel_blend_frames=panel_blend_frames, panel_min_face_area_ratio=panel_min_face_area_ratio, panel_pos_alpha=panel_pos_alpha, panel_size_alpha=panel_size_alpha)
        self.smoothed_cx, self.smoothed_cy = src_w / 2.0, src_h / 2.0
        self.target_cx, self.target_cy = self.smoothed_cx, self.smoothed_cy
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0
        self._panel_no_face_count = 0

    def _detect_motion(self, gray: np.ndarray):
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return [], None
        diff = cv2.GaussianBlur(cv2.absdiff(gray, self.prev_gray), (9, 9), 0)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)
        motion[self.overlay_detector.exclusion_mask > 0] = 0
        cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 0.006 * self.src_w * self.src_h:
                boxes.append((x, y, w, h))
        return boxes, motion

    def _compose_output(self, crop: np.ndarray, top_strip: Optional[np.ndarray], bottom_strip: Optional[np.ndarray]) -> np.ndarray:
        top_h, bottom_h = _overlay_heights(self.target_h, top_strip, bottom_strip, self.src_h, self.overlay_composite, self.preserve_bottom_overlay)
        mid_h = max(1, self.target_h - top_h - bottom_h)
        out = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        out[top_h:top_h + mid_h] = _resize_cover(crop, self.target_w, mid_h)
        if top_h > 0 and top_strip is not None and top_strip.size:
            out[:top_h] = cv2.resize(top_strip, (self.target_w, top_h), interpolation=cv2.INTER_AREA)
        if bottom_h > 0 and bottom_strip is not None and bottom_strip.size:
            out[self.target_h - bottom_h:] = cv2.resize(bottom_strip, (self.target_w, bottom_h), interpolation=cv2.INTER_AREA)
        return out

    def _process_panel(self, frame: np.ndarray) -> np.ndarray:
        assert self.panel_tracker is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        if self.frame_idx % self.panel_detection_stride == 0:
            raw = self.panel_tracker.detect_faces(frame, gray)
            top_ol, bot_ol = self.overlay_detector.top_overlay, self.overlay_detector.bottom_overlay
            faces = [(x, y, w, h) for x, y, w, h in raw if not ((top_ol and y + h / 2 < top_ol[1]) or (bot_ol and y + h / 2 > bot_ol[0]))]
            self.panel_tracker.update_detections(faces)
        else:
            self.panel_tracker.tick_extrapolation()
        active = [tf for tf in self.panel_tracker._tracked if tf.active]
        self._panel_no_face_count = 0 if active else self._panel_no_face_count + 1
        if self._panel_no_face_count > 90:
            return self._process_single(frame)
        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        top_h, bottom_h = _overlay_heights(self.target_h, top_strip, bottom_strip, self.src_h, self.overlay_composite, self.preserve_bottom_overlay)
        ph = max(1, self.target_h - top_h - bottom_h)
        panel = self.panel_tracker.render(frame, self.target_w, ph, self.panel_gap)
        out = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        out[top_h:top_h + ph] = panel
        if top_h > 0 and top_strip is not None and top_strip.size:
            out[:top_h] = cv2.resize(top_strip, (self.target_w, top_h), interpolation=cv2.INTER_AREA)
        if bottom_h > 0 and bottom_strip is not None and bottom_strip.size:
            out[self.target_h - bottom_h:] = cv2.resize(bottom_strip, (self.target_w, bottom_h), interpolation=cv2.INTER_AREA)
        return out

    def _process_single(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        is_cut = self.scene_detector.check(gray)
        if is_cut:
            self.smoothed_cx = (self.smoothed_cx + self.src_w / 2) / 2
            self.smoothed_cy = (self.smoothed_cy + self.src_h / 2) / 2
            if self.ball_tracker:
                self.ball_tracker.reset_position(self.src_w / 2, self.src_h / 2)
        if self.frame_idx % self.analysis_stride == 0:
            candidates = []
            play_top, play_bot = self.overlay_detector.get_play_area_bounds()
            try:
                faces = self.face_detector.detectMultiScale(gray[play_top:play_bot], scaleFactor=1.15, minNeighbors=4, minSize=(32, 32))
            except Exception:
                faces = []
            for x, y, w, h in faces[:3]:
                candidates.append((0.30, (x + w / 2, play_top + y + h / 2)))
            boxes, motion = self._detect_motion(gray)
            if boxes:
                x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
                x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
                candidates.append((0.20 if self.ball_tracker else 0.38, ((x0 + x1) / 2, (y0 + y1) / 2)))
            if self.ball_tracker:
                ball = self.ball_tracker.update(frame, gray, motion, self.overlay_detector.exclusion_mask)
                if ball:
                    bx, by, _, conf = ball
                    candidates.append((self.ball_weight * max(0.25, conf), (bx, by)))
            if candidates:
                sw = sum(w for w, _ in candidates)
                self.target_cx = sum(cx * w for w, (cx, _) in candidates) / max(sw, 1e-6)
                self.target_cy = sum(cy * w for w, (_, cy) in candidates) / max(sw, 1e-6)
            else:
                self.target_cx, self.target_cy = self.src_w / 2, self.src_h / 2
            self.target_cy = _clamp(self.target_cy, play_top + 12, play_bot - 12)
        self.prev_gray = gray
        dx, dy = self.target_cx - self.smoothed_cx, self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px:
            dx = 0.0
        if abs(dy) < self.deadzone_px * 0.45:
            dy = 0.0
        alpha = (1 - self.smooth_strength) * (3 if is_cut else 1)
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-self.max_pan_px * 0.45, min(self.max_pan_px * 0.45, dy * alpha))
        play_top, play_bot = self.overlay_detector.get_play_area_bounds()
        x0 = int(_clamp(round(self.smoothed_cx - self.crop_w / 2), 0, self.max_x))
        y0 = int(_clamp(round(self.smoothed_cy - self.crop_h / 2), play_top if play_bot - play_top >= self.crop_h else 0, (play_bot - self.crop_h) if play_bot - play_top >= self.crop_h else self.max_y))
        crop = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if crop.size == 0:
            crop = frame
        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        return self._compose_output(crop, top_strip, bottom_strip)

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        if self.auto_mode and self._auto_detector is not None and not self._auto_decided:
            self._auto_detector.feed(frame)
            if self._auto_detector.ready():
                self._auto_decided = True
                if self._auto_detector.result() == "panel" and self.panel_tracker is None:
                    self.panel_mode, self.ball_tracker = True, None
                    p = self._init
                    self.panel_tracker = PanelTracker(self.src_w, self.src_h, p["panel_max_faces"], p["panel_max_missing_frames"], p["panel_layout_hold_frames"], p["panel_blend_frames"], p["panel_min_face_area_ratio"], p["panel_pos_alpha"], p["panel_size_alpha"])
        if self.panel_mode and self.panel_tracker is not None:
            return self._process_panel(frame)
        return self._process_single(frame)

# -----------------------------------------------------------------------------
# Offline VOD
# -----------------------------------------------------------------------------
def create_vertical_master(source_path: str, output_path: str, target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H, smooth_strength: float = 0.975, analysis_stride: int = 4, deadzone_ratio: float = 0.05, max_pan_ratio: float = 0.012, sport_profile: str = "auto", ball_tracking: bool = True, overlay_composite: bool = True, preserve_bottom_overlay: bool = False, panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, auto_mode: bool = False, progress_cb: Optional[Callable[[float, str], None]] = None):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False, "Could not open input source"
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release(); return False, "Invalid source dimensions"
    reframer = SmoothReframer(src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, sport_profile, ball_tracking, overlay_composite=overlay_composite, preserve_bottom_overlay=preserve_bottom_overlay, panel_mode=panel_mode, panel_max_faces=panel_max_faces, panel_detection_stride=panel_detection_stride, panel_gap=panel_gap, auto_mode=auto_mode)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else DEFAULT_OUTPUT_FPS, (target_w, target_h))
    if not writer.isOpened():
        cap.release(); return False, "Could not create output file"
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(reframer.process(frame))
            i += 1
            if progress_cb and fc > 0 and i % 5 == 0:
                progress_cb(i / fc, f"Creating vertical master {i}/{fc}")
    finally:
        cap.release(); writer.release()
    return True, "Done"

# -----------------------------------------------------------------------------
# Cloudflare + sessions
# -----------------------------------------------------------------------------
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
    stats_lock: threading.Lock = field(default_factory=threading.Lock)
    error: str = ""


def _stats_snapshot(session: LiveSession) -> dict:
    lock = getattr(session, "stats_lock", None)
    if lock:
        with lock:
            return dict(getattr(session, "stats", {}) or {})
    return dict(getattr(session, "stats", {}) or {})


def _stats_update(session: LiveSession, values: dict) -> None:
    lock = getattr(session, "stats_lock", None)
    if lock:
        with lock:
            session.stats.update(values)
    else:
        session.stats.update(values)


def _stats_inc(session: LiveSession, key: str, amount: int | float = 1) -> None:
    lock = getattr(session, "stats_lock", None)
    if lock:
        with lock:
            session.stats[key] = session.stats.get(key, 0) + amount
    else:
        session.stats[key] = session.stats.get(key, 0) + amount


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
    headers = {"Authorization": f"Bearer {cfg.api_token}", "Content-Type": "application/json", "User-Agent": "DualFlow-Vertical-Cloudflare"}
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
    payload = {"meta": {"name": name}, "recording": {"mode": recording_mode, "timeoutSeconds": 0}, "preferLowLatency": bool(cfg.prefer_low_latency), "enabled": True}
    status, parsed = _cf_api_request(cfg, "POST", f"/accounts/{cfg.account_id}/stream/live_inputs", payload)
    if status not in (200, 201) or not parsed.get("success"):
        raise RuntimeError(f"Create live input failed: {parsed}")
    return parsed["result"]


def disable_live_input(cfg: CFStreamConfig, uid: str) -> None:
    _cf_api_request(cfg, "PUT", f"/accounts/{cfg.account_id}/stream/live_inputs/{uid}", {"enabled": False})


def build_public_playback_urls(cfg: CFStreamConfig, uid: str):
    base = f"https://customer-{cfg.customer_code}.cloudflarestream.com/{uid}"
    return f"{base}/manifest/video.m3u8" + ("?protocol=llhls" if cfg.prefer_low_latency else ""), f"{base}/manifest/video.mpd", f"{base}/iframe?autoplay=true&muted=true&controls=true&preload=metadata"


def _common_output_args(fps_int: int) -> list[str]:
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(fps_int), "-b:v", DEFAULT_VIDEO_BITRATE, "-maxrate", DEFAULT_MAXRATE, "-bufsize", DEFAULT_BUFSIZE, "-g", str(fps_int), "-keyint_min", str(fps_int), "-sc_threshold", "0", "-profile:v", "high", "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int}:min-keyint={fps_int}", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-flvflags", "no_duration_filesize", "-fflags", "nobuffer", "-flags", "low_delay", "-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0"]


def build_push_file_command(reframed_mp4: str, rtmps_url: str, stream_key: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-re", "-i", reframed_mp4]
    if _source_has_audio(reframed_mp4):
        cmd += ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"]
    return cmd + _common_output_args(fps_int) + ["-f", "flv", target]


def start_vod_to_live_push(cfg: CFStreamConfig, reframed_mp4: str, asset_name: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS) -> LiveSession:
    li = create_live_input(cfg, safe_token(Path(asset_name).stem))
    uid, rtmps_url, key = li["uid"], li["rtmps"]["url"], li["rtmps"]["streamKey"]
    hls, dash, iframe = build_public_playback_urls(cfg, uid)
    cmd = build_push_file_command(reframed_mp4, rtmps_url, key, loop_input, output_fps)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True)
    return LiveSession(uid, rtmps_url, key, hls, dash, iframe, cmd, proc, log_path, status="streaming")


def build_realtime_rtmps_push_command(target_w: int, target_h: int, fps: float, rtmps_url: str, stream_key: str):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    return ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{target_w}x{target_h}", "-r", str(fps_int), "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"] + _common_output_args(fps_int) + ["-f", "flv", target]

# -----------------------------------------------------------------------------
# Process helpers
# -----------------------------------------------------------------------------
def _read_exact(stream, nbytes: int) -> bytes:
    chunks, remaining = [], nbytes
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _read_frame_timeout(proc: subprocess.Popen, nbytes: int, timeout: float = INGEST_READ_TIMEOUT) -> Optional[bytes]:
    if proc is None or proc.stdout is None:
        return None
    timeout = max(INGEST_READ_TIMEOUT_MIN, min(float(timeout or INGEST_READ_TIMEOUT), INGEST_READ_TIMEOUT_MAX))
    if os.name == "nt":
        q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=1)
        def _reader():
            try:
                q.put(proc.stdout.read(nbytes), block=False)
            except Exception:
                with contextlib.suppress(Exception):
                    q.put(None, block=False)
        threading.Thread(target=_reader, daemon=True).start()
        try:
            data = q.get(timeout=timeout)
        except queue.Empty:
            return None
        return data if data and len(data) == nbytes else None
    fd, deadline, data = proc.stdout.fileno(), time.monotonic() + timeout, bytearray()
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


def _build_ingest_command(source: str, fps: float, pace_input: bool, loop_file: bool) -> list[str]:
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf = f"fps={fps_int},scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:force_original_aspect_ratio=decrease,pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:(ow-iw)/2:(oh-ih)/2:black"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-flags", "low_delay"]
    cmd += _source_input_args(source, pace_input=bool(pace_input) and not is_network_source(source), loop_file=loop_file)
    cmd += ["-an", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return cmd


def _open_ingest_process(source: str, fps: float, pace_input: bool, loop_file: bool, log_path: str) -> subprocess.Popen:
    cmd = _build_ingest_command(source, fps, pace_input, loop_file)
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
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()


def _make_placeholder_frame(w: int, h: int, text: str = "Starting stream...") -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (18, 22, 36)
    cv2.putText(frame, "Vertical stream", (28, max(48, h // 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (28, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 255), 2, cv2.LINE_AA)
    return frame

# -----------------------------------------------------------------------------
# Realtime worker
# -----------------------------------------------------------------------------
def _realtime_worker(session: LiveSession, source: str, target_w: int, target_h: int, delay_seconds: float, smooth_strength: float, analysis_stride: int, deadzone_ratio: float, max_pan_ratio: float, loop_file: bool, pace_input: bool, sport_profile: str, ball_tracking: bool, overlay_composite: bool, preserve_bottom_overlay: bool, panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, auto_mode: bool = False) -> None:
    session.status = "probing"
    start_ts = time.monotonic()
    info = probe_source(source)
    source_fps = float(info.get("fps") or DEFAULT_OUTPUT_FPS)
    if source_fps <= 0 or source_fps > 120:
        source_fps = DEFAULT_OUTPUT_FPS
    fps = DEFAULT_OUTPUT_FPS
    frame_interval = 1.0 / max(fps, 1.0)
    adaptive_read_timeout = max(INGEST_READ_TIMEOUT_MIN, min(INGEST_READ_TIMEOUT_MAX, max(INGEST_READ_TIMEOUT, frame_interval * 4.0)))
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_seconds = max(0.0, float(delay_seconds))
    max_buffer_frames = max(4, int(round(MAX_BUFFER_SECONDS * fps)))
    target_delay_frames = max(0, int(round(delay_seconds * fps)))
    _stats_update(session, {
        "fps": round(fps, 3), "fps_source": round(source_fps, 3), "fps_output": round(fps, 3), "ingest_read_timeout": round(adaptive_read_timeout, 3), "ingest_stall_policy": "timeout_is_stall_not_eof",
        "fps_in": 0.0, "fps_out": 0.0, "fps_process": 0.0, "ingest_fps_1s": 0.0, "output_fps_1s": 0.0, "process_fps_1s": 0.0,
        "delay_seconds_requested": round(delay_seconds, 2), "delay_seconds_configured": round(delay_seconds, 2), "target_delay_seconds": round(delay_seconds, 2), "target_delay_frames": target_delay_frames, "delay_frames": target_delay_frames, "effective_live_buffer_seconds": round(MAX_BUFFER_SECONDS, 2),
        "working_resolution": f"{src_w}x{src_h}", "source_reported_resolution": f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)}", "source_reported_fps": round(source_fps, 3),
        "sport_profile": sport_profile, "panel_mode": panel_mode, "auto_mode": auto_mode and not panel_mode, "frames_in": 0, "frames_processed": 0, "frames_out": 0, "frame_drops": 0, "input_drop_count": 0, "startup_buffer_fill_frames": 0, "output_underruns": 0,
        "write_failures": 0, "output_write_failures": 0, "buffer_len": 0, "buffer_seconds": 0.0, "buffer_seconds_est": 0.0, "buffer_fill_pct": 0.0, "placeholder_frames": 0, "source_stalls": 0, "consecutive_source_stalls": 0, "ingest_restarts": 0,
        "processing_ms": 0.0, "avg_process_ms": 0.0, "p95_process_ms": 0.0, "read_ms": 0.0, "avg_ingest_read_ms": 0.0, "p95_ingest_read_ms": 0.0, "write_ms": 0.0, "avg_output_write_ms": 0.0, "p95_output_write_ms": 0.0,
        "avg_schedule_drift_ms": 0.0, "p95_schedule_drift_ms": 0.0, "startup_ms_to_first_source_frame": 0.0, "startup_ms_to_first_live_frame": 0.0, "ball_confidence": 0.0, "overlay_top": False, "overlay_bottom": False, "panel_active_faces": 0, "panel_detector": "-",
        "ffmpeg_alive": False, "ingest_alive": False, "ffmpeg_returncode": None, "ingest_returncode": None, "mode": "panel" if panel_mode else "sports" if ball_tracking else "single", "health": "starting", "updated_at_ms": int(time.time() * 1000),
    })
    reframer = SmoothReframer(src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, sport_profile, False if panel_mode else ball_tracking, overlay_composite=overlay_composite, preserve_bottom_overlay=preserve_bottom_overlay, panel_mode=panel_mode, panel_max_faces=panel_max_faces, panel_detection_stride=panel_detection_stride, panel_gap=panel_gap, auto_mode=auto_mode and not panel_mode)
    buffer: collections.deque[tuple[float, np.ndarray]] = collections.deque()
    placeholder = _make_placeholder_frame(target_w, target_h)
    last_good_frame = placeholder.copy()
    try:
        session.proc = _start_output_process(session)
    except Exception as exc:
        session.status = "ffmpeg_start_failed"; session.error = str(exc); return
    try:
        prime = max(1, int(fps * LIVE_STARTUP_PRIME_SECONDS))
        if session.proc.stdin:
            for _ in range(prime):
                if session.stop_event.is_set():
                    break
                session.proc.stdin.write(placeholder.tobytes())
                _stats_inc(session, "placeholder_frames"); _stats_inc(session, "frames_out")
    except Exception as exc:
        session.status = "ffmpeg_pipe_broken"; session.error = f"Could not prime output: {exc}"; return
    ingest: Optional[subprocess.Popen] = None
    source_ended = False
    next_deadline = time.monotonic() + frame_interval
    fps_window_start = time.monotonic()
    win_in = win_out = win_proc = 0
    consecutive_stalls = 0
    live_ready = False
    first_source_seen = False
    first_live_written = False
    process_samples: collections.deque[float] = collections.deque(maxlen=120)
    read_samples: collections.deque[float] = collections.deque(maxlen=120)
    write_samples: collections.deque[float] = collections.deque(maxlen=120)
    drift_samples: collections.deque[float] = collections.deque(maxlen=120)

    def p95(vals):
        if not vals:
            return 0.0
        arr = sorted(vals)
        return float(arr[min(len(arr) - 1, int(round((len(arr) - 1) * 0.95)))])

    def restart_ingest() -> Optional[subprocess.Popen]:
        nonlocal ingest, consecutive_stalls
        _terminate_process(ingest)
        _stats_inc(session, "ingest_restarts")
        consecutive_stalls += 1
        _stats_update(session, {"consecutive_source_stalls": consecutive_stalls})
        if consecutive_stalls > LIVE_MAX_RECOVERABLE_STALLS and not is_network_source(source) and not loop_file:
            return None
        return _open_ingest_process(source, fps, pace_input, loop_file, session.log_path)

    try:
        session.status = "connecting_source"
        ingest = _open_ingest_process(source, fps, pace_input, loop_file, session.log_path)
        session.status = "streaming"
        _stats_update(session, {"health": "running"})
        while not session.stop_event.is_set():
            if ingest is None or ingest.poll() is not None:
                _stats_inc(session, "source_stalls")
                ingest = restart_ingest()
                if ingest is None:
                    session.status = "source_ended"; session.error = f"Source ended/unavailable: {source}"; break
            capture_time = time.monotonic()
            rs = time.monotonic()
            raw = _read_frame_timeout(ingest, frame_bytes, timeout=adaptive_read_timeout) if ingest else None
            read_ms = (time.monotonic() - rs) * 1000
            read_samples.append(read_ms)
            if raw is not None and len(raw) == frame_bytes:
                consecutive_stalls = 0
                _stats_update(session, {"consecutive_source_stalls": 0})
                _stats_inc(session, "frames_in"); win_in += 1
                if not first_source_seen:
                    first_source_seen = True
                    _stats_update(session, {"startup_ms_to_first_source_frame": round((time.monotonic() - start_ts) * 1000, 2)})
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))
                ps = time.monotonic()
                try:
                    processed = reframer.process(frame)
                except Exception as exc:
                    logger.exception("[ReframerError] %s", exc)
                    processed = last_good_frame
                process_samples.append((time.monotonic() - ps) * 1000)
                _stats_inc(session, "frames_processed"); win_proc += 1
                buffer.append((capture_time, processed.copy()))
                last_good_frame = processed.copy()
            else:
                _stats_inc(session, "source_stalls")
                consecutive_stalls += 1
                _stats_update(session, {"consecutive_source_stalls": consecutive_stalls})
                if consecutive_stalls >= LIVE_MAX_RECOVERABLE_STALLS:
                    restarted = restart_ingest()
                    if restarted is not None:
                        ingest = restarted
                    elif not loop_file and not is_network_source(source):
                        source_ended = True
            while len(buffer) > max_buffer_frames:
                buffer.popleft(); _stats_inc(session, "frame_drops"); _stats_inc(session, "input_drop_count")
            cutoff = time.monotonic() - delay_seconds
            if buffer and buffer[0][0] <= cutoff:
                out_frame = buffer.popleft()[1]
                live_ready = True
            elif buffer:
                out_frame = last_good_frame
                _stats_inc(session, "output_underruns" if live_ready else "startup_buffer_fill_frames")
            elif not source_ended:
                out_frame = last_good_frame
                _stats_inc(session, "output_underruns" if live_ready else "startup_buffer_fill_frames")
            elif source_ended and buffer:
                out_frame = buffer.popleft()[1]
            else:
                if not loop_file:
                    session.status = "source_ended"
                    break
                out_frame = last_good_frame
            ws = time.monotonic()
            try:
                if session.proc is None or session.proc.stdin is None or session.proc.poll() is not None:
                    raise RuntimeError("Output FFmpeg process is not running")
                session.proc.stdin.write(out_frame.tobytes())
                _stats_inc(session, "frames_out"); win_out += 1
                if not first_live_written:
                    first_live_written = True
                    _stats_update(session, {"startup_ms_to_first_live_frame": round((time.monotonic() - start_ts) * 1000, 2)})
            except Exception as exc:
                _stats_inc(session, "write_failures"); _stats_inc(session, "output_write_failures")
                session.status = "ffmpeg_pipe_broken"; session.error = str(exc); break
            write_ms = (time.monotonic() - ws) * 1000
            write_samples.append(write_ms)
            sleep_for = next_deadline - time.monotonic()
            drift_samples.append(max(0.0, -sleep_for * 1000))
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif -sleep_for > frame_interval * 2:
                next_deadline = time.monotonic()
                while len(buffer) > 1:
                    buffer.popleft(); _stats_inc(session, "frame_drops"); _stats_inc(session, "input_drop_count")
            next_deadline += frame_interval
            elapsed = time.monotonic() - fps_window_start
            if elapsed >= 1.0:
                fps_in, fps_out, fps_proc = round(win_in / elapsed, 1), round(win_out / elapsed, 1), round(win_proc / elapsed, 1)
                win_in = win_out = win_proc = 0
                fps_window_start = time.monotonic()
            else:
                snap = _stats_snapshot(session)
                fps_in, fps_out, fps_proc = snap.get("fps_in", 0.0), snap.get("fps_out", 0.0), snap.get("fps_process", 0.0)
            bsec = len(buffer) / max(fps, 1)
            p95_proc, p95_write = round(p95(process_samples), 2), round(p95(write_samples), 2)
            health = "healthy"
            if fps_out and fps_out < fps * 0.80:
                health = "output_fps_low"
            elif p95_proc > (1000 / fps) * 1.05:
                health = "processing_bottleneck"
            elif p95_write > 20:
                health = "rtmps_backpressure"
            elif _stats_snapshot(session).get("source_stalls", 0) > 0:
                health = "source_stalls_detected"
            _stats_update(session, {"read_ms": round(read_ms, 2), "processing_ms": round(process_samples[-1], 2) if process_samples else 0.0, "write_ms": round(write_ms, 2), "avg_process_ms": round(sum(process_samples) / max(len(process_samples), 1), 2), "p95_process_ms": p95_proc, "avg_ingest_read_ms": round(sum(read_samples) / max(len(read_samples), 1), 2), "p95_ingest_read_ms": round(p95(read_samples), 2), "avg_output_write_ms": round(sum(write_samples) / max(len(write_samples), 1), 2), "p95_output_write_ms": p95_write, "avg_schedule_drift_ms": round(sum(drift_samples) / max(len(drift_samples), 1), 2), "p95_schedule_drift_ms": round(p95(drift_samples), 2), "buffer_len": len(buffer), "buffer_seconds": round(bsec, 3), "buffer_seconds_est": round(bsec, 3), "buffer_fill_pct": round(100 * len(buffer) / max(max_buffer_frames, 1), 1), "fps_in": fps_in, "ingest_fps_1s": fps_in, "fps_out": fps_out, "output_fps_1s": fps_out, "fps_actual": fps_out, "fps_process": fps_proc, "process_fps_1s": fps_proc, "ffmpeg_alive": bool(session.proc is not None and session.proc.poll() is None), "ingest_alive": bool(ingest is not None and ingest.poll() is None), "ffmpeg_returncode": None if session.proc is None else session.proc.poll(), "ingest_returncode": None if ingest is None else ingest.poll(), "ball_confidence": round(reframer.ball_tracker.conf, 3) if reframer.ball_tracker else 0.0, "overlay_top": reframer.overlay_detector.top_overlay is not None, "overlay_bottom": reframer.overlay_detector.bottom_overlay is not None, "panel_active_faces": reframer.panel_tracker.active_count if reframer.panel_tracker else 0, "panel_detector": getattr(reframer.panel_tracker, "detector_backend_name", "-") if reframer.panel_tracker else "-", "mode": "panel" if reframer.panel_mode else "sports" if reframer.ball_tracker is not None else "single", "health": health, "updated_at_ms": int(time.time() * 1000)})
    except Exception as exc:
        logger.exception("[WORKER CRASH] %s", exc)
        session.status = "worker_error"; session.error = str(exc)
    finally:
        _terminate_process(ingest)
        with contextlib.suppress(Exception):
            if session.proc and session.proc.stdin:
                session.proc.stdin.close()
        _terminate_process(session.proc)
        _stats_update(session, {"updated_at_ms": int(time.time() * 1000)})
        if session.status not in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}:
            session.status = "stopped"


def start_realtime_delayed_vertical_push(cfg: CFStreamConfig, source: str, asset_name: str, target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H, delay_seconds: float = 0.0, smooth_strength: float = 0.975, analysis_stride: int = 4, deadzone_ratio: float = 0.05, max_pan_ratio: float = 0.012, loop_file: bool = False, pace_input: bool = True, sport_profile: str = "auto", ball_tracking: bool = True, overlay_composite: bool = True, preserve_bottom_overlay: bool = False, panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, auto_mode: bool = False) -> LiveSession:
    li = create_live_input(cfg, safe_token(Path(asset_name).stem))
    uid, rtmps_url, key = li["uid"], li["rtmps"]["url"], li["rtmps"]["streamKey"]
    hls, dash, iframe = build_public_playback_urls(cfg, uid)
    cmd = build_realtime_rtmps_push_command(target_w, target_h, DEFAULT_OUTPUT_FPS, rtmps_url, key)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    session = LiveSession(uid, rtmps_url, key, hls, dash, iframe, cmd, None, log_path)
    worker = threading.Thread(target=_realtime_worker, args=(session, source, target_w, target_h, delay_seconds, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, loop_file, pace_input, sport_profile, ball_tracking, overlay_composite, preserve_bottom_overlay, panel_mode, panel_max_faces, panel_detection_stride, panel_gap, auto_mode), daemon=True)
    session.worker = worker
    worker.start()
    return session


def cleanup_old_logs(directory: str = "/tmp", max_age_seconds: int = CLEANUP_LOG_MAX_AGE_SECONDS) -> int:
    now, removed = time.time(), 0
    try:
        for name in os.listdir(directory):
            if name.endswith(".log"):
                path = os.path.join(directory, name)
                with contextlib.suppress(Exception):
                    if now - os.path.getmtime(path) > max_age_seconds:
                        os.remove(path); removed += 1
    except Exception:
        pass
    return removed


def stop_live_session(cfg: CFStreamConfig, session: Optional[LiveSession]) -> None:
    if not session:
        return
    session.stop_event.set()
    with contextlib.suppress(Exception):
        if session.worker and session.worker.is_alive():
            session.worker.join(timeout=8)
    _terminate_process(session.proc)
    with contextlib.suppress(Exception):
        disable_live_input(cfg, session.uid)
    cleanup_old_logs()


def read_log_tail(path: str, max_chars: int = 12000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            return fp.read()[-max_chars:]
    except Exception:
        return ""


# =============================================================================
# P0 ARCHITECTURE PATCH: decoupled live pipeline
# -----------------------------------------------------------------------------
# This overrides the earlier synchronous realtime worker. The new design starts
# RTMPS output immediately and decouples ingest, verticalization, and output clock:
#   ingest thread    -> latest raw frame slot
#   processor thread -> latest vertical frame slot, drops stale input
#   output clock     -> fixed 30 FPS write, repeats latest vertical frame if needed
# This keeps Cloudflare playback alive and low-buffer even when CV processing spikes.
# =============================================================================

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
    auto_mode: bool = False,
) -> None:
    start_ts = time.monotonic()
    fps = DEFAULT_OUTPUT_FPS
    fps_int = max(24, min(60, int(round(fps))))
    frame_interval = 1.0 / fps_int
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    adaptive_read_timeout = max(INGEST_READ_TIMEOUT_MIN, min(INGEST_READ_TIMEOUT_MAX, max(INGEST_READ_TIMEOUT, frame_interval * 4.0)))
    delay_seconds = max(0.0, min(float(delay_seconds or 0.0), MAX_BUFFER_SECONDS))
    max_buffer_frames = max(2, int(round(max(delay_seconds, 0.10) * fps_int)))

    # Shared slots. We intentionally keep latest-only semantics to avoid live latency build-up.
    lock = threading.Lock()
    latest_input = {"seq": 0, "ts": 0.0, "frame": None}
    latest_output = {"seq": 0, "ts": 0.0, "frame": _make_placeholder_frame(target_w, target_h)}
    stop = session.stop_event
    input_buffer: collections.deque[tuple[int, float, np.ndarray]] = collections.deque(maxlen=max_buffer_frames)

    counters = {
        "frames_in": 0,
        "frames_processed": 0,
        "frames_out": 0,
        "frames_repeated": 0,
        "frames_dropped_input": 0,
        "frames_dropped_processing": 0,
        "source_stalls": 0,
        "consecutive_source_stalls": 0,
        "ingest_restarts": 0,
        "write_failures": 0,
        "output_write_failures": 0,
        "frame_drops": 0,
        "input_drop_count": 0,
        "startup_buffer_fill_frames": 0,
        "output_underruns": 0,
    }
    process_samples: collections.deque[float] = collections.deque(maxlen=180)
    read_samples: collections.deque[float] = collections.deque(maxlen=180)
    write_samples: collections.deque[float] = collections.deque(maxlen=180)
    drift_samples: collections.deque[float] = collections.deque(maxlen=180)
    win = {"t": time.monotonic(), "in": 0, "proc": 0, "out": 0}

    def _p95(vals) -> float:
        if not vals:
            return 0.0
        arr = sorted(vals)
        return float(arr[min(len(arr) - 1, int(round((len(arr) - 1) * 0.95)))])

    _stats_update(session, {
        "pipeline_arch": "decoupled_fixed_output_clock_v2",
        "health": "starting",
        "fps": fps_int,
        "fps_source": 0.0,
        "fps_output": fps_int,
        "fps_in": 0.0,
        "fps_process": 0.0,
        "fps_out": 0.0,
        "ingest_fps_1s": 0.0,
        "process_fps_1s": 0.0,
        "output_fps_1s": 0.0,
        "delay_seconds_requested": round(delay_seconds, 2),
        "delay_seconds_configured": round(delay_seconds, 2),
        "target_delay_seconds": round(delay_seconds, 2),
        "target_delay_frames": int(round(delay_seconds * fps_int)),
        "effective_live_buffer_seconds": round(max_buffer_frames / fps_int, 3),
        "ingest_read_timeout": round(adaptive_read_timeout, 3),
        "ingest_stall_policy": "timeout_is_stall_not_eof",
        "buffer_policy": "latest_frame_drop_stale_repeat_last",
        "working_resolution": f"{src_w}x{src_h}",
        "sport_profile": sport_profile,
        "panel_mode": bool(panel_mode),
        "auto_mode": bool(auto_mode and not panel_mode),
        "mode": "panel" if panel_mode else "sports" if ball_tracking else "single",
        "frames_in": 0,
        "frames_processed": 0,
        "frames_out": 0,
        "frames_repeated": 0,
        "frames_dropped_input": 0,
        "frames_dropped_processing": 0,
        "frame_drops": 0,
        "input_drop_count": 0,
        "startup_buffer_fill_frames": 0,
        "output_underruns": 0,
        "source_stalls": 0,
        "consecutive_source_stalls": 0,
        "ingest_restarts": 0,
        "write_failures": 0,
        "output_write_failures": 0,
        "processing_ms": 0.0,
        "avg_process_ms": 0.0,
        "p95_process_ms": 0.0,
        "read_ms": 0.0,
        "avg_ingest_read_ms": 0.0,
        "p95_ingest_read_ms": 0.0,
        "write_ms": 0.0,
        "avg_output_write_ms": 0.0,
        "p95_output_write_ms": 0.0,
        "avg_schedule_drift_ms": 0.0,
        "p95_schedule_drift_ms": 0.0,
        "buffer_len": 0,
        "buffer_seconds": 0.0,
        "buffer_seconds_est": 0.0,
        "buffer_fill_pct": 0.0,
        "latest_frame_age_ms": 0.0,
        "process_budget_exceeded_count": 0,
        "ball_confidence": 0.0,
        "panel_active_faces": 0,
        "panel_detector": "-",
        "ffmpeg_alive": False,
        "ingest_alive": False,
        "ffmpeg_returncode": None,
        "ingest_returncode": None,
        "startup_ms_to_first_live_frame": 0.0,
        "startup_ms_to_first_source_frame": 0.0,
        "updated_at_ms": int(time.time() * 1000),
    })

    try:
        session.proc = _start_output_process(session)
        session.status = "streaming"
        _stats_update(session, {"health": "running", "ffmpeg_alive": True})
    except Exception as exc:
        session.status = "ffmpeg_start_failed"
        session.error = str(exc)
        return

    # Start immediately: prime Cloudflare/RTMPS with a few placeholder frames.
    try:
        prime = max(1, int(fps_int * LIVE_STARTUP_PRIME_SECONDS))
        if session.proc.stdin:
            for _ in range(prime):
                session.proc.stdin.write(latest_output["frame"].tobytes())
                counters["frames_out"] += 1
                counters["frames_repeated"] += 1
    except Exception as exc:
        session.status = "ffmpeg_pipe_broken"
        session.error = f"Could not prime output: {exc}"
        return

    ingest_holder = {"proc": None}

    def _restart_ingest() -> Optional[subprocess.Popen]:
        _terminate_process(ingest_holder.get("proc"))
        counters["ingest_restarts"] += 1
        if counters["consecutive_source_stalls"] > MAX_CONSECUTIVE_STALLS_NON_LOOP and not is_network_source(source) and not loop_file:
            return None
        try:
            return _open_ingest_process(source, fps_int, pace_input, loop_file, session.log_path)
        except Exception as exc:
            session.error = f"Ingest restart failed: {exc}"
            return None

    def ingest_loop() -> None:
        try:
            ingest_holder["proc"] = _open_ingest_process(source, fps_int, pace_input, loop_file, session.log_path)
            first = True
            while not stop.is_set():
                proc = ingest_holder.get("proc")
                if proc is None or proc.poll() is not None:
                    counters["source_stalls"] += 1
                    counters["consecutive_source_stalls"] += 1
                    proc = _restart_ingest()
                    ingest_holder["proc"] = proc
                    if proc is None:
                        if not loop_file and not is_network_source(source):
                            session.status = "source_ended"
                            stop.set()
                            break
                        time.sleep(0.05)
                        continue
                rs = time.monotonic()
                raw = _read_frame_timeout(proc, frame_bytes, timeout=adaptive_read_timeout)
                read_ms = (time.monotonic() - rs) * 1000.0
                read_samples.append(read_ms)
                if raw is None or len(raw) != frame_bytes:
                    counters["source_stalls"] += 1
                    counters["consecutive_source_stalls"] += 1
                    if counters["consecutive_source_stalls"] >= 3:
                        proc = _restart_ingest()
                        ingest_holder["proc"] = proc
                    continue
                counters["consecutive_source_stalls"] = 0
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
                ts = time.monotonic()
                with lock:
                    if len(input_buffer) == input_buffer.maxlen:
                        counters["frames_dropped_input"] += 1
                        counters["frame_drops"] += 1
                        counters["input_drop_count"] += 1
                    latest_input["seq"] += 1
                    latest_input["ts"] = ts
                    latest_input["frame"] = frame
                    input_buffer.append((latest_input["seq"], ts, frame))
                counters["frames_in"] += 1
                win["in"] += 1
                if first:
                    first = False
                    _stats_update(session, {"startup_ms_to_first_source_frame": round((time.monotonic() - start_ts) * 1000.0, 2)})
        except Exception as exc:
            logger.exception("[INGEST THREAD ERROR] %s", exc)
            session.error = f"ingest_error: {exc}"

    def processor_loop() -> None:
        try:
            reframer = SmoothReframer(
                src_w, src_h, target_w, target_h,
                smooth_strength=smooth_strength,
                analysis_stride=analysis_stride,
                deadzone_ratio=deadzone_ratio,
                max_pan_ratio=max_pan_ratio,
                sport_profile=sport_profile,
                ball_tracking=(False if panel_mode else ball_tracking),
                overlay_composite=overlay_composite,
                preserve_bottom_overlay=preserve_bottom_overlay,
                panel_mode=panel_mode,
                panel_max_faces=panel_max_faces,
                panel_detection_stride=panel_detection_stride,
                panel_gap=panel_gap,
                auto_mode=(auto_mode and not panel_mode),
            )
            last_seq = 0
            budget_ms = 1000.0 / fps_int
            while not stop.is_set():
                item = None
                with lock:
                    if input_buffer:
                        # Latest-frame policy: drop stale frames before processing.
                        if len(input_buffer) > 1:
                            dropped = len(input_buffer) - 1
                            counters["frames_dropped_processing"] += dropped
                            counters["frame_drops"] += dropped
                            while len(input_buffer) > 1:
                                input_buffer.popleft()
                        item = input_buffer.popleft()
                    elif latest_input["frame"] is not None and latest_input["seq"] != last_seq:
                        item = (latest_input["seq"], latest_input["ts"], latest_input["frame"])
                if item is None:
                    time.sleep(0.002)
                    continue
                seq, ts, frame = item
                if seq == last_seq:
                    continue
                last_seq = seq
                ps = time.monotonic()
                try:
                    out = reframer.process(frame)
                except Exception as exc:
                    logger.exception("[REFRAMER ERROR] %s", exc)
                    out = latest_output["frame"]
                proc_ms = (time.monotonic() - ps) * 1000.0
                process_samples.append(proc_ms)
                if proc_ms > budget_ms:
                    _stats_inc(session, "process_budget_exceeded_count")
                with lock:
                    latest_output["seq"] += 1
                    latest_output["ts"] = time.monotonic()
                    latest_output["frame"] = out.copy()
                counters["frames_processed"] += 1
                win["proc"] += 1
                _stats_update(session, {
                    "ball_confidence": round(reframer.ball_tracker.conf, 3) if reframer.ball_tracker is not None else 0.0,
                    "panel_active_faces": reframer.panel_tracker.active_count if reframer.panel_tracker else 0,
                    "panel_detector": getattr(reframer.panel_tracker, "detector_backend_name", "-") if reframer.panel_tracker else "-",
                    "mode": "panel" if reframer.panel_mode else "sports" if reframer.ball_tracker is not None else "single",
                })
        except Exception as exc:
            logger.exception("[PROCESSOR THREAD ERROR] %s", exc)
            session.error = f"processor_error: {exc}"

    ingest_thread = threading.Thread(target=ingest_loop, daemon=True, name="ingest_loop")
    processor_thread = threading.Thread(target=processor_loop, daemon=True, name="processor_loop")
    ingest_thread.start()
    processor_thread.start()

    # Fixed output clock: never wait for processing. Repeat latest frame when needed.
    next_deadline = time.monotonic()
    last_written_seq = -1
    first_live = True
    try:
        while not stop.is_set():
            with lock:
                out_frame = latest_output["frame"].copy()
                out_seq = latest_output["seq"]
                blen = len(input_buffer)
                latest_age_ms = (time.monotonic() - latest_output["ts"]) * 1000.0 if latest_output["ts"] else 0.0
            if out_seq == last_written_seq:
                counters["frames_repeated"] += 1
                if first_live:
                    counters["startup_buffer_fill_frames"] += 1
                else:
                    counters["output_underruns"] += 1
            last_written_seq = out_seq
            ws = time.monotonic()
            try:
                if session.proc is None or session.proc.stdin is None or session.proc.poll() is not None:
                    raise RuntimeError("Output FFmpeg process is not running")
                session.proc.stdin.write(out_frame.tobytes())
                counters["frames_out"] += 1
                win["out"] += 1
                if first_live:
                    first_live = False
                    _stats_update(session, {"startup_ms_to_first_live_frame": round((time.monotonic() - start_ts) * 1000.0, 2)})
            except Exception as exc:
                counters["write_failures"] += 1
                counters["output_write_failures"] += 1
                session.status = "ffmpeg_pipe_broken"
                session.error = str(exc)
                break
            write_samples.append((time.monotonic() - ws) * 1000.0)

            sleep_for = next_deadline - time.monotonic()
            drift_ms = max(0.0, -sleep_for * 1000.0)
            drift_samples.append(drift_ms)
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif -sleep_for > frame_interval * 3:
                next_deadline = time.monotonic()
            next_deadline += frame_interval

            now = time.monotonic()
            if now - win["t"] >= 1.0:
                elapsed = now - win["t"]
                fps_in = round(win["in"] / elapsed, 1)
                fps_proc = round(win["proc"] / elapsed, 1)
                fps_out = round(win["out"] / elapsed, 1)
                win["t"], win["in"], win["proc"], win["out"] = now, 0, 0, 0
                p95_proc = round(_p95(process_samples), 2)
                p95_write = round(_p95(write_samples), 2)
                health = "healthy"
                if fps_out < fps_int * 0.90:
                    health = "output_fps_low"
                elif p95_proc > (1000.0 / fps_int) * 2.5:
                    health = "processing_tail_latency_high"
                elif p95_write > 20:
                    health = "rtmps_backpressure"
                elif counters["source_stalls"] > 0:
                    health = "source_stalls_detected"
                _stats_update(session, {
                    "health": health,
                    "fps_in": fps_in,
                    "ingest_fps_1s": fps_in,
                    "fps_process": fps_proc,
                    "process_fps_1s": fps_proc,
                    "fps_out": fps_out,
                    "output_fps_1s": fps_out,
                    "fps_actual": fps_out,
                    **counters,
                    "read_ms": round(read_samples[-1], 2) if read_samples else 0.0,
                    "avg_ingest_read_ms": round(sum(read_samples) / max(len(read_samples), 1), 2),
                    "p95_ingest_read_ms": round(_p95(read_samples), 2),
                    "processing_ms": round(process_samples[-1], 2) if process_samples else 0.0,
                    "avg_process_ms": round(sum(process_samples) / max(len(process_samples), 1), 2),
                    "p95_process_ms": p95_proc,
                    "write_ms": round(write_samples[-1], 2) if write_samples else 0.0,
                    "avg_output_write_ms": round(sum(write_samples) / max(len(write_samples), 1), 2),
                    "p95_output_write_ms": p95_write,
                    "avg_schedule_drift_ms": round(sum(drift_samples) / max(len(drift_samples), 1), 2),
                    "p95_schedule_drift_ms": round(_p95(drift_samples), 2),
                    "buffer_len": blen,
                    "buffer_seconds": round(blen / fps_int, 3),
                    "buffer_seconds_est": round(blen / fps_int, 3),
                    "buffer_fill_pct": round(100.0 * blen / max(max_buffer_frames, 1), 1),
                    "latest_frame_age_ms": round(latest_age_ms, 2),
                    "ffmpeg_alive": bool(session.proc is not None and session.proc.poll() is None),
                    "ingest_alive": bool(ingest_holder.get("proc") is not None and ingest_holder["proc"].poll() is None),
                    "ffmpeg_returncode": None if session.proc is None else session.proc.poll(),
                    "ingest_returncode": None if ingest_holder.get("proc") is None else ingest_holder["proc"].poll(),
                    "updated_at_ms": int(time.time() * 1000),
                })
    except Exception as exc:
        logger.exception("[OUTPUT CLOCK ERROR] %s", exc)
        session.status = "worker_error"
        session.error = str(exc)
    finally:
        stop.set()
        _terminate_process(ingest_holder.get("proc"))
        with contextlib.suppress(Exception):
            if session.proc and session.proc.stdin:
                session.proc.stdin.close()
        _terminate_process(session.proc)
        _stats_update(session, {"updated_at_ms": int(time.time() * 1000)})
        if session.status not in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}:
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
    panel_detection_stride: int = 2,
    panel_gap: int = 4,
    auto_mode: bool = False,
) -> LiveSession:
    li = create_live_input(cfg, safe_token(Path(asset_name).stem))
    uid = li["uid"]
    rtmps_url = li["rtmps"]["url"]
    key = li["rtmps"]["streamKey"]
    hls, dash, iframe = build_public_playback_urls(cfg, uid)
    cmd = build_realtime_rtmps_push_command(target_w, target_h, DEFAULT_OUTPUT_FPS, rtmps_url, key)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    session = LiveSession(uid, rtmps_url, key, hls, dash, iframe, cmd, None, log_path)
    worker = threading.Thread(
        target=_realtime_worker,
        args=(session, source, target_w, target_h, delay_seconds, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, loop_file, pace_input, sport_profile, ball_tracking, overlay_composite, preserve_bottom_overlay, panel_mode, panel_max_faces, panel_detection_stride, panel_gap, auto_mode),
        daemon=True,
        name="decoupled_realtime_worker",
    )
    session.worker = worker
    worker.start()
    return session
