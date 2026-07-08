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

try:
    import psutil  # optional analytics
except Exception:
    psutil = None

try:
    from ultralytics import YOLO as _YOLO  # optional P1 sports-ball detector
except Exception:
    _YOLO = None

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

# Constants
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
MAX_BUFFER_SECONDS = 0.35
LIVE_STARTUP_PRIME_SECONDS = 0.25
INGEST_READ_TIMEOUT = 0.75
INGEST_READ_TIMEOUT_MIN = 0.25
INGEST_READ_TIMEOUT_MAX = 1.25
CLEANUP_LOG_MAX_AGE_SECONDS = 6 * 60 * 60
MAX_CONSECUTIVE_STALLS_NON_LOOP = 12

# P0/P1 runtime tuning
LIVE_OUTPUT_FPS_MIN = 24
LIVE_OUTPUT_FPS_MAX = 30
LIVE_BACKLOG_SOFT_FRAMES = 2
LIVE_BACKLOG_HARD_FRAMES = 5
LIVE_MAX_ACCEPTABLE_FRAME_AGE_MS = 220
YOLO_BALL_DETECT_EVERY_N = 6
YOLO_BALL_MODEL_CANDIDATES = ("yolo11n.pt", "yolov8n.pt", "yolov8s.pt")



def _normalize_output_fps(value: float | int | str | None, *, live: bool = False) -> int:
    """Normalize fps for streaming paths while avoiding accidental live overload."""
    try:
        fps = int(round(float(value or DEFAULT_OUTPUT_FPS)))
    except Exception:
        fps = int(round(DEFAULT_OUTPUT_FPS))
    hi = LIVE_OUTPUT_FPS_MAX if live else 60
    return max(LIVE_OUTPUT_FPS_MIN, min(hi, fps))


def _redact_url(value: str) -> str:
    """Redact stream keys and URL credentials before writing commands to logs."""
    if not value:
        return value
    text = str(value)
    text = re.sub(r"(rtmps?://[^\s]+/)[^/\s]+$", r"\1<redacted-stream-key>", text)
    text = re.sub(r"(?i)(api[_-]?token|token|key|sig|signature|access_token)=([^&\s]+)", r"\1=<redacted>", text)
    text = re.sub(r"(?i)(password|pass|pwd)=([^&\s]+)", r"\1=<redacted>", text)
    return text


def _redact_cmd(cmd: list[str]) -> str:
    return " ".join(_redact_url(part) for part in (cmd or []))


def _audio_source_input_args(source: str, fps: float, *, loop_audio: bool = False, pace_audio: bool = True) -> list[str]:
    """Build input args for preserving source audio in the RTMPS output muxer."""
    if not source:
        return []
    is_net = is_network_source(source)
    args = ["-thread_queue_size", "512"]
    if loop_audio and not is_net:
        args += ["-stream_loop", "-1"]
    if pace_audio and not is_net:
        args += ["-re"]
    if is_net:
        args += ["-rw_timeout", "15000000"]
    if str(source).lower().startswith(("http://", "https://")):
        args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2"]
    args += ["-i", source]
    return args


def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "", value).strip(".-") or "stream"


def is_network_source(source: str) -> bool:
    s = (source or " ").lower().strip()
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
            elif s.get("codec_type") == "audio":
                res["has_audio"] = True
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
        return False


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


class OverlayDetector:
    def __init__(
        self, src_w: int, src_h: int, top_scan_ratio: float = 0.18, bottom_scan_ratio: float = 0.14,
        warmup_frames: int = 18, stable_diff_threshold: float = 11.0, row_text_threshold: float = 0.018, overlay_hold_frames: int = 10,
    ):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.top_scan_h = max(24, int(round(src_h * top_scan_ratio)))
        self.bottom_scan_h = max(24, int(round(src_h * bottom_scan_ratio)))
        self.warmup_frames = max(4, int(warmup_frames))
        self.stable_diff_threshold = float(stable_diff_threshold)
        self.row_text_threshold = float(row_text_threshold)
        self.overlay_hold_frames = max(1, int(overlay_hold_frames))

        self.top_avg: Optional[np.ndarray] = None
        self.bottom_avg: Optional[np.ndarray] = None
        self.frame_count = 0
        self.top_overlay: Optional[tuple[int, int]] = None
        self.bottom_overlay: Optional[tuple[int, int]] = None
        self.top_hold = 0
        self.bottom_hold = 0
        self.exclusion_mask = np.zeros((self.src_h, self.src_w), dtype=np.uint8)

    def _row_edge_density(self, gray_patch: np.ndarray) -> np.ndarray:
        return cv2.Canny(gray_patch, 60, 180).mean(axis=1) / 255.0

    def _row_sat_mean(self, bgr_patch: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)[:, :, 1].mean(axis=1) / 255.0

    def _row_green_ratio(self, bgr_patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([30, 30, 25], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))
        return (green > 0).mean(axis=1)

    def _detect_overlay_range(self, current_bgr: np.ndarray, avg_bgr: np.ndarray, *, top: bool) -> Optional[tuple[int, int]]:
        curr_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        avg_gray = cv2.cvtColor(np.clip(avg_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff = np.abs(curr_gray.astype(np.float32) - avg_gray.astype(np.float32))
        row_diff = diff.mean(axis=1)
        row_edges = self._row_edge_density(curr_gray)
        row_sat = self._row_sat_mean(current_bgr)
        row_green = self._row_green_ratio(current_bgr)

        stable = row_diff < self.stable_diff_threshold
        texty = row_edges > self.row_text_threshold
        colorful = row_sat > 0.10
        not_field = row_green < 0.28
        active = stable & texty & (colorful | not_field) if top else stable & texty & colorful & not_field

        if active.any():
            kernel = np.ones(5, dtype=np.uint8)
            active_u8 = np.convolve(active.astype(np.uint8), kernel, mode='same') > 1
        else:
            active_u8 = active

        idx = np.where(active_u8)[0]
        if idx.size == 0:
            return None

        runs, start, prev = [], idx[0], idx[0]
        for v in idx[1:]:
            if v == prev + 1:
                prev = v
            else:
                runs.append((start, prev + 1))
                start, prev = v, v
        runs.append((start, prev + 1))
        runs.sort(key=lambda x: (x[1] - x[0]), reverse=True)

        y0, y1 = runs[0]
        band_h = y1 - y0

        if top:
            if y0 > int(0.06 * current_bgr.shape[0]) or band_h < 12:
                return None
            y0, y1 = max(0, y0 - 6), min(current_bgr.shape[0], y1 + 6)
            if (y1 - y0) > int(0.16 * self.src_h):
                y1 = y0 + int(0.16 * self.src_h)
            return (y0, y1)

        if band_h < 10 or band_h > int(0.065 * self.src_h) or y1 < int(0.50 * current_bgr.shape[0]):
            return None
        return (max(0, y0 - 4), min(current_bgr.shape[0], y1 + 4))

    def update(self, frame: np.ndarray) -> None:
        self.frame_count += 1
        top_patch = frame[:self.top_scan_h].astype(np.float32)
        bot_patch = frame[self.src_h - self.bottom_scan_h:].astype(np.float32)

        if self.top_avg is None:
            self.top_avg, self.bottom_avg = top_patch.copy(), bot_patch.copy()
            return

        alpha = 0.60 if self.frame_count <= self.warmup_frames else 0.93
        self.top_avg = alpha * self.top_avg + (1.0 - alpha) * top_patch
        self.bottom_avg = alpha * self.bottom_avg + (1.0 - alpha) * bot_patch

        if self.frame_count >= self.warmup_frames:
            top_range = self._detect_overlay_range(frame[:self.top_scan_h], self.top_avg, top=True)
            bot_local = self._detect_overlay_range(frame[self.src_h - self.bottom_scan_h:], self.bottom_avg, top=False)
            bot_range = (self.src_h - self.bottom_scan_h + bot_local[0], self.src_h - self.bottom_scan_h + bot_local[1]) if bot_local else None

            if top_range:
                self.top_overlay, self.top_hold = top_range, self.overlay_hold_frames
            elif self.top_hold > 0:
                self.top_hold -= 1
            else:
                self.top_overlay = None

            if bot_range:
                self.bottom_overlay, self.bottom_hold = bot_range, self.overlay_hold_frames
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
        top_y = (self.top_overlay[1] if self.top_overlay else 0) + 6
        bot_y = (self.bottom_overlay[0] if self.bottom_overlay else self.src_h) - 6
        if bot_y <= top_y + 32:
            return 0, self.src_h
        return min(self.src_h - 32, top_y), max(32, bot_y)

    def extract_top_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.top_overlay:
            return None
        strip = frame[self.top_overlay[0]:self.top_overlay[1]]
        return strip.copy() if strip.size else None

    def extract_bottom_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.bottom_overlay:
            return None
        strip = frame[self.bottom_overlay[0]:self.bottom_overlay[1]]
        return strip.copy() if strip.size else None


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
        is_cut = False
        if self.prev_hist is not None and self.cooldown <= 0:
            corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
            mean_pixel_diff = float(np.mean(cv2.absdiff(gray, self.prev_gray))) if self.prev_gray is not None else 0.0
            if (1.0 - corr) > self.hist_diff_thresh and mean_pixel_diff > self.pixel_diff_thresh:
                is_cut, self.cooldown = True, self.cooldown_frames
        self.prev_hist, self.prev_gray = hist, gray.copy()
        return is_cut


class BallTracker:
    def __init__(self, src_w: int, src_h: int, sport_profile: str = "auto"):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.sport = (sport_profile or "auto").strip().lower()
        if self.sport not in {"basketball", "cricket", "soccer"}:
            self.sport = "generic"

        self.cx, self.cy = src_w / 2.0, src_h / 2.0
        self.radius, self.conf, self.vx, self.vy = 0.0, 0.0, 0.0, 0.0
        self.missing_count, self.max_missing = 0, 12
        min_dim = min(src_w, src_h)
        self.min_r = max(3, int(round(min_dim * 0.006)))
        self.max_r = max(self.min_r + 4, int(round(min_dim * 0.038)))
        self.gate_radius = max(self.src_w * 0.18, 40.0)
        self._hough_param2 = {"basketball": 32, "cricket": 32, "soccer": 26}.get(self.sport, 24)
        self._first_detection = True
        self._frame_idx = 0
        self._last_source = "none"
        self._yolo_enabled = YOLOBallDetector.available()

    def _build_field_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        if self.sport not in {"soccer", "cricket"}:
            return None
        mask = cv2.inRange(hsv, np.array([30, 30, 30], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
        kernel = np.ones((7, 7), np.uint8)
        return cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2), kernel, iterations=3)

    def _candidates_hough(self, gray: np.ndarray) -> list[tuple[float, float, float]]:
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (7, 7), 1.5), cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(16, int(min(self.src_w, self.src_h) * 0.035)),
            param1=110, param2=self._hough_param2, minRadius=self.min_r, maxRadius=self.max_r,
        )
        return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0][:25]] if circles is not None else []

    def _candidates_contour(self, gray: np.ndarray, field_mask: Optional[np.ndarray]) -> list[tuple[float, float, float]]:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 150)
        if field_mask is not None:
            edges = cv2.bitwise_and(edges, field_mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        results = []
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
            if peri > 0 and 0.55 <= (4.0 * math.pi * area / (peri * peri)):
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if self.min_r <= r <= self.max_r * 1.3:
                    results.append((float(cx), float(cy), float(r)))
        results.sort(key=lambda t: abs(t[2] - (self.min_r + self.max_r) / 2.0))
        return results[:20]

    def _candidates_color_blob(self, hsv: np.ndarray, field_mask: Optional[np.ndarray]) -> list[tuple[float, float, float]]:
        masks = []
        if self.sport == "basketball":
            masks.append(cv2.inRange(hsv, np.array([3, 80, 80]), np.array([22, 255, 255])))
        elif self.sport == "cricket":
            masks.extend([
                cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([165, 100, 60]), np.array([179, 255, 255])),
                cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 45, 255]))
            ])
        elif self.sport == "soccer":
            masks.append(cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 60, 255])))
        else:
            masks.append(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 55, 255])))

        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        if field_mask is not None and self.sport in {"soccer", "cricket"}:
            combined = cv2.bitwise_and(combined, field_mask)

        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1), kernel, iterations=1)
        results = []
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area, peri = cv2.contourArea(c), cv2.arcLength(c, True)
            if peri > 0 and 0.40 <= (4.0 * math.pi * area / (peri * peri)):
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if self.min_r * 0.7 <= r <= self.max_r * 1.5:
                    results.append((float(cx), float(cy), float(r)))
        results.sort(key=lambda t: abs(t[2] - (self.min_r + self.max_r) / 2.0))
        return results[:15]

    def _score_candidate(self, cx: float, cy: float, radius: float, hsv: np.ndarray, motion_mask: Optional[np.ndarray], source: str) -> float:
        r_int = max(3, int(radius))
        patch = hsv[
            max(0, int(cy - r_int)):min(self.src_h, int(cy + r_int + 1)),
            max(0, int(cx - r_int)):min(self.src_w, int(cx + r_int + 1))
        ]
        color_score = 0.0
        if patch.size > 0:
            mh, ms, mv = patch.reshape(-1, 3).mean(axis=0)
            if self.sport == "basketball":
                color_score = 1.0 if (5 <= mh <= 22 and ms >= 80 and mv >= 70) else (0.6 if (3 <= mh <= 28 and ms >= 50 and mv >= 50) else 0.0)
            elif self.sport == "cricket":
                color_score = max(1.0 if (ms <= 45 and mv >= 170) else 0.0, 1.0 if ((mh <= 10 or mh >= 165) and ms >= 90 and mv >= 50) else 0.0)
            elif self.sport == "soccer":
                color_score = 0.9 if (ms <= 55 and mv >= 160) else (0.5 if (ms <= 80 and mv >= 130) else 0.0)
            else:
                color_score = 0.3 if mv >= 150 else 0.0

        motion_score = 0.0
        if motion_mask is not None:
            mr = max(5, int(radius * 2.0))
            mp = motion_mask[max(0, int(cy - mr)):min(self.src_h, int(cy + mr + 1)), max(0, int(cx - mr)):min(self.src_w, int(cx + mr + 1))]
            if mp.size > 0:
                motion_score = float(np.count_nonzero(mp)) / float(mp.size)

        dist = math.hypot(cx - (self.cx + self.vx), cy - (self.cy + self.vy))
        proximity_score = max(0.0, 1.0 - dist / self.gate_radius)
        size_score = max(0.0, 1.0 - (abs(radius - (self.min_r + self.max_r) / 2.0) / max((self.min_r + self.max_r) / 2.0, 1.0)))
        source_bonus = 0.15 if source == "multi" else 0.0

        weights = {"cricket": (0.25, 0.25, 0.25, 0.15), "basketball": (0.35, 0.20, 0.22, 0.13), "soccer": (0.22, 0.28, 0.28, 0.12)}.get(self.sport, (0.20, 0.30, 0.30, 0.10))
        return float(weights[0] * color_score + weights[1] * motion_score + weights[2] * proximity_score + weights[3] * size_score + 0.10 * source_bonus)

    def update(self, frame: np.ndarray, gray: np.ndarray, motion_mask: Optional[np.ndarray], exclusion_mask: Optional[np.ndarray] = None) -> Optional[tuple[float, float, float, float]]:
        self._frame_idx += 1
        if self._yolo_enabled and (self._frame_idx % YOLO_BALL_DETECT_EVERY_N == 1 or self.conf < 0.22 or self._first_detection):
            yolo_hit = YOLOBallDetector.detect_ball(frame)
            if yolo_hit is not None:
                cx, cy, radius, conf = yolo_hit
                self.vx = (cx - self.cx) * 0.35 + self.vx * 0.65
                self.vy = (cy - self.cy) * 0.35 + self.vy * 0.65
                self.cx, self.cy = cx, cy
                self.radius = float(_clamp(radius, self.min_r, self.max_r * 1.5))
                self.conf = max(self.conf * 0.70, conf)
                self.missing_count = 0
                self._first_detection = False
                self._last_source = "yolo"
                return (self.cx, self.cy, self.radius, self.conf)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        det_gray = gray.copy()
        if exclusion_mask is not None:
            det_gray[exclusion_mask > 0] = 0

        field_mask = self._build_field_mask(hsv)
        raw_candidates = []
        for cx, cy, r in self._candidates_hough(det_gray): raw_candidates.append((cx, cy, r, "hough"))
        for cx, cy, r in self._candidates_contour(det_gray, field_mask): raw_candidates.append((cx, cy, r, "contour"))
        for cx, cy, r in self._candidates_color_blob(hsv, field_mask): raw_candidates.append((cx, cy, r, "color"))

        if exclusion_mask is not None:
            raw_candidates = [(cx, cy, r, src) for cx, cy, r, src in raw_candidates if not (0 <= int(round(cy)) < self.src_h and 0 <= int(round(cx)) < self.src_w and exclusion_mask[int(round(cy)), int(round(cx))] > 0)]

        clusters, used = [], [False] * len(raw_candidates)
        merge_dist = max(self.max_r * 2.5, 20.0)
        for i, (cx1, cy1, r1, s1) in enumerate(raw_candidates):
            if used[i]: continue
            group_cx, group_cy, group_r, sources = [cx1], [cy1], [r1], {s1}
            used[i] = True
            for j in range(i + 1, len(raw_candidates)):
                if not used[j] and math.hypot(cx1 - raw_candidates[j][0], cy1 - raw_candidates[j][1]) < merge_dist:
                    group_cx.append(raw_candidates[j][0])
                    group_cy.append(raw_candidates[j][1])
                    group_r.append(raw_candidates[j][2])
                    sources.add(raw_candidates[j][3])
                    used[j] = True
            clusters.append((sum(group_cx)/len(group_cx), sum(group_cy)/len(group_cy), sum(group_r)/len(group_r), "multi" if len(sources) > 1 else list(sources)[0]))

        best, best_score = None, -1.0
        for cx, cy, r, src in clusters:
            score = self._score_candidate(cx, cy, r, hsv, motion_mask, src)
            if score > best_score:
                best_score, best = score, (cx, cy, r, score)

        if best is None or best_score < 0.22:
            self.missing_count += 1
            self.conf *= 0.88
            if self.missing_count > self.max_missing:
                self.conf = 0.0
            return None

        cx, cy, r, score = best
        pos_alpha = {"basketball": 0.65, "cricket": 0.55, "soccer": 0.60}.get(self.sport, 0.60)
        vel_alpha = pos_alpha * 0.75
        self.vx = vel_alpha * self.vx + (1.0 - vel_alpha) * (cx - self.cx)
        self.vy = vel_alpha * self.vy + (1.0 - vel_alpha) * (cy - self.cy)
        self.cx = pos_alpha * self.cx + (1.0 - pos_alpha) * cx
        self.cy = pos_alpha * self.cy + (1.0 - pos_alpha) * cy
        self.radius = (0.6 * self.radius + 0.4 * r) if self.radius > 0 else r

        if self._first_detection:
            self.conf = max(0.40, min(1.0, 0.7 * 0.40 + 0.35 * score))
            self._first_detection = False
        else:
            self.conf = min(1.0, 0.7 * self.conf + 0.35 * score)

        self.missing_count = 0
        return (self.cx, self.cy, self.radius, self.conf)

    def reset_position(self, cx: float, cy: float) -> None:
        self.cx, self.cy, self.vx, self.vy = cx, cy, 0.0, 0.0
        self.conf *= 0.3
        self._first_detection = True


class YOLOBallDetector:
    """Optional lightweight sports-ball detector."""
    _model = None
    _loaded = False

    @classmethod
    def available(cls) -> bool:
        if _YOLO is None:
            return False
        if cls._loaded:
            return cls._model is not None
        cls._loaded = True
        for cand in YOLO_BALL_MODEL_CANDIDATES:
            if os.path.exists(cand):
                try:
                    cls._model = _YOLO(cand)
                    logger.info("Loaded optional YOLO ball detector: %s", cand)
                    return True
                except Exception:
                    continue
        return False

    @classmethod
    def detect_ball(cls, frame: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        if not cls.available() or frame is None or frame.size == 0:
            return None
        try:
            h, w = frame.shape[:2]
            det_w = min(w, 960)
            scale = det_w / float(w)
            det_h = max(2, int(round(h * scale)))
            det = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
            res = cls._model(det, verbose=False, conf=0.20)[0]
            boxes = getattr(res, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                return None
            best = None
            best_score = -1.0
            inv = 1.0 / scale
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except Exception:
                    continue
                if cls_id != 32:  # COCO sports ball
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                radius = max(bw, bh) * 0.5 * inv
                cx = (x1 + x2) * 0.5 * inv
                cy = (y1 + y2) * 0.5 * inv
                score = conf
                if score > best_score:
                    best_score = score
                    best = (float(cx), float(cy), float(radius), float(conf))
            return best
        except Exception:
            return None


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
    missing_frames: int = 0
    active: bool = True
    _pos_alpha: float = 0.90
    _size_alpha: float = 0.92

    def update(self, det_x: float, det_y: float, det_w: float, det_h: float) -> None:
        self.raw_x, self.raw_y, self.raw_w, self.raw_h = det_x, det_y, det_w, det_h
        self.sx = self._pos_alpha * self.sx + (1.0 - self._pos_alpha) * det_x
        self.sy = self._pos_alpha * self.sy + (1.0 - self._pos_alpha) * det_y
        self.sw = self._size_alpha * self.sw + (1.0 - self._size_alpha) * det_w
        self.sh = self._size_alpha * self.sh + (1.0 - self._size_alpha) * det_h
        self.missing_frames = 0
        self.active = True

    def extrapolate(self) -> None:
        self.missing_frames += 1
        if self.missing_frames > 24:
            self.active = False

    def tick_smooth(self) -> None:
        pass


def _match_faces_to_detections(tracked: list[_TrackedFace], detections: list[tuple[float, float, float, float]], max_dist: float) -> tuple[dict[int, int], list[int], list[int]]:
    matched, used_det = {}, set()
    for ti, tf in enumerate(tracked):
        best_d, best_di = max_dist, -1
        for di, (dx, dy, dw, dh) in enumerate(detections):
            if di in used_det: continue
            d = math.hypot(tf.sx - (dx + dw / 2.0), tf.sy - (dy + dh / 2.0))
            if d < best_d:
                best_d, best_di = d, di
        if best_di >= 0:
            matched[ti] = best_di
            used_det.add(best_di)
    return matched, [ti for ti in range(len(tracked)) if ti not in matched], [di for di in range(len(detections)) if di not in used_det]


@dataclass
class _PanelCell:
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int


def _compute_panel_layout(n: int, canvas_w: int, canvas_h: int, gap: int = 4) -> list[_PanelCell]:
    n = max(1, min(n, 4))
    if n == 1:
        return [_PanelCell(0, 0, canvas_w, canvas_h)]
    if n == 2:
        row_h = (canvas_h - gap) // 2
        return [_PanelCell(0, 0, canvas_w, row_h), _PanelCell(0, row_h + gap, canvas_w, canvas_h - row_h - gap)]
    if n == 3:
        top_h, col_w = (canvas_h - gap) // 2, (canvas_w - gap) // 2
        return [_PanelCell(0, 0, canvas_w, top_h), _PanelCell(0, top_h + gap, col_w, canvas_h - top_h - gap), _PanelCell(col_w + gap, top_h + gap, canvas_w - col_w - gap, canvas_h - top_h - gap)]
    row_h, col_w = (canvas_h - gap) // 2, (canvas_w - gap) // 2
    return [
        _PanelCell(0, 0, col_w, row_h),
        _PanelCell(col_w + gap, 0, canvas_w - col_w - gap, row_h),
        _PanelCell(0, row_h + gap, col_w, canvas_h - row_h - gap),
        _PanelCell(col_w + gap, row_h + gap, canvas_w - col_w - gap, canvas_h - row_h - gap)
    ]


class PanelTracker:
    def __init__(self, src_w: int, src_h: int, max_faces: int = 4, max_missing_frames: int = 24,
                 layout_hold_frames: int = 24, blend_frames: int = 18, min_face_area_ratio: float = 0.0012,
                 pos_alpha: float = 0.90, size_alpha: float = 0.92):
        self.src_w, self.src_h = src_w, src_h
        self.max_faces, self.max_missing_frames = max_faces, max_missing_frames
        self.layout_hold_frames, self.blend_frames = layout_hold_frames, blend_frames
        self.min_face_area = min_face_area_ratio * src_w * src_h
        self.pos_alpha, self.size_alpha = pos_alpha, size_alpha

        self._tracked: list[_TrackedFace] = []
        self._next_id, self._active_count, self._candidate_count, self._candidate_hold = 0, 0, 0, 0
        self._blend_remaining = 0
        self._prev_output: Optional[np.ndarray] = None
        self._match_dist = max(src_w, src_h) * 0.18
        self._canvas_buffer: Optional[np.ndarray] = None
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self.detector_backend_name = "haar"
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
        return max(1, len([tf for tf in self._tracked if tf.active]))

    @staticmethod
    def _nms_faces(faces: list[tuple[int, int, int, int]], iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
        if len(faces) <= 1: return faces
        boxes = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        keep, used = [], [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            keep.append(boxes[i])
            used[i] = True
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                x2, y2, w2, h2 = boxes[j]
                inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union > iou_thresh:
                    used[j] = True
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
                        score = float(det.score[0]) if getattr(det, "score", None) else 0.0
                        if score < 0.60: continue
                        rb = det.location_data.relative_bounding_box
                        x, y, ww, hh = int(rb.xmin * sw * inv), int(rb.ymin * sh * inv), int(rb.width * sw * inv), int(rb.height * sh * inv)
                        x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
                        ww, hh = min(ww, w - x), min(hh, h - y)
                        if ww >= 48 and hh >= 48 and 0.60 <= ww / max(hh, 1) <= 1.60:
                            faces.append((x, y, ww, hh))
                if faces:
                    self.detector_backend_name = "mediapipe"
                    return self._nms_faces(faces, 0.35)
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
        return self._nms_faces([(x, y, w, h) for x, y, w, h in faces if 0.65 <= w / max(h, 1) <= 1.50], 0.4)

    def update_detections(self, raw_faces: list[tuple[int, int, int, int]]) -> None:
        faces = [(float(x), float(y), float(w), float(h)) for x, y, w, h in raw_faces if w * h >= self.min_face_area]
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:self.max_faces]
        matched, unmatched_t, unmatched_d = _match_faces_to_detections(self._tracked, faces, self._match_dist)

        for ti, di in matched.items():
            dx, dy, dw, dh = faces[di]
            self._tracked[ti].update(dx + dw / 2.0, dy + dh / 2.0, dw, dh)

        for ti in unmatched_t:
            self._tracked[ti].extrapolate()

        for di in unmatched_d:
            dx, dy, dw, dh = faces[di]
            self._tracked.append(_TrackedFace(self._next_id, dx + dw / 2.0, dy + dh / 2.0, dw, dh, dx + dw / 2.0, dy + dh / 2.0, dw, dh, _pos_alpha=self.pos_alpha, _size_alpha=self.size_alpha))
            self._next_id += 1

        self._tracked = [tf for tf in self._tracked if tf.missing_frames <= self.max_missing_frames]
        visible = [tf for tf in self._tracked if tf.active]
        current_n = max(1, min(len(visible), self.max_faces))

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

    def _crop_person(self, frame: np.ndarray, face: _TrackedFace, cell_w: int, cell_h: int, zoom_factor: float = 1.0) -> np.ndarray:
        fh, fw = frame.shape[:2]
        cell_ar = (cell_w / max(cell_h, 1)) if cell_w > 0 and cell_h > 0 else 9.0 / 16.0
        crop_w = max(face.sw * 3.0 * zoom_factor, face.sw * 2.5)
        crop_h = crop_w / max(cell_ar, 0.01)
        crop_w, crop_h = min(crop_w, fw * 0.95), min(crop_h, fh * 0.95)

        if crop_w / max(crop_h, 1) > cell_ar:
            crop_w = crop_h * cell_ar
        else:
            crop_h = crop_w / max(cell_ar, 0.01)

        cx, cy = face.sx, face.sy - face.sh * 0.35
        x0, y0 = int(round(cx - crop_w / 2.0)), int(round(cy - crop_h * 0.42))
        x1, y1 = int(round(x0 + crop_w)), int(round(y0 + crop_h))

        if x0 < 0: x1 -= x0; x0 = 0
        if y0 < 0: y1 -= y0; y0 = 0
        if x1 > fw: x0 -= (x1 - fw); x1 = fw
        if y1 > fh: y0 -= (y1 - fh); y1 = fh

        x0, y0 = max(0, x0), max(0, y0)
        actual_w, actual_h = x1 - x0, y1 - y0

        if actual_w > 0 and actual_h > 0:
            actual_ar = actual_w / actual_h
            if actual_ar > cell_ar * 1.02:
                trim = max(0, min(int((actual_w - actual_h * cell_ar) / 2), (actual_w // 2) - 5))
                x0 += trim; x1 -= trim
            elif actual_ar < cell_ar * 0.98:
                trim = max(0, min(int((actual_h - actual_w / cell_ar) / 2), (actual_h // 2) - 5))
                y0 += trim; y1 -= trim

        x0, y0 = max(0, min(x0, fw - 1)), max(0, min(y0, fh - 1))
        x1, y1 = max(x0 + 1, min(x1, fw)), max(y0 + 1, min(y1, fh))

        if x1 <= x0 or y1 <= y0:
            c_w, c_h = int(fw * 0.45), int(fw * 0.45 / max(cell_ar, 0.01))
            c_h = min(c_h, fh)
            c_w = int(c_h * cell_ar)
            c_x0 = min(max(0, int(cx - c_w / 2)), max(0, fw - c_w))
            c_y0 = min(max(0, int(cy - c_h / 2)), max(0, fh - c_h))
            return frame[c_y0:c_y0 + c_h, c_x0:c_x0 + c_w]
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _draw_dividers(canvas: np.ndarray, cells: list[_PanelCell], gap: int) -> None:
        if gap < 2: return
        h, w = canvas.shape[:2]
        xs, ys = {c.dst_x for c in cells if c.dst_x > 0}, {c.dst_y for c in cells if c.dst_y > 0}
        for x in xs:
            canvas[:, max(0, x - gap // 2):min(w, x + gap // 2)] = (8, 8, 8)
        for y in ys:
            canvas[max(0, y - gap // 2):min(h, y + gap // 2), :] = (8, 8, 8)

    def render(self, source_frame: np.ndarray, canvas_w: int, canvas_h: int, gap: int = 4) -> np.ndarray:
        if self._active_count == 0:
            self._active_count = 1
        all_active = sorted([tf for tf in self._tracked if tf.active], key=lambda tf: tf.sx)

        if not all_active:
            fallback = _resize_cover(source_frame, canvas_w, canvas_h)
            if self._blend_remaining > 0 and self._prev_output is not None and self._prev_output.shape[:2] == fallback.shape[:2]:
                alpha = self._blend_remaining / max(self.blend_frames, 1)
                fallback = cv2.addWeighted(self._prev_output, alpha, fallback, 1.0 - alpha, 0)
                self._blend_remaining -= 1
            self._prev_output = fallback.copy()
            return fallback.copy()

        n = min(self._active_count, max(1, len(all_active)))
        cells = _compute_panel_layout(n, canvas_w, canvas_h, gap)
        active_faces = all_active[:n]

        if self._canvas_buffer is None or self._canvas_buffer.shape != (canvas_h, canvas_w, 3):
            self._canvas_buffer = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        else:
            self._canvas_buffer[:] = 0

        canvas = self._canvas_buffer
        avg_size = sum(f.sw * f.sh for f in active_faces) / max(len(active_faces), 1)

        for idx, cell in enumerate(cells):
            if idx < len(active_faces):
                face = active_faces[idx]
                zoom = _clamp(1.0 / max(math.sqrt((face.sw * face.sh) / max(avg_size, 1.0)), 0.5), 0.7, 1.4)
                crop = self._crop_person(source_frame, face, cell.dst_w, cell.dst_h, zoom)
                canvas[cell.dst_y:cell.dst_y + cell.dst_h, cell.dst_x:cell.dst_x + cell.dst_w] = _resize_cover(crop, cell.dst_w, cell.dst_h)
            else:
                canvas[cell.dst_y:cell.dst_y + cell.dst_h, cell.dst_x:cell.dst_x + cell.dst_w] = 0

        self._draw_dividers(canvas, cells, gap)
        canvas_copy = canvas.copy()

        if self._blend_remaining > 0 and self._prev_output is not None and self._prev_output.shape[:2] == canvas_copy.shape[:2]:
            alpha = self._blend_remaining / max(self.blend_frames, 1)
            output = cv2.addWeighted(self._prev_output, alpha, canvas_copy, 1.0 - alpha, 0)
            self._blend_remaining -= 1
        else:
            output = canvas_copy

        self._prev_output = output.copy()
        return output.copy()


class AutoModeDetector:
    def __init__(self, probe_frames: int = 45, min_faces: int = 2, panel_ratio: float = 0.55):
        self.probe_frames, self.min_faces, self.panel_ratio = max(8, probe_frames), min_faces, panel_ratio
        self._face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._panel_hits, self._fed = 0, 0

    def feed(self, frame: np.ndarray) -> None:
        if self._fed >= self.probe_frames: return
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
    def __init__(
        self, src_w: int, src_h: int, target_w: int, target_h: int, smooth_strength: float = 0.975, analysis_stride: int = 4,
        deadzone_ratio: float = 0.05, max_pan_ratio: float = 0.012, sport_profile: str = "auto", ball_tracking: bool = True,
        ball_weight: float = 0.55, context_bias: float = 0.20, overlay_composite: bool = True, preserve_bottom_overlay: bool = False,
        panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4,
        panel_max_missing_frames: int = 24, panel_layout_hold_frames: int = 24, panel_blend_frames: int = 18,
        panel_min_face_area_ratio: float = 0.0012, panel_pos_alpha: float = 0.90, panel_size_alpha: float = 0.92,
        overlay_stride: int = 2, auto_mode: bool = False,
    ):
        self.src_w, self.src_h, self.target_w, self.target_h = int(src_w), int(src_h), int(target_w), int(target_h)
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.max_x, self.max_y = max(0, src_w - self.crop_w), max(0, src_h - self.crop_h)
        self.smooth_strength, self.analysis_stride = float(smooth_strength), max(1, int(analysis_stride))
        self.deadzone_px, self.max_pan_px = max(8.0, self.crop_w * deadzone_ratio), max(2.0, self.crop_w * max_pan_ratio)
        self.ball_weight, self.context_bias = ball_weight, context_bias
        self.overlay_composite, self.preserve_bottom_overlay = overlay_composite, preserve_bottom_overlay
        self.sport_profile = (sport_profile or "auto").strip().lower()

        self.panel_mode = bool(panel_mode)
        self.panel_gap, self.panel_detection_stride = int(panel_gap), max(1, int(panel_detection_stride))
        self.overlay_stride = max(1, int(overlay_stride))
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.saliency = None
        try:
            if hasattr(cv2, "saliency"):
                self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            pass

        self.overlay_detector = OverlayDetector(src_w, src_h)
        self.scene_detector = SceneChangeDetector()
        self.ball_tracker = None if self.panel_mode or not ball_tracking else BallTracker(src_w, src_h, sport_profile=self.sport_profile)
        self.panel_tracker = None
        if self.panel_mode:
            self.panel_tracker = PanelTracker(src_w, src_h, panel_max_faces, panel_max_missing_frames, panel_layout_hold_frames, panel_blend_frames, panel_min_face_area_ratio, panel_pos_alpha, panel_size_alpha)

        self._auto_detector = AutoModeDetector() if auto_mode else None
        self._auto_decided = False
        self._init_panel = dict(
            panel_max_faces=panel_max_faces, panel_max_missing_frames=panel_max_missing_frames,
            panel_layout_hold_frames=panel_layout_hold_frames, panel_blend_frames=panel_blend_frames,
            panel_min_face_area_ratio=panel_min_face_area_ratio, panel_pos_alpha=panel_pos_alpha, panel_size_alpha=panel_size_alpha
        )

        self.smoothed_cx, self.smoothed_cy = src_w / 2.0, src_h / 2.0
        self.target_cx, self.target_cy = self.smoothed_cx, self.smoothed_cy
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0
        self._panel_no_face_count = 0

    def _detect_motion(self, gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return [], None
        diff = cv2.GaussianBlur(cv2.absdiff(gray, self.prev_gray), (9, 9), 0)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)
        if self.overlay_detector.exclusion_mask is not None:
            motion[self.overlay_detector.exclusion_mask > 0] = 0
        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 0.006 * self.src_w * self.src_h:
                boxes.append((x, y, w, h))
        return boxes, motion

    def _compose_output(self, play_crop: np.ndarray, top_strip: Optional[np.ndarray], bottom_strip: Optional[np.ndarray]) -> np.ndarray:
        top_h, bottom_h = _overlay_heights(self.target_h, top_strip, bottom_strip, self.src_h, self.overlay_composite, self.preserve_bottom_overlay)
        mid_h = max(1, self.target_h - top_h - bottom_h)
        output = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        output[top_h:top_h + mid_h] = _resize_cover(play_crop, self.target_w, mid_h)
        if top_h > 0 and top_strip is not None and top_strip.size:
            output[:top_h] = cv2.resize(top_strip, (self.target_w, top_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, top_h - 1), (self.target_w - 1, top_h - 1), (10, 10, 10), 1)
        if bottom_h > 0 and bottom_strip is not None and bottom_strip.size:
            output[self.target_h - bottom_h:] = cv2.resize(bottom_strip, (self.target_w, bottom_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, self.target_h - bottom_h), (self.target_w - 1, self.target_h - bottom_h), (10, 10, 10), 1)
        return output

    def _process_panel(self, frame: np.ndarray) -> np.ndarray:
        assert self.panel_tracker is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        play_top, play_bot = self.overlay_detector.get_play_area_bounds()

        if self.frame_idx % self.panel_detection_stride == 0:
            try:
                raw_faces = self.panel_tracker.detect_faces(frame, gray)
                top_ol, bot_ol = self.overlay_detector.top_overlay, self.overlay_detector.bottom_overlay
                adjusted = [(x, y, w, h) for x, y, w, h in raw_faces if not ((top_ol and (y + h / 2.0) < top_ol[1]) or (bot_ol and (y + h / 2.0) > bot_ol[0]))]
                self.panel_tracker.update_detections(adjusted)
            except Exception:
                self.panel_tracker.tick_extrapolation()
        else:
            self.panel_tracker.tick_extrapolation()

        active_faces = [tf for tf in self.panel_tracker._tracked if tf.active]
        self._panel_no_face_count = 0 if active_faces else self._panel_no_face_count + 1
        if self._panel_no_face_count > 90:
            return self._process_single(frame)

        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        top_h, bottom_h = _overlay_heights(self.target_h, top_strip, bottom_strip, self.src_h, self.overlay_composite, self.preserve_bottom_overlay)
        panel_canvas_h = max(1, self.target_h - top_h - bottom_h)
        panel_frame = self.panel_tracker.render(frame, self.target_w, panel_canvas_h, self.panel_gap)
        return self._compose_output(panel_frame, top_strip, bottom_strip)

    def _process_single(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        is_cut = self.scene_detector.check(gray)
        if is_cut:
            self.smoothed_cx = (self.smoothed_cx + self.src_w / 2.0) / 2.0
            self.smoothed_cy = (self.smoothed_cy + self.src_h / 2.0) / 2.0
            if self.ball_tracker:
                self.ball_tracker.reset_position(self.src_w / 2.0, self.src_h / 2.0)

        if self.frame_idx % self.analysis_stride == 0:
            candidates = []
            play_top, play_bot = self.overlay_detector.get_play_area_bounds()
            try:
                faces = self.face_detector.detectMultiScale(gray[play_top:play_bot], scaleFactor=1.15, minNeighbors=4, minSize=(32, 32))
            except Exception:
                faces = []
            for x, y, w, h in faces[:3]:
                candidates.append((0.30, (x + w / 2.0, play_top + y + h / 2.0)))

            motion_boxes, motion_mask = self._detect_motion(gray)
            if motion_boxes:
                x0, y0 = min(b[0] for b in motion_boxes), min(b[1] for b in motion_boxes)
                x1, y1 = max(b[0] + b[2] for b in motion_boxes), max(b[1] + b[3] for b in motion_boxes)
                candidates.append((0.20 if self.ball_tracker else 0.38, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))
            else:
                motion_mask = None

            if self.saliency is not None:
                try:
                    success, sal_map = self.saliency.computeSaliency(frame)
                    if success:
                        sal_map = (sal_map * 255).astype("uint8")
                        if self.overlay_detector.exclusion_mask is not None:
                            sal_map[self.overlay_detector.exclusion_mask > 0] = 0
                        _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                            if w * h > 0.02 * self.src_w * self.src_h:
                                candidates.append((0.08, (x + w / 2.0, y + h / 2.0)))
                except Exception:
                    pass

            if self.ball_tracker:
                ball = self.ball_tracker.update(frame, gray, motion_mask, self.overlay_detector.exclusion_mask)
                if ball:
                    bx, by, _, conf = ball
                    candidates.append((self.ball_weight * max(0.25, conf), (bx, by)))
                    if motion_boxes:
                        mx0, my0 = min(b[0] for b in motion_boxes), min(b[1] for b in motion_boxes)
                        mx1, my1 = max(b[0] + b[2] for b in motion_boxes), max(b[1] + b[3] for b in motion_boxes)
                        candidates.append((self.context_bias, ((mx0 + mx1) / 2.0, (my0 + my1) / 2.0)))

            if candidates:
                sw = sum(w for w, _ in candidates)
                self.target_cx = sum(cx * w for w, (cx, _) in candidates) / max(sw, 1e-6)
                self.target_cy = sum(cy * w for w, (_, cy) in candidates) / max(sw, 1e-6)
            else:
                self.target_cx, self.target_cy = self.src_w / 2.0, self.src_h / 2.0

            self.target_cy = _clamp(self.target_cy, play_top + 12, play_bot - 12)

        self.prev_gray = gray
        dx, dy = self.target_cx - self.smoothed_cx, self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px: dx = 0.0
        if abs(dy) < self.deadzone_px * 0.45: dy = 0.0

        alpha = (1 - self.smooth_strength) * (3 if is_cut else 1)
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-self.max_pan_px * 0.45, min(self.max_pan_px * 0.45, dy * alpha))

        play_top, play_bot = self.overlay_detector.get_play_area_bounds()
        x0 = int(_clamp(round(self.smoothed_cx - self.crop_w / 2), 0, self.max_x))
        play_h = play_bot - play_top
        if play_h >= self.crop_h:
            y0 = int(_clamp(round(self.smoothed_cy - self.crop_h / 2), play_top, play_bot - self.crop_h))
        else:
            y0 = int(_clamp(round(self.smoothed_cy - self.crop_h / 2), 0, self.max_y))
            if self.overlay_detector.top_overlay and y0 < play_top: y0 = min(self.max_y, play_top)
            if self.overlay_detector.bottom_overlay and y0 + self.crop_h > play_bot: y0 = max(0, play_bot - self.crop_h)

        crop = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if crop.size == 0: crop = frame

        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        return self._compose_output(crop, top_strip, bottom_strip)

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        if self._auto_detector is not None and not self._auto_decided:
            self._auto_detector.feed(frame)
            if self._auto_detector.ready():
                self._auto_decided = True
                if self._auto_detector.result() == "panel" and self.panel_tracker is None:
                    self.panel_mode, self.ball_tracker = True, None
                    p = self._init_panel
                    self.panel_tracker = PanelTracker(self.src_w, self.src_h, p["panel_max_faces"], p["panel_max_missing_frames"], p["panel_layout_hold_frames"], p["panel_blend_frames"], p["panel_min_face_area_ratio"], p["panel_pos_alpha"], p["panel_size_alpha"])

        if self.panel_mode and self.panel_tracker is not None:
            return self._process_panel(frame)
        return self._process_single(frame)


def create_vertical_master(
    source_path: str, output_path: str, target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H,
    smooth_strength: float = 0.975, analysis_stride: int = 4, deadzone_ratio: float = 0.05, max_pan_ratio: float = 0.012,
    sport_profile: str = "auto", ball_tracking: bool = True, overlay_composite: bool = True, preserve_bottom_overlay: bool = False,
    panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4,
    auto_mode: bool = False, progress_cb: Optional[Callable[[float, str], None]] = None,
):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return False, "Could not open input source"
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or DEFAULT_OUTPUT_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_w <= 0 or src_h <= 0:
        cap.release()
        return False, "Invalid source dimensions"

    reframer = SmoothReframer(
        src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio,
        sport_profile, ball_tracking, overlay_composite=overlay_composite, preserve_bottom_overlay=preserve_bottom_overlay,
        panel_mode=panel_mode, panel_max_faces=panel_max_faces, panel_detection_stride=panel_detection_stride,
        panel_gap=panel_gap, auto_mode=auto_mode,
    )

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else DEFAULT_OUTPUT_FPS, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        return False, "Could not create output file"

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
    stats_lock: threading.Lock = field(default_factory=threading.Lock)
    error: str = ""


def _stats_snapshot(session: LiveSession) -> dict:
    with getattr(session, "stats_lock", threading.Lock()):
        return dict(getattr(session, "stats", {}) or {})


def _stats_update(session: LiveSession, values: dict) -> None:
    with getattr(session, "stats_lock", threading.Lock()):
        session.stats.update(values)


def _stats_inc(session: LiveSession, key: str, amount: int | float = 1) -> None:
    with getattr(session, "stats_lock", threading.Lock()):
        session.stats[key] = session.stats.get(key, 0) + amount


def _resource_snapshot() -> dict:
    if psutil is None:
        return {}
    try:
        proc = psutil.Process(os.getpid())
        return {
            "cpu_percent": round(proc.cpu_percent(interval=None), 1),
            "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "threads": proc.num_threads(),
        }
    except Exception:
        return {}


def cfstream_config_from_inputs(account_id: str, api_token: str, customer_code: str, prefer_low_latency: bool = False) -> CFStreamConfig:
    if not account_id: raise ValueError("Cloudflare account ID is required.")
    if not api_token: raise ValueError("Cloudflare API token is required.")
    if not customer_code: raise ValueError("Cloudflare customer code is required.")
    code = customer_code.strip().replace("customer-", "").replace(".cloudflarestream.com", "").strip("/")
    return CFStreamConfig(account_id.strip(), api_token.strip(), code, bool(prefer_low_latency))


def _cf_api_request(cfg: CFStreamConfig, method: str, path: str, payload: Optional[dict] = None):
    url = f"https://api.cloudflare.com/client/v4{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {cfg.api_token}",
        "Content-Type": "application/json",
        "User-Agent": "DualFlow-Vertical-Cloudflare"
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (_safe_json_loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        parsed = _safe_json_loads(body) if body else {"success": False, "errors": [{"message": body}]}
        return exc.code, parsed


def create_live_input(cfg: CFStreamConfig, name: str, recording_mode: str = "automatic") -> dict:
    payload = {
        "meta": {"name": name},
        "recording": {"mode": recording_mode, "timeoutSeconds": 0},
        "preferLowLatency": bool(cfg.prefer_low_latency),
        "enabled": True
    }
    status, parsed = _cf_api_request(cfg, "POST", f"/accounts/{cfg.account_id}/stream/live_inputs", payload)
    if status not in (200, 201) or not parsed.get("success"):
        raise RuntimeError(f"Create live input failed: {parsed}")
    return parsed["result"]


def disable_live_input(cfg: CFStreamConfig, uid: str) -> None:
    _cf_api_request(cfg, "PUT", f"/accounts/{cfg.account_id}/stream/live_inputs/{uid}", {"enabled": False})


def build_public_playback_urls(cfg: CFStreamConfig, uid: str):
    base = f"https://customer-{cfg.customer_code}.cloudflarestream.com/{uid}"
    return (
        f"{base}/manifest/video.m3u8" + ("?protocol=llhls" if cfg.prefer_low_latency else ""),
        f"{base}/manifest/video.mpd",
        f"{base}/iframe?autoplay=true&muted=true&controls=true&preload=metadata"
    )


def _common_output_args(fps_int: int) -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-r", str(fps_int),
        "-bf", "0",
        "-refs", "1",
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int * 2),
        "-keyint_min", str(fps_int * 2),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params",
        (
            f"nal-hrd=cbr:"
            f"force-cfr=1:"
            f"scenecut=0:"
            f"keyint={fps_int * 2}:"
            f"min-keyint={fps_int * 2}:"
            f"rc-lookahead=0:"
            f"bframes=0:"
            f"ref=1:"
            f"sync-lookahead=0"
        ),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-ac", "2",
        "-af", "aresample=async=1:first_pts=0",
        "-flvflags", "no_duration_filesize",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-flush_packets", "1",
        "-muxdelay", "0",
        "-muxpreload", "0",
    ]


def build_push_file_command(reframed_mp4: str, rtmps_url: str, stream_key: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = _normalize_output_fps(output_fps, live=False)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"] + (["-stream_loop", "-1"] if loop_input else []) + ["-re", "-i", reframed_mp4]
    cmd += (["-map", "0:v:0", "-map", "0:a:0"] if _source_has_audio(reframed_mp4) else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"])
    return cmd + _common_output_args(fps_int) + ["-f", "flv", target]


def start_vod_to_live_push(cfg: CFStreamConfig, reframed_mp4: str, asset_name: str, loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS) -> LiveSession:
    li = create_live_input(cfg, safe_token(Path(asset_name).stem))
    uid, rtmps_url, key = li["uid"], li["rtmps"]["url"], li["rtmps"]["streamKey"]
    hls, dash, iframe = build_public_playback_urls(cfg, uid)
    cmd = build_push_file_command(reframed_mp4, rtmps_url, key, loop_input, output_fps)
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True)
    return LiveSession(uid, rtmps_url, key, hls, dash, iframe, cmd, proc, log_path, status="streaming")


def build_realtime_rtmps_push_command(
    target_w: int,
    target_h: int,
    fps: float,
    rtmps_url: str,
    stream_key: str,
    source: Optional[str] = None,
    *,
    loop_audio: bool = False,
    pace_audio: bool = True,
):
    """Build the RTMPS output command for processed raw-video frames.

    Raw vertical frames are input #0 from stdin. When source audio exists, the
    original source becomes input #1 and is mapped explicitly. The output does
    not use -shortest; the worker owns lifetime by closing stdin, which avoids
    premature audio EOF stopping the live push.
    """
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = _normalize_output_fps(fps, live=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
        "-thread_queue_size", "512",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{target_w}x{target_h}", "-r", str(fps_int), "-i", "-",
    ]

    if source and _source_has_audio(source):
        cmd += _audio_source_input_args(source, fps_int, loop_audio=loop_audio, pace_audio=pace_audio)
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"]

    return cmd + _common_output_args(fps_int) + ["-f", "flv", target]


def _read_frame_timeout(proc: subprocess.Popen, nbytes: int, timeout: float = INGEST_READ_TIMEOUT) -> Optional[bytes]:
    if proc is None or proc.stdout is None:
        return None
    timeout = max(INGEST_READ_TIMEOUT_MIN, min(float(timeout or INGEST_READ_TIMEOUT), INGEST_READ_TIMEOUT_MAX))
    if os.name == "nt":
        q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=1)
        def _reader():
            with contextlib.suppress(Exception):
                q.put(proc.stdout.read(nbytes), block=False)
        threading.Thread(target=_reader, daemon=True).start()
        try:
            data = q.get(timeout=timeout)
        except queue.Empty:
            return None
        return data if data and len(data) == nbytes else None

    fd, deadline, data = proc.stdout.fileno(), time.monotonic() + timeout, bytearray()
    while len(data) < nbytes:
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        try:
            ready, _, _ = select.select([fd], [], [], min(rem, 0.05))
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
    fps_int = _normalize_output_fps(fps, live=True)
    vf = (
        f"fps={fps_int},"
        f"scale={WORKING_INPUT_W}:{WORKING_INPUT_H}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={WORKING_INPUT_W}:{WORKING_INPUT_H}:"
        f"(ow-iw)/2:(oh-ih)/2:black"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-flags", "low_delay",
    ] + _source_input_args(
        source,
        pace_input=bool(pace_input) and not is_network_source(source),
        loop_file=loop_file,
    ) + [
        "-an",  # Audio is handled directly by the output FFmpeg process
        "-vf", vf,
        "-pix_fmt", "bgr24",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-f", "rawvideo",
        "pipe:1",
    ]


def _open_ingest_process(source: str, fps: float, pace_input: bool, loop_file: bool, log_path: str) -> subprocess.Popen:
    cmd = _build_ingest_command(source, fps, pace_input, loop_file)
    log_fp = open(log_path, "a", encoding="utf-8")
    log_fp.write("\n=== INGEST CMD ===\n" + _redact_cmd(cmd) + "\n")
    log_fp.flush()
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fp, bufsize=0)


def _start_output_process(session: LiveSession) -> subprocess.Popen:
    log_fp = open(session.log_path, "a", encoding="utf-8")
    log_fp.write("\n=== PUSH CMD ===\n" + _redact_cmd(session.ffmpeg_cmd) + "\n")
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
    cv2.putText(frame, "Vertical stream", (28, max(48, h//10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return frame


def _realtime_worker(
    session: LiveSession, source: str, target_w: int, target_h: int, delay_seconds: float, smooth_strength: float,
    analysis_stride: int, deadzone_ratio: float, max_pan_ratio: float, loop_file: bool, pace_input: bool,
    sport_profile: str, ball_tracking: bool, overlay_composite: bool, preserve_bottom_overlay: bool,
    panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, auto_mode: bool = False,
    output_fps: float = DEFAULT_OUTPUT_FPS
) -> None:
    start_ts = time.monotonic()
    fps_int = _normalize_output_fps(output_fps, live=True)
    frame_interval = 1.0 / fps_int
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
    frame_bytes = src_w * src_h * 3
    delay_seconds = max(0.0, min(float(delay_seconds or 0.0), MAX_BUFFER_SECONDS))
    max_buffer_frames = max(LIVE_BACKLOG_HARD_FRAMES + 2, int(round(max(delay_seconds, 0.5) * fps_int)))

    input_lock = threading.Lock()
    output_lock = threading.Lock()
    new_frame_evt = threading.Event()
    stop = session.stop_event

    latest_output = {"seq": 0, "ts": 0.0, "frame": _make_placeholder_frame(target_w, target_h)}
    input_buffer: collections.deque[tuple[int, float, np.ndarray]] = collections.deque(maxlen=max_buffer_frames)

    counters = {k: 0 for k in ["frames_in", "frames_processed", "frames_out", "frames_repeated", "frames_dropped_input", "frames_dropped_processing", "source_stalls", "consecutive_source_stalls", "ingest_restarts", "write_failures", "output_write_failures", "frame_drops", "input_drop_count", "startup_buffer_fill_frames", "output_underruns"]}
    process_samples = collections.deque(maxlen=180)
    read_samples = collections.deque(maxlen=180)
    write_samples = collections.deque(maxlen=180)
    drift_samples = collections.deque(maxlen=180)
    win = {"t": time.monotonic(), "in": 0, "proc": 0, "out": 0}

    def p95(v):
        if not v: return 0.0
        a = sorted(v)
        return float(a[min(len(a)-1, int(round((len(a)-1)*.95)))])

    _stats_update(session, {
        "pipeline_arch": "decoupled_smooth_microbuffer_v4", "buffer_policy": "adaptive_fifo_keep_context_drop_stale",
        "health": "starting", "fps": fps_int, "fps_output": fps_int, "effective_live_buffer_seconds": round(max_buffer_frames/fps_int, 3),
        "working_resolution": f"{src_w}x{src_h}", "mode": "panel" if panel_mode else "sports" if ball_tracking else "single",
        **counters, "updated_at_ms": int(time.time() * 1000), "ingest_read_timeout": INGEST_READ_TIMEOUT
    })

    try:
        session.proc = _start_output_process(session)
        time.sleep(0.05)
        if session.proc.poll() is not None:
            session.status = "ffmpeg_start_failed"
            session.error = read_log_tail(session.log_path, 4000)
            return
        session.status = "streaming"
    except Exception as exc:
        session.status = "ffmpeg_start_failed"
        session.error = str(exc)
        return

    with contextlib.suppress(Exception):
        for _ in range(max(1, int(fps_int * LIVE_STARTUP_PRIME_SECONDS))):
            if session.proc.poll() is not None:
                session.status = "ffmpeg_pipe_broken"
                session.error = read_log_tail(session.log_path, 4000)
                return
            session.proc.stdin.write(latest_output["frame"].tobytes())
            counters["frames_out"] += 1

    ingest_holder = {"proc": None}

    def restart_ingest():
        _terminate_process(ingest_holder.get("proc"))
        counters["ingest_restarts"] += 1
        if counters["consecutive_source_stalls"] > MAX_CONSECUTIVE_STALLS_NON_LOOP and not is_network_source(source) and not loop_file:
            return None
        return _open_ingest_process(source, fps_int, pace_input, loop_file, session.log_path)

    def ingest_loop():
        seq = 0
        ingest_holder["proc"] = restart_ingest()
        first = True
        while not stop.is_set():
            proc = ingest_holder.get("proc")
            if proc is None or proc.poll() is not None:
                counters["source_stalls"] += 1
                counters["consecutive_source_stalls"] += 1
                ingest_holder["proc"] = restart_ingest()
                time.sleep(0.02)
                continue

            rs = time.monotonic()
            raw = _read_frame_timeout(proc, frame_bytes, INGEST_READ_TIMEOUT)
            read_samples.append((time.monotonic() - rs) * 1000)

            if raw is None or len(raw) != frame_bytes:
                counters["source_stalls"] += 1
                counters["consecutive_source_stalls"] += 1
                if counters["consecutive_source_stalls"] >= 3:
                    ingest_holder["proc"] = restart_ingest()
                continue

            counters["consecutive_source_stalls"] = 0
            seq += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
            ts = time.monotonic()

            with input_lock:
                if len(input_buffer) == input_buffer.maxlen:
                    counters["frames_dropped_input"] += 1
                    counters["frame_drops"] += 1
                    counters["input_drop_count"] += 1
                input_buffer.append((seq, ts, frame))
                counters["frames_in"] += 1
                win["in"] += 1

            if first:
                first = False
                _stats_update(session, {"startup_ms_to_first_source_frame": round((time.monotonic() - start_ts) * 1000, 2)})

    runtime_stride = analysis_stride

    def proc_loop():
        nonlocal runtime_stride
        reframer = SmoothReframer(
            src_w, src_h, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio,
            sport_profile, False if panel_mode else ball_tracking, overlay_composite=overlay_composite,
            preserve_bottom_overlay=preserve_bottom_overlay, panel_mode=panel_mode, panel_max_faces=panel_max_faces,
            panel_detection_stride=panel_detection_stride, panel_gap=panel_gap, auto_mode=auto_mode and not panel_mode
        )
        while not stop.is_set():
            with input_lock:
                if not input_buffer:
                    item = None
                    skipped = 0
                else:
                    backlog = len(input_buffer)
                    skipped = 0
                    if backlog > LIVE_BACKLOG_HARD_FRAMES:
                        skipped = max(0, backlog - LIVE_BACKLOG_SOFT_FRAMES)
                        while len(input_buffer) > LIVE_BACKLOG_SOFT_FRAMES:
                            input_buffer.popleft()
                    elif backlog > LIVE_BACKLOG_SOFT_FRAMES:
                        skipped = max(0, backlog - LIVE_BACKLOG_SOFT_FRAMES)
                        while len(input_buffer) > LIVE_BACKLOG_SOFT_FRAMES:
                            input_buffer.popleft()
                    item = input_buffer.popleft()

            if item is None:
                time.sleep(0.002)
                continue

            if skipped > 0:
                counters["frames_dropped_processing"] += skipped

            seq, ts, frame = item
            ps = time.monotonic()
            try:
                out = reframer.process(frame)
            except Exception as exc:
                logger.exception("[REFRAMER ERROR] %s", exc)
                out = latest_output["frame"]

            process_samples.append((time.monotonic() - ps) * 1000)
            p95_proc = p95(process_samples)
            if p95_proc > 85:
                runtime_stride = min(runtime_stride + 1, 10)
            elif p95_proc < 45:
                runtime_stride = max(runtime_stride - 1, 2)
            reframer.analysis_stride = runtime_stride

            with output_lock:
                latest_output.update(seq=latest_output["seq"] + 1, ts=time.monotonic(), frame=out)
                new_frame_evt.set()

            counters["frames_processed"] += 1
            win["proc"] += 1
            _stats_update(session, {
                "ball_confidence": round(reframer.ball_tracker.conf, 3) if reframer.ball_tracker else 0.0,
                "ball_source": getattr(reframer.ball_tracker, "_last_source", "-") if reframer.ball_tracker else "-",
                "panel_active_faces": reframer.panel_tracker.active_count if reframer.panel_tracker else 0,
                "panel_detector": getattr(reframer.panel_tracker, "detector_backend_name", "-") if reframer.panel_tracker else "-",
                "analysis_stride_runtime": runtime_stride,
                "process_queue_age_ms": round((time.monotonic() - ts) * 1000, 2),
            })

    threading.Thread(target=ingest_loop, daemon=True, name="ingest_loop").start()
    threading.Thread(target=proc_loop, daemon=True, name="processor_loop").start()

    next_deadline = time.monotonic()
    last_seq = -1
    try:
        while not stop.is_set():
            with output_lock:
                out_frame = latest_output["frame"]
                out_seq = latest_output["seq"]
                age = (time.monotonic() - latest_output["ts"]) * 1000 if latest_output["ts"] else 0

            with input_lock:
                blen = len(input_buffer)

            if out_seq == last_seq:
                counters["frames_repeated"] += 1
                counters["output_underruns"] += 1
                if age > LIVE_MAX_ACCEPTABLE_FRAME_AGE_MS:
                    out_frame = _make_placeholder_frame(target_w, target_h, "Re-syncing live feed...")
            last_seq = out_seq

            ws = time.monotonic()
            try:
                if session.proc is None or session.proc.stdin is None or session.proc.poll() is not None:
                    raise RuntimeError("Output FFmpeg process is not running")
                session.proc.stdin.write(out_frame.tobytes())
                counters["frames_out"] += 1
                win["out"] += 1
            except Exception as exc:
                counters["write_failures"] += 1
                counters["output_write_failures"] += 1
                session.status = "ffmpeg_pipe_broken"
                session.error = str(exc)
                break

            write_samples.append((time.monotonic() - ws) * 1000)
            sleep_for = next_deadline - time.monotonic()
            drift_samples.append(max(0, -sleep_for * 1000))

            if sleep_for > 0:
                new_frame_evt.wait(timeout=sleep_for)
                new_frame_evt.clear()
            elif -sleep_for > frame_interval * 3:
                next_deadline = time.monotonic()
            next_deadline += frame_interval

            now = time.monotonic()
            if now - win["t"] >= 1:
                elapsed = now - win["t"]
                fps_in = round(win["in"] / elapsed, 1)
                fps_proc = round(win["proc"] / elapsed, 1)
                fps_out = round(win["out"] / elapsed, 1)
                win["t"], win["in"], win["proc"], win["out"] = now, 0, 0, 0

                health = "healthy" if fps_out >= fps_int * 0.90 else "output_fps_low"
                _stats_update(session, {
                    "health": health, "fps_in": fps_in, "ingest_fps_1s": fps_in, "fps_process": fps_proc, "process_fps_1s": fps_proc,
                    "fps_out": fps_out, "output_fps_1s": fps_out, "fps_actual": fps_out, **counters,
                    "read_ms": round(read_samples[-1], 2) if read_samples else 0, "p95_ingest_read_ms": round(p95(read_samples), 2),
                    "processing_ms": round(process_samples[-1], 2) if process_samples else 0, "p95_process_ms": round(p95(process_samples), 2),
                    "write_ms": round(write_samples[-1], 2) if write_samples else 0, "p95_output_write_ms": round(p95(write_samples), 2),
                    "avg_schedule_drift_ms": round(sum(drift_samples) / max(len(drift_samples), 1), 2),
                    "p95_schedule_drift_ms": round(p95(drift_samples), 2), "buffer_len": blen, "buffer_seconds": round(blen / fps_int, 3),
                    "buffer_seconds_est": round(blen / fps_int, 3), "buffer_fill_pct": round(100 * blen / max(max_buffer_frames, 1), 1),
                    "latest_frame_age_ms": round(age, 2), "ffmpeg_alive": session.proc.poll() is None,
                    "ingest_alive": ingest_holder.get("proc") is not None and ingest_holder.get("proc").poll() is None,
                    "updated_at_ms": int(time.time() * 1000)
                })
    finally:
        stop.set()
        _terminate_process(ingest_holder.get("proc"))
        with contextlib.suppress(Exception):
            if session.proc and session.proc.stdin:
                session.proc.stdin.close()
        _terminate_process(session.proc)
        if session.status not in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}:
            session.status = "stopped"


def start_realtime_delayed_vertical_push(
    cfg: CFStreamConfig, source: str, asset_name: str, target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 0.0, smooth_strength: float = 0.975, analysis_stride: int = 4, deadzone_ratio: float = 0.05,
    max_pan_ratio: float = 0.012, loop_file: bool = False, pace_input: bool = True, sport_profile: str = "auto",
    ball_tracking: bool = True, overlay_composite: bool = True, preserve_bottom_overlay: bool = False,
    panel_mode: bool = False, panel_max_faces: int = 4, panel_detection_stride: int = 2, panel_gap: int = 4, auto_mode: bool = False,
    output_fps: Optional[float] = None
) -> LiveSession:
    li = create_live_input(cfg, safe_token(Path(asset_name).stem))
    uid, rtmps_url, key = li["uid"], li["rtmps"]["url"], li["rtmps"]["streamKey"]
    hls, dash, iframe = build_public_playback_urls(cfg, uid)
    
    src_meta = probe_source(source)
    selected_output_fps = _normalize_output_fps(output_fps or src_meta.get("fps") or DEFAULT_OUTPUT_FPS, live=True)
    
    # Pass source to map audio directly if available
    audio_source = source if src_meta.get("has_audio") else None
    cmd = build_realtime_rtmps_push_command(target_w, target_h, selected_output_fps, rtmps_url, key, source=audio_source, loop_audio=loop_file, pace_audio=pace_input)
    
    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    session = LiveSession(uid, rtmps_url, key, hls, dash, iframe, cmd, None, log_path)
    _stats_update(session, {
        "selected_output_fps": selected_output_fps, "source_fps": src_meta.get("fps", 0.0),
        "source_width": src_meta.get("width", 0), "source_height": src_meta.get("height", 0)
    })

    session.worker = threading.Thread(
        target=_realtime_worker,
        args=(session, source, target_w, target_h, delay_seconds, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, loop_file, pace_input, sport_profile, ball_tracking, overlay_composite, preserve_bottom_overlay, panel_mode, panel_max_faces, panel_detection_stride, panel_gap, auto_mode, selected_output_fps),
        daemon=True,
        name="decoupled_realtime_worker"
    )
    session.worker.start()
    return session


def cleanup_old_logs(directory: str = "/tmp", max_age_seconds: int = CLEANUP_LOG_MAX_AGE_SECONDS) -> int:
    removed = 0
    now = time.time()
    with contextlib.suppress(Exception):
        for name in os.listdir(directory):
            if name.endswith(".log"):
                p = os.path.join(directory, name)
                with contextlib.suppress(Exception):
                    if now - os.path.getmtime(p) > max_age_seconds:
                        os.remove(p)
                        removed += 1
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
