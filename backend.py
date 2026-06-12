from __future__ import annotations

import collections
import json
import math
import queue
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_W = 540
DEFAULT_TARGET_H = 960
MAX_UPLOAD_MB = 400
WORKING_INPUT_W = 960   # live-optimized working width
WORKING_INPUT_H = 540   # live-optimized working height
PLACEHOLDER_FPS = 30.0
DEFAULT_OUTPUT_FPS = 30.0
DEFAULT_VIDEO_BITRATE = "3500k"
DEFAULT_MAXRATE = "3500k"
DEFAULT_BUFSIZE = "3500k"  # live-safe tighter bufsize

LIVE_ANALYZE_US_NETWORK = 3000000
LIVE_ANALYZE_US_FILE = 8000000
LIVE_PROBESIZE_NETWORK = 3000000
LIVE_PROBESIZE_FILE = 8000000
LIVE_INGEST_QUEUE_SECONDS = 1.0
LIVE_PROCESS_QUEUE_SECONDS = 1.0
LIVE_JITTER_MARGIN_SECONDS = 0.50
LIVE_METRICS_WINDOW_SECONDS = 6.0
LIVE_RECONNECT_DELAY_MAX = 2
LIVE_NETWORK_RW_TIMEOUT_US = 15000000
LIVE_OUTPUT_PRESET = "superfast"
LIVE_OUTPUT_TUNE = "zerolatency"
LIVE_REPEAT_LAST_FRAME_ON_UNDERRUN = True

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
    network = is_network_source(source)
    analyze = LIVE_ANALYZE_US_NETWORK if network else LIVE_ANALYZE_US_FILE
    probesize = LIVE_PROBESIZE_NETWORK if network else LIVE_PROBESIZE_FILE
    args: list[str] = [
        "-fflags", "+genpts+discardcorrupt" + ("+nobuffer" if network else ""),
        "-analyzeduration", str(analyze),
        "-probesize", str(probesize),
    ]
    if loop_file and not network:
        args += ["-stream_loop", "-1"]
    if pace_input and not network:
        args += ["-re"]
    if network:
        args += ["-rw_timeout", str(LIVE_NETWORK_RW_TIMEOUT_US), "-flags", "low_delay"]
    ls = source.lower().strip()
    if ls.startswith(("http://", "https://")):
        args += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", str(LIVE_RECONNECT_DELAY_MAX)]
    if ls.startswith(("rtmp://", "rtmps://")):
        args += ["-rtmp_live", "live"]
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
        "-analyzeduration", str(LIVE_ANALYZE_US_NETWORK if is_network_source(source) else LIVE_ANALYZE_US_FILE),
        "-probesize", str(LIVE_PROBESIZE_NETWORK if is_network_source(source) else LIVE_PROBESIZE_FILE),
        source,
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


# ---------------------------------------------------------------------------
# Overlay / scorecard logic
# ---------------------------------------------------------------------------
class OverlayDetector:
    def __init__(
        self,
        src_w: int,
        src_h: int,
        top_scan_ratio: float = 0.18,
        bottom_scan_ratio: float = 0.14,
        warmup_frames: int = 18,
        stable_diff_threshold: float = 11.0,
        row_text_threshold: float = 0.018,
        overlay_hold_frames: int = 10,
    ):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
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
        edges = cv2.Canny(gray_patch, 60, 180)
        return edges.mean(axis=1) / 255.0

    def _row_sat_mean(self, bgr_patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].mean(axis=1) / 255.0

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

        if top:
            active = stable & texty & (colorful | not_field)
        else:
            active = stable & texty & colorful & not_field

        if active.any():
            kernel = np.ones(5, dtype=np.uint8)
            active_u8 = np.convolve(active.astype(np.uint8), kernel, mode='same') > 1
        else:
            active_u8 = active

        idx = np.where(active_u8)[0]
        if idx.size == 0:
            return None

        runs = []
        start = idx[0]
        prev = idx[0]
        for v in idx[1:]:
            if v == prev + 1:
                prev = v
            else:
                runs.append((start, prev + 1))
                start = v
                prev = v
        runs.append((start, prev + 1))
        runs.sort(key=lambda x: (x[1] - x[0]), reverse=True)
        y0, y1 = runs[0]
        band_h = y1 - y0

        if top:
            if y0 > int(0.06 * current_bgr.shape[0]):
                return None
            if band_h < 12:
                return None
            y0 = max(0, y0 - 6)
            y1 = min(current_bgr.shape[0], y1 + 6)
            if (y1 - y0) > int(0.16 * self.src_h):
                y1 = y0 + int(0.16 * self.src_h)
            return (y0, y1)

        if band_h < 10:
            return None
        if band_h > int(0.065 * self.src_h):
            return None
        if y1 < int(0.50 * current_bgr.shape[0]):
            return None
        y0 = max(0, y0 - 4)
        y1 = min(current_bgr.shape[0], y1 + 4)
        return (y0, y1)

    def update(self, frame_bgr: np.ndarray) -> None:
        self.frame_count += 1
        top_patch = frame_bgr[:self.top_scan_h].astype(np.float32)
        bot_patch = frame_bgr[self.src_h - self.bottom_scan_h:].astype(np.float32)

        if self.top_avg is None:
            self.top_avg = top_patch.copy()
            self.bottom_avg = bot_patch.copy()
            return

        alpha = 0.60 if self.frame_count <= self.warmup_frames else 0.93
        self.top_avg = alpha * self.top_avg + (1.0 - alpha) * top_patch
        self.bottom_avg = alpha * self.bottom_avg + (1.0 - alpha) * bot_patch

        if self.frame_count >= self.warmup_frames:
            top_range = self._detect_overlay_range(frame_bgr[:self.top_scan_h], self.top_avg, top=True)
            bot_local = self._detect_overlay_range(frame_bgr[self.src_h - self.bottom_scan_h:], self.bottom_avg, top=False)
            bot_range = None
            if bot_local is not None:
                bot_range = (self.src_h - self.bottom_scan_h + bot_local[0], self.src_h - self.bottom_scan_h + bot_local[1])

            if top_range is not None:
                self.top_overlay = top_range
                self.top_hold = self.overlay_hold_frames
            elif self.top_hold > 0:
                self.top_hold -= 1
            else:
                self.top_overlay = None

            if bot_range is not None:
                self.bottom_overlay = bot_range
                self.bottom_hold = self.overlay_hold_frames
            elif self.bottom_hold > 0:
                self.bottom_hold -= 1
            else:
                self.bottom_overlay = None

        self.exclusion_mask[:] = 0
        if self.top_overlay is not None:
            self.exclusion_mask[self.top_overlay[0]:self.top_overlay[1], :] = 255
        if self.bottom_overlay is not None:
            self.exclusion_mask[self.bottom_overlay[0]:self.bottom_overlay[1], :] = 255

    def get_play_area_bounds(self) -> tuple[int, int]:
        top_y = self.top_overlay[1] if self.top_overlay is not None else 0
        bot_y = self.bottom_overlay[0] if self.bottom_overlay is not None else self.src_h
        top_y = min(self.src_h - 32, top_y + 6)
        bot_y = max(32, bot_y - 6)
        if bot_y <= top_y + 32:
            top_y = 0
            bot_y = self.src_h
        return top_y, bot_y

    def extract_top_strip(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.top_overlay is None:
            return None
        y0, y1 = self.top_overlay
        strip = frame_bgr[y0:y1]
        return strip.copy() if strip.size else None

    def extract_bottom_strip(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.bottom_overlay is None:
            return None
        y0, y1 = self.bottom_overlay
        strip = frame_bgr[y0:y1]
        return strip.copy() if strip.size else None


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

    def check(self, gray: np.ndarray) -> bool:
        if self.cooldown > 0:
            self.cooldown -= 1
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        is_cut = False
        if self.prev_hist is not None and self.cooldown <= 0:
            corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
            hist_diff = 1.0 - corr
            mean_pixel_diff = 0.0
            if self.prev_gray is not None:
                mean_pixel_diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
            if hist_diff > self.hist_diff_thresh and mean_pixel_diff > self.pixel_diff_thresh:
                is_cut = True
                self.cooldown = self.cooldown_frames
        self.prev_hist = hist
        self.prev_gray = gray.copy()
        return is_cut


# ---------------------------------------------------------------------------
# Ball tracker
# ---------------------------------------------------------------------------
class BallTracker:
    def __init__(self, src_w: int, src_h: int, sport_profile: str = "auto"):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.sport = (sport_profile or "auto").strip().lower()
        if self.sport not in {"basketball", "cricket", "soccer"}:
            self.sport = "generic"
        self.cx = src_w / 2.0
        self.cy = src_h / 2.0
        self.radius = 0.0
        self.conf = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.missing_count = 0
        self.max_missing = 12
        min_dim = min(src_w, src_h)
        self.min_r = max(3, int(round(min_dim * 0.006)))
        self.max_r = max(self.min_r + 4, int(round(min_dim * 0.038)))
        self.gate_radius = max(self.src_w * 0.18, 40.0)
        self._hough_param2 = {"basketball": 32, "cricket": 32, "soccer": 26}.get(self.sport, 24)
        self._first_detection = True

    def _build_field_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        if self.sport not in {"soccer", "cricket"}:
            return None
        lower_green = np.array([30, 30, 30], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=3)
        return mask

    def _candidates_hough(self, gray: np.ndarray) -> list[tuple[float, float, float]]:
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(16, int(min(self.src_w, self.src_h) * 0.035)),
            param1=110,
            param2=self._hough_param2,
            minRadius=self.min_r, maxRadius=self.max_r,
        )
        if circles is None:
            return []
        return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0][:25]]

    def _candidates_contour(self, gray: np.ndarray, field_mask: Optional[np.ndarray]) -> list[tuple[float, float, float]]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 50, 150)
        if field_mask is not None:
            edges = cv2.bitwise_and(edges, field_mask)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: list[tuple[float, float, float]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * math.pi * area / (peri * peri)
            if circularity < 0.55:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if self.min_r <= r <= self.max_r * 1.3:
                results.append((float(cx), float(cy), float(r)))
        results.sort(key=lambda t: abs(t[2] - (self.min_r + self.max_r) / 2.0))
        return results[:20]

    def _candidates_color_blob(self, hsv: np.ndarray, field_mask: Optional[np.ndarray]) -> list[tuple[float, float, float]]:
        masks: list[np.ndarray] = []
        if self.sport == "basketball":
            masks.append(cv2.inRange(hsv, np.array([3, 80, 80]), np.array([22, 255, 255])))
        elif self.sport == "cricket":
            masks.append(cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255])))
            masks.append(cv2.inRange(hsv, np.array([165, 100, 60]), np.array([179, 255, 255])))
            masks.append(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 45, 255])))
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
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.dilate(combined, kernel, iterations=1)
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: list[tuple[float, float, float]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * math.pi * area / (peri * peri)
            if circularity < 0.40:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if self.min_r * 0.7 <= r <= self.max_r * 1.5:
                results.append((float(cx), float(cy), float(r)))
        results.sort(key=lambda t: abs(t[2] - (self.min_r + self.max_r) / 2.0))
        return results[:15]

    def _score_candidate(self, cx: float, cy: float, radius: float, hsv: np.ndarray, motion_mask: Optional[np.ndarray], source: str) -> float:
        r_int = max(3, int(radius))
        x0 = max(0, int(cx - r_int))
        y0 = max(0, int(cy - r_int))
        x1 = min(self.src_w, int(cx + r_int + 1))
        y1 = min(self.src_h, int(cy + r_int + 1))
        patch = hsv[y0:y1, x0:x1]
        color_score = 0.0
        if patch.size > 0:
            mh, ms, mv = patch.reshape(-1, 3).mean(axis=0)
            if self.sport == "basketball":
                if 5 <= mh <= 22 and ms >= 80 and mv >= 70:
                    color_score = 1.0
                elif 3 <= mh <= 28 and ms >= 50 and mv >= 50:
                    color_score = 0.6
            elif self.sport == "cricket":
                white_s = 1.0 if (ms <= 45 and mv > 170) else 0.0
                red_s = 1.0 if ((mh <= 10 or mh > 165) and ms > 90 and mv >= 50) else 0.0
                color_score = max(white_s, red_s)
            elif self.sport == "soccer":
                if ms <= 55 and mv > 160:
                    color_score = 0.9
                elif ms <= 80 and mv > 130:
                    color_score = 0.5
            else:
                color_score = 0.3 if mv > 150 else 0.0

        motion_score = 0.0
        if motion_mask is not None:
            mr = max(5, int(radius * 2.0))
            mx0 = max(0, int(cx - mr))
            my0 = max(0, int(cy - mr))
            mx1 = min(self.src_w, int(cx + mr + 1))
            my1 = min(self.src_h, int(cy + mr + 1))
            mp = motion_mask[my0:my1, mx0:mx1]
            if mp.size > 0:
                motion_score = float(np.count_nonzero(mp)) / float(mp.size)

        pred_x = self.cx + self.vx
        pred_y = self.cy + self.vy
        dist = math.hypot(cx - pred_x, cy - pred_y)
        proximity_score = max(0.0, 1.0 - dist / self.gate_radius)

        expected_r = (self.min_r + self.max_r) / 2.0
        size_dev = abs(radius - expected_r) / max(expected_r, 1.0)
        size_score = max(0.0, 1.0 - size_dev)
        source_bonus = 0.15 if source == "multi" else 0.0

        if self.sport == "cricket":
            score = 0.25 * color_score + 0.25 * motion_score + 0.25 * proximity_score + 0.15 * size_score + 0.10 * source_bonus
        elif self.sport == "basketball":
            score = 0.35 * color_score + 0.20 * motion_score + 0.22 * proximity_score + 0.13 * size_score + 0.10 * source_bonus
        elif self.sport == "soccer":
            score = 0.22 * color_score + 0.28 * motion_score + 0.28 * proximity_score + 0.12 * size_score + 0.10 * source_bonus
        else:
            score = 0.20 * color_score + 0.30 * motion_score + 0.30 * proximity_score + 0.10 * size_score + 0.10 * source_bonus
        return float(score)

    def update(self, frame: np.ndarray, gray: np.ndarray, motion_mask: Optional[np.ndarray], exclusion_mask: Optional[np.ndarray] = None) -> Optional[tuple[float, float, float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        det_gray = gray.copy()
        if exclusion_mask is not None:
            det_gray[exclusion_mask > 0] = 0
        field_mask = self._build_field_mask(hsv)

        raw_candidates: list[tuple[float, float, float, str]] = []
        for cx, cy, r in self._candidates_hough(det_gray):
            raw_candidates.append((cx, cy, r, "hough"))
        for cx, cy, r in self._candidates_contour(det_gray, field_mask):
            raw_candidates.append((cx, cy, r, "contour"))
        for cx, cy, r in self._candidates_color_blob(hsv, field_mask):
            raw_candidates.append((cx, cy, r, "color"))

        if exclusion_mask is not None:
            filtered = []
            for cx, cy, r, src in raw_candidates:
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= iy < self.src_h and 0 <= ix < self.src_w and exclusion_mask[iy, ix] == 0:
                    filtered.append((cx, cy, r, src))
            raw_candidates = filtered

        clusters: list[tuple[float, float, float, str]] = []
        used = [False] * len(raw_candidates)
        merge_dist = max(self.max_r * 2.5, 20.0)
        for i, (cx1, cy1, r1, s1) in enumerate(raw_candidates):
            if used[i]:
                continue
            group_cx = [cx1]
            group_cy = [cy1]
            group_r = [r1]
            sources = {s1}
            used[i] = True
            for j in range(i + 1, len(raw_candidates)):
                if used[j]:
                    continue
                cx2, cy2, r2, s2 = raw_candidates[j]
                if math.hypot(cx1 - cx2, cy1 - cy2) < merge_dist:
                    group_cx.append(cx2)
                    group_cy.append(cy2)
                    group_r.append(r2)
                    sources.add(s2)
                    used[j] = True
            avg_cx = sum(group_cx) / len(group_cx)
            avg_cy = sum(group_cy) / len(group_cy)
            avg_r = sum(group_r) / len(group_r)
            src_tag = "multi" if len(sources) > 1 else list(sources)[0]
            clusters.append((avg_cx, avg_cy, avg_r, src_tag))

        best: Optional[tuple[float, float, float, float]] = None
        best_score = -1.0
        for cx, cy, r, src in clusters:
            score = self._score_candidate(cx, cy, r, hsv, motion_mask, src)
            if score > best_score:
                best_score = score
                best = (cx, cy, r, score)

        if best is None or best_score < 0.15:
            self.missing_count += 1
            self.conf *= 0.88
            if self.missing_count > self.max_missing:
                self.conf = 0.0
            return None

        cx, cy, r, score = best
        new_vx = cx - self.cx
        new_vy = cy - self.cy
        pos_alpha = {"basketball": 0.65, "cricket": 0.55, "soccer": 0.60}.get(self.sport, 0.60)
        vel_alpha = pos_alpha * 0.75

        self.vx = vel_alpha * self.vx + (1.0 - vel_alpha) * new_vx
        self.vy = vel_alpha * self.vy + (1.0 - vel_alpha) * new_vy
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
        self.cx = cx
        self.cy = cy
        self.vx = 0.0
        self.vy = 0.0
        self.conf *= 0.3
        self._first_detection = True


# ---------------------------------------------------------------------------
# Panel discussion mode
# ---------------------------------------------------------------------------
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
        self.raw_x = det_x
        self.raw_y = det_y
        self.raw_w = det_w
        self.raw_h = det_h
        a_p = self._pos_alpha
        a_s = self._size_alpha
        self.sx = a_p * self.sx + (1.0 - a_p) * det_x
        self.sy = a_p * self.sy + (1.0 - a_p) * det_y
        self.sw = a_s * self.sw + (1.0 - a_s) * det_w
        self.sh = a_s * self.sh + (1.0 - a_s) * det_h
        self.missing_frames = 0
        self.active = True

    def extrapolate(self) -> None:
        self.missing_frames += 1
        if self.missing_frames > 12:
            self.active = False

    def tick_smooth(self) -> None:
        pass


def _face_centre(tf: _TrackedFace) -> tuple[float, float]:
    return tf.sx, tf.sy


def _match_faces_to_detections(tracked: list[_TrackedFace], detections: list[tuple[float, float, float, float]], max_dist: float) -> tuple[dict[int, int], list[int], list[int]]:
    matched: dict[int, int] = {}
    used_det: set[int] = set()
    for ti, tf in enumerate(tracked):
        best_d = max_dist
        best_di = -1
        tx, ty = _face_centre(tf)
        for di, (dx, dy, dw, dh) in enumerate(detections):
            if di in used_det:
                continue
            d = math.hypot(tx - (dx + dw / 2.0), ty - (dy + dh / 2.0))
            if d < best_d:
                best_d = d
                best_di = di
        if best_di >= 0:
            matched[ti] = best_di
            used_det.add(best_di)
    unmatched_tracked = [ti for ti in range(len(tracked)) if ti not in matched]
    unmatched_det = [di for di in range(len(detections)) if di not in used_det]
    return matched, unmatched_tracked, unmatched_det


@dataclass
class _PanelCell:
    dst_x: int
    dst_y: int
    dst_w: int
    dst_h: int


def _compute_panel_layout(n: int, canvas_w: int, canvas_h: int, gap: int = 4) -> list[_PanelCell]:
    n = max(1, min(n, 4))
    cells: list[_PanelCell] = []
    if n == 1:
        cells.append(_PanelCell(0, 0, canvas_w, canvas_h))
    elif n == 2:
        row_h = (canvas_h - gap) // 2
        cells.append(_PanelCell(0, 0, canvas_w, row_h))
        cells.append(_PanelCell(0, row_h + gap, canvas_w, canvas_h - row_h - gap))
    elif n == 3:
        top_h = (canvas_h - gap) // 2
        bot_h = canvas_h - top_h - gap
        col_w = (canvas_w - gap) // 2
        cells.append(_PanelCell(0, 0, canvas_w, top_h))
        cells.append(_PanelCell(0, top_h + gap, col_w, bot_h))
        cells.append(_PanelCell(col_w + gap, top_h + gap, canvas_w - col_w - gap, bot_h))
    else:
        row_h = (canvas_h - gap) // 2
        col_w = (canvas_w - gap) // 2
        cells.append(_PanelCell(0, 0, col_w, row_h))
        cells.append(_PanelCell(col_w + gap, 0, canvas_w - col_w - gap, row_h))
        cells.append(_PanelCell(0, row_h + gap, col_w, canvas_h - row_h - gap))
        cells.append(_PanelCell(col_w + gap, row_h + gap, canvas_w - col_w - gap, canvas_h - row_h - gap))
    return cells


class PanelTracker:
    def __init__(
        self,
        src_w: int,
        src_h: int,
        max_faces: int = 4,
        max_missing_frames: int = 24,
        layout_hold_frames: int = 15,
        blend_frames: int = 10,
        min_face_area_ratio: float = 0.0012,
        pos_alpha: float = 0.90,
        size_alpha: float = 0.92,
    ):
        self.src_w = src_w
        self.src_h = src_h
        self.max_faces = max_faces
        self.max_missing_frames = max_missing_frames
        self.layout_hold_frames = layout_hold_frames
        self.blend_frames = blend_frames
        self.min_face_area = min_face_area_ratio * src_w * src_h
        self.pos_alpha = pos_alpha
        self.size_alpha = size_alpha

        self._tracked: list[_TrackedFace] = []
        self._next_id = 0
        self._active_count = 0
        self._candidate_count = 0
        self._candidate_hold = 0
        self._blend_remaining = 0
        self._prev_output: Optional[np.ndarray] = None
        self._match_dist = max(src_w, src_h) * 0.18
        self._canvas_buffer: Optional[np.ndarray] = None

        self.face_detector: Optional[cv2.CascadeClassifier] = None
        self.profile_detector: Optional[cv2.CascadeClassifier] = None
        try:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            pass
        try:
            self.profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        except Exception:
            pass
        self.use_mediapipe = True
        self._mp_available = False
        self._mp_face = None
        self.detector_backend_name = "haar"
        try:
            import mediapipe as mp
            self._mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.60)
            self._mp_available = True
            self.detector_backend_name = "mediapipe"
        except Exception:
            self._mp_face = None
            self._mp_available = False
            self.detector_backend_name = "haar"

    @staticmethod
    def _nms_faces(faces: list[tuple[int, int, int, int]], iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
        if len(faces) <= 1:
            return faces
        boxes = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        keep: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            keep.append(boxes[i])
            used[i] = True
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                ix0 = max(x1, x2)
                iy0 = max(y1, y2)
                ix1 = min(x1 + w1, x2 + w2)
                iy1 = min(y1 + h1, y2 + h2)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union > iou_thresh:
                    used[j] = True
        return keep

    def detect_faces(self, frame: np.ndarray, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.use_mediapipe and self._mp_available and self._mp_face is not None:
            try:
                h, w = frame.shape[:2]
                det_scale = 1.0 if w <= 960 else 960.0 / w
                small = cv2.resize(frame, (int(round(w * det_scale)), int(round(h * det_scale))), interpolation=cv2.INTER_AREA) if det_scale < 1.0 else frame
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                res = self._mp_face.process(rgb)
                mp_faces: list[tuple[int, int, int, int]] = []
                sh, sw = small.shape[:2]
                inv = 1.0 / det_scale
                if res and res.detections:
                    for det in res.detections:
                        score = float(det.score[0]) if getattr(det, 'score', None) else 0.0
                        if score < 0.60:
                            continue
                        rb = det.location_data.relative_bounding_box
                        x = int(round(rb.xmin * sw * inv))
                        y = int(round(rb.ymin * sh * inv))
                        ww = int(round(rb.width * sw * inv))
                        hh = int(round(rb.height * sh * inv))
                        if ww <= 0 or hh <= 0:
                            continue
                        x = max(0, min(x, w - 1))
                        y = max(0, min(y, h - 1))
                        ww = min(ww, w - x)
                        hh = min(hh, h - y)
                        if ww < 48 or hh < 48:
                            continue
                        ar = ww / max(hh, 1)
                        if 0.60 <= ar <= 1.60:
                            mp_faces.append((x, y, ww, hh))
                if mp_faces:
                    self.detector_backend_name = "mediapipe"
                    return self._nms_faces(mp_faces, iou_thresh=0.35)
            except Exception:
                pass

        det_scale = 0.5
        small_gray = cv2.resize(gray, None, fx=det_scale, fy=det_scale, interpolation=cv2.INTER_AREA)
        inv = 1.0 / det_scale
        min_sz = (24, 24)
        faces: list[tuple[int, int, int, int]] = []
        if self.face_detector is not None:
            try:
                detected = self.face_detector.detectMultiScale(small_gray, scaleFactor=1.10, minNeighbors=5, minSize=min_sz)
                if len(detected):
                    faces.extend([(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in detected])
            except Exception:
                pass
        if self.profile_detector is not None:
            try:
                detected = self.profile_detector.detectMultiScale(small_gray, scaleFactor=1.10, minNeighbors=5, minSize=min_sz)
                if len(detected):
                    faces.extend([(int(x * inv), int(y * inv), int(w * inv), int(h * inv)) for (x, y, w, h) in detected])
            except Exception:
                pass
        filtered = []
        for (x, y, w, h) in faces:
            ar = w / max(h, 1)
            if 0.65 <= ar <= 1.50:
                filtered.append((x, y, w, h))
        self.detector_backend_name = "haar"
        return self._nms_faces(filtered, iou_thresh=0.4)

    def update_detections(self, raw_faces: list[tuple[int, int, int, int]]) -> None:
        faces = [(float(x), float(y), float(w), float(h)) for (x, y, w, h) in raw_faces if w * h >= self.min_face_area]
        if len(faces) > self.max_faces:
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            faces = faces[:self.max_faces]

        matched, unmatched_t, unmatched_d = _match_faces_to_detections(self._tracked, faces, self._match_dist)
        for ti, di in matched.items():
            dx, dy, dw, dh = faces[di]
            self._tracked[ti].update(dx + dw / 2.0, dy + dh / 2.0, dw, dh)
        for ti in unmatched_t:
            self._tracked[ti].extrapolate()
        for di in unmatched_d:
            dx, dy, dw, dh = faces[di]
            cx, cy = dx + dw / 2.0, dy + dh / 2.0
            tf = _TrackedFace(
                face_id=self._next_id, sx=cx, sy=cy, sw=dw, sh=dh,
                raw_x=cx, raw_y=cy, raw_w=dw, raw_h=dh,
                _pos_alpha=self.pos_alpha, _size_alpha=self.size_alpha,
            )
            self._next_id += 1
            self._tracked.append(tf)

        max_tracked = self.max_faces * 3
        if len(self._tracked) > max_tracked:
            self._tracked.sort(key=lambda tf: (tf.active, -tf.missing_frames), reverse=True)
            self._tracked = self._tracked[:self.max_faces * 2]
        self._tracked = [tf for tf in self._tracked if tf.missing_frames <= self.max_missing_frames]

        visible = [tf for tf in self._tracked if tf.active]
        current_n = max(1, min(len(visible), self.max_faces))
        if current_n != self._candidate_count:
            self._candidate_count = current_n
            self._candidate_hold = 0
        else:
            self._candidate_hold += 1
        if self._candidate_hold >= self.layout_hold_frames:
            if current_n != self._active_count:
                self._active_count = current_n
                self._blend_remaining = self.blend_frames

    def tick_extrapolation(self) -> None:
        for tf in self._tracked:
            tf.tick_smooth()

    def render(self, source_frame: np.ndarray, canvas_w: int, canvas_h: int, gap: int = 4) -> np.ndarray:
        if self._active_count == 0:
            self._active_count = 1
        all_active = sorted([tf for tf in self._tracked if tf.active], key=lambda tf: tf.sx)
        if len(all_active) == 0:
            fallback_frame = _resize_cover(source_frame, canvas_w, canvas_h)
            if self._blend_remaining > 0 and self._prev_output is not None:
                prev = self._prev_output
                if prev.shape[:2] == fallback_frame.shape[:2]:
                    alpha = self._blend_remaining / max(self.blend_frames, 1)
                    fallback_frame = cv2.addWeighted(prev, alpha, fallback_frame, 1.0 - alpha, 0)
                self._blend_remaining -= 1
            self._prev_output = fallback_frame.copy()
            return fallback_frame.copy()

        n = min(self._active_count, max(1, len(all_active)))
        cells = _compute_panel_layout(n, canvas_w, canvas_h, gap=gap)
        active_faces = all_active[:n]

        if self._canvas_buffer is None or self._canvas_buffer.shape != (canvas_h, canvas_w, 3):
            self._canvas_buffer = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        else:
            self._canvas_buffer[:] = 0
        canvas = self._canvas_buffer

        face_sizes = [f.sw * f.sh for f in active_faces]
        avg_size = sum(face_sizes) / max(len(face_sizes), 1)

        for idx, cell in enumerate(cells):
            if idx < len(active_faces):
                face = active_faces[idx]
                size_ratio = math.sqrt((face.sw * face.sh) / max(avg_size, 1.0))
                zoom_factor = _clamp(1.0 / max(size_ratio, 0.5), 0.7, 1.4)
                crop = self._crop_person(source_frame, face, cell.dst_w, cell.dst_h, zoom_factor)
                rendered = _resize_cover(crop, cell.dst_w, cell.dst_h)
                canvas[cell.dst_y: cell.dst_y + cell.dst_h, cell.dst_x: cell.dst_x + cell.dst_w] = rendered
            else:
                canvas[cell.dst_y: cell.dst_y + cell.dst_h, cell.dst_x: cell.dst_x + cell.dst_w] = 0

        self._draw_dividers(canvas, cells, gap)
        canvas_copy = canvas.copy()
        if self._blend_remaining > 0 and self._prev_output is not None:
            prev = self._prev_output
            if prev.shape[:2] == canvas_copy.shape[:2]:
                alpha = self._blend_remaining / max(self.blend_frames, 1)
                output = cv2.addWeighted(prev, alpha, canvas_copy, 1.0 - alpha, 0)
            else:
                output = canvas_copy
            self._blend_remaining -= 1
        else:
            output = canvas_copy
        self._prev_output = output.copy()
        return output.copy()

    def _crop_person(self, frame: np.ndarray, face: _TrackedFace, cell_w: int = 0, cell_h: int = 0, zoom_factor: float = 1.0) -> np.ndarray:
        fh, fw = frame.shape[:2]
        cx, cy = face.sx, face.sy
        fw_f, fh_f = face.sw, face.sh
        cell_ar = (cell_w / max(cell_h, 1)) if cell_w > 0 and cell_h > 0 else 9.0 / 16.0
        base_crop_w = fw_f * 3.0 * zoom_factor
        min_crop_w = fw_f * 2.5
        crop_w = max(base_crop_w, min_crop_w)
        crop_h = crop_w / max(cell_ar, 0.01)
        crop_w = min(crop_w, fw * 0.95)
        crop_h = min(crop_h, fh * 0.95)
        if crop_w / max(crop_h, 1) > cell_ar:
            crop_w = crop_h * cell_ar
        else:
            crop_h = crop_w / max(cell_ar, 0.01)

        cy_shifted = cy - fh_f * 0.35
        x0 = int(round(cx - crop_w / 2.0))
        y0 = int(round(cy_shifted - crop_h * 0.42))
        x1 = int(round(x0 + crop_w))
        y1 = int(round(y0 + crop_h))
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > fw:
            x0 -= (x1 - fw)
            x1 = fw
        if y1 > fh:
            y0 -= (y1 - fh)
            y1 = fh
        x0 = max(0, x0)
        y0 = max(0, y0)

        actual_w = x1 - x0
        actual_h = y1 - y0
        if actual_w > 0 and actual_h > 0:
            actual_ar = actual_w / actual_h
            if actual_ar > cell_ar * 1.02:
                trim = int((actual_w - actual_h * cell_ar) / 2)
                trim = max(0, min(trim, (actual_w // 2) - 5))
                x0 += trim
                x1 -= trim
            elif actual_ar < cell_ar * 0.98:
                trim = int((actual_h - actual_w / cell_ar) / 2)
                trim = max(0, min(trim, (actual_h // 2) - 5))
                y0 += trim
                y1 -= trim

        x0 = max(0, min(x0, fw - 1))
        y0 = max(0, min(y0, fh - 1))
        x1 = max(x0 + 1, min(x1, fw))
        y1 = max(y0 + 1, min(y1, fh))
        if x1 <= x0 or y1 <= y0:
            c_w = int(fw * 0.45)
            c_h = int(c_w / max(cell_ar, 0.01))
            c_h = min(c_h, fh)
            c_w = int(c_h * cell_ar)
            c_x0 = max(0, int(cx - c_w / 2))
            c_y0 = max(0, int(cy - c_h / 2))
            c_x0 = min(c_x0, max(0, fw - c_w))
            c_y0 = min(c_y0, max(0, fh - c_h))
            return frame[c_y0:c_y0 + c_h, c_x0:c_x0 + c_w]
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _draw_dividers(canvas: np.ndarray, cells: list[_PanelCell], gap: int) -> None:
        if gap < 2:
            return
        h, w = canvas.shape[:2]
        xs: set[int] = set()
        ys: set[int] = set()
        for c in cells:
            if c.dst_x > 0:
                xs.add(c.dst_x)
            if c.dst_y > 0:
                ys.add(c.dst_y)
        color = (8, 8, 8)
        for x in xs:
            x0 = max(0, x - gap // 2)
            x1 = min(w, x0 + gap)
            canvas[:, x0:x1] = color
        for y in ys:
            y0 = max(0, y - gap // 2)
            y1 = min(h, y0 + gap)
            canvas[y0:y1, :] = color

    @property
    def active_count(self) -> int:
        return max(1, self._active_count)

    @property
    def tracked_faces(self) -> list[_TrackedFace]:
        return list(self._tracked)


# ---------------------------------------------------------------------------
# Smooth reframer
# ---------------------------------------------------------------------------
class SmoothReframer:
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
        ball_weight: float = 0.55,
        context_bias: float = 0.20,
        overlay_composite: bool = True,
        preserve_bottom_overlay: bool = False,
        panel_mode: bool = False,
        panel_max_faces: int = 4,
        panel_detection_stride: int = 2,
        panel_gap: int = 4,
        panel_max_missing_frames: int = 24,
        panel_layout_hold_frames: int = 15,
        panel_blend_frames: int = 10,
        panel_min_face_area_ratio: float = 0.0012,
        panel_pos_alpha: float = 0.90,
        panel_size_alpha: float = 0.92,
        overlay_stride: int = 2,
        enable_saliency: bool = True,
    ):
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.crop_w, self.crop_h = _vertical_crop_box(src_w, src_h)
        self.max_x = max(0, src_w - self.crop_w)
        self.max_y = max(0, src_h - self.crop_h)
        self.smooth_strength = float(smooth_strength)
        self.analysis_stride = max(1, int(analysis_stride))
        self.deadzone_px = max(8.0, self.crop_w * deadzone_ratio)
        self.max_pan_px = max(2.0, self.crop_w * max_pan_ratio)
        self.ball_weight = float(ball_weight)
        self.context_bias = float(context_bias)
        self.overlay_composite = bool(overlay_composite)
        self.preserve_bottom_overlay = bool(preserve_bottom_overlay)
        self.sport_profile = (sport_profile or "auto").strip().lower()

        self.panel_mode = bool(panel_mode)
        self.panel_gap = int(panel_gap)
        self.panel_detection_stride = max(1, int(panel_detection_stride))
        self.overlay_stride = max(1, int(overlay_stride))

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.saliency = None
        if enable_saliency:
            try:
                if hasattr(cv2, "saliency"):
                    self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            except Exception:
                self.saliency = None

        self.overlay_detector = OverlayDetector(src_w, src_h)
        self.scene_detector = SceneChangeDetector()
        self.ball_tracker = None if self.panel_mode else (BallTracker(src_w, src_h, sport_profile=self.sport_profile) if ball_tracking else None)
        self.panel_tracker: Optional[PanelTracker] = None
        if self.panel_mode:
            self.panel_tracker = PanelTracker(
                src_w=src_w,
                src_h=src_h,
                max_faces=panel_max_faces,
                max_missing_frames=panel_max_missing_frames,
                layout_hold_frames=panel_layout_hold_frames,
                blend_frames=panel_blend_frames,
                min_face_area_ratio=panel_min_face_area_ratio,
                pos_alpha=panel_pos_alpha,
                size_alpha=panel_size_alpha,
            )

        self.smoothed_cx = src_w / 2.0
        self.smoothed_cy = src_h / 2.0
        self.target_cx = self.smoothed_cx
        self.target_cy = self.smoothed_cy
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0
        self._panel_no_face_count = 0

    def _detect_motion(self, gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.prev_gray is None:
            return [], None
        diff = cv2.absdiff(gray, self.prev_gray)
        diff = cv2.GaussianBlur(diff, (9, 9), 0)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)
        excl = self.overlay_detector.exclusion_mask
        if excl is not None:
            motion[excl > 0] = 0
        cnts, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.006 * self.src_w * self.src_h
        boxes: list[tuple[int, int, int, int]] = []
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area:
                boxes.append((x, y, w, h))
        return boxes, motion

    def _compose_output(self, play_crop: np.ndarray, top_strip: Optional[np.ndarray], bottom_strip: Optional[np.ndarray]) -> np.ndarray:
        top_h = 0
        if self.overlay_composite and top_strip is not None and top_strip.size:
            strip_h_src = top_strip.shape[0]
            top_h = int(round(self.target_h * 0.082))
            if strip_h_src < 0.05 * self.src_h:
                top_h = int(round(self.target_h * 0.075))
            top_h = int(_clamp(top_h, 42, int(self.target_h * 0.11)))
        bottom_h = 0
        if self.overlay_composite and self.preserve_bottom_overlay and bottom_strip is not None and bottom_strip.size:
            bottom_h = int(_clamp(int(round(self.target_h * 0.045)), 20, int(self.target_h * 0.07)))
        mid_h = max(1, self.target_h - top_h - bottom_h)
        output = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        play_render = _resize_cover(play_crop, self.target_w, mid_h)
        output[top_h:top_h + mid_h, :] = play_render
        if top_h > 0 and top_strip is not None and top_strip.size:
            output[:top_h, :] = cv2.resize(top_strip, (self.target_w, top_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, top_h - 1), (self.target_w - 1, top_h - 1), (10, 10, 10), 1)
        if bottom_h > 0 and bottom_strip is not None and bottom_strip.size:
            output[self.target_h - bottom_h:, :] = cv2.resize(bottom_strip, (self.target_w, bottom_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, self.target_h - bottom_h), (self.target_w - 1, self.target_h - bottom_h), (10, 10, 10), 1)
        return output

    def _process_panel(self, frame: np.ndarray) -> np.ndarray:
        assert self.panel_tracker is not None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        if self.frame_idx % self.panel_detection_stride == 0:
            try:
                raw_faces = self.panel_tracker.detect_faces(frame, gray)
                adjusted = []
                top_ol = self.overlay_detector.top_overlay
                bot_ol = self.overlay_detector.bottom_overlay
                for (x, y, w, h) in raw_faces:
                    face_cy = y + h / 2.0
                    in_top = top_ol is not None and face_cy < top_ol[1]
                    in_bot = bot_ol is not None and face_cy > bot_ol[0]
                    if not (in_top or in_bot):
                        adjusted.append((x, y, w, h))
                self.panel_tracker.update_detections(adjusted)
            except Exception:
                self.panel_tracker.tick_extrapolation()
        else:
            self.panel_tracker.tick_extrapolation()

        active_faces = [tf for tf in self.panel_tracker._tracked if tf.active]
        if len(active_faces) == 0:
            self._panel_no_face_count += 1
        else:
            self._panel_no_face_count = 0
        if self._panel_no_face_count > 90:
            return self._process_single(frame)

        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        top_h = int(_clamp(int(round(self.target_h * 0.082)), 42, int(self.target_h * 0.11))) if (self.overlay_composite and top_strip is not None and top_strip.size) else 0
        bottom_h = int(_clamp(int(round(self.target_h * 0.045)), 20, int(self.target_h * 0.07))) if (self.overlay_composite and self.preserve_bottom_overlay and bottom_strip is not None and bottom_strip.size) else 0
        panel_canvas_h = max(1, self.target_h - top_h - bottom_h)
        panel_frame = self.panel_tracker.render(source_frame=frame, canvas_w=self.target_w, canvas_h=panel_canvas_h, gap=self.panel_gap)
        output = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        output[top_h: top_h + panel_canvas_h, :] = panel_frame
        if top_h > 0 and top_strip is not None and top_strip.size:
            output[:top_h, :] = cv2.resize(top_strip, (self.target_w, top_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, top_h - 1), (self.target_w - 1, top_h - 1), (10, 10, 10), 1)
        if bottom_h > 0 and bottom_strip is not None and bottom_strip.size:
            output[self.target_h - bottom_h:, :] = cv2.resize(bottom_strip, (self.target_w, bottom_h), interpolation=cv2.INTER_AREA)
            cv2.line(output, (0, self.target_h - bottom_h), (self.target_w - 1, self.target_h - bottom_h), (10, 10, 10), 1)
        return output

    def _process_single(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.overlay_detector.update(frame)
        is_cut = self.scene_detector.check(gray)
        if is_cut:
            self.smoothed_cx = (self.smoothed_cx + self.src_w / 2.0) / 2.0
            self.smoothed_cy = (self.smoothed_cy + self.src_h / 2.0) / 2.0
            if self.ball_tracker is not None:
                self.ball_tracker.reset_position(self.src_w / 2.0, self.src_h / 2.0)

        if self.frame_idx % self.analysis_stride == 0:
            candidates: list[tuple[float, tuple[float, float]]] = []
            excl = self.overlay_detector.exclusion_mask
            play_top, play_bot = self.overlay_detector.get_play_area_bounds()
            try:
                play_gray = gray[play_top:play_bot, :]
                faces = self.face_detector.detectMultiScale(play_gray, scaleFactor=1.15, minNeighbors=4, minSize=(32, 32))
            except Exception:
                faces = []
            for (x, y, w, h) in faces[:3]:
                candidates.append((0.30, (x + w / 2.0, play_top + y + h / 2.0)))

            motion_boxes, motion_mask = self._detect_motion(gray)
            if motion_boxes:
                x0 = min(p[0] for p in motion_boxes)
                y0 = min(p[1] for p in motion_boxes)
                x1 = max(p[0] + p[2] for p in motion_boxes)
                y1 = max(p[1] + p[3] for p in motion_boxes)
                motion_w = 0.20 if self.ball_tracker is not None else 0.38
                candidates.append((motion_w, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))
            else:
                motion_mask = None

            if self.saliency is not None:
                try:
                    success, sal_map = self.saliency.computeSaliency(frame)
                    if success:
                        sal_map = (sal_map * 255).astype("uint8")
                        if excl is not None:
                            sal_map[excl > 0] = 0
                        _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            if w * h > 0.02 * self.src_w * self.src_h:
                                candidates.append((0.08, (x + w / 2.0, y + h / 2.0)))
                except Exception:
                    pass

            if self.ball_tracker is not None:
                ball = self.ball_tracker.update(frame, gray, motion_mask, excl)
                if ball is not None:
                    bx, by, _br, bconf = ball
                    candidates.append((self.ball_weight * max(0.25, bconf), (bx, by)))
                    if motion_boxes:
                        mx0 = min(p[0] for p in motion_boxes)
                        my0 = min(p[1] for p in motion_boxes)
                        mx1 = max(p[0] + p[2] for p in motion_boxes)
                        my1 = max(p[1] + p[3] for p in motion_boxes)
                        candidates.append((self.context_bias, ((mx0 + mx1) / 2.0, (my0 + my1) / 2.0)))

            if candidates:
                ws = sum(w for w, _ in candidates)
                self.target_cx = sum(cx * w for w, (cx, _) in candidates) / max(ws, 1e-6)
                self.target_cy = sum(cy * w for w, (_, cy) in candidates) / max(ws, 1e-6)
            else:
                self.target_cx = self.src_w / 2.0
                self.target_cy = self.src_h / 2.0

            y_margin = max(12, int(0.02 * self.src_h))
            self.target_cy = _clamp(self.target_cy, play_top + y_margin, play_bot - y_margin)

        self.prev_gray = gray
        dx = self.target_cx - self.smoothed_cx
        dy = self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px:
            dx = 0.0
        if abs(dy) < self.deadzone_px * 0.45:
            dy = 0.0
        alpha = (1.0 - self.smooth_strength) * (3.0 if is_cut else 1.0)
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-(self.max_pan_px * 0.45), min(self.max_pan_px * 0.45, dy * alpha))

        play_top, play_bot = self.overlay_detector.get_play_area_bounds()
        x0 = int(round(self.smoothed_cx - self.crop_w / 2.0))
        x0 = int(_clamp(x0, 0, self.max_x))
        play_h = play_bot - play_top
        if play_h >= self.crop_h:
            min_y = play_top
            max_y = play_bot - self.crop_h
            y0 = int(round(self.smoothed_cy - self.crop_h / 2.0))
            y0 = int(_clamp(y0, min_y, max_y))
        else:
            y0 = int(_clamp(round(self.smoothed_cy - self.crop_h / 2.0), 0, self.max_y))
            if self.overlay_detector.top_overlay is not None and y0 < play_top:
                y0 = min(self.max_y, play_top)
            if self.overlay_detector.bottom_overlay is not None and y0 + self.crop_h > play_bot:
                y0 = max(0, play_bot - self.crop_h)

        crop = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if crop.size == 0:
            crop = frame
        top_strip = self.overlay_detector.extract_top_strip(frame) if self.overlay_composite else None
        bottom_strip = self.overlay_detector.extract_bottom_strip(frame) if self.overlay_composite else None
        return self._compose_output(crop, top_strip, bottom_strip)

    def process(self, frame: np.ndarray) -> np.ndarray:
        self.frame_idx += 1
        return self._process_panel(frame) if (self.panel_mode and self.panel_tracker is not None) else self._process_single(frame)


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
        enable_saliency=True,
    )
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else DEFAULT_OUTPUT_FPS, (target_w, target_h))
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
        "-c:v", "libx264", "-preset", LIVE_OUTPUT_PRESET, "-tune", LIVE_OUTPUT_TUNE,
        "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr",
        "-r", str(fps_int),
        "-b:v", DEFAULT_VIDEO_BITRATE,
        "-maxrate", DEFAULT_MAXRATE,
        "-bufsize", DEFAULT_BUFSIZE,
        "-g", str(fps_int),
        "-keyint_min", str(fps_int),
        "-sc_threshold", "0",
        "-profile:v", "high",
        "-x264-params", f"nal-hrd=cbr:force-cfr=1:scenecut=0:keyint={fps_int}:min-keyint={fps_int}",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-flvflags", "no_duration_filesize",
        "-f", "flv", target,
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


def _build_ingest_command(source: str, fps: float, pace_input: bool, loop_file: bool, working_w: int = WORKING_INPUT_W, working_h: int = WORKING_INPUT_H) -> list[str]:
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    vf = (
        f"fps={fps_int},"
        f"scale={working_w}:{working_h}:force_original_aspect_ratio=decrease,"
        f"pad={working_w}:{working_h}:(ow-iw)/2:(oh-ih)/2:black"
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


def _open_ingest_process(source: str, fps: float, pace_input: bool, loop_file: bool, log_path: str, working_w: int = WORKING_INPUT_W, working_h: int = WORKING_INPUT_H) -> subprocess.Popen:
    cmd = _build_ingest_command(source, fps=fps, pace_input=pace_input, loop_file=loop_file, working_w=working_w, working_h=working_h)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== INGEST CMD ===\n" + " ".join(cmd) + "\n")
        f.flush()
    log_fp = open(log_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fp, bufsize=0)


def _start_output_process(session: LiveSession) -> subprocess.Popen:
    with open(session.log_path, "a", encoding="utf-8") as f:
        f.write("\n=== PUSH CMD ===\n" + " ".join(session.ffmpeg_cmd) + "\n")
        f.flush()
    log_fp = open(session.log_path, "a", encoding="utf-8")
    return subprocess.Popen(session.ffmpeg_cmd, stdin=subprocess.PIPE, stdout=log_fp, stderr=subprocess.STDOUT, bufsize=0)


# ---------------------------------------------------------------------------
# Live pipeline telemetry helpers
# ---------------------------------------------------------------------------
class RollingMetric:
    def __init__(self, max_seconds: float = LIVE_METRICS_WINDOW_SECONDS):
        self.max_seconds = float(max_seconds)
        self.samples: collections.deque[tuple[float, float]] = collections.deque()

    def add(self, value: float, ts: Optional[float] = None) -> None:
        now = time.monotonic() if ts is None else float(ts)
        self.samples.append((now, float(value)))
        self._trim(now)

    def _trim(self, now: Optional[float] = None) -> None:
        ref = time.monotonic() if now is None else float(now)
        cutoff = ref - self.max_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def values(self) -> list[float]:
        self._trim()
        return [v for _, v in self.samples]

    def avg(self) -> float:
        vals = self.values()
        return float(sum(vals) / len(vals)) if vals else 0.0

    def p95(self) -> float:
        vals = self.values()
        if not vals:
            return 0.0
        if len(vals) == 1:
            return float(vals[0])
        return float(np.percentile(np.asarray(vals, dtype=np.float32), 95))


@dataclass
class LiveCounters:
    frames_in: int = 0
    frames_processed: int = 0
    frames_out: int = 0
    placeholder_frames: int = 0
    source_stalls: int = 0
    output_underruns: int = 0
    input_drop_count: int = 0
    process_drop_count: int = 0


@dataclass
class LiveSharedState:
    stop_event: threading.Event
    raw_queue: queue.Queue
    processed_queue: queue.Queue
    counters: LiveCounters = field(default_factory=LiveCounters)
    ingest_read_ms: RollingMetric = field(default_factory=RollingMetric)
    process_ms: RollingMetric = field(default_factory=RollingMetric)
    output_write_ms: RollingMetric = field(default_factory=RollingMetric)
    schedule_drift_ms: RollingMetric = field(default_factory=RollingMetric)
    frames_in_stamps: collections.deque = field(default_factory=collections.deque)
    frames_processed_stamps: collections.deque = field(default_factory=collections.deque)
    frames_out_stamps: collections.deque = field(default_factory=collections.deque)
    ingest_proc: Optional[subprocess.Popen] = None
    output_proc: Optional[subprocess.Popen] = None
    ingest_eof: bool = False
    ingest_error: str = ""
    output_error: str = ""
    last_good_frame: Optional[np.ndarray] = None
    live_jitter_margin_frames: int = 0
    live_delay_frames: int = 0
    live_queue_max_frames: int = 0

    def _trim_stamps(self, dq: collections.deque, now: Optional[float] = None) -> None:
        ref = time.monotonic() if now is None else float(now)
        cutoff = ref - 1.0
        while dq and dq[0] < cutoff:
            dq.popleft()

    def mark_in(self) -> None:
        now = time.monotonic()
        self.counters.frames_in += 1
        self.frames_in_stamps.append(now)
        self._trim_stamps(self.frames_in_stamps, now)

    def mark_processed(self) -> None:
        now = time.monotonic()
        self.counters.frames_processed += 1
        self.frames_processed_stamps.append(now)
        self._trim_stamps(self.frames_processed_stamps, now)

    def mark_out(self) -> None:
        now = time.monotonic()
        self.counters.frames_out += 1
        self.frames_out_stamps.append(now)
        self._trim_stamps(self.frames_out_stamps, now)

    def fps_in_1s(self) -> float:
        self._trim_stamps(self.frames_in_stamps)
        return float(len(self.frames_in_stamps))

    def fps_process_1s(self) -> float:
        self._trim_stamps(self.frames_processed_stamps)
        return float(len(self.frames_processed_stamps))

    def fps_out_1s(self) -> float:
        self._trim_stamps(self.frames_out_stamps)
        return float(len(self.frames_out_stamps))


def _queue_put_drop_oldest(q: queue.Queue, item, counter_attr: list[int]) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                _ = q.get_nowait()
                counter_attr[0] += 1
            except queue.Empty:
                return


def _queue_get_one(q: queue.Queue):
    try:
        return q.get_nowait()
    except queue.Empty:
        return None


def _poll_alive(proc: Optional[subprocess.Popen]) -> bool:
    return bool(proc is not None and proc.poll() is None)


def _ingest_loop(shared: LiveSharedState, source: str, fps: float, loop_file: bool, pace_input: bool, log_path: str, working_w: int, working_h: int) -> None:
    frame_bytes = int(working_w * working_h * 3)
    local_drop = [0]
    try:
        shared.ingest_proc = _open_ingest_process(source, fps=fps, pace_input=pace_input, loop_file=loop_file, log_path=log_path, working_w=working_w, working_h=working_h)
        while not shared.stop_event.is_set():
            t0 = time.monotonic()
            raw = _read_exact(shared.ingest_proc.stdout, frame_bytes) if shared.ingest_proc and shared.ingest_proc.stdout else b""
            shared.ingest_read_ms.add((time.monotonic() - t0) * 1000.0)
            if len(raw) < frame_bytes:
                shared.counters.source_stalls += 1
                shared.ingest_eof = True
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((working_h, working_w, 3)).copy()
            _queue_put_drop_oldest(shared.raw_queue, frame, local_drop)
            shared.counters.input_drop_count += local_drop[0]
            local_drop[0] = 0
            shared.mark_in()
    except Exception as exc:
        shared.ingest_error = str(exc)
        shared.ingest_eof = True
    finally:
        try:
            if shared.ingest_proc and shared.ingest_proc.poll() is None:
                shared.ingest_proc.terminate()
                try:
                    shared.ingest_proc.wait(timeout=3)
                except Exception:
                    shared.ingest_proc.kill()
        except Exception:
            pass


def _process_loop(
    shared: LiveSharedState,
    target_w: int,
    target_h: int,
    smooth_strength: float,
    analysis_stride: int,
    deadzone_ratio: float,
    max_pan_ratio: float,
    sport_profile: str,
    ball_tracking: bool,
    overlay_composite: bool,
    preserve_bottom_overlay: bool,
    panel_mode: bool,
    panel_max_faces: int,
    panel_detection_stride: int,
    panel_gap: int,
    working_w: int,
    working_h: int,
) -> None:
    local_drop = [0]
    reframer = SmoothReframer(
        working_w, working_h, target_w, target_h,
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
        enable_saliency=False,  # live-safe default
    )
    try:
        while not shared.stop_event.is_set():
            if shared.ingest_eof and shared.raw_queue.empty():
                break
            try:
                frame = shared.raw_queue.get(timeout=0.20)
            except queue.Empty:
                continue
            t0 = time.monotonic()
            out = reframer.process(frame)
            shared.process_ms.add((time.monotonic() - t0) * 1000.0)
            _queue_put_drop_oldest(shared.processed_queue, out, local_drop)
            shared.counters.process_drop_count += local_drop[0]
            local_drop[0] = 0
            shared.mark_processed()
    except Exception as exc:
        shared.output_error = str(exc)


def _publish_metrics(session: LiveSession, shared: LiveSharedState, delay_frames: int, fps: float, panel_mode: bool = False, delay_buffer_len: int = 0) -> None:
    buffer_len = delay_buffer_len
    buffer_seconds = float(buffer_len / max(fps, 1.0))
    buffer_fill_pct = float(100.0 * buffer_len / max(delay_frames, 1))
    session.stats.update({
        "frames_in": shared.counters.frames_in,
        "frames_processed": shared.counters.frames_processed,
        "frames_out": shared.counters.frames_out,
        "buffer_len": buffer_len,
        "buffer_fill_pct": round(buffer_fill_pct, 1),
        "buffer_seconds": round(buffer_seconds, 2),
        "delay_frames": delay_frames,
        "delay_seconds": round(delay_frames / max(fps, 1.0), 2),
        "placeholder_frames": shared.counters.placeholder_frames,
        "source_stalls": shared.counters.source_stalls,
        "output_underruns": shared.counters.output_underruns,
        "input_drop_count": shared.counters.input_drop_count,
        "process_drop_count": shared.counters.process_drop_count,
        "ingest_fps_1s": round(shared.fps_in_1s(), 2),
        "process_fps_1s": round(shared.fps_process_1s(), 2),
        "output_fps_1s": round(shared.fps_out_1s(), 2),
        "avg_process_ms": round(shared.process_ms.avg(), 2),
        "p95_process_ms": round(shared.process_ms.p95(), 2),
        "avg_output_write_ms": round(shared.output_write_ms.avg(), 2),
        "p95_output_write_ms": round(shared.output_write_ms.p95(), 2),
        "avg_ingest_read_ms": round(shared.ingest_read_ms.avg(), 2),
        "p95_ingest_read_ms": round(shared.ingest_read_ms.p95(), 2),
        "avg_schedule_drift_ms": round(shared.schedule_drift_ms.avg(), 2),
        "p95_schedule_drift_ms": round(shared.schedule_drift_ms.p95(), 2),
        "ffmpeg_alive": _poll_alive(shared.output_proc),
        "ingest_alive": _poll_alive(shared.ingest_proc),
        "panel_mode": bool(panel_mode),
        "working_resolution": f"{WORKING_INPUT_W}x{WORKING_INPUT_H}",
        "queue_target_frames": shared.live_delay_frames,
        "queue_max_frames": shared.live_queue_max_frames,
        "queue_jitter_margin_frames": shared.live_jitter_margin_frames,
    })


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
    fps = float(DEFAULT_OUTPUT_FPS)
    working_w, working_h = WORKING_INPUT_W, WORKING_INPUT_H
    delay_frames = max(1, int(round(delay_seconds * fps)))
    jitter_margin_frames = max(6, int(round(fps * LIVE_JITTER_MARGIN_SECONDS)))
    queue_max_frames = max(delay_frames + jitter_margin_frames + 2, int(round(fps * 1.5)))
    raw_q_max = max(6, int(round(fps * LIVE_INGEST_QUEUE_SECONDS)))
    proc_q_max = max(queue_max_frames, int(round(fps * LIVE_PROCESS_QUEUE_SECONDS)))

    shared = LiveSharedState(stop_event=session.stop_event, raw_queue=queue.Queue(maxsize=raw_q_max), processed_queue=queue.Queue(maxsize=proc_q_max))
    shared.live_delay_frames = delay_frames
    shared.live_jitter_margin_frames = jitter_margin_frames
    shared.live_queue_max_frames = proc_q_max

    session.stats = {
        "fps": round(fps, 3),
        "delay_frames": delay_frames,
        "delay_seconds": round(delay_frames / max(fps, 1.0), 2),
        "working_resolution": f"{working_w}x{working_h}",
        "source_reported_resolution": f"{int(info.get('width') or 0)}x{int(info.get('height') or 0)}",
        "sport_profile": sport_profile,
        "panel_mode": panel_mode,
        "queue_max_frames": proc_q_max,
        "queue_jitter_margin_frames": jitter_margin_frames,
    }

    placeholder = _make_placeholder_frame(target_w, target_h)
    shared.last_good_frame = placeholder.copy()
    try:
        session.proc = _start_output_process(session)
        shared.output_proc = session.proc
    except Exception as exc:
        session.status = "ffmpeg_start_failed"
        session.error = str(exc)
        return

    try:
        if session.proc.stdin:
            prime_frames = max(1, min(15, int(round(fps * 0.5))))
            for _ in range(prime_frames):
                t0 = time.monotonic()
                session.proc.stdin.write(placeholder.tobytes())
                shared.output_write_ms.add((time.monotonic() - t0) * 1000.0)
                shared.counters.placeholder_frames += 1
                shared.mark_out()
    except Exception as exc:
        session.status = "ffmpeg_pipe_broken"
        session.error = f"Could not prime output: {exc}"
        return

    ingest_thread = threading.Thread(target=_ingest_loop, args=(shared, source, fps, loop_file, pace_input, session.log_path, working_w, working_h), daemon=True)
    process_thread = threading.Thread(
        target=_process_loop,
        args=(shared, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio,
              sport_profile, ball_tracking, overlay_composite, preserve_bottom_overlay, panel_mode,
              panel_max_faces, panel_detection_stride, panel_gap, working_w, working_h),
        daemon=True,
    )
    ingest_thread.start()
    process_thread.start()

    session.status = "buffering"
    frame_interval = 1.0 / fps
    next_deadline = time.monotonic()
    second_tick = time.monotonic()
    delay_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=queue_max_frames)

    try:
        while not session.stop_event.is_set():
            while True:
                item = _queue_get_one(shared.processed_queue)
                if item is None:
                    break
                delay_buffer.append(item)

            if len(delay_buffer) >= delay_frames:
                session.status = "streaming"
            elif shared.ingest_eof and len(delay_buffer) == 0 and shared.raw_queue.empty() and shared.processed_queue.empty():
                session.status = "source_ended"
                break
            else:
                session.status = "buffering"

            while len(delay_buffer) > (delay_frames + jitter_margin_frames):
                delay_buffer.popleft()
                shared.counters.process_drop_count += 1

            if session.proc is None or session.proc.stdin is None or session.proc.poll() is not None:
                session.status = "ffmpeg_pipe_broken"
                session.error = shared.output_error or "Output ffmpeg exited"
                break

            if session.status == "streaming" and len(delay_buffer) >= delay_frames:
                frame_to_write = delay_buffer.popleft()
                shared.last_good_frame = frame_to_write
            elif session.status == "buffering":
                if LIVE_REPEAT_LAST_FRAME_ON_UNDERRUN and shared.last_good_frame is not None and shared.counters.frames_out > 0:
                    frame_to_write = shared.last_good_frame
                    shared.counters.output_underruns += 1
                else:
                    frame_to_write = placeholder
                    shared.counters.placeholder_frames += 1
            else:
                frame_to_write = shared.last_good_frame if shared.last_good_frame is not None else placeholder

            now = time.monotonic()
            drift_ms = max(0.0, (now - next_deadline) * 1000.0)
            shared.schedule_drift_ms.add(drift_ms)
            if now < next_deadline:
                time.sleep(next_deadline - now)
            write_t0 = time.monotonic()
            try:
                session.proc.stdin.write(frame_to_write.tobytes())
            except Exception as exc:
                shared.output_error = str(exc)
                session.status = "ffmpeg_pipe_broken"
                session.error = str(exc)
                break
            shared.output_write_ms.add((time.monotonic() - write_t0) * 1000.0)
            shared.mark_out()
            next_deadline += frame_interval
            if next_deadline < time.monotonic() - (frame_interval * 2.0):
                next_deadline = time.monotonic() + frame_interval

            if (time.monotonic() - second_tick) >= 1.0:
                _publish_metrics(session, shared, delay_frames, fps, panel_mode=panel_mode, delay_buffer_len=len(delay_buffer))
                second_tick = time.monotonic()

        _publish_metrics(session, shared, delay_frames, fps, panel_mode=panel_mode, delay_buffer_len=len(delay_buffer))
    except Exception as exc:
        session.status = "worker_error"
        session.error = str(exc)
    finally:
        session.stop_event.set()
        try:
            if ingest_thread.is_alive():
                ingest_thread.join(timeout=2)
        except Exception:
            pass
        try:
            if process_thread.is_alive():
                process_thread.join(timeout=2)
        except Exception:
            pass
        try:
            if session.proc and session.proc.stdin:
                session.proc.stdin.close()
        except Exception:
            pass
        try:
            if shared.ingest_proc and shared.ingest_proc.poll() is None:
                shared.ingest_proc.terminate()
                try:
                    shared.ingest_proc.wait(timeout=3)
                except Exception:
                    shared.ingest_proc.kill()
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
            session, source, target_w, target_h, delay_seconds, smooth_strength,
            analysis_stride, deadzone_ratio, max_pan_ratio, loop_file, pace_input,
            sport_profile, ball_tracking, overlay_composite, preserve_bottom_overlay,
            panel_mode, panel_max_faces, panel_detection_stride, panel_gap,
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
