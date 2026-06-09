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


# ---------------------------------------------------------------------------
# Overlay / Scorecard Detector
# ---------------------------------------------------------------------------

class OverlayDetector:
    """Detects static broadcast overlays (scorecards, tickers, bug logos)
    at the top and bottom of the source frame. Generates exclusion masks
    so that overlay regions do not influence reframing decisions, and
    optionally extracts the scorecard strip for compositing into the
    vertical output."""

    def __init__(self, src_w: int, src_h: int, top_scan_ratio: float = 0.15,
                 bottom_scan_ratio: float = 0.12, stability_frames: int = 25,
                 edge_density_thresh: float = 0.025, diff_thresh: float = 12.0):
        self.src_w = src_w
        self.src_h = src_h
        self.top_scan_h = max(8, int(src_h * top_scan_ratio))
        self.bottom_scan_h = max(8, int(src_h * bottom_scan_ratio))
        self.stability_frames = max(5, stability_frames)
        self.edge_density_thresh = edge_density_thresh
        self.diff_thresh = diff_thresh

        # Running accumulators (float32 for precision)
        self.top_acc: Optional[np.ndarray] = None
        self.bottom_acc: Optional[np.ndarray] = None
        self.frame_count = 0

        # Detected stable overlay bounds (y0, y1) relative to source
        self.top_overlay: Optional[tuple[int, int]] = None      # (0, h)
        self.bottom_overlay: Optional[tuple[int, int]] = None    # (y, src_h)

        # Exclusion mask (same size as source, uint8, 255 = exclude)
        self.exclusion_mask = np.zeros((src_h, src_w), dtype=np.uint8)

    def _edge_density(self, gray_patch: np.ndarray) -> float:
        edges = cv2.Canny(gray_patch, 60, 180)
        return float(np.count_nonzero(edges)) / max(1.0, float(edges.size))

    def _has_text_like_content(self, gray_patch: np.ndarray) -> bool:
        density = self._edge_density(gray_patch)
        return density >= self.edge_density_thresh

    def update(self, gray: np.ndarray) -> None:
        h, w = gray.shape[:2]
        top_strip = gray[:self.top_scan_h, :].astype(np.float32)
        bottom_strip = gray[h - self.bottom_scan_h:, :].astype(np.float32)

        if self.top_acc is None:
            self.top_acc = top_strip.copy()
            self.bottom_acc = bottom_strip.copy()
            self.frame_count = 1
            return

        # Exponential moving average
        alpha = 0.92
        self.top_acc = alpha * self.top_acc + (1.0 - alpha) * top_strip
        self.bottom_acc = alpha * self.bottom_acc + (1.0 - alpha) * bottom_strip
        self.frame_count += 1

        if self.frame_count < self.stability_frames:
            return

        # Check top: compare current strip to running average
        top_diff = np.mean(np.abs(top_strip - self.top_acc))
        if top_diff < self.diff_thresh and self._has_text_like_content(gray[:self.top_scan_h, :]):
            self.top_overlay = (0, self.top_scan_h)
        else:
            self.top_overlay = None

        # Check bottom
        bot_diff = np.mean(np.abs(bottom_strip - self.bottom_acc))
        if bot_diff < self.diff_thresh and self._has_text_like_content(gray[h - self.bottom_scan_h:, :]):
            self.bottom_overlay = (h - self.bottom_scan_h, h)
        else:
            self.bottom_overlay = None

        # Rebuild exclusion mask
        self.exclusion_mask[:] = 0
        if self.top_overlay is not None:
            self.exclusion_mask[self.top_overlay[0]:self.top_overlay[1], :] = 255
        if self.bottom_overlay is not None:
            self.exclusion_mask[self.bottom_overlay[0]:self.bottom_overlay[1], :] = 255

    def extract_scorecard_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return the top overlay region as a BGR image strip, or None."""
        if self.top_overlay is None:
            return None
        y0, y1 = self.top_overlay
        strip = frame[y0:y1, :]
        if strip.size == 0:
            return None
        return strip.copy()

    def extract_bottom_strip(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return the bottom overlay region as a BGR image strip, or None."""
        if self.bottom_overlay is None:
            return None
        y0, y1 = self.bottom_overlay
        strip = frame[y0:y1, :]
        if strip.size == 0:
            return None
        return strip.copy()

    def get_play_area_bounds(self) -> tuple[int, int]:
        """Return (top_y, bottom_y) of the actual play area excluding overlays."""
        top_y = self.top_overlay[1] if self.top_overlay else 0
        bot_y = self.bottom_overlay[0] if self.bottom_overlay else self.src_h
        return top_y, bot_y


# ---------------------------------------------------------------------------
# Scene-change detector
# ---------------------------------------------------------------------------

class SceneChangeDetector:
    """Detects hard camera cuts in broadcast footage so the reframer can
    reset smoothing gently instead of wildly jumping."""

    def __init__(self, hist_diff_thresh: float = 0.55, pixel_diff_thresh: float = 45.0,
                 cooldown_frames: int = 8):
        self.hist_diff_thresh = hist_diff_thresh
        self.pixel_diff_thresh = pixel_diff_thresh
        self.cooldown_frames = cooldown_frames
        self.prev_hist: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.cooldown = 0

    def check(self, gray: np.ndarray) -> bool:
        """Return True if a scene change / hard cut is detected."""
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
# Ball Tracker (multi-method, sport-aware)
# ---------------------------------------------------------------------------

class BallTracker:
    """Multi-method ball detection tuned for basketball, cricket, and soccer.
    Uses up to three detection methods and fuses results:
      1. HoughCircles for circular shape
      2. Contour circularity analysis on filtered masks
      3. Color-blob detection in HSV space
    Temporal smoothing with velocity prediction reduces jitter."""

    def __init__(self, src_w: int, src_h: int, sport_profile: str = "auto"):
        self.src_w = src_w
        self.src_h = src_h
        self.sport = (sport_profile or "auto").strip().lower()
        if self.sport not in {"basketball", "cricket", "soccer"}:
            self.sport = "generic"

        # State
        self.cx = src_w / 2.0
        self.cy = src_h / 2.0
        self.radius = 0.0
        self.conf = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.missing_count = 0
        self.max_missing = 12  # frames before confidence fully decays

        # Size bounds (relative to min dimension)
        min_dim = min(src_w, src_h)
        self.min_r = max(3, int(round(min_dim * 0.006)))
        self.max_r = max(self.min_r + 4, int(round(min_dim * 0.038)))

        # Gating distance for temporal prediction
        self.gate_radius = max(self.src_w * 0.35, 80.0)

    def _build_field_mask(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """Build a mask of the playing field (green areas) for soccer/cricket."""
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
            param1=110, param2=15,
            minRadius=self.min_r, maxRadius=self.max_r,
        )
        if circles is None:
            return []
        return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0][:25]]

    def _candidates_contour(self, gray: np.ndarray, field_mask: Optional[np.ndarray]) -> list[tuple[float, float, float]]:
        """Find small circular contours on an edge+threshold image."""
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
        """Detect ball-colored blobs using sport-specific HSV ranges."""
        masks: list[np.ndarray] = []
        if self.sport == "basketball":
            masks.append(cv2.inRange(hsv, np.array([3, 80, 80]), np.array([22, 255, 255])))
        elif self.sport == "cricket":
            # Red ball
            masks.append(cv2.inRange(hsv, np.array([0, 100, 60]), np.array([10, 255, 255])))
            masks.append(cv2.inRange(hsv, np.array([165, 100, 60]), np.array([179, 255, 255])))
            # White ball
            masks.append(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 45, 255])))
        elif self.sport == "soccer":
            # White / light ball
            masks.append(cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 60, 255])))
        else:
            # Generic bright objects
            masks.append(cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 55, 255])))

        if not masks:
            return []

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

    def _score_candidate(self, cx: float, cy: float, radius: float,
                         hsv: np.ndarray, motion_mask: Optional[np.ndarray],
                         source: str) -> float:
        """Score a ball candidate based on multiple cues."""
        # --- Color score ---
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
                white_s = 1.0 if (ms <= 45 and mv >= 170) else 0.0
                red_s = 1.0 if ((mh <= 10 or mh >= 165) and ms >= 90 and mv >= 50) else 0.0
                color_score = max(white_s, red_s)
            elif self.sport == "soccer":
                if ms <= 55 and mv >= 160:
                    color_score = 0.9
                elif ms <= 80 and mv >= 130:
                    color_score = 0.5
            else:
                color_score = 0.3 if mv >= 150 else 0.0

        # --- Motion score ---
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

        # --- Proximity to prediction ---
        pred_x = self.cx + self.vx
        pred_y = self.cy + self.vy
        dist = math.hypot(cx - pred_x, cy - pred_y)
        proximity_score = max(0.0, 1.0 - dist / self.gate_radius)

        # --- Size score (prefer expected ball size) ---
        expected_r = (self.min_r + self.max_r) / 2.0
        size_dev = abs(radius - expected_r) / max(expected_r, 1.0)
        size_score = max(0.0, 1.0 - size_dev)

        # --- Source bonus (multiple detection methods found same area) ---
        source_bonus = 0.0
        if source == "multi":
            source_bonus = 0.15

        # --- Sport-specific weighting ---
        if self.sport == "cricket":
            score = 0.25 * color_score + 0.25 * motion_score + 0.25 * proximity_score + 0.15 * size_score + 0.10 * source_bonus
        elif self.sport == "basketball":
            score = 0.35 * color_score + 0.20 * motion_score + 0.22 * proximity_score + 0.13 * size_score + 0.10 * source_bonus
        elif self.sport == "soccer":
            score = 0.22 * color_score + 0.28 * motion_score + 0.28 * proximity_score + 0.12 * size_score + 0.10 * source_bonus
        else:
            score = 0.20 * color_score + 0.30 * motion_score + 0.30 * proximity_score + 0.10 * size_score + 0.10 * source_bonus

        return float(score)

    def update(self, frame: np.ndarray, gray: np.ndarray,
               motion_mask: Optional[np.ndarray],
               exclusion_mask: Optional[np.ndarray] = None) -> Optional[tuple[float, float, float, float]]:
        """Run detection and return (cx, cy, radius, confidence) or None."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply exclusion mask to gray for detection (zero-out overlay regions)
        det_gray = gray.copy()
        if exclusion_mask is not None:
            det_gray[exclusion_mask > 0] = 0

        field_mask = self._build_field_mask(hsv)

        # Collect candidates from all methods
        raw_candidates: list[tuple[float, float, float, str]] = []
        for cx, cy, r in self._candidates_hough(det_gray):
            raw_candidates.append((cx, cy, r, "hough"))
        for cx, cy, r in self._candidates_contour(det_gray, field_mask):
            raw_candidates.append((cx, cy, r, "contour"))
        for cx, cy, r in self._candidates_color_blob(hsv, field_mask):
            raw_candidates.append((cx, cy, r, "color"))

        # Filter out candidates that fall in exclusion zones
        if exclusion_mask is not None:
            filtered = []
            for cx, cy, r, src in raw_candidates:
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= iy < self.src_h and 0 <= ix < self.src_w:
                    if exclusion_mask[iy, ix] == 0:
                        filtered.append((cx, cy, r, src))
            raw_candidates = filtered

        # Cluster nearby candidates & mark "multi" source
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

        # Score each cluster
        best: Optional[tuple[float, float, float, float]] = None
        best_score = -1.0
        for cx, cy, r, src in clusters:
            score = self._score_candidate(cx, cy, r, hsv, motion_mask, src)
            if score > best_score:
                best_score = score
                best = (cx, cy, r, score)

        # Acceptance threshold
        if best is None or best_score < 0.15:
            self.missing_count += 1
            self.conf *= 0.88
            if self.missing_count > self.max_missing:
                self.conf = 0.0
            return None

        cx, cy, r, score = best

        # Update velocity with smoothing
        new_vx = cx - self.cx
        new_vy = cy - self.cy
        self.vx = 0.5 * self.vx + 0.5 * new_vx
        self.vy = 0.5 * self.vy + 0.5 * new_vy

        # Smooth position update (heavier smoothing to reduce jitter)
        self.cx = 0.6 * self.cx + 0.4 * cx
        self.cy = 0.6 * self.cy + 0.4 * cy
        self.radius = (0.6 * self.radius + 0.4 * r) if self.radius > 0 else r
        self.conf = min(1.0, 0.7 * self.conf + 0.55 * score)
        self.missing_count = 0

        return (self.cx, self.cy, self.radius, self.conf)

    def reset_position(self, cx: float, cy: float) -> None:
        """Soft reset after a scene change."""
        self.cx = cx
        self.cy = cy
        self.vx = 0.0
        self.vy = 0.0
        self.conf *= 0.3


# ---------------------------------------------------------------------------
# Smooth Reframer (v2 — overlay-aware, ball-tracking, scene-cut-safe)
# ---------------------------------------------------------------------------

class SmoothReframer:
    """Stable vertical reframer with:
      - multi-cue focus: faces, motion clusters, saliency, ball tracking
      - overlay/scorecard detection and exclusion from reframing decisions
      - optional scorecard compositing into vertical output
      - scene-change detection with gentle reset
      - fixed output resolution (target_w x target_h)
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
        ball_weight: float = 0.55,
        context_bias: float = 0.20,
        overlay_composite: bool = True,
    ):
        self.src_w, self.src_h = int(src_w), int(src_h)
        self.target_w, self.target_h = int(target_w), int(target_h)
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
        self.sport_profile = (sport_profile or "auto").strip().lower()

        # Sub-components
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.saliency = None
        try:
            if hasattr(cv2, "saliency"):
                self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            pass

        self.overlay_detector = OverlayDetector(src_w, src_h)
        self.scene_detector = SceneChangeDetector()
        self.ball_tracker = BallTracker(src_w, src_h, sport_profile=self.sport_profile) if ball_tracking else None

        # Smoothing state
        self.smoothed_cx = src_w / 2.0
        self.smoothed_cy = src_h / 2.0
        self.target_cx = self.smoothed_cx
        self.target_cy = self.smoothed_cy
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0

    def _detect_motion(self, gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.prev_gray is None:
            return [], None
        diff = cv2.absdiff(gray, self.prev_gray)
        diff = cv2.GaussianBlur(diff, (9, 9), 0)
        _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.dilate(motion, None, iterations=2)

        # Zero-out overlay regions so they don't create false motion
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

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update overlay detector every frame for stability
        self.overlay_detector.update(gray)

        # Scene change detection
        is_cut = self.scene_detector.check(gray)
        if is_cut:
            # Gentle reset: move smoothed position halfway to center
            self.smoothed_cx = (self.smoothed_cx + self.src_w / 2.0) / 2.0
            self.smoothed_cy = (self.smoothed_cy + self.src_h / 2.0) / 2.0
            if self.ball_tracker is not None:
                self.ball_tracker.reset_position(self.src_w / 2.0, self.src_h / 2.0)

        # Analysis (every N frames)
        if self.frame_idx % self.analysis_stride == 0:
            candidates: list[tuple[float, tuple[float, float]]] = []
            excl = self.overlay_detector.exclusion_mask
            play_top, play_bot = self.overlay_detector.get_play_area_bounds()

            # --- Faces ---
            try:
                play_gray = gray[play_top:play_bot, :]
                faces = self.face_detector.detectMultiScale(
                    play_gray, scaleFactor=1.15, minNeighbors=4, minSize=(32, 32),
                )
            except Exception:
                faces = []
            for (x, y, w, h) in faces[:3]:
                # Offset y back to full-frame coordinates
                candidates.append((0.35, (x + w / 2.0, play_top + y + h / 2.0)))

            # --- Motion ---
            motion_boxes, motion_mask = self._detect_motion(gray)
            if motion_boxes:
                x0 = min(p[0] for p in motion_boxes)
                y0 = min(p[1] for p in motion_boxes)
                x1 = max(p[0] + p[2] for p in motion_boxes)
                y1 = max(p[1] + p[3] for p in motion_boxes)
                candidates.append((0.30, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)))
                # Strongest single motion region
                bx, by, bw, bh = motion_boxes[0]
                candidates.append((0.15, (bx + bw / 2.0, by + bh / 2.0)))
            else:
                motion_mask = None

            # --- Saliency ---
            if self.saliency is not None:
                try:
                    success, sal_map = self.saliency.computeSaliency(frame)
                    if success:
                        sal_map = (sal_map * 255).astype("uint8")
                        # Zero overlays in saliency
                        if excl is not None:
                            sal_map[excl > 0] = 0
                        _, thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            c = max(cnts, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            if w * h > 0.02 * self.src_w * self.src_h:
                                candidates.append((0.10, (x + w / 2.0, y + h / 2.0)))
                except Exception:
                    pass

            # --- Ball tracking ---
            if self.ball_tracker is not None:
                ball = self.ball_tracker.update(frame, gray, motion_mask, excl)
                if ball is not None:
                    bx, by, _br, bconf = ball
                    effective_weight = self.ball_weight * max(0.25, bconf)
                    candidates.append((effective_weight, (bx, by)))
                    # Context around motion to keep players visible
                    if motion_boxes:
                        mx0 = min(p[0] for p in motion_boxes)
                        my0 = min(p[1] for p in motion_boxes)
                        mx1 = max(p[0] + p[2] for p in motion_boxes)
                        my1 = max(p[1] + p[3] for p in motion_boxes)
                        candidates.append((self.context_bias, ((mx0 + mx1) / 2.0, (my0 + my1) / 2.0)))

            # --- Compute weighted target ---
            if candidates:
                ws = sum(w for w, _ in candidates)
                self.target_cx = sum(cx * w for w, (cx, _) in candidates) / max(ws, 1e-6)
                self.target_cy = sum(cy * w for w, (_, cy) in candidates) / max(ws, 1e-6)
            else:
                self.target_cx = self.src_w / 2.0
                self.target_cy = self.src_h / 2.0

        self.prev_gray = gray

        # --- Smooth panning ---
        dx = self.target_cx - self.smoothed_cx
        dy = self.target_cy - self.smoothed_cy
        if abs(dx) < self.deadzone_px:
            dx = 0.0
        if abs(dy) < self.deadzone_px * 0.45:
            dy = 0.0

        # On scene cuts, allow faster repositioning
        alpha = (1.0 - self.smooth_strength) * (3.0 if is_cut else 1.0)
        self.smoothed_cx += max(-self.max_pan_px, min(self.max_pan_px, dx * alpha))
        self.smoothed_cy += max(-(self.max_pan_px * 0.45), min(self.max_pan_px * 0.45, dy * alpha))

        # --- Crop ---
        x0 = int(round(self.smoothed_cx - self.crop_w / 2.0))
        y0 = int(round(self.smoothed_cy - self.crop_h / 2.0))
        x0 = int(_clamp(x0, 0, self.max_x))
        y0 = int(_clamp(y0, 0, self.max_y))

        crop = frame[y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        if crop.size == 0:
            crop = frame

        output = cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_CUBIC)

        # --- Composite scorecard strip ---
        if self.overlay_composite:
            score_strip = self.overlay_detector.extract_scorecard_strip(frame)
            if score_strip is not None:
                strip_h_target = max(24, int(self.target_h * 0.065))
                strip_resized = cv2.resize(score_strip, (self.target_w, strip_h_target), interpolation=cv2.INTER_AREA)
                output[:strip_h_target, :] = strip_resized

            bottom_strip = self.overlay_detector.extract_bottom_strip(frame)
            if bottom_strip is not None:
                bstrip_h_target = max(20, int(self.target_h * 0.05))
                bstrip_resized = cv2.resize(bottom_strip, (self.target_w, bstrip_h_target), interpolation=cv2.INTER_AREA)
                output[self.target_h - bstrip_h_target:, :] = bstrip_resized

        self.frame_idx += 1
        return output


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


def cfstream_config_from_inputs(
    account_id: str, api_token: str, customer_code: str,
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
    reframed_mp4: str, rtmps_url: str, stream_key: str,
    loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS,
):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(output_fps or DEFAULT_OUTPUT_FPS))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    if loop_input:
        cmd += ["-stream_loop", "-1"]
    cmd += [
        "-re", "-i", reframed_mp4,
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
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
        "-f", "flv", target,
    ]
    return cmd


def start_vod_to_live_push(
    cfg: CFStreamConfig, reframed_mp4: str, asset_name: str,
    loop_input: bool = True, output_fps: float = DEFAULT_OUTPUT_FPS,
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
    target_w: int, target_h: int, fps: float, rtmps_url: str, stream_key: str,
):
    target = rtmps_url.rstrip("/") + "/" + stream_key
    fps_int = max(24, min(60, int(round(fps or DEFAULT_OUTPUT_FPS))))
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{target_w}x{target_h}",
        "-r", str(fps_int), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-shortest", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
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
    cv2.putText(frame, "Vertical stream", (28, max(48, target_h // 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (28, target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Cloudflare live input priming", (28, target_h // 2 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 195, 230), 1, cv2.LINE_AA)
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
    session: LiveSession, source: str,
    target_w: int, target_h: int, delay_seconds: float,
    smooth_strength: float, analysis_stride: int,
    deadzone_ratio: float, max_pan_ratio: float,
    loop_file: bool, pace_input: bool,
    sport_profile: str, ball_tracking: bool, overlay_composite: bool,
) -> None:
    session.status = "probing"
    info = probe_source(source)
    fps = DEFAULT_OUTPUT_FPS
    src_w, src_h = WORKING_INPUT_W, WORKING_INPUT_H
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
        src_w, src_h, target_w, target_h,
        smooth_strength=smooth_strength,
        analysis_stride=analysis_stride,
        deadzone_ratio=deadzone_ratio,
        max_pan_ratio=max_pan_ratio,
        sport_profile=sport_profile,
        ball_tracking=ball_tracking,
        overlay_composite=overlay_composite,
    )
    buffer: collections.deque = collections.deque(maxlen=max(delay_frames + 240, 600))
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
            prime = int(max(1.0, min(delay_seconds / 2.0, 3.0)) * fps)
            for _ in range(prime):
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
                    session.error = f"Source unavailable: {source}"
                else:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))
                    buffer.append(reframer.process(frame))
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
                next_deadline = time.monotonic()

            if frames_out % int(max(1.0, fps)) == 0:
                ball_conf = 0.0
                if reframer.ball_tracker is not None:
                    ball_conf = round(reframer.ball_tracker.conf, 3)
                session.stats.update({
                    "frames_in": frames_in,
                    "frames_out": frames_out,
                    "buffer_len": len(buffer),
                    "delay_seconds": round(delay_frames / max(fps, 1.0), 2),
                    "placeholder_frames": placeholder_frames,
                    "source_stalls": source_stalls,
                    "ball_confidence": ball_conf,
                    "overlay_top": reframer.overlay_detector.top_overlay is not None,
                    "overlay_bottom": reframer.overlay_detector.bottom_overlay is not None,
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
    cfg: CFStreamConfig, source: str, asset_name: str,
    target_w: int = DEFAULT_TARGET_W, target_h: int = DEFAULT_TARGET_H,
    delay_seconds: float = 20.0,
    smooth_strength: float = 0.975,
    analysis_stride: int = 4,
    deadzone_ratio: float = 0.05,
    max_pan_ratio: float = 0.012,
    loop_file: bool = False,
    pace_input: bool = True,
    sport_profile: str = "auto",
    ball_tracking: bool = True,
    overlay_composite: bool = True,
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
            loop_file, pace_input, sport_profile, ball_tracking, overlay_composite,
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
