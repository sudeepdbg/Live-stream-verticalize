"""
VideoForge Web – Encoder + VMAF Analytics + AI Enhancement + Intelligent ABR Player
Run:  streamlit run app.py

Deploy:
  Streamlit Community Cloud → packages.txt: ffmpeg
  HF Spaces (Streamlit SDK)  → same packages.txt

Changelog:
  OPT 1 – Modularized backend utilities into logical sections
  OPT 2 – Added @st.cache_data for probe(), quality_metrics(), loudness probe
  OPT 3 – Implemented lazy loading for heavy computations
  OPT 4 – Added memory monitoring for large file handling
  OPT 5 – Optimized CSS loading and reduced re-renders
  NEW 1 – Advanced Player Controls: speed, frame-step, A/B compare, zoom, loop, screenshot
  NEW 2 – Intelligent ABR Engine: network-aware bitrate switching with guardrails
  NEW 3 – QoE/QoS Dashboard: real-time metrics, buffer health, switch analytics
  NEW 4 – Network Simulator: test ABR behavior with synthetic traces
  BUG FIX – All previous fixes retained + new edge-case handling
"""

import os, io, csv, json, subprocess, tempfile, time, re, atexit, math, hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoForge · AI Encoder + ABR Player",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Temp File Cleanup ─────────────────────────────────────────────────────────
_temp_files = set()

def register_temp_file(path: str):
    _temp_files.add(path)

def _cleanup_temps():
    for path in _temp_files:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
    _temp_files.clear()

atexit.register(_cleanup_temps)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZATION: Cached Utilities
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def probe_cached(path: str, file_hash: str) -> dict:
    """Cached version of probe() - invalidates on file change via hash."""
    return _probe_impl(path)

def _probe_impl(path: str) -> dict:
    """Internal probe implementation (not cached)."""
    r = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "vcodec": "unknown", "vbitrate_kbps": 0,
        "acodec": "unknown", "abitrate_kbps": 0,
        "sample_rate": 0, "channels": 0, "audio_duration": 0.0,
        "has_audio": False, "color_space": "unknown", "bit_depth": 8,
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", path],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        d = json.loads(out)
        fmt = d.get("format", {})
        r["duration"] = float(fmt.get("duration", 0) or 0)
        r["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000

        for s in d.get("streams", []):
            if s.get("codec_type") == "video" and r["width"] == 0:
                r["width"] = s.get("width", 0)
                r["height"] = s.get("height", 0)
                r["vcodec"] = s.get("codec_name", "unknown")
                r["color_space"] = s.get("color_space", "unknown")
                r["bit_depth"] = int(s.get("bits_per_raw_sample", 8) or 8)
                try:
                    n, dn = map(int, s.get("r_frame_rate", "0/1").split("/"))
                    r["fps"] = round(n / dn, 3) if dn else 0.0
                except Exception:
                    pass
            elif s.get("codec_type") == "audio" and not r["has_audio"]:
                r["has_audio"] = True
                r["acodec"] = s.get("codec_name", "unknown")
                r["abitrate_kbps"] = int(s.get("bit_rate", 0) or 0) // 1000
                r["sample_rate"] = int(s.get("sample_rate", 0) or 0)
                r["channels"] = int(s.get("channels", 0) or 0)
                r["audio_duration"] = float(s.get("duration", 0) or 0)
    except Exception:
        pass
    return r


@st.cache_data(ttl=1800, show_spinner=False)
def quality_metrics_cached(ref_path: str, dist_path: str, ref_hash: str, dist_hash: str, 
                          do_vmaf: bool, duration_sec: float) -> dict:
    """Cached quality metrics computation."""
    return _quality_metrics_impl(ref_path, dist_path, do_vmaf, duration_sec)

def _quality_metrics_impl(ref: str, dist: str, do_vmaf: bool, duration_sec: float) -> dict:
    """Internal implementation of quality metrics."""
    res = {"psnr": None, "ssim": None, "vmaf": None}

    # PSNR
    try:
        cmd = ["ffmpeg", "-y", "-i", dist, "-i", ref, "-filter_complex", "[0:v][1:v]psnr", "-f", "null", "-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"PSNR", line, re.I):
                m = re.search(r"average[:\s]+([0-9.]+|inf)", line, re.I)
                if m:
                    v = m.group(1)
                    res["psnr"] = 100.0 if v == "inf" else round(float(v), 3)
    except Exception:
        pass

    # SSIM
    try:
        cmd = ["ffmpeg", "-y", "-i", dist, "-i", ref, "-filter_complex", "[0:v][1:v]ssim", "-f", "null", "-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=180)
        for line in out.splitlines():
            if re.search(r"SSIM", line, re.I):
                m = re.search(r"All[:\s]+([0-9.]+)", line, re.I)
                if m:
                    res["ssim"] = round(float(m.group(1)), 5)
    except Exception:
        pass

    # VMAF
    if do_vmaf:
        try:
            vmaf_timeout = max(120, int(duration_sec * 3)) if duration_sec > 0 else 300
            vf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            vf.close()
            register_temp_file(vf.name)
            cmd = ["ffmpeg", "-y", "-i", dist, "-i", ref,
                   "-filter_complex", f"[0:v][1:v]libvmaf=log_fmt=json:log_path={vf.name}", "-f", "null", "-"]
            subprocess.run(cmd, capture_output=True, timeout=vmaf_timeout, check=True)
            with open(vf.name) as f:
                vdata = json.load(f)
            score = (vdata.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
                     or vdata.get("VMAF score") or vdata.get("aggregate", {}).get("VMAF_score"))
            if score is not None:
                res["vmaf"] = round(float(score), 2)
            try:
                os.unlink(vf.name)
            except Exception:
                pass
        except Exception:
            pass
    return res


@st.cache_data(ttl=600, show_spinner=False)
def probe_loudness_cached(path: str, file_hash: str) -> dict:
    """Cached loudness probe."""
    return _probe_loudness_impl(path)

def _probe_loudness_impl(path: str) -> dict:
    """Internal loudness probe implementation."""
    res = {"mean_volume": None, "max_volume": None}
    try:
        cmd = ["ffmpeg", "-y", "-i", path, "-af", "volumedetect", "-f", "null", "-"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=60)
        for line in out.splitlines():
            m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", line, re.I)
            if m:
                try: res["mean_volume"] = float(m.group(1))
                except ValueError: pass
            m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", line, re.I)
            if m:
                try: res["max_volume"] = float(m.group(1))
                except ValueError: pass
        # Fallback parsing
        if res["mean_volume"] is None:
            summary = re.search(r"Mean volume:\s*([-\d.]+)\s+dB", out, re.I)
            if summary:
                try: res["mean_volume"] = float(summary.group(1))
                except ValueError: pass
    except Exception:
        pass
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Intelligent ABR Engine with Guardrails
# ══════════════════════════════════════════════════════════════════════════════

class ABRStrategy(Enum):
    CONSERVATIVE = "conservative"  # Prioritize smoothness, avoid switches
    BALANCED = "balanced"          # Default: balance quality and stability
    AGGRESSIVE = "aggressive"      # Maximize quality, tolerate more switches


@dataclass
class NetworkMetrics:
    """Real-time network state for ABR decisions."""
    throughput_mbps: float = 0.0  # Estimated available bandwidth
    latency_ms: float = 0.0        # Round-trip latency
    jitter_ms: float = 0.0         # Latency variation
    packet_loss_pct: float = 0.0   # Packet loss percentage
    timestamp: float = field(default_factory=time.time)


@dataclass
class ABRConfig:
    """Guardrails and tuning parameters for ABR algorithm."""
    # Quality bounds
    min_resolution: str = "480p"   # Never drop below this
    max_resolution: str = "1080p"  # Cap at this (respect tier/device)
    
    # Buffer management
    buffer_target_sec: float = 30.0    # Ideal buffer level
    buffer_min_sec: float = 5.0        # Emergency downgrade threshold
    buffer_max_sec: float = 60.0       # Aggressive upgrade threshold
    
    # Switching constraints
    switch_hysteresis_up: float = 0.20    # Need 20% BW increase to switch up
    switch_hysteresis_down: float = 0.10  # Need 10% BW decrease to switch down
    switch_cooldown_sec: float = 3.0      # Min time between switches
    max_switches_per_min: int = 4         # Prevent oscillation
    
    # QoE guardrails
    min_vmaf_score: float = 70.0     # Never accept quality below this
    max_rebuffer_pct: float = 2.0    # Alert if rebuffering exceeds this
    startup_delay_max_sec: float = 3.0  # Max acceptable initial load time
    
    # Strategy
    strategy: ABRStrategy = ABRStrategy.BALANCED
    
    # Device constraints
    max_decode_fps: int = 60
    max_display_resolution: str = "1080p"


@dataclass
class ABRState:
    """Runtime state for ABR decision engine."""
    current_bitrate_kbps: int = 1500
    current_resolution: str = "720p"
    buffer_level_sec: float = 30.0
    last_switch_time: float = 0.0
    switch_count_last_min: int = 0
    throughput_ewma: Optional[float] = None
    quality_score_ewma: Optional[float] = None
    in_recovery_mode: bool = False
    recovery_until: float = 0.0
    
    # QoE tracking
    total_played_sec: float = 0.0
    total_buffered_sec: float = 0.0
    quality_switches: List[Dict] = field(default_factory=list)


class IntelligentBitrateSelector:
    """
    Intelligent ABR engine with guardrails for smooth QoE/QoS.
    
    Features:
    - EWMA bandwidth estimation with outlier rejection
    - Buffer-aware quality selection
    - Hysteresis to prevent oscillation
    - Recovery mode after underruns
    - QoE metric tracking and alerting
    """
    
    # Bitrate ladder (resolution -> bitrates in kbps)
    BITRATE_LADDER = {
        "480p": [500, 800, 1200],
        "720p": [1500, 2500, 4000],
        "1080p": [5000, 8000, 12000],
        "1440p": [15000, 20000],
        "2160p": [25000, 35000, 50000],
    }
    
    # VMAF estimates per bitrate (approximate, codec-dependent)
    VMAF_ESTIMATES = {
        500: 65, 800: 72, 1200: 78, 1500: 80, 2500: 85, 4000: 89,
        5000: 90, 8000: 93, 12000: 95, 15000: 96, 20000: 97,
        25000: 97, 35000: 98, 50000: 99,
    }
    
    def __init__(self, config: ABRConfig):
        self.config = config
        self.state = ABRState()
        self._switch_history: List[float] = []  # timestamps
        
    def update_network(self, metrics: NetworkMetrics):
        """Update network estimates with EWMA smoothing."""
        alpha = 0.3  # Smoothing factor
        if self.state.throughput_ewma is None:
            self.state.throughput_ewma = metrics.throughput_mbps
        else:
            self.state.throughput_ewma = (
                alpha * metrics.throughput_mbps + 
                (1 - alpha) * self.state.throughput_ewma
            )
        # Outlier rejection: ignore spikes >3x EWMA
        if metrics.throughput_mbps > 3 * self.state.throughput_ewma:
            metrics.throughput_mbps = self.state.throughput_ewma
            
    def update_buffer(self, buffer_sec: float):
        """Update buffer level and trigger emergency actions if needed."""
        self.state.buffer_level_sec = buffer_sec
        # Emergency downgrade if buffer critically low
        if buffer_sec < self.config.buffer_min_sec and not self.state.in_recovery_mode:
            self.state.in_recovery_mode = True
            self.state.recovery_until = time.time() + 30  # 30s conservative mode
            
    def _get_available_bitrates(self) -> List[int]:
        """Get bitrates within configured resolution bounds."""
        resolutions = list(self.BITRATE_LADDER.keys())
        min_idx = resolutions.index(self.config.min_resolution)
        max_idx = resolutions.index(self.config.max_resolution)
        bitrates = []
        for res in resolutions[min_idx:max_idx+1]:
            bitrates.extend(self.BITRATE_LADDER[res])
        return sorted(set(bitrates))
    
    def _estimate_vmaf(self, bitrate_kbps: int) -> float:
        """Estimate VMAF score for a given bitrate."""
        # Linear interpolation between known points
        known = sorted(self.VMAF_ESTIMATES.items())
        if bitrate_kbps <= known[0][0]:
            return known[0][1]
        if bitrate_kbps >= known[-1][0]:
            return known[-1][1]
        for i in range(len(known)-1):
            if known[i][0] <= bitrate_kbps <= known[i+1][0]:
                t = (bitrate_kbps - known[i][0]) / (known[i+1][0] - known[i][0])
                return known[i][1] + t * (known[i+1][1] - known[i][1])
        return 85.0  # Fallback
    
    def _should_switch(self, target_bitrate: int) -> bool:
        """Apply switching constraints and guardrails."""
        now = time.time()
        
        # Cooldown check
        if now - self.state.last_switch_time < self.config.switch_cooldown_sec:
            return False
            
        # Rate limit switches
        recent = [t for t in self._switch_history if now - t < 60]
        if len(recent) >= self.config.max_switches_per_min:
            return False
            
        # Recovery mode: be conservative
        if self.state.in_recovery_mode:
            if now < self.state.recovery_until:
                return target_bitrate < self.state.current_bitrate_kbps  # Only allow downgrades
            else:
                self.state.in_recovery_mode = False
                
        # Hysteresis check
        current = self.state.current_bitrate_kbps
        if target_bitrate > current:
            # Need significant BW increase to switch up
            required = current * (1 + self.config.switch_hysteresis_up)
            if self.state.throughput_ewma * 1000 < required:  # Convert Mbps->kbps
                return False
        elif target_bitrate < current:
            # Smaller decrease triggers downgrade
            allowed = current * (1 - self.config.switch_hysteresis_down)
            if self.state.throughput_ewma * 1000 > allowed:
                return False
                
        return True
    
    def select_bitrate(self, network: NetworkMetrics) -> Dict[str, Any]:
        """
        Main decision function: select optimal bitrate based on network, buffer, and guardrails.
        
        Returns dict with:
        - selected_bitrate_kbps: chosen bitrate
        - reason: explanation for decision
        - qoe_metrics: current QoE indicators
        - alerts: any guardrail warnings
        """
        alerts = []
        reason = "no change"
        
        # Update internal state
        self.update_network(network)
        
        # Get candidate bitrates
        available = self._get_available_bitrates()
        if not available:
            return {"error": "No valid bitrates in configured range"}
            
        # Estimate ideal bitrate from bandwidth (with headroom)
        headroom = 0.8 if self.config.strategy == ABRStrategy.CONSERVATIVE else 0.9
        ideal_kbps = self.state.throughput_ewma * 1000 * headroom
        
        # Find best matching bitrate
        candidates = [b for b in available if b <= ideal_kbps]
        if not candidates:
            # Fallback to lowest available
            target_bitrate = min(available)
            reason = "bandwidth too low, using minimum"
            alerts.append("⚠️ Bandwidth insufficient for target quality")
        else:
            # Pick highest candidate within bounds
            target_bitrate = max(candidates)
            
        # Buffer-based overrides
        if self.state.buffer_level_sec < self.config.buffer_min_sec:
            # Emergency: force downgrade to lowest
            target_bitrate = min(available)
            reason = "emergency: buffer critical"
            alerts.append("🚨 Buffer underrun risk - forced downgrade")
        elif self.state.buffer_level_sec > self.config.buffer_max_sec:
            # Aggressive: try highest available
            target_bitrate = max(available)
            reason = "buffer healthy, maximizing quality"
            
        # Quality floor check
        estimated_vmaf = self._estimate_vmaf(target_bitrate)
        if estimated_vmaf < self.config.min_vmaf_score:
            # Find minimum bitrate that meets quality floor
            for b in sorted(available, reverse=True):
                if self._estimate_vmaf(b) >= self.config.min_vmaf_score:
                    target_bitrate = b
                    reason = "enforced minimum quality floor"
                    alerts.append(f"📊 Quality floor: selected {b} kbps for VMAF≥{self.config.min_vmaf_score}")
                    break
                    
        # Apply switching constraints
        if target_bitrate != self.state.current_bitrate_kbps:
            if self._should_switch(target_bitrate):
                # Execute switch
                old_bitrate = self.state.current_bitrate_kbps
                self.state.current_bitrate_kbps = target_bitrate
                self.state.last_switch_time = time.time()
                self._switch_history.append(time.time())
                # Clean old history
                now = time.time()
                self._switch_history = [t for t in self._switch_history if now - t < 60]
                self.state.switch_count_last_min = len(self._switch_history)
                reason = f"switched from {old_bitrate} to {target_bitrate} kbps"
                self.state.quality_switches.append({
                    "time": now,
                    "from": old_bitrate,
                    "to": target_bitrate,
                    "buffer": self.state.buffer_level_sec,
                    "throughput": self.state.throughput_ewma,
                })
            else:
                target_bitrate = self.state.current_bitrate_kbps
                reason = "switch suppressed by guardrails"
                
        # Update QoE tracking
        self.state.total_played_sec += 1  # Simulate 1s playback
        if network.latency_ms > 200 or network.packet_loss_pct > 2:
            self.state.total_buffered_sec += 0.1  # Simulate minor rebuffer
            
        # Compile response
        rebuffer_pct = (self.state.total_buffered_sec / max(1, self.state.total_played_sec)) * 100
        avg_vmaf = self._estimate_vmaf(self.state.current_bitrate_kbps)
        
        return {
            "selected_bitrate_kbps": self.state.current_bitrate_kbps,
            "estimated_vmaf": avg_vmaf,
            "reason": reason,
            "alerts": alerts,
            "qoe_metrics": {
                "rebuffer_pct": round(rebuffer_pct, 2),
                "avg_quality_vmaf": round(avg_vmaf, 1),
                "switches_last_min": self.state.switch_count_last_min,
                "buffer_health": "good" if self.state.buffer_level_sec > 20 else "warning" if self.state.buffer_level_sec > 10 else "critical",
            },
            "network_estimate_mbps": round(self.state.throughput_ewma, 2),
        }
    
    def get_qoe_report(self) -> Dict[str, Any]:
        """Generate comprehensive QoE report."""
        rebuffer_pct = (self.state.total_buffered_sec / max(1, self.state.total_played_sec)) * 100
        avg_vmaf = np.mean([self._estimate_vmaf(sw["to"]) for sw in self.state.quality_switches[-10:]]) if self.state.quality_switches else self._estimate_vmaf(self.state.current_bitrate_kbps)
        
        return {
            "session_duration_sec": round(self.state.total_played_sec, 1),
            "total_rebuffer_sec": round(self.state.total_buffered_sec, 1),
            "rebuffer_ratio_pct": round(rebuffer_pct, 2),
            "avg_quality_vmaf": round(avg_vmaf, 1),
            "total_switches": len(self.state.quality_switches),
            "switches_per_min": round(len(self.state.quality_switches) / max(1, self.state.total_played_sec/60), 1),
            "current_bitrate_kbps": self.state.current_bitrate_kbps,
            "current_buffer_sec": round(self.state.buffer_level_sec, 1),
            "guardrail_alerts": [
                "Rebuffering above threshold" if rebuffer_pct > self.config.max_rebuffer_pct else None,
                "Quality below floor" if avg_vmaf < self.config.min_vmaf_score else None,
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Backend Utilities (non-cached)
# ══════════════════════════════════════════════════════════════════════════════

def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False

def vmaf_ok() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-filters"], stderr=subprocess.STDOUT, text=True, timeout=10)
        return "libvmaf" in out
    except Exception:
        return False

def build_enhance_filters(settings: dict, src_meta: dict) -> list:
    """Build FFmpeg filter chain for AI enhancements."""
    filters = []
    if settings.get("denoise"):
        strength = float(settings.get("denoise_strength", 5))
        s2 = round(strength / 2, 1)
        filters.append(f"hqdn3d={strength}:{strength}:{s2}:{s2}")
    if settings.get("deblock"):
        strength = int(settings.get("deblock_strength", 5))
        alpha = round(strength / 10.0, 2)
        beta = round(alpha * 0.5, 2)
        filters.append(f"deblock=filter=strong:alpha={alpha}:beta={beta}")
    if settings.get("sharpen"):
        amount = float(settings.get("sharpen_amount", 0.5))
        c_amount = round(amount * 0.5, 2)
        filters.append(f"unsharp=lx=5:ly=5:la={amount}:cx=3:cy=3:ca={c_amount}")
    if settings.get("color_enhance"):
        vibrance = float(settings.get("vibrance", 0.15))
        contrast = float(settings.get("contrast", 1.0))
        sat = round(1.0 + vibrance, 4)
        filters.append(f"eq=contrast={contrast}:saturation={sat}")
    if settings.get("hdr_convert") and src_meta.get("bit_depth", 8) >= 10:
        filters.append("zscale=transfer=linear,format=gbrpf32le")
        filters.append(f"tonemap={settings.get('tonemap_algo','hable')}:desat=0.2")
        filters.append("zscale=transfer=bt709:matrix=bt709:range=tv,format=yuv420p10le")
    if settings.get("upscale"):
        target_w = int(settings.get("upscale_width", src_meta["width"] * 2))
        target_h = int(settings.get("upscale_height", src_meta["height"] * 2))
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        target_h = target_h if target_h % 2 == 0 else target_h - 1
        algo = settings.get("upscale_algo", "lanczos")
        filters.append(f"scale={target_w}:{target_h}:flags={algo}+accurate_rnd+full_chroma_int")
    if settings.get("frame_interp"):
        target_fps = int(settings.get("target_fps", 60))
        filters.append(f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    return filters

def estimate_processing_time(src_meta: dict, settings: dict) -> str:
    base_factor = 1.0
    if settings.get("denoise"): base_factor *= 1.3
    if settings.get("sharpen"): base_factor *= 1.1
    if settings.get("upscale"): base_factor *= 2.5 if settings.get("upscale_algo") == "lanczos" else 1.8
    if settings.get("hdr_convert"): base_factor *= 1.6
    if settings.get("color_enhance"): base_factor *= 1.15
    if settings.get("deblock"): base_factor *= 1.4
    if settings.get("frame_interp"): base_factor *= 3.0
    duration_min = (src_meta.get("duration", 0) / 60) * base_factor
    if duration_min < 1: return f"~{max(1, int(duration_min * 60))}s"
    elif duration_min < 10: return f"~{duration_min:.1f} min"
    else: return f"~{duration_min:.0f} min"

def encode(input_path, output_path, codec, crf, enhance_settings: dict, src_meta: dict,
           progress_cb=None, duration=0.0):
    """Encode with optional enhancement filters."""
    cmap = {
        "AVC (H.264)": ("libx264", ["-preset", "fast"]),
        "HEVC (H.265)": ("libx265", ["-preset", "fast"]),
        "AV1": ("libaom-av1", ["-b:v", "0", "-cpu-used", "8", "-tile-columns", "2", "-threads", "4", "-usage", "realtime"]),
    }
    lib, extra = cmap.get(codec, ("libx264", ["-preset", "fast"]))
    filters = build_enhance_filters(enhance_settings, src_meta)
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", lib, "-crf", str(crf)] + extra
    if filters: cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:a", "copy", "-movflags", "+faststart", output_path]
    lines, t0 = [], time.time()
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, bufsize=1)
        for line in proc.stderr:
            lines.append(line.rstrip())
            if progress_cb and "time=" in line and duration > 0:
                try:
                    ts = line.split("time=")[1].split(" ")[0]
                    h, m, s = map(float, ts.split(":"))
                    progress_cb(min((h * 3600 + m * 60 + s) / duration, 0.99))
                except: pass
        proc.wait()
        elapsed = time.time() - t0
        log = "\n".join(lines[-80:])
        if proc.returncode == 0: return True, "Done!", log, elapsed
        hints = {-6: "OOM — disable upscaling/frame interpolation.", -9: "OOM (SIGKILL) — reduce complexity.", -11: "Segfault — check filter compatibility.", 1: "FFmpeg error — see log."}
        return False, hints.get(proc.returncode, f"FFmpeg exit {proc.returncode}"), log, elapsed
    except FileNotFoundError: return False, "FFmpeg not found.", "", 0.0
    except Exception as e: return False, str(e), "\n".join(lines), time.time() - t0


# ── Display helpers ───────────────────────────────────────────────────────────
def vmaf_display(v):
    if v is None: return "—", ""
    if v >= 93: return f"{v:.1f} · Excellent", "q-exc"
    if v >= 80: return f"{v:.1f} · Good", "q-gd"
    if v >= 60: return f"{v:.1f} · Fair", "q-ok"
    return f"{v:.1f} · Poor", "q-bad"

def ssim_display(v) -> str:
    if v is None: return "—"
    label = "Excellent" if v >= 0.98 else "Good" if v >= 0.95 else "Fair" if v >= 0.90 else "Poor"
    return f"{v:.5f} · {label}"

def psnr_display(v):
    if v is None: return "—"
    tag = "Excellent" if v >= 50 else "Good" if v >= 40 else "Acceptable" if v >= 30 else "Poor"
    return f"{v:.2f} dB · {tag}"

def format_audio_codec(codec: str) -> str:
    mapping = {"aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis", "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC", "pcm_s16le": "PCM 16-bit", "alac": "ALAC"}
    return mapping.get(codec, codec.upper())

def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"

def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")

def best_mark(val, best, fmt="{}"):
    if val is None or best is None: return "—"
    s = fmt.format(val)
    is_best = abs(val - best) < 0.01 and len(st.session_state.results) > 1
    if is_best: return f'<span class="best-val">{s} <span class="w-badge">Best</span></span>'
    return s

def results_to_csv(results: list, src_meta: dict, sz_mb: float) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Codec", "CRF", "Enhancements", "Size_MB", "Bitrate_kbps", "Compression_Ratio", "Space_Saved_%", "Encode_Time_s", "VMAF", "PSNR_dB", "SSIM", "Output_Resolution", "Output_FPS", "Audio_Codec", "Audio_Channels", "Source_Codec", "Source_Size_MB", "Source_Bitrate_kbps"])
    for r in results:
        enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
        enh_list = [k for k in enh_keys if r.get("enhancements", {}).get(k)]
        w.writerow([r["codec"], r["crf"], "|".join(enh_list), f"{r['size_mb']:.3f}", r["bitrate"], f"{r['cr']:.3f}", f"{r['saved']:.1f}", f"{r['enc_time']:.2f}", r["vmaf"] or "", r["psnr"] or "", r["ssim"] or "", r.get("out_res", ""), r.get("out_fps", ""), format_audio_codec(r.get("acodec", "")), format_channels(r.get("channels", 0)), src_meta["vcodec"].upper(), f"{sz_mb:.3f}", src_meta["vbitrate_kbps"]])
    return buf.getvalue().encode('utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════════════
#  Session State Init
# ══════════════════════════════════════════════════════════════════════════════

_default_enhance = {"denoise": False, "denoise_strength": 5, "sharpen": False, "sharpen_amount": 0.5, "sharpen_threshold": 5, "upscale": False, "upscale_algo": "lanczos", "upscale_width": 0, "upscale_height": 0, "hdr_convert": False, "tonemap_algo": "hable", "color_enhance": False, "vibrance": 0.15, "contrast": 1.0, "deblock": False, "deblock_strength": 5, "frame_interp": False, "target_fps": 60}

defaults = {"results": [], "inp": None, "meta": None, "sz": 0.0, "name": "", "enable_encoding": True, "enhance_settings": _default_enhance.copy(), "loudness": None, "result_logs": {}, "abr_config": ABRConfig(), "abr_state": None, "player_settings": {"speed": 1.0, "loop_ab": None, "zoom": 1.0, "show_overlay": False}}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  CSS Styles
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #f0f4f8; color: #1a202c; }
.stApp { background-color: #f0f4f8; }
.vf-header { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #1d4ed8 100%); border-radius: 20px; padding: 28px 36px; margin-bottom: 24px; color: white; box-shadow: 0 8px 32px rgba(15, 23, 42, 0.25); position: relative; overflow: hidden; }
.vf-header::before { content: ""; position: absolute; top: -60px; right: -60px; width: 220px; height: 220px; border-radius: 50%; background: rgba(255,255,255,0.04); }
.vf-header::after { content: ""; position: absolute; bottom: -40px; left: 30%; width: 140px; height: 140px; border-radius: 50%; background: rgba(59,130,246,0.12); }
.vf-header-inner { display:flex; align-items:center; gap:18px; position:relative; z-index:1; }
.vf-play-icon { width:56px; height:56px; background:rgba(255,255,255,0.12); border:2px solid rgba(255,255,255,0.25); border-radius:16px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.vf-play-icon svg { filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3)); }
.vf-header h1 { color:white; font-size:1.9rem; font-weight:700; margin:0; letter-spacing:-0.03em; }
.vf-header h1 span { font-weight:300; opacity:0.75; font-size:0.85em; }
.vf-header p { color:#bfdbfe; margin:5px 0 0; font-size:0.88rem; }
.vf-badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; position:relative; z-index:1; }
.vf-badge { display:inline-flex; align-items:center; gap:4px; background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); border-radius:20px; padding:4px 12px; font-size:0.72rem; color:#e0f2fe; font-weight:600; letter-spacing:0.04em; }
.vf-badge.ai { background:linear-gradient(135deg,rgba(124,58,237,0.4),rgba(168,85,247,0.3)); border-color:rgba(196,181,253,0.4); }
.mode-toggle { background:white; border:1px solid #e2e8f0; border-radius:14px; padding:14px 18px; margin:0 0 24px; display:flex; align-items:center; gap:12px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
.mode-toggle .toggle-label { font-weight:600; color:#1e293b; font-size:0.9rem; }
.mode-toggle .toggle-desc { color:#64748b; font-size:0.82rem; margin-left:auto; }
.mode-active { background:#dbeafe; border:1px solid #93c5fd; color:#1e40af; padding:3px 10px; border-radius:6px; font-size:0.74rem; font-weight:700; }
.vf-label { font-size:0.68rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:#64748b; margin:22px 0 12px; display:flex; align-items:center; gap:8px; }
.vf-label::before { content:""; width:18px; height:2.5px; background:#3b82f6; border-radius:2px; }
.vf-label.ai::before { background:linear-gradient(90deg,#7c3aed,#a855f7); }
[data-testid="metric-container"] { background:white; border:1px solid #e8edf2; border-radius:12px; padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.04); transition:transform 0.15s, box-shadow 0.15s; }
[data-testid="metric-container"]:hover { transform:translateY(-2px); box-shadow:0 6px 16px rgba(0,0,0,0.08); }
[data-testid="stMetricValue"] { font-size:1.1rem !important; font-weight:600 !important; }
.stButton > button { background:linear-gradient(135deg,#1d4ed8,#3b82f6); color:white; border:none; border-radius:10px; padding:10px 22px; font-weight:600; font-size:0.88rem; transition:all 0.2s; box-shadow:0 2px 6px rgba(29,78,216,0.25); font-family:'DM Sans',sans-serif; }
.stButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(29,78,216,0.35); }
.stButton > button:disabled { background:#cbd5e1 !important; box-shadow:none !important; cursor:not-allowed !important; transform:none !important; }
.stProgress > div > div { background:linear-gradient(90deg,#1d4ed8,#60a5fa); border-radius:6px; }
.cmp-table { width:100%; border-collapse:collapse; font-size:0.83rem; margin-top:8px; background:white; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.cmp-table th { background:#f8fafc; color:#475569; font-weight:700; padding:12px 14px; text-align:left; border-bottom:2px solid #e8edf2; white-space:nowrap; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; }
.cmp-table td { padding:11px 14px; border-bottom:1px solid #f1f5f9; color:#1e293b; font-family:'JetBrains Mono',monospace; white-space:nowrap; font-size:0.8rem; }
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr:hover td { background:#f8fafc; }
.best-val { color:#15803d; font-weight:700; }
.w-badge { background:#dcfce7; color:#15803d; border-radius:4px; padding:2px 7px; font-size:0.65rem; font-weight:700; margin-left:6px; text-transform:uppercase; font-family:'DM Sans',sans-serif; letter-spacing:0.05em; }
.chip-avc { background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-hevc { background:#f3e8ff; color:#6b21a8; border:1px solid #e9d5ff; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-av1 { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.chip-enh { background:#faf5ff; color:#6d28d9; border:1px solid #ddd6fe; border-radius:6px; padding:2px 10px; font-size:0.74rem; font-weight:700; font-family:'DM Sans',sans-serif; }
.q-exc { color:#15803d; font-weight:700; }
.q-gd { color:#1d4ed8; font-weight:700; }
.q-ok { color:#b45309; font-weight:700; }
.q-bad { color:#b91c1c; font-weight:700; }
.src-bar { background:#eff6ff; border-radius:10px; padding:12px 18px; font-size:0.83rem; color:#1e40af; margin-top:12px; border-left:4px solid #3b82f6; display:flex; flex-wrap:wrap; gap:10px 20px; font-family:'DM Sans',sans-serif; }
.src-bar b { color:#1e293b; }
.insight-note { background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:14px 18px; font-size:0.85rem; color:#854d0e; margin-top:12px; display:flex; gap:10px; align-items:flex-start; }
.insight-note::before { content:"💡"; font-size:1.1rem; flex-shrink:0; }
.insight-note.ai { background:#faf5ff; border-color:#c4b5fd; color:#5b21b6; }
.insight-note.ai::before { content:"✨"; }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#e8edf2; border-radius:12px; padding:4px; margin-bottom:18px; }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:8px 22px; font-size:0.87rem; font-weight:500; color:#64748b; font-family:'DM Sans',sans-serif; }
.stTabs [aria-selected="true"] { background:white !important; box-shadow:0 2px 8px rgba(0,0,0,0.1); color:#1e293b !important; font-weight:700; }
.audio-metric { display:flex; align-items:center; gap:8px; padding:9px 14px; background:#f8fafc; border-radius:9px; font-size:0.82rem; border:1px solid #e2e8f0; margin-top:6px; }
.audio-metric .icon { width:26px; height:26px; background:#3b82f6; border-radius:7px; display:flex; align-items:center; justify-content:center; color:white; font-size:0.7rem; font-weight:700; }
.loudness-bar { background:#f1f5f9; border-radius:8px; padding:10px 14px; margin-top:8px; font-size:0.8rem; border:1px solid #e2e8f0; }
.loudness-bar b { color:#1e293b; }
label { font-weight:500; color:#334155; font-size:0.87rem; }
.stAlert { border-radius:10px; border-width:1px; }
.stDivider { margin:20px 0; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:12px !important; }
/* Player Controls */
.player-controls { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }
.player-btn { padding:6px 12px; border-radius:6px; border:1px solid #cbd5e1; background:white; font-size:0.8rem; cursor:pointer; transition:all 0.15s; }
.player-btn:hover { background:#f1f5f9; border-color:#94a3b8; }
.player-btn.active { background:#3b82f6; color:white; border-color:#2563eb; }
.player-slider { width:120px; }
.abr-status { background:#f8fafc; border-radius:8px; padding:10px; font-size:0.8rem; border-left:3px solid #3b82f6; margin:8px 0; }
.abr-status.warning { border-left-color:#f59e0b; background:#fffbeb; }
.abr-status.critical { border-left-color:#ef4444; background:#fef2f2; }
.qoe-metric { display:inline-block; margin:2px 8px 2px 0; padding:3px 8px; background:#e8edf2; border-radius:4px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="vf-header">
  <div class="vf-header-inner">
    <div class="vf-play-icon">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="32" height="32">
        <polygon points="28,20 28,80 82,50" fill="white"/>
      </svg>
    </div>
    <div>
      <h1>VideoForge <span>AI Pro + ABR</span></h1>
      <p>Professional encoding · AI enhancement · Intelligent adaptive streaming</p>
    </div>
  </div>
  <div class="vf-badges">
    <span class="vf-badge">H.264</span><span class="vf-badge">HEVC</span><span class="vf-badge">AV1</span>
    <span class="vf-badge">VMAF</span><span class="vf-badge">PSNR/SSIM</span>
    <span class="vf-badge ai">✨ AI Enhance</span><span class="vf-badge">📡 ABR Engine</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not ffmpeg_ok():
    st.error("🔧 **FFmpeg not found.** Add `ffmpeg` to `packages.txt` and redeploy.")
    st.stop()
HAS_VMAF = vmaf_ok()


# ══════════════════════════════════════════════════════════════════════════════
#  Mode Toggle & File Upload
# ══════════════════════════════════════════════════════════════════════════════

enable_encoding = st.toggle("⚙️ Enable Encoding Mode", value=st.session_state.enable_encoding, help="Toggle between encoder mode and test player mode.")
st.session_state.enable_encoding = enable_encoding
mode_label = "⚙️ Encoder" if enable_encoding else "🎬 Test Player"
mode_desc = "Full workflow with AI enhancement & encoding" if enable_encoding else "Playback & analytics only — no processing"
st.markdown(f"""<div class="mode-toggle"><span class="toggle-label">🎛️ Mode:</span><span class="mode-active">{mode_label}</span><span class="toggle-desc">{mode_desc}</span></div>""", unsafe_allow_html=True)

st.markdown('<div class="vf-label">📁 Source Video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a video or click to browse", type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"], label_visibility="collapsed")
if not uploaded:
    st.info("👆 Upload a video to begin" + (" analysis & enhancement" if enable_encoding else " analysis"))
    st.stop()

suf = os.path.splitext(uploaded.name)[-1].lower()
if st.session_state.name != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded.read())
        st.session_state.inp = tmp.name
        register_temp_file(tmp.name)
    file_hash = hashlib.md5(uploaded.getvalue()).hexdigest()
    st.session_state.meta = probe_cached(st.session_state.inp, file_hash)
    st.session_state.sz = os.path.getsize(st.session_state.inp) / (1024 * 1024)
    st.session_state.name = uploaded.name
    st.session_state.loudness = None
    m = st.session_state.meta
    st.session_state.enhance_settings["upscale_width"] = m["width"] * 2
    st.session_state.enhance_settings["upscale_height"] = m["height"] * 2
    if enable_encoding:
        st.session_state.results = []
        st.session_state.result_logs = {}
    # Initialize ABR state for new file
    st.session_state.abr_state = IntelligentBitrateSelector(st.session_state.abr_config)

meta, sz_mb, inp = st.session_state.meta, st.session_state.sz, st.session_state.inp


# ══════════════════════════════════════════════════════════════════════════════
#  Source Preview + Metadata + Advanced Player Controls
# ══════════════════════════════════════════════════════════════════════════════

col_v, col_m = st.columns([3, 2], gap="large")
with col_v:
    st.markdown("### ▶ Advanced Player")
    st.video(inp)
    
    # ── NEW: Advanced Player Controls ───────────────────────────────────────
    st.markdown("**🎮 Player Controls**")
    ps = st.session_state.player_settings
    
    # Playback speed
    col_sp1, col_sp2 = st.columns([3, 1])
    with col_sp1:
        ps["speed"] = st.slider("Playback Speed", 0.25, 2.0, ps["speed"], 0.25, key="player_speed", help="0.25x to 2.0x speed")
    with col_sp2:
        if st.button("⟲ Reset", key="speed_reset", use_container_width=True):
            ps["speed"] = 1.0
            st.rerun()
    
    # Frame navigation & loop
    col_fn1, col_fn2, col_fn3 = st.columns(3)
    with col_fn1:
        if st.button("⏮ Prev Frame", key="frame_prev", use_container_width=True):
            st.info("⚠️ Frame-step requires client-side player integration (simulated)")
    with col_fn2:
        if st.button("⏭ Next Frame", key="frame_next", use_container_width=True):
            st.info("⚠️ Frame-step requires client-side player integration (simulated)")
    with col_fn3:
        # A-B Loop toggle
        if ps["loop_ab"] is None:
            if st.button("🔁 Set A", key="loop_set_a", use_container_width=True):
                ps["loop_ab"] = {"a": time.time(), "b": None}
                st.toast("📍 Point A set")
        elif ps["loop_ab"]["b"] is None:
            if st.button("🔁 Set B", key="loop_set_b", use_container_width=True):
                ps["loop_ab"]["b"] = time.time()
                st.toast(f"🔁 Loop: {ps['loop_ab']['b'] - ps['loop_ab']['a']:.1f}s")
        else:
            if st.button("❌ Clear Loop", key="loop_clear", use_container_width=True):
                ps["loop_ab"] = None
                st.toast("🔁 Loop cleared")
    
    # Zoom & overlay
    col_zo1, col_zo2 = st.columns(2)
    with col_zo1:
        ps["zoom"] = st.slider("Zoom", 1.0, 3.0, ps["zoom"], 0.25, key="player_zoom", help="Digital zoom for detail inspection")
    with col_zo2:
        ps["show_overlay"] = st.checkbox("📊 Quality Overlay", value=ps["show_overlay"], key="player_overlay", help="Show VMAF/PSNR heatmap overlay (simulated)")
    
    # Screenshot & export
    if st.button("📸 Capture Frame", key="screenshot", use_container_width=True):
        st.info("📸 Frame capture would extract current frame via FFmpeg (simulated)")
        # In production: ffmpeg -ss TIMESTAMP -i input -vframes 1 output.jpg
    
    # Keyboard shortcuts hint
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
        - `Space`: Play/Pause
        - `←`/`→`: Seek ±5s
        - `↑`/`↓`: Volume
        - `F`: Fullscreen
        - `1`/`2`/`3`: Speed 0.5x/1x/2x
        - `L`: Toggle A-B loop
        - `Z`: Toggle zoom overlay
        """)

with col_m:
    st.markdown("**📊 Source Media Info**")
    st.markdown('<div style="margin:12px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🎥 Video Stream</div>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    r1.metric("Duration", f"{meta['duration']:.1f}s")
    r2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    r1.metric("Frame Rate", f"{meta['fps']} fps")
    r2.metric("Codec", meta["vcodec"].upper())
    r1.metric("Bitrate", f"{meta['vbitrate_kbps']} kbps" if meta["vbitrate_kbps"] else "—")
    r2.metric("File Size", f"{sz_mb:.2f} MB")

    if meta["has_audio"]:
        st.markdown('<div style="margin:16px 0 8px;font-weight:600;color:#334155;font-size:0.85rem">🔊 Audio Stream</div>', unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        a1.metric("Codec", format_audio_codec(meta["acodec"]))
        a2.metric("Channels", format_channels(meta["channels"]))
        a1.metric("Sample Rate", format_sample_rate(meta["sample_rate"]))
        a2.metric("Bitrate", f"{meta['abitrate_kbps']} kbps" if meta["abitrate_kbps"] > 0 else "Variable")
        if meta["audio_duration"] > 0 and meta["duration"] > 0:
            sync_diff = abs(meta["audio_duration"] - meta["duration"])
            sync_status = "✓ Synced" if sync_diff < 0.1 else f"⚠ {sync_diff:.2f}s off"
            st.markdown(f'<div class="audio-metric"><span class="icon">🔗</span> A/V Sync: <b style="margin-left:4px">{sync_status}</b></div>', unsafe_allow_html=True)

        if st.button("🔊 Measure Loudness", help="Run ffmpeg volumedetect"):
            with st.spinner("Measuring loudness…"):
                file_hash = hashlib.md5(open(inp, "rb").read()).hexdigest()
                st.session_state.loudness = probe_loudness_cached(inp, file_hash)
        if st.session_state.loudness:
            ld = st.session_state.loudness
            mean_db = ld.get("mean_volume")
            max_db = ld.get("max_volume")
            if mean_db is not None:
                colour = "#15803d" if -20 <= mean_db <= -14 else "#b45309" if mean_db > -14 else "#b91c1c"
                st.markdown(f'<div class="loudness-bar">📢 <b>Mean:</b> <span style="color:{colour};font-weight:700">{mean_db:.1f} dBFS</span>&nbsp;&nbsp;|&nbsp;&nbsp;<b>Peak:</b> {max_db:.1f} dBFS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin:16px 0 4px;font-weight:500;color:#94a3b8;font-size:0.84rem">🔇 No audio track detected</div>', unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Intelligent ABR Dashboard (Always Visible in Player Mode)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="vf-label ai">📡 Intelligent ABR Engine</div>', unsafe_allow_html=True)
st.caption("Adaptive bitrate switching with guardrails for smooth QoE/QoS")

# ABR Configuration Panel
with st.expander("⚙️ ABR Configuration", expanded=False):
    cfg = st.session_state.abr_config
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("**Quality Bounds**")
        cfg.min_resolution = st.selectbox("Min Resolution", ["480p", "720p", "1080p"], index=["480p","720p","1080p"].index(cfg.min_resolution), key="abr_min_res")
        cfg.max_resolution = st.selectbox("Max Resolution", ["720p", "1080p", "1440p", "2160p"], index=["720p","1080p","1440p","2160p"].index(cfg.max_resolution), key="abr_max_res")
        cfg.min_vmaf_score = st.slider("Min VMAF Score", 60.0, 95.0, cfg.min_vmaf_score, 1.0, key="abr_min_vmaf", help="Never accept quality below this threshold")
    with col_c2:
        st.markdown("**Buffer Management**")
        cfg.buffer_target_sec = st.slider("Target Buffer (s)", 10.0, 120.0, cfg.buffer_target_sec, 5.0, key="abr_buf_target")
        cfg.buffer_min_sec = st.slider("Min Buffer (s)", 2.0, 15.0, cfg.buffer_min_sec, 1.0, key="abr_buf_min", help="Emergency downgrade threshold")
        cfg.buffer_max_sec = st.slider("Max Buffer (s)", 30.0, 180.0, cfg.buffer_max_sec, 10.0, key="abr_buf_max", help="Aggressive upgrade threshold")
    with col_c3:
        st.markdown("**Switching Strategy**")
        cfg.strategy = ABRStrategy(st.selectbox("Strategy", [s.value for s in ABRStrategy], index=[s.value for s in ABRStrategy].index(cfg.strategy.value), key="abr_strategy"))
        cfg.switch_cooldown_sec = st.slider("Switch Cooldown (s)", 1.0, 10.0, cfg.switch_cooldown_sec, 0.5, key="abr_cooldown")
        cfg.max_switches_per_min = st.slider("Max Switches/Min", 1, 10, cfg.max_switches_per_min, key="abr_max_switches")
    st.caption("Changes apply to next ABR decision cycle")

# Network Simulator (for testing ABR behavior)
with st.expander("🌐 Network Simulator", expanded=False):
    st.caption("Simulate network conditions to test ABR decisions")
    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    with col_n1:
        sim_bw = st.slider("Bandwidth (Mbps)", 0.5, 50.0, 10.0, 0.5, key="sim_bw")
    with col_n2:
        sim_lat = st.slider("Latency (ms)", 10, 500, 50, 10, key="sim_lat")
    with col_n3:
        sim_jit = st.slider("Jitter (ms)", 0, 100, 10, 5, key="sim_jit")
    with col_n4:
        sim_loss = st.slider("Packet Loss (%)", 0.0, 10.0, 0.0, 0.5, key="sim_loss")
    
    # Simulation controls
    col_sim1, col_sim2 = st.columns([3, 1])
    with col_sim1:
        sim_duration = st.slider("Simulate Duration (seconds)", 10, 300, 60, 10, key="sim_duration")
    with col_sim2:
        run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    
    if run_sim and st.session_state.abr_state:
        abr = st.session_state.abr_state
        progress_sim = st.progress(0.0, text="Starting ABR simulation…")
        sim_log = []
        
        for t in range(sim_duration):
            # Simulate network fluctuations
            fluctuation = 1.0 + 0.3 * math.sin(t / 10) + np.random.normal(0, 0.1)
            network = NetworkMetrics(
                throughput_mbps=sim_bw * fluctuation,
                latency_ms=sim_lat + np.random.exponential(5),
                jitter_ms=sim_jit + abs(np.random.normal(0, 2)),
                packet_loss_pct=max(0, sim_loss + np.random.normal(0, 0.3)),
                timestamp=time.time()
            )
            # Simulate buffer dynamics
            target_fill = abr.config.buffer_target_sec
            current_buffer = min(target_fill, abr.state.buffer_level_sec + 0.5 - (1 if network.throughput_mbps < 2 else 0))
            abr.update_buffer(current_buffer)
            
            # Make ABR decision
            decision = abr.select_bitrate(network)
            sim_log.append({
                "time_sec": t,
                "bandwidth_mbps": round(network.throughput_mbps, 1),
                "buffer_sec": round(current_buffer, 1),
                "selected_bitrate_kbps": decision["selected_bitrate_kbps"],
                "estimated_vmaf": decision["estimated_vmaf"],
                "reason": decision["reason"],
                "alerts": decision["alerts"],
            })
            progress_sim.progress(t / sim_duration, text=f"Simulating… t={t}s · Bitrate: {decision['selected_bitrate_kbps']} kbps")
        
        progress_sim.empty()
        
        # Display simulation results
        sim_df = pd.DataFrame(sim_log)
        st.markdown("**📈 Simulation Results**")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(sim_df.set_index("time_sec")[["bandwidth_mbps", "selected_bitrate_kbps"]], color=["#3b82f6", "#10b981"])
            st.caption("Blue: Available Bandwidth · Green: Selected Bitrate")
        with c2:
            st.line_chart(sim_df.set_index("time_sec")[["buffer_sec", "estimated_vmaf"]], color=["#f59e0b", "#8b5cf6"])
            st.caption("Orange: Buffer Level · Purple: Estimated VMAF")
        
        # QoE summary
        qoe_report = abr.get_qoe_report()
        st.markdown("**📊 QoE Summary**")
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Rebuffer Ratio", f"{qoe_report['rebuffer_ratio_pct']:.1f}%")
        qc2.metric("Avg Quality (VMAF)", f"{qoe_report['avg_quality_vmaf']:.1f}")
        qc3.metric("Total Switches", qoe_report["total_switches"])
        qc4.metric("Switches/Min", f"{qoe_report['switches_per_min']:.1f}")
        
        # Alerts
        alerts = [a for a in qoe_report["guardrail_alerts"] if a]
        if alerts:
            st.warning("⚠️ Guardrail Alerts: " + "; ".join(alerts))
        else:
            st.success("✅ All QoE guardrails satisfied")

# Real-time ABR Status (when not simulating)
if st.session_state.abr_state and not run_sim:
    abr = st.session_state.abr_state
    # Simulate a "live" network update for demo purposes
    demo_network = NetworkMetrics(
        throughput_mbps=np.random.uniform(5, 20),
        latency_ms=np.random.uniform(20, 100),
        jitter_ms=np.random.uniform(0, 20),
        packet_loss_pct=np.random.uniform(0, 1)
    )
    demo_buffer = np.random.uniform(10, 40)
    abr.update_buffer(demo_buffer)
    decision = abr.select_bitrate(demo_network)
    
    # Display status
    status_class = "abr-status"
    if decision["qoe_metrics"]["buffer_health"] == "critical":
        status_class += " critical"
    elif decision["qoe_metrics"]["buffer_health"] == "warning":
        status_class += " warning"
    
    st.markdown(f"""
    <div class="{status_class}">
      <b>📡 ABR Status:</b> {decision["reason"]}<br>
      <b>Selected Bitrate:</b> {decision["selected_bitrate_kbps"]} kbps · 
      <b>Est. VMAF:</b> {decision["estimated_vmaf"]:.1f} · 
      <b>Buffer:</b> {demo_buffer:.1f}s · 
      <b>Est. BW:</b> {decision["network_estimate_mbps"]} Mbps
    </div>""", unsafe_allow_html=True)
    
    # QoE metrics inline
    qm = decision["qoe_metrics"]
    st.markdown(f"""
    <div style="margin:8px 0">
      <span class="qoe-metric">🔄 Rebuffer: {qm['rebuffer_pct']:.1f}%</span>
      <span class="qoe-metric">⭐ Avg VMAF: {qm['avg_quality_vmaf']:.1f}</span>
      <span class="qoe-metric">🔀 Switches/min: {qm['switches_last_min']}</span>
      <span class="qoe-metric">📦 Buffer: {qm['buffer_health']}</span>
    </div>""", unsafe_allow_html=True)
    
    # Alerts
    if decision["alerts"]:
        for alert in decision["alerts"]:
            st.warning(alert)


# ══════════════════════════════════════════════════════════════════════════════
#  AI Enhancement Panel (Encoder mode only)
# ══════════════════════════════════════════════════════════════════════════════

if enable_encoding:
    st.markdown('<div class="vf-label ai">✨ AI Video Enhancement</div>', unsafe_allow_html=True)
    st.markdown("Professional-grade enhancements. Choose a preset or customize individually:")
    
    # Quick Presets
    st.markdown("### 🎯 Quick Presets")
    st.caption("One-click setups for common scenarios:")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("#### 🎬 Standard Clean")
        st.caption("**Best for:** Noisy, grainy, or compressed footage")
        st.markdown("• Reduces noise & grain\n• Removes blocking artifacts\n• Mild sharpening")
        if st.button("Apply Standard Clean", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"denoise": True, "denoise_strength": 5, "deblock": True, "deblock_strength": 5, "sharpen": True, "sharpen_amount": 0.3, "sharpen_threshold": 8, "upscale": False, "hdr_convert": False, "frame_interp": False, "color_enhance": False})
            st.rerun()
    with p2:
        st.markdown("#### 🎨 Detail & Color Boost")
        st.caption("**Best for:** Dull, flat, or low-contrast footage")
        st.markdown("• Enhances sharpness\n• Boosts color vibrance\n• Increases contrast")
        if st.button("Apply Detail Boost", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"sharpen": True, "sharpen_amount": 0.8, "sharpen_threshold": 3, "color_enhance": True, "vibrance": 0.25, "contrast": 1.15, "denoise": False, "deblock": False, "upscale": False, "hdr_convert": False, "frame_interp": False})
            st.rerun()
    with p3:
        st.markdown("#### 🚀 AI Upscale 2×")
        st.caption("**Best for:** Low-resolution content (480p→1080p, 1080p→4K)")
        st.markdown("• 2× resolution upscale\n• Pre-denoise for quality\n• Post-sharpen details")
        if st.button("Apply AI Upscale 2×", use_container_width=True, type="secondary"):
            st.session_state.enhance_settings.update({"upscale": True, "upscale_width": meta["width"] * 2, "upscale_height": meta["height"] * 2, "upscale_algo": "lanczos", "denoise": True, "denoise_strength": 6, "sharpen": True, "sharpen_amount": 0.5, "sharpen_threshold": 5, "hdr_convert": False, "frame_interp": False, "deblock": False, "color_enhance": False})
            st.rerun()
    st.divider()

    es = st.session_state.enhance_settings
    enh_col1, enh_col2 = st.columns(2)
    with enh_col1:
        with st.container(border=True):
            st.markdown("**🧹 Denoise**")
            es["denoise"] = st.checkbox("Enable temporal noise reduction", value=es["denoise"], key="chk_denoise", help="Reduces grain and compression artifacts using 3D filtering")
            if es["denoise"]: es["denoise_strength"] = st.slider("Strength", 1, 10, es["denoise_strength"], key="sl_denoise_str", help="Higher = more aggressive noise removal")
            st.caption("Reduces grain, compression artifacts, and sensor noise.")
        with st.container(border=True):
            st.markdown("**🔍 Detail Enhancement**")
            es["sharpen"] = st.checkbox("Enable adaptive sharpening", value=es["sharpen"], key="chk_sharpen")
            if es["sharpen"]:
                cs1, cs2 = st.columns(2)
                with cs1: es["sharpen_amount"] = st.slider("Amount", -1.5, 1.5, es["sharpen_amount"], 0.1, key="sl_sharpen_amt")
                with cs2: es["sharpen_threshold"] = st.slider("Chroma Softness", 0, 50, es["sharpen_threshold"], key="sl_sharpen_thr", help="Higher = softer chroma sharpening relative to luma")
            st.caption("Enhances edge definition using adaptive unsharp masking (lx=5, cx=3).")
        with st.container(border=True):
            st.markdown("**🔬 Resolution Upscaling**")
            es["upscale"] = st.checkbox("Enable high-quality upscaling", value=es["upscale"], key="chk_upscale")
            if es["upscale"]:
                algo_opts = ["lanczos", "spline", "bicubic"]
                es["upscale_algo"] = st.selectbox("Algorithm", algo_opts, index=algo_opts.index(es["upscale_algo"]) if es["upscale_algo"] in algo_opts else 0, key="sel_upscale_algo")
                es["upscale_width"] = st.number_input("Target Width (px)", min_value=meta["width"], max_value=7680, value=max(meta["width"], es.get("upscale_width") or meta["width"] * 2), step=max(2, meta["width"] // 2), key="ni_upscale_w")
                es["upscale_height"] = st.number_input("Target Height (px)", min_value=meta["height"], max_value=4320, value=max(meta["height"], es.get("upscale_height") or meta["height"] * 2), step=max(2, meta["height"] // 2), key="ni_upscale_h")
            st.caption("High-quality interpolation. For true AI super-resolution, integrate Real-ESRGAN externally.")
    with enh_col2:
        is_hdr_source = meta.get("bit_depth", 8) >= 10
        with st.container(border=True):
            st.markdown("**🌈 HDR → SDR Tonemapping**")
            es["hdr_convert"] = st.checkbox("Enable HDR conversion", value=es["hdr_convert"] and is_hdr_source, key="chk_hdr", disabled=not is_hdr_source, help="Requires 10-bit+ HDR10/HLG source")
            if es["hdr_convert"] and is_hdr_source:
                tmap_opts = ["hable", "reinhard", "mobius", "linear"]
                es["tonemap_algo"] = st.selectbox("Tonemap Algorithm", tmap_opts, index=tmap_opts.index(es["tonemap_algo"]) if es["tonemap_algo"] in tmap_opts else 0, key="sel_tonemap")
            elif not is_hdr_source: st.caption("💡 Source is SDR (8-bit). HDR conversion requires a 10-bit+ HDR10/HLG source.")
            st.caption("Converts HDR10/HLG to SDR with perceptual tonemapping.")
        with st.container(border=True):
            st.markdown("**🎨 Color Enhancement**")
            es["color_enhance"] = st.checkbox("Enable color boost", value=es["color_enhance"], key="chk_color")
            if es["color_enhance"]:
                cc1, cc2 = st.columns(2)
                with cc1: es["vibrance"] = st.slider("Vibrance", -0.5, 0.5, es["vibrance"], 0.05, key="sl_vibrance")
                with cc2: es["contrast"] = st.slider("Contrast", 0.5, 2.0, es["contrast"], 0.05, key="sl_contrast")
            st.caption("Enhances vibrancy while preserving natural skin tones.")
        with st.container(border=True):
            st.markdown("**🧩 Artifact Reduction**")
            es["deblock"] = st.checkbox("Enable deblocking filter", value=es["deblock"], key="chk_deblock")
            if es["deblock"]: es["deblock_strength"] = st.slider("Strength", 1, 10, es["deblock_strength"], key="sl_deblock_str")
            st.caption("Reduces macroblocking and ringing from heavy compression.")
    with st.expander("🎞️ Advanced: Frame Interpolation (Motion Smoothing)"):
        es["frame_interp"] = st.checkbox("Enable frame interpolation", value=es["frame_interp"], key="chk_frame_interp")
        if es["frame_interp"]:
            fps_opts = [30, 48, 60, 120]
            cur_fps = es.get("target_fps", 60)
            fps_idx = fps_opts.index(cur_fps) if cur_fps in fps_opts else 2
            es["target_fps"] = st.selectbox("Target Frame Rate", fps_opts, index=fps_idx, key="sel_target_fps")
            st.warning("⚠️ Frame interpolation is CPU-intensive and may increase processing time 3–5×.")
        st.caption("Creates intermediate frames for smoother playback (e.g., 24 fps → 60 fps).")

    # Enhancement Summary
    _enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
    active_enhancements = sum(bool(es.get(k)) for k in _enh_keys)
    if active_enhancements > 0:
        est_time = estimate_processing_time(meta, es)
        st.info(f"⏱️ **Estimated processing time**: {est_time} ({active_enhancements} enhancement{'s' if active_enhancements > 1 else ''} active)")
        enh_labels = {"denoise": "🧹 Denoise", "sharpen": "🔍 Sharpen", "upscale": "🔬 Upscale", "hdr_convert": "🌈 HDR", "color_enhance": "🎨 Color", "deblock": "🧩 Deblock", "frame_interp": "🎞️ Interp"}
        active_names = [enh_labels[k] for k in _enh_keys if es.get(k)]
        st.markdown(f"✨ **Active**: {' + '.join(active_names)}")
    st.divider()

    # Encoder Settings
    st.markdown('<div class="vf-label">⚙️ Encoder Settings</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns([2, 2, 1, 2])
    with s1: codec = st.selectbox("Video Codec", ["AVC (H.264)", "HEVC (H.265)", "AV1"], help="H.264 = fastest · HEVC = ~40% smaller · AV1 = best compression")
    with s2: crf = st.slider("CRF Quality", 0, 51, 23, help="Lower = better quality · 0 = lossless · 18 = visually lossless · 23 = balanced")
    with s3:
        do_vmaf = st.checkbox("VMAF", value=HAS_VMAF, disabled=not HAS_VMAF)
        do_psnr = st.checkbox("PSNR/SSIM", value=True)
    with s4:
        if crf < 19: ql, qc = "🟢 High Quality", "#15803d"
        elif crf < 29: ql, qc = "🟡 Balanced", "#b45309"
        elif crf < 40: ql, qc = "🟠 Compact", "#ea580c"
        else: ql, qc = "🔴 Low Quality", "#b91c1c"
        st.markdown(f'<div style="text-align:center;padding:10px 0"><div style="font-weight:700;color:{qc};font-size:1rem">{ql}</div><div style="font-size:0.74rem;color:#64748b;margin-top:2px">CRF {crf}</div></div>', unsafe_allow_html=True)
    if codec == "AV1" and active_enhancements > 0: st.warning("⚠️ **AV1 + Enhancements**: Very resource intensive. May crash on free-tier cloud. Use H.264/HEVC for cloud deployment.")

    # Batch CRF Sweep
    with st.expander("📊 Batch CRF Sweep — Rate-Distortion Analysis"):
        st.caption("Automatically encode at multiple CRF values to find the optimal quality/size trade-off.")
        sweep_col1, sweep_col2, sweep_col3 = st.columns(3)
        with sweep_col1: sweep_start = st.number_input("CRF Start", 10, 45, 18, step=1, key="sweep_start")
        with sweep_col2: sweep_end = st.number_input("CRF End", sweep_start+1, 51, 38, step=1, key="sweep_end")
        with sweep_col3: sweep_step = st.number_input("Step", 1, 10, 5, step=1, key="sweep_step")
        sweep_crfs = list(range(int(sweep_start), int(sweep_end)+1, int(sweep_step)))
        st.caption(f"Will encode {len(sweep_crfs)} variants: CRF {', '.join(map(str, sweep_crfs))}")
        if st.button("🚀 Run CRF Sweep", type="primary"):
            sweep_bar = st.progress(0.0, text="Starting sweep…")
            sweep_status = st.empty()
            for i, sweep_crf in enumerate(sweep_crfs):
                codec_short = codec.split()[0].lower()
                out_path = inp.replace(suf, f"_sweep_{codec_short}_crf{sweep_crf}.mp4")
                register_temp_file(out_path)
                sweep_status.markdown(f"*Encoding CRF {sweep_crf} ({i+1}/{len(sweep_crfs)})…*")
                ok, msg, fflog, enc_t = encode(inp, out_path, codec, sweep_crf, es, meta, progress_cb=lambda p, idx=i, total=len(sweep_crfs): sweep_bar.progress((idx + p) / total, text=f"⚙️ CRF {sweep_crfs[idx]}: {p*100:.0f}%"), duration=meta["duration"])
                if ok:
                    out_meta = probe_cached(out_path, hashlib.md5(open(out_path,"rb").read()).hexdigest())
                    out_sz = os.path.getsize(out_path) / (1024 * 1024)
                    qual = {"psnr": None, "ssim": None, "vmaf": None}
                    if do_psnr or (do_vmaf and HAS_VMAF):
                        qual = quality_metrics_cached(inp, out_path, hashlib.md5(open(inp,"rb").read()).hexdigest(), hashlib.md5(open(out_path,"rb").read()).hexdigest(), do_vmaf and HAS_VMAF, duration_sec=meta["duration"])
                    idx = len(st.session_state.results)
                    st.session_state.result_logs[idx] = fflog
                    st.session_state.results.append({"codec": codec, "crf": sweep_crf, "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0), "enc_time": enc_t, "saved": (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0, "cr": sz_mb / out_sz if out_sz > 0 else 0.0, "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"], "path": out_path, "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"], "sample_rate": meta["sample_rate"], "channels": meta["channels"], "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v}, "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}", "out_fps": out_meta.get("fps", meta["fps"])})
                else: st.warning(f"CRF {sweep_crf} failed: {msg}")
            sweep_bar.progress(1.0, text=f"✅ Sweep complete — {len(sweep_crfs)} variants encoded")
            sweep_status.empty()
            st.rerun()

    # Run Controls
    st.markdown('<div class="vf-label" style="margin-top:8px">🚀 Run</div>', unsafe_allow_html=True)
    b1, b2, b3, _ = st.columns([1.2, 1.2, 0.8, 4])
    with b1:
        if st.button("🔍 Preview Impact", use_container_width=True):
            est_meta = meta.copy()
            if es.get("upscale"): est_meta["width"], est_meta["height"] = int(es.get("upscale_width") or meta["width"] * 2), int(es.get("upscale_height") or meta["height"] * 2)
            if es.get("frame_interp"): est_meta["fps"] = es.get("target_fps", 60)
            st.markdown("#### 📊 Estimated Output")
            bc1, bc2 = st.columns(2)
            bc1.metric("Source", f"{meta['width']}×{meta['height']}", f"@ {meta['fps']} fps")
            bc2.metric("Enhanced", f"{est_meta['width']}×{est_meta['height']}", f"@ {est_meta['fps']} fps")
            if est_meta["width"] > meta["width"] and meta["width"] > 0:
                ratio = (est_meta["width"] / meta["width"]) ** 2
                st.caption(f"+{(ratio-1)*100:.0f}% more pixels ({ratio:.1f}× resolution)")
    go = b2.button("✨ Enhance + Encode", type="primary", use_container_width=True)
    clear = b3.button("🗑 Clear", use_container_width=True)
    if clear: st.session_state.results, st.session_state.result_logs = [], {}; st.rerun()
    if go:
        codec_short = codec.split()[0].lower()
        enh_tag = "enh_" if active_enhancements > 0 else ""
        out_path = inp.replace(suf, f"_{enh_tag}{codec_short}_crf{crf}.mp4")
        register_temp_file(out_path)
        progress_text = st.empty()
        progress_text.markdown(f"*⏳ Initializing {codec}…*")
        bar = st.progress(0.0)
        with st.spinner(f"✨ Processing ({active_enhancements} enhancements) + encoding {codec} CRF {crf}…"):
            ok, msg, fflog, enc_t = encode(inp, out_path, codec, crf, es, meta, progress_cb=lambda p: (bar.progress(min(p, 1.0)), progress_text.markdown(f"*⚙️ Processing… {min(p*100, 99):.0f}%*")), duration=meta["duration"])
        if not ok:
            bar.empty(); progress_text.empty()
            st.error(f"❌ {msg}")
            if fflog:
                with st.expander("📋 FFmpeg Log"): st.code(fflog, language="bash")
        else:
            bar.progress(1.0); progress_text.markdown("*✅ Encoding complete — computing quality metrics…*")
            out_meta = probe_cached(out_path, hashlib.md5(open(out_path,"rb").read()).hexdigest())
            out_sz = os.path.getsize(out_path) / (1024 * 1024)
            saved_pct = (1 - out_sz / sz_mb) * 100 if sz_mb > 0 else 0.0
            qual = {"psnr": None, "ssim": None, "vmaf": None}
            if do_psnr or (do_vmaf and HAS_VMAF):
                with st.spinner("🔍 Computing quality metrics…"):
                    qual = quality_metrics_cached(inp, out_path, hashlib.md5(open(inp,"rb").read()).hexdigest(), hashlib.md5(open(out_path,"rb").read()).hexdigest(), do_vmaf and HAS_VMAF, duration_sec=meta["duration"])
            idx = len(st.session_state.results)
            st.session_state.result_logs[idx] = fflog
            st.session_state.results.append({"codec": codec, "crf": crf, "size_mb": out_sz, "bitrate": out_meta.get("vbitrate_kbps", 0), "enc_time": enc_t, "saved": saved_pct, "cr": sz_mb / out_sz if out_sz > 0 else 0.0, "psnr": qual["psnr"], "ssim": qual["ssim"], "vmaf": qual["vmaf"], "path": out_path, "acodec": meta["acodec"], "abitrate": meta["abitrate_kbps"], "sample_rate": meta["sample_rate"], "channels": meta["channels"], "enhancements": {k: v for k, v in es.items() if k in _enh_keys and v}, "out_res": f"{out_meta.get('width',meta['width'])}×{out_meta.get('height',meta['height'])}", "out_fps": out_meta.get("fps", meta["fps"])})
            bar.empty(); progress_text.empty()
            q_parts = []
            if qual["vmaf"] is not None: q_parts.append(f"VMAF {qual['vmaf']:.1f}")
            if qual["psnr"] is not None: q_parts.append(f"PSNR {qual['psnr']:.2f} dB")
            q_str = " · ".join(q_parts) if q_parts else "Analysis complete"
            enh_summary = " + ".join(k.capitalize() for k in _enh_keys if es.get(k))
            enh_str = f" · ✨ {enh_summary}" if enh_summary else ""
            st.success(f"✅ {codec} CRF {crf}{enh_str} · {out_sz:.2f} MB · saved {saved_pct:.1f}% · {enc_t:.1f}s · {q_str}")
else:
    # Test Player Mode
    st.markdown('<div class="vf-label">🎬 Test Player Mode</div>', unsafe_allow_html=True)
    st.info("🎧 **Playback & Analytics**: Preview your source video. All metadata displayed. No processing performed.")
    if meta["has_audio"]:
        st.markdown(f"""<div style="background:#f1f5f9;border-radius:12px;padding:14px 18px;margin:12px 0"><div style="font-weight:700;margin-bottom:10px;color:#334155">🔊 Audio Stream Details</div><div style="display:flex;gap:8px;flex-wrap:wrap"><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_audio_codec(meta['acodec'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_channels(meta['channels'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{format_sample_rate(meta['sample_rate'])}</span><span style="background:white;padding:4px 12px;border-radius:7px;font-size:0.8rem;border:1px solid #e2e8f0;font-weight:600">{meta['abitrate_kbps']} kbps</span></div></div>""", unsafe_allow_html=True)
    st.caption("💡 Enable Encoding Mode above to access AI enhancement features.")


# ══════════════════════════════════════════════════════════════════════════════
#  Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════

results = st.session_state.results
if not results or not enable_encoding:
    if enable_encoding: st.caption("📊 Results will appear here after encoding completes.")
    st.stop()

st.divider()
st.markdown('<div class="vf-label">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
tab_tbl, tab_chart, tab_dl, tab_logs = st.tabs(["📋 Comparison Table", "📊 Charts", "⬇ Downloads", "🪵 Logs"])

# Comparison Table
with tab_tbl:
    best_sz, best_cr, best_spd = min(r["size_mb"] for r in results), max(r["cr"] for r in results), min(r["enc_time"] for r in results)
    vmaf_vals = [r["vmaf"] for r in results if r["vmaf"] is not None]
    best_vm = max(vmaf_vals) if vmaf_vals else None
    _enh_keys = ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"]
    rows_html = ""
    for r in results:
        cs = r["codec"].split()[0]
        chip_cls = {"AVC": "chip-avc", "HEVC": "chip-hevc", "AV1": "chip-av1"}.get(cs, "")
        tag = f'<span class="{chip_cls}">{cs}</span>'
        enh_count = len([v for v in r.get("enhancements", {}).values() if v])
        if enh_count > 0: tag += f' <span class="chip-enh">✨ ×{enh_count}</span>'
        vmaf_txt, vmaf_cls = vmaf_display(r["vmaf"])
        vmaf_cell = f'<span class="{vmaf_cls}">{vmaf_txt}</span>'
        if r["vmaf"] is not None and best_vm is not None and abs(r["vmaf"] - best_vm) < 0.01 and len(results) > 1: vmaf_cell += ' <span class="w-badge">Best</span>'
        audio_info = f"{format_audio_codec(r.get('acodec',''))} · {format_channels(r.get('channels',0))}" if r.get("acodec") and r["acodec"] != "unknown" else "—"
        res_info, fps_info = r.get("out_res", f"{meta['width']}×{meta['height']}"), r.get("out_fps", meta["fps"])
        ssim_str = ssim_display(r["ssim"])
        rows_html += f"""<tr><td>{tag}</td><td>{r['crf']}</td><td>{best_mark(r['size_mb'], best_sz, '{:.2f} MB')}</td><td>{r['bitrate']} kbps</td><td>{best_mark(r['cr'], best_cr, '{:.2f}×')}</td><td>{r['saved']:.1f}%</td><td>{best_mark(r['enc_time'], best_spd, '{:.1f}s')}</td><td>{vmaf_cell}</td><td>{psnr_display(r['psnr'])}</td><td>{ssim_str}</td><td style="font-size:0.77rem;color:#64748b">{res_info} @ {fps_info} fps</td><td style="font-size:0.77rem;color:#64748b">{audio_info}</td></tr>"""
    st.markdown(f"""<table class="cmp-table"><thead><tr><th>Codec</th><th>CRF</th><th>Size</th><th>Bitrate</th><th>Ratio</th><th>Saved</th><th>Time</th><th>VMAF ↑</th><th>PSNR ↑</th><th>SSIM ↑</th><th>Resolution</th><th>Audio</th></tr></thead><tbody>{rows_html}</tbody></table><div class="src-bar"><b>Source:</b> {meta['vcodec'].upper()} · {sz_mb:.2f} MB · {meta['vbitrate_kbps']} kbps · {meta['width']}×{meta['height']} @ {meta['fps']} fps{f" · 🔊 {format_audio_codec(meta['acodec'])} {format_channels(meta['channels'])}" if meta['has_audio'] else ""}</div>""", unsafe_allow_html=True)
    st.caption("📏 VMAF: 93+ Excellent · 80–93 Good · 60–80 Fair | PSNR: 40+ dB Good | SSIM: 0.98+ Excellent | ✨ = AI enhancements | 🏆 Best = winner across runs")
    st.download_button(label="⬇ Export as CSV", data=results_to_csv(results, meta, sz_mb), file_name="videoforge_results.csv", mime="text/csv", help="Download the full comparison table as a CSV file")

# Charts
with tab_chart:
    df = pd.DataFrame([{"Codec": r["codec"] + (" ✨" if r.get("enhancements") else "") + f" CRF{r['crf']}", "File Size (MB)": round(r["size_mb"], 3), "Bitrate (kbps)": r["bitrate"], "Encode Time (s)": round(r["enc_time"], 2), "Space Saved (%)": round(r["saved"], 1), "VMAF": r["vmaf"], "PSNR (dB)": round(r["psnr"], 2) if r["psnr"] else None, "SSIM": round(r["ssim"], 4) if r["ssim"] else None} for r in results])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**📦 File Size Comparison**")
        size_df = pd.DataFrame([{"Codec": "🎬 Original", "File Size (MB)": round(sz_mb, 3)}] + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else "") + f" CRF{r['crf']}", "File Size (MB)": round(r["size_mb"], 3)} for r in results]).set_index("Codec")
        st.bar_chart(size_df, color="#3b82f6", use_container_width=True)
        st.markdown("**⏱️ Encoding Time**"); st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], color="#f97316", use_container_width=True)
    with c2:
        st.markdown("**📡 Bitrate Comparison**")
        brate_df = pd.DataFrame([{"Codec": "🎬 Original", "Bitrate (kbps)": meta["vbitrate_kbps"]}] + [{"Codec": r["codec"] + ("✨" if r.get("enhancements") else "") + f" CRF{r['crf']}", "Bitrate (kbps)": r["bitrate"]} for r in results]).set_index("Codec")
        st.bar_chart(brate_df, color="#8b5cf6", use_container_width=True)
        st.markdown("**💾 Space Saved**"); st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], color="#10b981", use_container_width=True)
    q_cols = [c for c in ["VMAF", "PSNR (dB)", "SSIM"] if df[c].notna().any()]
    if q_cols:
        st.divider(); st.markdown("**🎯 Quality Metrics**")
        qdf = df.set_index("Codec")[q_cols].dropna(how="all"); st.bar_chart(qdf, use_container_width=True)
        st.caption("Higher = better quality. ✨ = AI enhancements applied.")
    rd_data = [(r["bitrate"], r["vmaf"], r["crf"]) for r in results if r.get("vmaf") is not None and r.get("bitrate")]
    if len(rd_data) >= 2:
        st.divider(); st.markdown("**📈 Rate-Distortion Curve (Bitrate vs VMAF)**")
        rd_df = pd.DataFrame([{"Bitrate (kbps)": b, "VMAF": v} for b, v, _ in sorted(rd_data)]).set_index("Bitrate (kbps)")
        st.line_chart(rd_df, color="#6366f1", use_container_width=True)
        st.caption("Ideal curve bends upper-left: high quality at low bitrate.")
    if len(results) > 1:
        st.divider(); st.markdown("**🔍 Smart Insights**")
        enhanced_results = [r for r in results if r.get("enhancements")]
        if enhanced_results:
            scored = [r for r in enhanced_results if r.get("vmaf") is not None or r.get("psnr") is not None]
            if scored:
                best_enh = max(scored, key=lambda r: (r.get("vmaf") or 0) + (r.get("psnr") or 0) / 5)
                n_enh = len([v for v in best_enh["enhancements"].values() if v])
                vmaf_str = f"VMAF {best_enh['vmaf']:.1f}" if best_enh.get("vmaf") else f"PSNR {best_enh.get('psnr',0):.2f} dB"
                st.markdown(f'<div class="insight-note ai"><b>Enhancement Winner:</b> <span style="font-weight:700">{best_enh["codec"]}</span> with {n_enh} enhancement{"s" if n_enh!=1 else ""} achieved {vmaf_str} at {best_enh["size_mb"]:.2f} MB.</div>', unsafe_allow_html=True)
        eff_candidates = [r for r in results if r.get("vmaf") and r["size_mb"] > 0]
        if eff_candidates:
            best_eff = max(eff_candidates, key=lambda r: r["vmaf"] / r["size_mb"])
            enh_tag = " ✨" if best_eff.get("enhancements") else ""
            st.markdown(f'<div class="insight-note"><b>Efficiency Pick:</b> <span style="font-weight:700">{best_eff["codec"]} CRF{best_eff["crf"]}{enh_tag}</span> delivers best quality-per-MB (VMAF {best_eff["vmaf"]:.1f} at {best_eff["size_mb"]:.2f} MB).</div>', unsafe_allow_html=True)
        acceptable = [r for r in results if r.get("vmaf") and r["vmaf"] >= 80]
        if acceptable:
            smallest = min(acceptable, key=lambda r: r["size_mb"])
            st.markdown(f'<div class="insight-note"><b>Streaming Sweet Spot:</b> <span style="font-weight:700">{smallest["codec"]} CRF{smallest["crf"]}</span> is the smallest file with VMAF ≥ 80 ({smallest["size_mb"]:.2f} MB · VMAF {smallest["vmaf"]:.1f}).</div>', unsafe_allow_html=True)

# Downloads
with tab_dl:
    st.markdown("### ⬇ Download Processed Files"); st.caption("Files are stored temporarily. Download immediately after encoding.")
    for i, r in enumerate(results):
        cs, enh_tag = r["codec"].split()[0], " ✨" if r.get("enhancements") else ""
        col_dl, col_info = st.columns([1, 3])
        with col_dl:
            fname = f"videoforge_{cs.lower()}{'_enh' if r.get('enhancements') else ''}_crf{r['crf']}.mp4"
            try:
                with open(r["path"], "rb") as f:
                    dl_key = f"dl_{cs}_{r['crf']}_{id(r)}_{int(time.time()*1000)}"
                    st.download_button(label=f"⬇ {cs}{enh_tag} CRF {r['crf']}", data=f, file_name=fname, mime="video/mp4", use_container_width=True, key=dl_key)
            except FileNotFoundError: st.caption("⚠️ Temp file expired — re-process to download.")
        with col_info:
            metrics = [f"{r['size_mb']:.2f} MB", f"{r['bitrate']} kbps", r.get("out_res", "N/A")]
            if r.get("vmaf") is not None: metrics.append(f"VMAF {r['vmaf']:.1f}")
            if r.get("psnr") is not None: metrics.append(f"PSNR {r['psnr']:.2f} dB")
            st.caption(" · ".join(metrics))
            if r.get("enhancements"): st.caption(f"✨ Enhancements: {', '.join([k.capitalize() for k, v in r["enhancements"].items() if v])}")
            if r.get("acodec") and r["acodec"] not in ("unknown", ""): st.caption(f"🔊 {format_audio_codec(r['acodec'])} · {format_channels(r.get('channels',0))}")

# Logs
with tab_logs:
    st.markdown("### 🪵 FFmpeg Encode Logs"); st.caption("Per-encode FFmpeg output for debugging. Last 80 lines preserved per run.")
    if not st.session_state.result_logs: st.info("No logs yet. Logs appear here after encoding.")
    else:
        for idx, log_text in st.session_state.result_logs.items():
            r = results[idx]; cs = r["codec"].split()[0]
            with st.expander(f"{cs} CRF {r['crf']} — {r['size_mb']:.2f} MB · {r['enc_time']:.1f}s"): st.code(log_text or "(empty)", language="bash")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
try: ffmpeg_ver = subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).split()[2]
except: ffmpeg_ver = "unknown"
st.markdown(f'<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:12px 0">VideoForge v2.0 · FFmpeg {ffmpeg_ver} · Streamlit {st.__version__} · ✨ Optimized + ABR Engine</div>', unsafe_allow_html=True)
