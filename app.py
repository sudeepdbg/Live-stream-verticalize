from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(page_title="Dual Flow Vertical Live → Cloudflare Stream", page_icon="📱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
.hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;}
.chip,.chip-ok,.chip-warn,.chip-bad {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; margin-right:6px; margin-top:6px;}
.chip {border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1;}
.chip-ok {border:1px solid rgba(34,197,94,.35); background:rgba(34,197,94,.10); color:#4ade80;}
.chip-warn {border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.10); color:#fbbf24;}
.chip-bad {border:1px solid rgba(239,68,68,.35); background:rgba(239,68,68,.10); color:#f87171;}
.panel-box {border:1px solid rgba(99,102,241,.35); border-radius:14px; padding:14px 16px; background:rgba(99,102,241,.06); margin-top:8px;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _stat(stats: dict, *keys: str, default: Any = 0) -> Any:
    for key in keys:
        if key in stats and stats.get(key) is not None:
            return stats.get(key)
    return default


def _fmt_ms(v: Any) -> str:
    return "-" if v is None else f"{_num(v):.1f} ms"


def _fmt_sec(v: Any) -> str:
    return "-" if v is None else f"{_num(v):.2f}s"


def _fmt_pct(v: Any) -> str:
    return "-" if v is None else f"{_num(v):.1f}%"


def _chip(label: str, level: str = "ok") -> str:
    klass = {"ok": "chip-ok", "warn": "chip-warn", "bad": "chip-bad", "neutral": "chip"}.get(level, "chip")
    return f"<span class='{klass}'>{label}</span>"


def _safe_stats(session) -> dict:
    if not session:
        return {}
    try:
        if hasattr(backend, "_stats_snapshot"):
            return backend._stats_snapshot(session)
    except Exception:
        pass
    lock = getattr(session, "stats_lock", None)
    if lock:
        with lock:
            return dict(getattr(session, "stats", {}) or {})
    return dict(getattr(session, "stats", {}) or {})


def _valid_session(session) -> bool:
    return bool(session and getattr(session, "uid", None) and getattr(session, "hls_url", None) and getattr(session, "iframe_url", None) and getattr(session, "log_path", None) and hasattr(session, "stop_event"))


def _append_snapshot(session) -> None:
    if not session:
        return
    uid = getattr(session, "uid", "unknown") or "unknown"
    stats = _safe_stats(session)
    if st.session_state.analytics_uid != uid:
        st.session_state.analytics_uid = uid
        st.session_state.analytics_history = []
        st.session_state.analytics_seen_keys = []
    ts_ms = _int(_stat(stats, "updated_at_ms", default=int(time.time() * 1000)))
    key = f"{uid}:{ts_ms}:{_int(_stat(stats, 'frames_out', default=0))}:{_int(_stat(stats, 'frames_in', default=0))}:{getattr(session, 'status', 'unknown')}"
    if key in st.session_state.analytics_seen_keys:
        return
    snap = {
        "ts_ms": ts_ms,
        "time": time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000.0)),
        "status": getattr(session, "status", "unknown"),
        "health": _stat(stats, "health", default="-"),
        "pipeline_arch": _stat(stats, "pipeline_arch", default="-"),
        "buffer_policy": _stat(stats, "buffer_policy", default="-"),
        "frames_in": _int(_stat(stats, "frames_in", default=0)),
        "frames_processed": _int(_stat(stats, "frames_processed", default=0)),
        "frames_out": _int(_stat(stats, "frames_out", default=0)),
        "frames_repeated": _int(_stat(stats, "frames_repeated", default=0)),
        "frames_dropped_input": _int(_stat(stats, "frames_dropped_input", default=0)),
        "frames_dropped_processing": _int(_stat(stats, "frames_dropped_processing", default=0)),
        "fps_in": _num(_stat(stats, "fps_in", "ingest_fps_1s", default=0)),
        "fps_process": _num(_stat(stats, "fps_process", "process_fps_1s", default=0)),
        "fps_out": _num(_stat(stats, "fps_out", "output_fps_1s", "fps_actual", default=0)),
        "processing_ms": _num(_stat(stats, "processing_ms", "avg_process_ms", default=0)),
        "p95_process_ms": _num(_stat(stats, "p95_process_ms", default=0)),
        "read_ms": _num(_stat(stats, "read_ms", "avg_ingest_read_ms", default=0)),
        "p95_read_ms": _num(_stat(stats, "p95_ingest_read_ms", default=0)),
        "write_ms": _num(_stat(stats, "write_ms", "avg_output_write_ms", default=0)),
        "p95_write_ms": _num(_stat(stats, "p95_output_write_ms", default=0)),
        "drift_ms": _num(_stat(stats, "avg_schedule_drift_ms", default=0)),
        "p95_drift_ms": _num(_stat(stats, "p95_schedule_drift_ms", default=0)),
        "buffer_len": _int(_stat(stats, "buffer_len", default=0)),
        "buffer_seconds": _num(_stat(stats, "buffer_seconds", "buffer_seconds_est", default=0)),
        "buffer_fill_pct": _num(_stat(stats, "buffer_fill_pct", default=0)),
        "latest_frame_age_ms": _num(_stat(stats, "latest_frame_age_ms", default=0)),
        "startup_buffer_fill_frames": _int(_stat(stats, "startup_buffer_fill_frames", default=0)),
        "output_underruns": _int(_stat(stats, "output_underruns", default=0)),
        "frame_drops": _int(_stat(stats, "frame_drops", "input_drop_count", default=0)),
        "source_stalls": _int(_stat(stats, "source_stalls", default=0)),
        "consecutive_source_stalls": _int(_stat(stats, "consecutive_source_stalls", default=0)),
        "ingest_restarts": _int(_stat(stats, "ingest_restarts", default=0)),
        "write_failures": _int(_stat(stats, "write_failures", "output_write_failures", default=0)),
        "ball_confidence": _num(_stat(stats, "ball_confidence", default=0)),
        "panel_active_faces": _int(_stat(stats, "panel_active_faces", default=0)),
        "panel_detector": _stat(stats, "panel_detector", default="-"),
        "mode": _stat(stats, "mode", default="-"),
        "ingest_read_timeout": _num(_stat(stats, "ingest_read_timeout", default=0)),
        "ffmpeg_alive": bool(_stat(stats, "ffmpeg_alive", default=False)),
        "ingest_alive": bool(_stat(stats, "ingest_alive", default=False)),
    }
    st.session_state.analytics_history.append(snap)
    st.session_state.analytics_history = st.session_state.analytics_history[-300:]
    st.session_state.analytics_seen_keys.append(key)
    st.session_state.analytics_seen_keys = st.session_state.analytics_seen_keys[-500:]


def _render_health(stats: dict, status: str) -> None:
    fps_target = _num(_stat(stats, "fps", default=backend.DEFAULT_OUTPUT_FPS), backend.DEFAULT_OUTPUT_FPS)
    fps_out = _num(_stat(stats, "fps_out", "output_fps_1s", "fps_actual", default=0))
    p95_proc = _num(_stat(stats, "p95_process_ms", "processing_ms", default=0))
    p95_write = _num(_stat(stats, "p95_output_write_ms", "write_ms", default=0))
    stalls = _int(_stat(stats, "source_stalls", default=0))
    drops = _int(_stat(stats, "frame_drops", "input_drop_count", default=0))
    repeats = _int(_stat(stats, "frames_repeated", default=0))
    health = str(_stat(stats, "health", default="-"))
    chips = [
        _chip(status.replace("_", " ").title(), "ok" if status == "streaming" else "warn" if status in {"priming_output", "connecting_source", "buffering"} else "bad"),
        _chip(f"Health: {health}", "ok" if health == "healthy" else "warn" if health in {"running", "source_stalls_detected", "processing_tail_latency_high"} else "bad"),
        _chip("Output cadence healthy", "ok") if fps_out >= fps_target * 0.90 else _chip("Output FPS low", "bad"),
        _chip("RTMPS write healthy", "ok") if p95_write <= 20 else _chip("RTMPS backpressure", "bad"),
        _chip("Source healthy", "ok") if stalls == 0 else _chip("Source ingest unstable", "bad"),
        _chip("Repeating frames", "warn") if repeats > 0 else _chip("No repeats yet", "ok"),
        _chip("Dropping to keep live", "warn") if drops > 0 else _chip("No drops", "ok"),
        _chip("Processing tail high", "warn") if p95_proc > 80 else _chip("Processing tail ok", "ok"),
    ]
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def _render_analytics(session) -> None:
    stats = _safe_stats(session)
    status = getattr(session, "status", "unknown")
    st.markdown("### 6) Live analytics & tuning")
    _render_health(stats, status)
    c = st.columns(4)
    c[0].metric("Output FPS", f"{_num(_stat(stats, 'fps_out', 'output_fps_1s', 'fps_actual', default=0)):.1f}")
    c[1].metric("Process FPS", f"{_num(_stat(stats, 'fps_process', 'process_fps_1s', default=0)):.1f}")
    c[2].metric("Ingest FPS", f"{_num(_stat(stats, 'fps_in', 'ingest_fps_1s', default=0)):.1f}")
    c[3].metric("Pipeline", _stat(stats, "pipeline_arch", default="-"))
    c = st.columns(4)
    c[0].metric("P95 process", _fmt_ms(_stat(stats, "p95_process_ms", default=0)))
    c[1].metric("Latest frame age", _fmt_ms(_stat(stats, "latest_frame_age_ms", default=0)))
    c[2].metric("Repeated frames", _int(_stat(stats, "frames_repeated", default=0)))
    c[3].metric("Frame drops", _int(_stat(stats, "frame_drops", "input_drop_count", default=0)))
    c = st.columns(4)
    c[0].metric("Input drops", _int(_stat(stats, "frames_dropped_input", default=0)))
    c[1].metric("Processing drops", _int(_stat(stats, "frames_dropped_processing", default=0)))
    c[2].metric("Buffer", _fmt_sec(_stat(stats, "buffer_seconds", "buffer_seconds_est", default=0)))
    c[3].metric("Buffer fill", _fmt_pct(_stat(stats, "buffer_fill_pct", default=0)))
    c = st.columns(4)
    c[0].metric("P95 write", _fmt_ms(_stat(stats, "p95_output_write_ms", default=0)))
    c[1].metric("P95 drift", _fmt_ms(_stat(stats, "p95_schedule_drift_ms", default=0)))
    c[2].metric("Source stalls", _int(_stat(stats, "source_stalls", default=0)))
    c[3].metric("Ingest restarts", _int(_stat(stats, "ingest_restarts", default=0)))
    c = st.columns(4)
    c[0].metric("Ball confidence", f"{_num(_stat(stats, 'ball_confidence', default=0)):.2f}")
    c[1].metric("Mode", _stat(stats, "mode", default="-"))
    c[2].metric("Panel faces", _int(_stat(stats, "panel_active_faces", default=0)))
    c[3].metric("Panel detector", _stat(stats, "panel_detector", default="-"))
    if st.session_state.analytics_history:
        df = pd.DataFrame(st.session_state.analytics_history)
        tabs = st.tabs(["Pipeline charts", "Buffers & counters", "Detection", "Raw stats"])
        with tabs[0]:
            c1, c2 = st.columns(2)
            c1.line_chart(df.set_index("time")[["fps_in", "fps_process", "fps_out"]], height=260)
            c2.line_chart(df.set_index("time")[["processing_ms", "p95_process_ms", "write_ms", "p95_write_ms"]], height=260)
        with tabs[1]:
            cols = [x for x in ["buffer_len", "buffer_seconds", "buffer_fill_pct", "latest_frame_age_ms"] if x in df.columns]
            st.line_chart(df.set_index("time")[cols], height=240)
            cols = [x for x in ["frames_repeated", "frame_drops", "frames_dropped_input", "frames_dropped_processing", "output_underruns", "source_stalls"] if x in df.columns]
            st.line_chart(df.set_index("time")[cols], height=240)
        with tabs[2]:
            st.line_chart(df.set_index("time")[["ball_confidence", "panel_active_faces"]], height=260)
        with tabs[3]:
            st.dataframe(df.tail(80), use_container_width=True, hide_index=True)
            st.json(stats)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, value in {
    "input_path": None, "meta": None, "reframed_path": None, "live_session": None,
    "cf_account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
    "cf_api_token": os.getenv("CLOUDFLARE_STREAM_API_TOKEN", ""),
    "cf_customer_code": os.getenv("CLOUDFLARE_STREAM_CUSTOMER_CODE", ""),
    "cf_low_latency": False,
    "analytics_history": [], "analytics_uid": None, "analytics_seen_keys": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = value
if not isinstance(st.session_state.analytics_seen_keys, list):
    st.session_state.analytics_seen_keys = list(st.session_state.analytics_seen_keys)[-300:]
if not _valid_session(st.session_state.get("live_session")):
    st.session_state.live_session = None

st.markdown("""
<div class='hero'><div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
<div><div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>Dual Flow Vertical Live → Cloudflare Stream</div>
<div style='font-size:.98rem; color:#cbd5e1;'>Smooth micro-buffer live architecture: immediate start, fixed output clock, bounded FIFO, low latency, and richer analytics.</div></div>
<div><span class='chip'>smooth micro-buffer</span><span class='chip'>sports live tuned</span><span class='chip'>thread-safe analytics</span><span class='chip'>Cloudflare</span></div>
</div></div>
""", unsafe_allow_html=True)

if not backend.ffmpeg_ok():
    st.error("FFmpeg / ffprobe not found. Add FFmpeg to your runtime and redeploy.")
    st.stop()

# Credentials
st.subheader("1) Cloudflare Stream credentials")
creds_set = bool(st.session_state.cf_account_id and st.session_state.cf_api_token and st.session_state.cf_customer_code)
if creds_set:
    masked = st.session_state.cf_account_id[-6:] if len(st.session_state.cf_account_id) > 6 else "***"
    st.markdown(f"<span class='chip-ok'>✓ Credentials saved</span> <span class='chip'>Account: ...{masked}</span>", unsafe_allow_html=True)
with st.expander("Edit credentials" if creds_set else "Enter credentials", expanded=not creds_set):
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.cf_account_id = st.text_input("Cloudflare account ID", value=st.session_state.cf_account_id)
        st.session_state.cf_api_token = st.text_input("Cloudflare Stream API token", value=st.session_state.cf_api_token, type="password")
    with c2:
        st.session_state.cf_customer_code = st.text_input("Cloudflare Stream customer code", value=st.session_state.cf_customer_code, help="Enter only the code, not the full domain.")
        st.session_state.cf_low_latency = st.checkbox("Prefer LL-HLS where available", value=st.session_state.cf_low_latency)
cf_cfg = None
if st.session_state.cf_account_id and st.session_state.cf_api_token and st.session_state.cf_customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(st.session_state.cf_account_id, st.session_state.cf_api_token, st.session_state.cf_customer_code, st.session_state.cf_low_latency)
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning("Fill in Account ID, Stream API token, and customer code to continue.")

# Source/workflow
st.subheader("2) Workflow and source")
workflow = st.radio("Workflow", ["VOD → Live (full-file verticalize first)", "Realtime live vertical push"], horizontal=True)
source_kind = st.radio("Source type", ["Upload file", "RTMP URL", "SRT URL", "Local path / arbitrary URL"], horizontal=True)
source_value = None
uploaded_name = "source"
if source_kind == "Upload file":
    upl = st.file_uploader(f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)", type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"])
    if upl:
        if getattr(upl, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {upl.size / (1024*1024):.1f} MB — keep it <= {backend.MAX_UPLOAD_MB} MB.")
            st.stop()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upl.name)[-1].lower()) as tmp:
            tmp.write(upl.read())
            st.session_state.input_path = tmp.name
        uploaded_name = upl.name
        source_value = st.session_state.input_path
elif source_kind == "RTMP URL":
    source_value = st.text_input("RTMP source URL", placeholder="rtmp://host/app/stream")
elif source_kind == "SRT URL":
    source_value = st.text_input("SRT source URL", placeholder="srt://host:port?mode=caller")
else:
    source_value = st.text_input("Local file path or arbitrary URL", placeholder="/mount/src/file.mp4 or https://example.com/file.mp4")

# Settings
st.subheader("3) Reframing settings")
left, right = st.columns(2)
with left:
    target_w = st.selectbox("Vertical output width", [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))
    st.caption(f"Output: {target_w} × {target_h}")
with right:
    delay_seconds = st.slider("Live delay / micro-buffer (seconds)", 0.0, 2.0, 0.0, 0.25) if workflow == "Realtime live vertical push" else 0.0
    loop_file = st.checkbox("Loop file source when it ends", value=True)

s1, s2, s3, s4 = st.columns(4)
smooth_strength = s1.slider("Smoothness", 0.90, 0.995, 0.975, 0.005)
analysis_stride = s2.slider("Analysis stride", 2, 10, 6, 1, help="Use 6 for smoother live sports; increase if p95 process is high.")
deadzone_ratio = s3.slider("Deadzone ratio", 0.02, 0.10, 0.05, 0.01)
max_pan_ratio = s4.slider("Max pan ratio", 0.005, 0.03, 0.012, 0.001)

m1, m2, m3 = st.columns(3)
sport_profile = m1.selectbox("Sport profile", ["auto", "soccer", "basketball", "cricket"], index=0)
panel_mode = st.checkbox("Enable panel discussion mode", value=False)
ball_tracking = m2.checkbox("Ball tracking", value=(not panel_mode), disabled=panel_mode)
if panel_mode: ball_tracking = False
overlay_composite = m2.checkbox("Overlay composite", value=True)
preserve_bottom_overlay = m3.checkbox("Preserve bottom overlay", value=True)
auto_mode = m3.checkbox("Auto mode detection", value=False, disabled=panel_mode)
if panel_mode: auto_mode = False
panel_max_faces, panel_detection_stride, panel_gap = 4, 2, 4
if panel_mode:
    st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    panel_max_faces = p1.slider("Max faces", 1, 4, 4, 1)
    panel_detection_stride = p2.slider("Detection stride", 1, 8, 3, 1)
    panel_gap = p3.slider("Panel gap", 0, 16, 4, 2)
    st.markdown("</div>", unsafe_allow_html=True)

# Metadata
if source_value and source_kind in ("Upload file", "Local path / arbitrary URL") and os.path.exists(str(source_value)):
    st.session_state.meta = backend.probe_source(str(source_value))
elif source_value and isinstance(source_value, str) and source_value.lower().startswith(("rtmp://", "rtmps://", "srt://", "http://", "https://")):
    with st.spinner("Probing source metadata..."):
        st.session_state.meta = backend.probe_source(str(source_value))
if st.session_state.meta:
    meta = st.session_state.meta
    c1, c2, c3 = st.columns(3)
    c1.metric("Source resolution", f"{int(meta.get('width', 0))}x{int(meta.get('height', 0))}")
    c2.metric("FPS", f"{meta.get('fps', 0)}")
    c3.metric("Duration", f"{float(meta.get('duration', 0.0)):.1f}s")
progress_bar = st.progress(0.0, text="Waiting")

# Actions
if workflow == "VOD → Live (full-file verticalize first)":
    st.markdown("### 4A) VOD → Live")
    vod_disabled = not bool(source_value) or source_kind in ("RTMP URL", "SRT URL")
    if st.button("Create vertical master", disabled=vod_disabled):
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        def _cb(pct: float, msg: str) -> None:
            progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=msg)
        try:
            ok, msg = backend.create_vertical_master(str(source_value), out_path, target_w=target_w, target_h=target_h, smooth_strength=smooth_strength, analysis_stride=analysis_stride, deadzone_ratio=deadzone_ratio, max_pan_ratio=max_pan_ratio, sport_profile=sport_profile, ball_tracking=ball_tracking, overlay_composite=overlay_composite, preserve_bottom_overlay=preserve_bottom_overlay, panel_mode=panel_mode, panel_max_faces=panel_max_faces, panel_detection_stride=panel_detection_stride, panel_gap=panel_gap, auto_mode=auto_mode, progress_cb=_cb)
            if ok:
                st.session_state.reframed_path = out_path
                progress_bar.progress(1.0, text="Vertical master complete")
                st.success("Vertical master created.")
            else:
                progress_bar.progress(0.0, text="Failed")
                st.error(msg)
        except Exception as exc:
            progress_bar.progress(0.0, text="Failed")
            st.error(f"Could not create vertical master: {exc}")
    if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
        st.video(st.session_state.reframed_path)
    if st.button("Start VOD → Live push", disabled=not (cf_cfg and st.session_state.reframed_path)):
        try:
            if st.session_state.live_session: backend.stop_live_session(cf_cfg, st.session_state.live_session)
            st.session_state.live_session = backend.start_vod_to_live_push(cf_cfg, st.session_state.reframed_path, uploaded_name, loop_input=loop_file)
            st.session_state.analytics_uid = None; st.session_state.analytics_history = []; st.session_state.analytics_seen_keys = []
            st.success("VOD → Live push started.")
        except Exception as exc: st.error(f"Could not start VOD → Live push: {exc}")
else:
    st.markdown("### 4B) Realtime live vertical push")
    st.caption("Recommended: delay 0–0.25s, output 540 or 360, analysis stride 6, panel off for sports.")
    if st.button("Start realtime live push", disabled=not (cf_cfg and source_value)):
        try:
            if st.session_state.live_session: backend.stop_live_session(cf_cfg, st.session_state.live_session)
            st.session_state.live_session = backend.start_realtime_delayed_vertical_push(cf_cfg, str(source_value), uploaded_name if uploaded_name != "source" else (source_value or "source"), target_w=target_w, target_h=target_h, delay_seconds=float(delay_seconds), smooth_strength=float(smooth_strength), analysis_stride=int(analysis_stride), deadzone_ratio=float(deadzone_ratio), max_pan_ratio=float(max_pan_ratio), loop_file=(loop_file and source_kind in ("Upload file", "Local path / arbitrary URL")), pace_input=(source_kind in ("Upload file", "Local path / arbitrary URL")), sport_profile=sport_profile, ball_tracking=ball_tracking, overlay_composite=overlay_composite, preserve_bottom_overlay=preserve_bottom_overlay, panel_mode=panel_mode, panel_max_faces=panel_max_faces, panel_detection_stride=panel_detection_stride, panel_gap=panel_gap, auto_mode=auto_mode)
            st.session_state.analytics_uid = None; st.session_state.analytics_history = []; st.session_state.analytics_seen_keys = []
            st.success("Realtime live worker started.")
        except Exception as exc: st.error(f"Could not start realtime live push: {exc}")

# Controls/status
left, right = st.columns(2)
with left:
    if st.button("Stop current live push", disabled=not bool(st.session_state.live_session)):
        if cf_cfg and st.session_state.live_session: backend.stop_live_session(cf_cfg, st.session_state.live_session)
        st.session_state.live_session = None; st.session_state.analytics_uid = None; st.session_state.analytics_history = []; st.session_state.analytics_seen_keys = []
        st.info("Current live session stopped and Cloudflare input disabled.")
with right:
    auto_refresh = st.checkbox("Auto-refresh session status", value=True)

if st.session_state.live_session:
    session = st.session_state.live_session
    _append_snapshot(session)
    stats = _safe_stats(session)
    status = getattr(session, "status", "unknown")
    err = getattr(session, "error", "")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Cloudflare playback**")
    st.caption(f"Live input UID: {getattr(session, 'uid', '-')}")
    normal_hls = getattr(session, "hls_url", "")
    normal_hls = normal_hls.split("?")[0] if normal_hls else ""
    st.text_input("Normal HLS test playback URL", value=normal_hls, key="normal_hls_test_url")
    st.text_input("Cloudflare iframe player URL", value=getattr(session, "iframe_url", ""), key="iframe_player_url")
    c = st.columns(4)
    c[0].metric("Pipeline status", status)
    c[1].metric("Health", _stat(stats, "health", default="-"))
    c[2].metric("Mode", _stat(stats, "mode", default="-"))
    c[3].metric("Frames out", _int(_stat(stats, "frames_out", default=0)))
    if err: st.error(err)
    if status == "streaming": st.success("Actively pushing frames to Cloudflare.")
    elif status in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended", "ffmpeg_exited"}: st.error("Worker hit an error or source ended. Check logs.")
    with st.expander("FFmpeg push log", expanded=status in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended", "ffmpeg_exited"}):
        log_path = getattr(session, "log_path", "")
        st.code(backend.read_log_tail(log_path) if log_path else "(no log path available)", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)
    _render_analytics(session)
    st.subheader("7) In-app playback")
    iframe_url = getattr(session, "iframe_url", "")
    if iframe_url: components.iframe(iframe_url, height=760, scrolling=True)
    if auto_refresh:
        time.sleep(1.5)
        st.rerun()

st.divider()
st.markdown("### Recommended sports live settings")
st.markdown("- Realtime live vertical push\n- Delay: 0–0.25s\n- Panel mode: Off\n- Auto mode: Off\n- Ball tracking: On for sports, Off for control test\n- Analysis stride: 6 to start\n- If `p95_process_ms > 80ms`, test output width 360 or disable ball tracking for comparison.")
