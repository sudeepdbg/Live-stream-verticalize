from __future__ import annotations
import json
import os
import tempfile
import time
from collections import deque
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(
    page_title="Dual Flow Vertical Live  Cloudflare Stream",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      .card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px;
             background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92));
             color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
      .hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px;
             margin-bottom:16px;
             background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%),
                         linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;}
      .chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem;
             border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1;
             margin-right:6px; margin-top:6px;}
      .chip-ok {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem;
             border:1px solid rgba(34,197,94,.35); background:rgba(34,197,94,.10); color:#4ade80;
             margin-right:6px; margin-top:6px;}
      .chip-warn {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem;
             border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.10); color:#fbbf24;
             margin-right:6px; margin-top:6px;}
      .chip-bad {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem;
             border:1px solid rgba(239,68,68,.35); background:rgba(239,68,68,.10); color:#f87171;
             margin-right:6px; margin-top:6px;}
      .panel-box {border:1px solid rgba(99,102,241,.35); border-radius:14px; padding:14px 16px;
                  background:rgba(99,102,241,.06); margin-top:8px;}
      .mini-box {border:1px solid rgba(148,163,184,.18); border-radius:14px; padding:12px 14px;
                 background:rgba(2,6,23,.20);}
      .tiny {font-size:.82rem; color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def _fmt_ms(v: Any) -> str:
    if v is None:
        return "-"
    return f"{_num(v):.1f} ms"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{_num(v):.1f}%"


def _fmt_sec(v: Any) -> str:
    if v is None:
        return "-"
    return f"{_num(v):.2f}s"


def _health_chip_html(label: str, level: str = "ok") -> str:
    klass = {
        "ok": "chip-ok",
        "warn": "chip-warn",
        "bad": "chip-bad",
        "neutral": "chip",
    }.get(level, "chip")
    return f"<span class='{klass}'>{label}</span>"


def _analytics_log_tail(path: str, max_lines: int = 40) -> str:
    raw = backend.read_log_tail(path, max_chars=50000)
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if ln.startswith("[ANALYTICS]")]
    return "\n".join(lines[-max_lines:])


def _append_snapshot_from_session(session) -> None:
    if not session:
        return
    uid = getattr(session, "uid", "unknown") or "unknown"
    stats = getattr(session, "stats", {}) or {}

    if st.session_state.analytics_uid != uid:
        st.session_state.analytics_uid = uid
        st.session_state.analytics_history = []
        st.session_state.analytics_seen_keys = set()

    ts_ms = _int(stats.get("updated_at_ms") or int(time.time() * 1000))
    dedupe_key = (
        uid,
        ts_ms,
        _int(stats.get("frames_out", 0)),
        _int(stats.get("frames_in", 0)),
        str(getattr(session, "status", "unknown")),
    )
    if dedupe_key in st.session_state.analytics_seen_keys:
        return

    snap = {
        "ts_ms": ts_ms,
        "time": time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000.0)),
        "status": getattr(session, "status", "unknown"),
        "frames_in": _int(stats.get("frames_in", 0)),
        "frames_out": _int(stats.get("frames_out", 0)),
        "frames_processed": _int(stats.get("frames_processed", 0)),
        "buffer_len": _int(stats.get("buffer_len", 0)),
        "buffer_fill_pct": _num(stats.get("buffer_fill_pct", 0)),
        "buffer_seconds": _num(stats.get("buffer_seconds", 0)),
        "placeholder_frames": _int(stats.get("placeholder_frames", 0)),
        "source_stalls": _int(stats.get("source_stalls", 0)),
        "output_underruns": _int(stats.get("output_underruns", 0)),
        "input_drop_count": _int(stats.get("input_drop_count", 0)),
        "ingest_fps_1s": _num(stats.get("ingest_fps_1s", 0)),
        "process_fps_1s": _num(stats.get("process_fps_1s", 0)),
        "output_fps_1s": _num(stats.get("output_fps_1s", 0)),
        "avg_process_ms": _num(stats.get("avg_process_ms", 0)),
        "p95_process_ms": _num(stats.get("p95_process_ms", 0)),
        "avg_output_write_ms": _num(stats.get("avg_output_write_ms", 0)),
        "p95_output_write_ms": _num(stats.get("p95_output_write_ms", 0)),
        "avg_ingest_read_ms": _num(stats.get("avg_ingest_read_ms", 0)),
        "p95_ingest_read_ms": _num(stats.get("p95_ingest_read_ms", 0)),
        "avg_schedule_drift_ms": _num(stats.get("avg_schedule_drift_ms", 0)),
        "p95_schedule_drift_ms": _num(stats.get("p95_schedule_drift_ms", 0)),
        "ball_confidence": _num(stats.get("ball_confidence", 0)),
        "panel_active_faces": _int(stats.get("panel_active_faces", 0)),
        "panel_detector": stats.get("panel_detector", "-"),
        "ffmpeg_alive": bool(stats.get("ffmpeg_alive", False)),
        "ingest_alive": bool(stats.get("ingest_alive", False)),
    }
    st.session_state.analytics_history.append(snap)
    if len(st.session_state.analytics_history) > 300:
        st.session_state.analytics_history = st.session_state.analytics_history[-300:]
    st.session_state.analytics_seen_keys.add(dedupe_key)
    if len(st.session_state.analytics_seen_keys) > 500:
        st.session_state.analytics_seen_keys = set(list(st.session_state.analytics_seen_keys)[-300:])


def _render_stream_health(stats: dict, session_status: str) -> None:
    fps_target = _num(stats.get("fps", backend.DEFAULT_OUTPUT_FPS), backend.DEFAULT_OUTPUT_FPS)
    p95_proc = _num(stats.get("p95_process_ms", 0))
    p95_write = _num(stats.get("p95_output_write_ms", 0))
    p95_read = _num(stats.get("p95_ingest_read_ms", 0))
    p95_drift = _num(stats.get("p95_schedule_drift_ms", 0))
    output_fps = _num(stats.get("output_fps_1s", 0))
    buffer_fill = _num(stats.get("buffer_fill_pct", 0))
    underruns = _int(stats.get("output_underruns", 0))
    stalls = _int(stats.get("source_stalls", 0))
    drops = _int(stats.get("input_drop_count", 0))

    chips = []
    if session_status == "streaming":
        chips.append(_health_chip_html("Streaming", "ok"))
    elif session_status in {"buffering", "connecting_source", "priming_output"}:
        chips.append(_health_chip_html(session_status.replace("_", " ").title(), "warn"))
    else:
        chips.append(_health_chip_html(session_status.replace("_", " ").title(), "bad"))

    if p95_proc > (1000.0 / max(fps_target, 1.0)) * 0.9:
        chips.append(_health_chip_html("Processing bottleneck", "bad"))
    elif p95_proc > (1000.0 / max(fps_target, 1.0)) * 0.65:
        chips.append(_health_chip_html("Processing close to limit", "warn"))
    else:
        chips.append(_health_chip_html("Processing healthy", "ok"))

    if output_fps < fps_target * 0.90:
        chips.append(_health_chip_html("Output FPS low", "bad"))
    elif output_fps < fps_target * 0.97:
        chips.append(_health_chip_html("Output FPS slightly low", "warn"))
    else:
        chips.append(_health_chip_html("Output cadence healthy", "ok"))

    if p95_write > 20:
        chips.append(_health_chip_html("Encoder / RTMPS backpressure", "bad"))
    elif p95_write > 10:
        chips.append(_health_chip_html("Output write elevated", "warn"))
    else:
        chips.append(_health_chip_html("Output write healthy", "ok"))

    if p95_read > 40 or stalls > 0:
        chips.append(_health_chip_html("Source ingest unstable", "bad" if stalls > 0 else "warn"))
    else:
        chips.append(_health_chip_html("Source ingest healthy", "ok"))

    if buffer_fill < 10 and underruns > 0:
        chips.append(_health_chip_html("Buffer starvation", "bad"))
    elif buffer_fill > 85 and drops > 0:
        chips.append(_health_chip_html("Buffer pressure high", "warn"))
    else:
        chips.append(_health_chip_html("Buffer stable", "ok"))

    if p95_drift > 20:
        chips.append(_health_chip_html("Scheduler drift high", "warn"))
    else:
        chips.append(_health_chip_html("Scheduler drift normal", "ok"))

    st.markdown("".join(chips), unsafe_allow_html=True)

    guidance = []
    if p95_proc > 28:
        guidance.append("**Processing is the bottleneck**: reduce analysis stride, disable ball tracking for non-sports, or lower working resolution if needed.")
    if p95_write > 15:
        guidance.append("**Encoder / network push is slowing writes**: lower bitrate further or move to `superfast` preset if this persists.")
    if output_fps < fps_target * 0.9:
        guidance.append("**Output FPS is below target**: check both `p95_process_ms` and `p95_output_write_ms` to see whether CV or encode/push is the limiter.")
    if buffer_fill < 10 and underruns > 0:
        guidance.append("**Buffer is starving**: source is not arriving steadily enough or processing is not keeping up. This directly maps to visible buffering/live instability.")
    if stalls > 0:
        guidance.append("**Source stalls detected**: verify RTMP/SRT source stability, reconnect behavior, and network health on the ingest side.")
    if not guidance:
        guidance.append("Pipeline looks broadly healthy. If player buffering still occurs, the next lever to test is bitrate/profile tuning on the output stream and player/network conditions.")

    for tip in guidance[:4]:
        st.info(tip)


def _render_analytics_section(session) -> None:
    stats = getattr(session, "stats", {}) or {}
    history = st.session_state.analytics_history
    status = getattr(session, "status", "unknown")

    st.markdown("### 6) Analytics & tuning")
    _render_stream_health(stats, status)

    row1 = st.columns(4)
    row1[0].metric("Output FPS (1s)", f"{_num(stats.get('output_fps_1s', 0)):.2f}")
    row1[1].metric("Process FPS (1s)", f"{_num(stats.get('process_fps_1s', 0)):.2f}")
    row1[2].metric("Ingest FPS (1s)", f"{_num(stats.get('ingest_fps_1s', 0)):.2f}")
    row1[3].metric("Target FPS", f"{_num(stats.get('fps', backend.DEFAULT_OUTPUT_FPS)):.2f}")

    row2 = st.columns(4)
    row2[0].metric("Avg process", _fmt_ms(stats.get("avg_process_ms")))
    row2[1].metric("P95 process", _fmt_ms(stats.get("p95_process_ms")))
    row2[2].metric("Avg output write", _fmt_ms(stats.get("avg_output_write_ms")))
    row2[3].metric("P95 output write", _fmt_ms(stats.get("p95_output_write_ms")))

    row3 = st.columns(4)
    row3[0].metric("Avg ingest read", _fmt_ms(stats.get("avg_ingest_read_ms")))
    row3[1].metric("P95 ingest read", _fmt_ms(stats.get("p95_ingest_read_ms")))
    row3[2].metric("Avg scheduler drift", _fmt_ms(stats.get("avg_schedule_drift_ms")))
    row3[3].metric("P95 scheduler drift", _fmt_ms(stats.get("p95_schedule_drift_ms")))

    row4 = st.columns(4)
    row4[0].metric("Buffer fill", _fmt_pct(stats.get("buffer_fill_pct")))
    row4[1].metric("Buffer seconds", _fmt_sec(stats.get("buffer_seconds")))
    row4[2].metric("Target delay", _fmt_sec(stats.get("target_delay_seconds") or stats.get("delay_seconds_configured")))
    row4[3].metric("Output underruns", _int(stats.get("output_underruns", 0)))

    row5 = st.columns(4)
    row5[0].metric("Startup  source frame", _fmt_ms(stats.get("startup_ms_to_first_source_frame")))
    row5[1].metric("Startup  first live frame", _fmt_ms(stats.get("startup_ms_to_first_live_frame")))
    row5[2].metric("Input drops", _int(stats.get("input_drop_count", 0)))
    row5[3].metric("Write failures", _int(stats.get("output_write_failures", 0)))

    row6 = st.columns(4)
    row6[0].metric("FFmpeg alive", "Yes" if stats.get("ffmpeg_alive", False) else "No")
    row6[1].metric("Ingest alive", "Yes" if stats.get("ingest_alive", False) else "No")
    row6[2].metric("FFmpeg rc", stats.get("ffmpeg_returncode", "-"))
    row6[3].metric("Ingest rc", stats.get("ingest_returncode", "-"))

    if history:
        df = pd.DataFrame(history)
        tabs = st.tabs([
            "Pipeline charts",
            "Buffer & timing",
            "Detection / content",
            "Raw stats",
            "Analytics log",
        ])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**FPS trend**")
                st.line_chart(df.set_index("time")[["ingest_fps_1s", "process_fps_1s", "output_fps_1s"]], height=260)
            with c2:
                st.markdown("**Process vs output-write latency**")
                st.line_chart(df.set_index("time")[["avg_process_ms", "p95_process_ms", "avg_output_write_ms", "p95_output_write_ms"]], height=260)

        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Buffer trend**")
                st.line_chart(df.set_index("time")[["buffer_len", "buffer_fill_pct", "buffer_seconds"]], height=260)
            with c2:
                st.markdown("**Ingest read + scheduler drift**")
                st.line_chart(df.set_index("time")[["avg_ingest_read_ms", "p95_ingest_read_ms", "avg_schedule_drift_ms", "p95_schedule_drift_ms"]], height=260)

            c3, c4 = st.columns(2)
            with c3:
                latest = df.iloc[-1]
                st.markdown("**Current counters**")
                st.markdown(
                    f"<div class='mini-box'>"
                    f"<div><b>Placeholder frames:</b> {_int(latest.get('placeholder_frames', 0))}</div>"
                    f"<div><b>Source stalls:</b> {_int(latest.get('source_stalls', 0))}</div>"
                    f"<div><b>Output underruns:</b> {_int(latest.get('output_underruns', 0))}</div>"
                    f"<div><b>Input drops:</b> {_int(latest.get('input_drop_count', 0))}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown("**Interpretation**")
                st.caption(
                    "Low buffer fill + rising underruns = starvation. High process p95 = CV bottleneck. High output-write p95 = encoder / RTMPS push bottleneck."
                )

        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                show_cols = [c for c in ["ball_confidence", "panel_active_faces"] if c in df.columns]
                if show_cols:
                    st.markdown("**Content detection trend**")
                    st.line_chart(df.set_index("time")[show_cols], height=260)
            with c2:
                latest = df.iloc[-1]
                st.markdown("**Latest detection state**")
                st.markdown(
                    f"<div class='mini-box'>"
                    f"<div><b>Ball confidence:</b> {_num(latest.get('ball_confidence', 0)):.2f}</div>"
                    f"<div><b>Panel active faces:</b> {_int(latest.get('panel_active_faces', 0))}</div>"
                    f"<div><b>Panel detector:</b> {latest.get('panel_detector', '-')}</div>"
                    f"<div><b>FFmpeg alive:</b> {'Yes' if bool(latest.get('ffmpeg_alive', False)) else 'No'}</div>"
                    f"<div><b>Ingest alive:</b> {'Yes' if bool(latest.get('ingest_alive', False)) else 'No'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with tabs[3]:
            st.dataframe(df.tail(60), use_container_width=True, hide_index=True)
            with st.expander("Current stats JSON"):
                st.json(stats)

        with tabs[4]:
            log_path = getattr(session, "log_path", "")
            analytics_only = _analytics_log_tail(log_path)
            if analytics_only:
                st.code(analytics_only, language="json")
            else:
                st.caption("No analytics lines written yet.")


# -- Session state defaults ------------------------------------------------
for key, value in {
    "input_path": None,
    "meta": None,
    "reframed_path": None,
    "live_session": None,
    "cf_account_id": os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
    "cf_api_token": os.getenv("CLOUDFLARE_STREAM_API_TOKEN", ""),
    "cf_customer_code": os.getenv("CLOUDFLARE_STREAM_CUSTOMER_CODE", ""),
    "cf_low_latency": False,
    "analytics_history": [],
    "analytics_uid": None,
    "analytics_seen_keys": set(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Hard-reset incompatible live_session objects from older deployments.
ls = st.session_state.get("live_session")
if ls is not None and not all(hasattr(ls, a) for a in ["uid", "hls_url", "iframe_url", "log_path"]):
    st.session_state.live_session = None
    ls = None

# -- Hero ------------------------------------------------------------------
st.markdown(
    """
    <div class='hero'>
      <div style='display:flex; justify-content:space-between; gap:12px;
                  align-items:flex-start; flex-wrap:wrap;'>
        <div>
          <div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>
            Dual Flow Vertical Live → Cloudflare Stream
          </div>
          <div style='font-size:.98rem; color:#cbd5e1;'>
            Supports VOD → Live and delayed realtime. RTMP / SRT / URL sources handled in
            realtime mode. Panel discussion mode splits multi-person frames into a 9:16 grid.
          </div>
        </div>
        <div>
          <span class='chip'>workflow switch</span>
          <span class='chip'>source-type switch</span>
          <span class='chip'>panel discussion mode</span>
          <span class='chip'>sport profiles</span>
          <span class='chip'>placeholder priming</span>
          <span class='chip'>analytics dashboard</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not backend.ffmpeg_ok():
    st.error("FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.")
    st.stop()

# -- Section 1: Cloudflare credentials (persisted) -------------------------
st.subheader("1) Cloudflare Stream credentials")

creds_set = bool(
    st.session_state.cf_account_id
    and st.session_state.cf_api_token
    and st.session_state.cf_customer_code
)

if creds_set:
    masked = st.session_state.cf_account_id[-6:] if len(st.session_state.cf_account_id) > 6 else "***"
    st.markdown(
        f"<span class='chip-ok'>✓ Credentials saved</span>"
        f"<span class='chip'>Account: ...{masked}</span>",
        unsafe_allow_html=True,
    )

creds_label = "Edit credentials" if creds_set else "Enter credentials"
with st.expander(creds_label, expanded=not creds_set):
    left_cf, right_cf = st.columns(2)
    with left_cf:
        _acc = st.text_input(
            "Cloudflare account ID",
            value=st.session_state.cf_account_id,
            key="_cf_acc_input",
        )
        _tok = st.text_input(
            "Cloudflare Stream API token",
            value=st.session_state.cf_api_token,
            type="password",
            key="_cf_tok_input",
        )
    with right_cf:
        _code = st.text_input(
            "Cloudflare Stream customer code",
            value=st.session_state.cf_customer_code,
            help="Enter only the code, not the full domain.",
            key="_cf_code_input",
        )
        _ll = st.checkbox(
            "Prefer LL-HLS where available",
            value=st.session_state.cf_low_latency,
            key="_cf_ll_input",
        )
    st.session_state.cf_account_id = _acc
    st.session_state.cf_api_token = _tok
    st.session_state.cf_customer_code = _code
    st.session_state.cf_low_latency = _ll

account_id = st.session_state.cf_account_id
api_token = st.session_state.cf_api_token
customer_code = st.session_state.cf_customer_code
prefer_low_latency = st.session_state.cf_low_latency

cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(
            account_id, api_token, customer_code, prefer_low_latency
        )
        if not creds_set:
            st.success("Cloudflare Stream configuration looks valid.")
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning("Fill in Account ID, Stream API token, and customer code to continue.")

# -- Section 2: Source & workflow -------------------------------------------
st.subheader("2) Workflow and source")
workflow = st.radio(
    "Workflow",
    [
        "VOD -> Live (full-file verticalize first)",
        "Delayed realtime (frame-by-frame then delay buffer)",
    ],
    horizontal=True,
)
source_kind = st.radio(
    "Source type",
    ["Upload file", "RTMP URL", "SRT URL", "Local path / arbitrary URL"],
    horizontal=True,
)

source_value = None
uploaded_name = "source"

if source_kind == "Upload file":
    upl = st.file_uploader(
        f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)",
        type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"],
    )
    if upl:
        if getattr(upl, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(
                f"File is {upl.size / (1024*1024):.1f} MB -- keep it <= {backend.MAX_UPLOAD_MB} MB."
            )
            st.stop()
        suffix = os.path.splitext(upl.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upl.read())
            st.session_state.input_path = tmp.name
        uploaded_name = upl.name
        source_value = st.session_state.input_path
elif source_kind == "RTMP URL":
    source_value = st.text_input(
        "RTMP source URL",
        placeholder="rtmp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
    )
elif source_kind == "SRT URL":
    source_value = st.text_input(
        "SRT source URL", placeholder="srt://host:port?mode=caller"
    )
else:
    source_value = st.text_input(
        "Local file path or arbitrary URL",
        placeholder="/mount/src/... or https://...",
    )

# -- Section 3: Reframing settings -----------------------------------------
st.subheader("3) Reframing settings")

col_res, col_delay = st.columns(2)
with col_res:
    target_w = st.selectbox("Vertical output width", [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))
    st.caption(f"Output: {target_w} x {target_h}")
with col_delay:
    delay_seconds = st.slider("Output delay (seconds)", 5, 30, 20, 1)
    loop_file = st.checkbox("Loop file source when it ends", value=True)

st.markdown("**Motion & pan tuning**")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    smooth_strength = st.slider("Smoothness", 0.90, 0.995, 0.97, 0.005)
with col_s2:
    analysis_stride = st.slider("Analysis stride", 3, 10, 6, 1)
with col_s3:
    deadzone_ratio = st.slider("Deadzone ratio", 0.02, 0.10, 0.06, 0.01)
with col_s4:
    max_pan_ratio = st.slider("Max pan ratio", 0.005, 0.03, 0.01, 0.005)

st.markdown("**Content mode**")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    sport_profile = st.selectbox(
        "Sport profile",
        ["auto", "soccer", "basketball", "cricket"],
        index=0,
        help="Tunes ball colour and motion heuristics. Use 'auto' for non-sport content.",
    )
with col_m2:
    ball_tracking = st.checkbox(
        "Ball tracking",
        value=True,
        help="Disable for non-sport or panel content to reduce false positives.",
    )
    overlay_composite = st.checkbox(
        "Overlay composite",
        value=True,
        help="Detect and preserve top scorecard / bottom lower-third strips.",
    )
with col_m3:
    preserve_bottom_overlay = st.checkbox(
        "Preserve bottom overlay",
        value=False,
        help="Reserve a band at the bottom for lower-thirds/tickers.",
    )

# -- Panel discussion mode -------------------------------------------------
st.markdown("**Panel discussion mode**")
panel_mode = st.checkbox(
    "Enable panel mode",
    value=False,
    help=(
        "Splits the 9:16 frame into per-person sub-panels (up to 4). "
        "Best for talk-shows, interviews, and news panels. "
        "Disable ball tracking when using this mode."
    ),
)

panel_max_faces = 4
panel_detection_stride = 2
panel_gap = 4

if panel_mode:
    st.markdown("<div class='panel-box'>", unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        panel_max_faces = st.slider(
            "Max faces (panels)", 1, 4, 4, 1,
            help="Layout auto-selects 1-up / 2-up / 3-up / 4-up based on detected faces.",
        )
    with pc2:
        panel_detection_stride = st.slider(
            "Detection stride", 1, 6, 2, 1,
            help="Detect faces every N frames. Tracker smooths between detections.",
        )
    with pc3:
        panel_gap = st.slider(
            "Panel gap (px)", 0, 16, 4, 2,
            help="Pixel gap between panel cells.",
        )
    st.caption(
        "Smoothing: position alpha=0.90 | size alpha=0.92 | layout switch: 15 frames | "
        "face persistence: 24 frames | transition blend: 10 frames | MediaPipe/Haar fallback"
    )
    if ball_tracking:
        st.warning(
            "Ball tracking is enabled alongside panel mode. "
            "Consider disabling it for panel/talk-show content."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -- Source metadata -------------------------------------------------------
if source_value and source_kind in ("Upload file", "Local path / arbitrary URL") and os.path.exists(str(source_value)):
    st.session_state.meta = backend.probe_source(str(source_value))
elif source_value and isinstance(source_value, str) and source_value.lower().startswith(
    ("rtmp://", "rtmps://", "srt://", "http://", "https://")
):
    st.session_state.meta = backend.probe_source(str(source_value))
    st.info(
        "For URL-based ingest, metadata can be approximate at setup time. "
        "The pipeline primes Cloudflare immediately and continues buffering in the background."
    )

if st.session_state.meta:
    meta = st.session_state.meta
    ma, mb, mc = st.columns(3)
    ma.metric("Source resolution", f"{int(meta.get('width', 0))}x{int(meta.get('height', 0))}")
    mb.metric("FPS", f"{meta.get('fps', 0)}")
    mc.metric("Duration", f"{float(meta.get('duration', 0.0)):.1f}s")

# -- Section 4: Actions ----------------------------------------------------
progress_bar = st.progress(0.0, text="Waiting")

if workflow == "VOD -> Live (full-file verticalize first)":
    st.markdown("### 4A) VOD -> Live")
    if source_kind in ("RTMP URL", "SRT URL"):
        st.warning(
            "VOD -> Live is intended for file-like sources. "
            "Switch to Delayed realtime for RTMP/SRT URLs."
        )

    vod_disabled = not bool(source_value) or source_kind in ("RTMP URL", "SRT URL")
    if st.button("Create vertical master", disabled=vod_disabled):
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        def _cb(pct: float, msg: str) -> None:
            progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=msg)

        ok, msg = backend.create_vertical_master(
            str(source_value),
            out_path,
            target_w,
            target_h,
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
            progress_cb=_cb,
        )
        if ok:
            st.session_state.reframed_path = out_path
            progress_bar.progress(1.0, text="Vertical master complete")
            st.success("Vertical master created.")
        else:
            st.error(msg)

    if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
        st.video(st.session_state.reframed_path)

    if st.button(
        "Start VOD -> Live push",
        disabled=not (cf_cfg and st.session_state.reframed_path),
    ):
        if st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        session = backend.start_vod_to_live_push(
            cf_cfg,
            st.session_state.reframed_path,
            uploaded_name,
            loop_input=loop_file,
        )
        st.session_state.live_session = session
        st.success("VOD -> Live push started.")

else:
    st.markdown("### 4B) Delayed realtime")
    st.caption(
        "Horizontal file / RTMP / SRT -> frame-by-frame vertical output with ~20-25 s delay. "
        "Startup placeholder frames are sent immediately so Cloudflare does not stall."
    )

    rt_disabled = not (cf_cfg and source_value)
    if st.button("Start delayed realtime push", disabled=rt_disabled):
        if st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        session = backend.start_realtime_delayed_vertical_push(
            cf_cfg,
            str(source_value),
            uploaded_name if uploaded_name != "source" else (source_value or "source"),
            target_w=target_w,
            target_h=target_h,
            delay_seconds=float(delay_seconds),
            smooth_strength=float(smooth_strength),
            analysis_stride=int(analysis_stride),
            deadzone_ratio=float(deadzone_ratio),
            max_pan_ratio=float(max_pan_ratio),
            loop_file=(loop_file and source_kind in ("Upload file", "Local path / arbitrary URL")),
            pace_input=(source_kind in ("Upload file", "Local path / arbitrary URL")),
            sport_profile=sport_profile,
            ball_tracking=ball_tracking,
            overlay_composite=overlay_composite,
            preserve_bottom_overlay=preserve_bottom_overlay,
            panel_mode=panel_mode,
            panel_max_faces=panel_max_faces,
            panel_detection_stride=panel_detection_stride,
            panel_gap=panel_gap,
        )
        st.session_state.live_session = session
        st.success("Delayed realtime worker started.")

# -- Stop / auto-refresh controls -----------------------------------------
left_action, right_action = st.columns([1, 1])
with left_action:
    if st.button(
        "Stop current live push",
        disabled=not bool(st.session_state.live_session),
    ):
        if cf_cfg and st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        st.session_state.live_session = None
        st.session_state.analytics_uid = None
        st.session_state.analytics_history = []
        st.session_state.analytics_seen_keys = set()
        st.info("Current live session stopped and Cloudflare input disabled.")
with right_action:
    auto_refresh = st.checkbox("Auto-refresh session status", value=True)

# -- Section 5: Live session status & playback -----------------------------
if st.session_state.live_session:
    session = st.session_state.live_session
    _append_snapshot_from_session(session)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Shared Cloudflare playback section**")
    st.caption(f"Live input UID: {getattr(session, 'uid', '-')}")

    normal_hls = getattr(session, "hls_url", "")
    normal_hls = normal_hls.split("?")[0] if normal_hls else ""
    st.text_input(
        "Normal HLS test playback URL", value=normal_hls, key="normal_hls_test_url"
    )
    st.caption(
        "Use the normal HLS URL above in any open-source HLS player "
        "if LL-HLS playback looks unstable."
    )
    st.text_input(
        "Cloudflare iframe player URL",
        value=getattr(session, "iframe_url", ""),
        key="iframe_player_url",
    )

    stats = getattr(session, "stats", {}) or {}
    session_status = getattr(session, "status", "unknown")
    session_error = getattr(session, "error", "")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline status", session_status)
    m2.metric("Working source", stats.get("working_resolution", "-"))
    m3.metric("Delay frames", stats.get("delay_frames", "-"))
    m4.metric("Frames out", stats.get("frames_out", 0))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Frames in", stats.get("frames_in", 0))
    m6.metric("Buffer len", stats.get("buffer_len", 0))
    m7.metric("Placeholder frames", stats.get("placeholder_frames", 0))
    m8.metric("Source stalls", stats.get("source_stalls", 0))

    m9, m10, m11, m12 = st.columns(4)
    panel_faces = stats.get("panel_active_faces", None)
    panel_mode_active = stats.get("panel_mode", False)
    m9.metric(
        "Panel faces",
        panel_faces if panel_faces is not None else "---",
        help="Active tracked faces in panel mode (--- when panel mode is off).",
    )
    m10.metric("Ball confidence", f"{_num(stats.get('ball_confidence', 0.0)):.2f}")
    overlay_top = stats.get("overlay_top", False)
    overlay_bot = stats.get("overlay_bottom", False)
    m11.metric("Top overlay", "Y" if overlay_top else "N")
    m12.metric("Bottom overlay", "Y" if overlay_bot else "N")

    if session_error:
        st.error(session_error)

    if session_status in {"priming_output", "connecting_source", "buffering"}:
        st.warning(
            "Output started; Cloudflare is being primed. "
            "Wait a few seconds, then refresh the Cloudflare playback page."
        )
    elif session_status == "streaming":
        mode_label = "panel" if panel_mode_active else "single-crop"
        st.success(f"Actively pushing frames to Cloudflare ({mode_label} mode).")
    elif session_status in {
        "ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended", "ffmpeg_exited"
    }:
        st.error("Worker hit an error or source ended. Check the FFmpeg log and analytics below.")

    with st.expander(
        "FFmpeg push log",
        expanded=session_status in {
            "ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended", "ffmpeg_exited"
        },
    ):
        log_path = getattr(session, "log_path", "")
        if log_path:
            st.code(backend.read_log_tail(log_path) or "(empty)", language="bash")
        else:
            st.code("(no log path available)", language="bash")

    st.markdown("</div>", unsafe_allow_html=True)

    _render_analytics_section(session)

    st.subheader("7) In-app playback")
    iframe_url = getattr(session, "iframe_url", "")
    if iframe_url:
        components.iframe(iframe_url, height=760, scrolling=True)

    if auto_refresh:
        time.sleep(3)
        st.rerun()

# -- Footer ----------------------------------------------------------------
st.divider()
st.markdown("### What's included")
st.markdown(
    "- **Workflow switch** -- VOD->Live or delayed realtime\n"
    "- **Source-type switch** -- file upload, RTMP, SRT, local path / URL\n"
    "- **Panel discussion mode** -- 1-up / 2-up / 3-up / 4-up auto-layout with jitter-free tracking\n"
    "- **Sport profiles** -- auto / soccer / basketball / cricket with ball tracking\n"
    "- **Overlay composite** -- preserves top scorecard and optional bottom lower-third\n"
    "- **Cloudflare startup priming** -- placeholder frames avoid 'stream not started' errors\n"
    "- **Stale session protection** -- safe across Streamlit redeploys\n"
    "- **Persistent credentials** -- Cloudflare keys saved in session, edit anytime\n"
    "- **Analytics dashboard** -- FPS, p95 timings, buffer health, startup latency, raw stats, analytics log"
)
