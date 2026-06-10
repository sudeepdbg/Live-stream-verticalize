from __future__ import annotations

import os
import tempfile
import time

import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(
    page_title="Dual Flow Vertical Live → Cloudflare Stream",
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
      .panel-box {border:1px solid rgba(99,102,241,.35); border-radius:14px; padding:14px 16px;
                  background:rgba(99,102,241,.06); margin-top:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────────────────────
for key, value in {
    "input_path": None,
    "meta": None,
    "reframed_path": None,
    "live_session": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Hard-reset incompatible live_session objects from older deployments.
ls = st.session_state.get("live_session")
if ls is not None and not all(hasattr(ls, a) for a in ["uid", "hls_url", "iframe_url", "log_path"]):
    st.session_state.live_session = None
    ls = None

# ── Hero ─────────────────────────────────────────────────────────────────────
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
          <span class='chip'>shared playback section</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not backend.ffmpeg_ok():
    st.error("FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.")
    st.stop()

# ── Section 1: Cloudflare credentials ────────────────────────────────────────
st.subheader("1) Cloudflare Stream credentials")
left_cf, right_cf = st.columns(2)
with left_cf:
    account_id = st.text_input(
        "Cloudflare account ID",
        value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
    )
    api_token = st.text_input(
        "Cloudflare Stream API token",
        value=os.getenv("CLOUDFLARE_STREAM_API_TOKEN", ""),
        type="password",
    )
with right_cf:
    customer_code = st.text_input(
        "Cloudflare Stream customer code",
        value=os.getenv("CLOUDFLARE_STREAM_CUSTOMER_CODE", ""),
        help="Enter only the code, not the full domain.",
    )
    prefer_low_latency = st.checkbox("Prefer LL-HLS where available", value=False)

cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(
            account_id, api_token, customer_code, prefer_low_latency
        )
        st.success("Cloudflare Stream configuration looks valid.")
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning("Fill in Account ID, Stream API token, and customer code to continue.")

# ── Section 2: Source & workflow ──────────────────────────────────────────────
st.subheader("2) Workflow and source")
workflow = st.radio(
    "Workflow",
    [
        "VOD → Live (full-file verticalize first)",
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
                f"File is {upl.size / (1024*1024):.1f} MB — keep it ≤ {backend.MAX_UPLOAD_MB} MB."
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

# ── Section 3: Reframing settings ────────────────────────────────────────────
st.subheader("3) Reframing settings")

col_res, col_delay = st.columns(2)
with col_res:
    target_w = st.selectbox("Vertical output width", [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))
    st.caption(f"Output: {target_w} × {target_h}")
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
    ball_tracking = st.checkbox("Ball tracking", value=True,
        help="Disable for non-sport or panel content to reduce false positives.")
    overlay_composite = st.checkbox("Overlay composite", value=True,
        help="Detect and preserve top scorecard / bottom lower-third strips.")
with col_m3:
    preserve_bottom_overlay = st.checkbox("Preserve bottom overlay", value=False,
        help="Reserve a band at the bottom for lower-thirds/tickers.")

# ── Panel discussion mode ─────────────────────────────────────────────────────
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
        "Smoothing: position α=0.90 · size α=0.92 · layout switch: 25 frames · "
        "face persistence: 18 frames · transition blend: 10 frames"
    )
    if ball_tracking:
        st.warning(
            "Ball tracking is enabled alongside panel mode. "
            "Consider disabling it for panel/talk-show content."
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Source metadata ───────────────────────────────────────────────────────────
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
    ma.metric("Source resolution", f"{int(meta.get('width', 0))}×{int(meta.get('height', 0))}")
    mb.metric("FPS", f"{meta.get('fps', 0)}")
    mc.metric("Duration", f"{float(meta.get('duration', 0.0)):.1f}s")

# ── Section 4: Actions ────────────────────────────────────────────────────────
progress_bar = st.progress(0.0, text="Waiting")

if workflow == "VOD → Live (full-file verticalize first)":
    st.markdown("### 4A) VOD → Live")
    if source_kind in ("RTMP URL", "SRT URL"):
        st.warning(
            "VOD → Live is intended for file-like sources. "
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
        "Start VOD → Live push",
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
        st.success("VOD → Live push started.")

else:
    st.markdown("### 4B) Delayed realtime")
    st.caption(
        "Horizontal file / RTMP / SRT → frame-by-frame vertical output with ~20–25 s delay. "
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

# ── Stop / auto-refresh controls ─────────────────────────────────────────────
left_action, right_action = st.columns([1, 1])
with left_action:
    if st.button(
        "Stop current live push",
        disabled=not bool(st.session_state.live_session),
    ):
        if cf_cfg and st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        st.session_state.live_session = None
        st.info("Current live session stopped and Cloudflare input disabled.")
with right_action:
    auto_refresh = st.checkbox("Auto-refresh session status", value=True)

# ── Section 5: Live session status & playback ─────────────────────────────────
if st.session_state.live_session:
    session = st.session_state.live_session
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

    # ── Primary metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline status", session_status)
    m2.metric("Working source", stats.get("working_resolution", "-"))
    m3.metric("Delay frames", stats.get("delay_frames", "-"))
    m4.metric("Frames out", stats.get("frames_out", 0))

    # ── Secondary metrics row ──
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Frames in", stats.get("frames_in", 0))
    m6.metric("Buffer len", stats.get("buffer_len", 0))
    m7.metric("Placeholder frames", stats.get("placeholder_frames", 0))
    m8.metric("Source stalls", stats.get("source_stalls", 0))

    # ── Panel + detection row ──
    m9, m10, m11, m12 = st.columns(4)
    panel_faces = stats.get("panel_active_faces", None)
    panel_mode_active = stats.get("panel_mode", False)
    m9.metric(
        "Panel faces",
        panel_faces if panel_faces is not None else "—",
        help="Active tracked faces in panel mode (— when panel mode is off).",
    )
    m10.metric("Ball confidence", f"{stats.get('ball_confidence', 0.0):.2f}")
    overlay_top = stats.get("overlay_top", False)
    overlay_bot = stats.get("overlay_bottom", False)
    m11.metric("Top overlay", "✓" if overlay_top else "✗")
    m12.metric("Bottom overlay", "✓" if overlay_bot else "✗")

    # ── Status banners ──
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
        "ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"
    }:
        st.error("Worker hit an error or source ended. Check the FFmpeg log below.")

    with st.expander(
        "FFmpeg push log",
        expanded=session_status in {
            "ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"
        },
    ):
        log_path = getattr(session, "log_path", "")
        if log_path:
            st.code(backend.read_log_tail(log_path) or "(empty)", language="bash")
        else:
            st.code("(no log path available)", language="bash")

    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("5) In-app playback")
    iframe_url = getattr(session, "iframe_url", "")
    if iframe_url:
        iframe_html = (
            '<div style="position:relative;padding-top:177.78%;max-width:360px;margin:0 auto;">'
            + f'<iframe src="{iframe_url}" '
            + 'style="border:none;position:absolute;top:0;left:0;height:100%;width:100%;'
            + 'border-radius:16px;overflow:hidden;" '
            + 'allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;" '
            + 'allowfullscreen="true"></iframe>'
            + "</div>"
        )
        components.html(iframe_html, height=760, scrolling=False)

    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### What's included")
st.markdown(
    "- **Workflow switch** — VOD→Live or delayed realtime\n"
    "- **Source-type switch** — file upload, RTMP, SRT, local path / URL\n"
    "- **Panel discussion mode** — 1-up / 2-up / 3-up / 4-up auto-layout with jitter-free tracking\n"
    "- **Sport profiles** — auto / soccer / basketball / cricket with ball tracking\n"
    "- **Overlay composite** — preserves top scorecard and optional bottom lower-third\n"
    "- **Cloudflare startup priming** — placeholder frames avoid 'stream not started' errors\n"
    "- **Stale session protection** — safe across Streamlit redeploys\n"
    "- **Full stats dashboard** — faces, ball confidence, overlays, buffer, stalls"
)
