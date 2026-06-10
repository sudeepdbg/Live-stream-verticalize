from __future__ import annotations
import os, tempfile, time
import streamlit as st
import streamlit.components.v1 as components
import backend

st.set_page_config(page_title="Dual Flow Vertical Live - Cloudflare Stream", page_icon="\U0001f4f1", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container {padding-top:1rem;padding-bottom:2rem;}
  .card {border:1px solid rgba(148,163,184,.24);border-radius:18px;padding:18px;background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(15,23,42,.92));color:#e5e7eb;box-shadow:0 14px 30px rgba(2,6,23,.18);}
  .hero {border:1px solid rgba(59,130,246,.18);border-radius:22px;padding:20px 22px;margin-bottom:16px;background:radial-gradient(circle at top right,rgba(37,99,235,.18),transparent 28%),linear-gradient(180deg,#0b1220 0%,#0f172a 100%);color:white;}
  .chip {display:inline-block;padding:6px 12px;border-radius:999px;font-size:.78rem;border:1px solid rgba(148,163,184,.25);background:#0f172a;color:#cbd5e1;margin-right:6px;margin-top:6px;}
</style>""", unsafe_allow_html=True)

for key, val in {"input_path": None, "meta": None, "reframed_path": None, "live_session": None}.items():
    if key not in st.session_state:
        st.session_state[key] = val

ls = st.session_state.get("live_session")
if ls is not None and not all(hasattr(ls, a) for a in ["uid", "hls_url", "iframe_url", "log_path"]):
    st.session_state.live_session = None

st.markdown("""
<div class='hero'>
  <div style='display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;'>
    <div>
      <div style='font-size:1.85rem;font-weight:800;margin-bottom:4px;'>Dual Flow Vertical Live &rarr; Cloudflare Stream</div>
      <div style='font-size:.98rem;color:#cbd5e1;'>Supports both (A)&nbsp;VOD&nbsp;&rarr;&nbsp;Live and (B)&nbsp;delayed realtime. RTMP/SRT/URL sources handled in delayed realtime mode.</div>
    </div>
    <div>
      <span class='chip'>sports reframing</span><span class='chip'>panel discussion</span>
      <span class='chip'>overlay composite</span><span class='chip'>ball tracking</span><span class='chip'>RTMP/SRT support</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

if not backend.ffmpeg_ok():
    st.error("FFmpeg / ffprobe not found."); st.stop()

st.subheader("1) Cloudflare Stream Live settings")
lc, rc = st.columns(2)
with lc:
    account_id = st.text_input("Cloudflare account ID", value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
    api_token = st.text_input("Cloudflare Stream API token", value=os.getenv("CLOUDFLARE_STREAM_API_TOKEN", ""), type="password")
with rc:
    customer_code = st.text_input("Cloudflare Stream customer code", value=os.getenv("CLOUDFLARE_STREAM_CUSTOMER_CODE", ""), help="Enter only the code, not the full domain.")
    prefer_low_latency = st.checkbox("Prefer LL-HLS where available", value=False)

cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(account_id, api_token, customer_code, prefer_low_latency)
        st.success("Cloudflare Stream configuration looks valid.")
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning("Fill in Account ID, Stream API token, and customer code.")

st.subheader("2) Select workflow and source")
workflow = st.radio("Workflow", ["VOD \u2192 Live (full-file verticalize first)", "Delayed realtime (frame-by-frame then delay buffer)"], horizontal=True)
source_kind = st.radio("Source type", ["Upload file", "RTMP URL", "SRT URL", "Local path / arbitrary URL"], horizontal=True)

source_value = None; uploaded_name = "source"
if source_kind == "Upload file":
    upl = st.file_uploader(f"Upload source video (max {backend.MAX_UPLOAD_MB} MB)", type=["avi","mp4","mkv","mov","webm","flv","ts","m4v","mxf"])
    if upl:
        if getattr(upl, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {upl.size/(1024*1024):.1f} MB. Keep it \u2264 {backend.MAX_UPLOAD_MB} MB."); st.stop()
        suffix = os.path.splitext(upl.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upl.read()); st.session_state.input_path = tmp.name
        uploaded_name = upl.name; source_value = st.session_state.input_path
elif source_kind == "RTMP URL":
    source_value = st.text_input("RTMP source URL", placeholder="rtmp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov")
elif source_kind == "SRT URL":
    source_value = st.text_input("SRT source URL", placeholder="srt://host:port?mode=caller")
else:
    source_value = st.text_input("Local file path or URL", placeholder="/mount/src/video.mp4 or https://example.com/stream.m3u8")

st.subheader("Reframing settings")
cl, cr = st.columns(2)
with cl:
    smooth_strength = st.slider("Smoothness", 0.90, 0.995, 0.975, 0.005)
    analysis_stride = st.slider("Analysis stride (frames)", 2, 10, 4, 1)
    deadzone_ratio = st.slider("Deadzone ratio", 0.02, 0.10, 0.05, 0.01)
    max_pan_ratio = st.slider("Max pan ratio", 0.005, 0.03, 0.012, 0.001)
with cr:
    sport_profile = st.selectbox("Sport profile", ["auto", "soccer", "basketball", "cricket"], index=0, help="Select the sport for optimised ball tracking and crop behaviour.")
    ball_tracking = st.checkbox("Enable ball tracking", value=True, help="Multi-method ball detection for sports content.")
    overlay_composite = st.checkbox("Preserve broadcast overlays", value=True, help="Detect and composite scorecards / lower-thirds into the vertical output.")
    panel_mode = st.checkbox("Panel discussion mode", value=False, help="When enabled, detects multiple faces and renders a split-screen panel layout (2-4 people). Falls back to normal crop if fewer than 2 faces are stably detected. Uses heavy smoothing to prevent jitter.")

st.subheader("Output settings")
co1, co2, co3 = st.columns(3)
with co1:
    target_w = st.selectbox("Vertical output width", [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9)); st.caption(f"Output: {target_w}\u00d7{target_h}")
with co2:
    delay_seconds = st.slider("Output delay (seconds)", 5, 30, 20, 1)
with co3:
    loop_file = st.checkbox("Loop file source when it ends", value=True)

if source_value:
    is_file = source_kind in ("Upload file", "Local path / arbitrary URL") and os.path.exists(str(source_value))
    is_url = isinstance(source_value, str) and source_value.lower().startswith(("rtmp://", "rtmps://", "srt://", "http://", "https://"))
    if is_file or is_url:
        try:
            st.session_state.meta = backend.probe_source(str(source_value))
        except Exception as exc:
            st.warning(f"Could not probe source: {exc}"); st.session_state.meta = None
        if is_url:
            st.info("For URL-based ingest, metadata can be approximate. The pipeline primes Cloudflare immediately.")

if st.session_state.meta:
    meta = st.session_state.meta; a, b, c = st.columns(3)
    a.metric("Source resolution", f"{int(meta.get('width',0))}\u00d7{int(meta.get('height',0))}")
    b.metric("FPS", f"{meta.get('fps',0)}"); c.metric("Duration", f"{float(meta.get('duration',0.0)):.1f}s")

progress_bar = st.progress(0.0, text="Waiting")
def _progress_cb(pct, msg):
    progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=str(msg))

if workflow.startswith("VOD"):
    st.markdown("### 3A) VOD \u2192 Live")
    if source_kind in ("RTMP URL", "SRT URL"):
        st.warning("VOD \u2192 Live is for file sources. Switch to Delayed realtime for RTMP/SRT.")
    if st.button("Create vertical master", disabled=not bool(source_value) or source_kind in ("RTMP URL", "SRT URL")):
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        try:
            ok, msg = backend.create_vertical_master(str(source_value), out_path, target_w=target_w, target_h=target_h, smooth_strength=smooth_strength, analysis_stride=analysis_stride, deadzone_ratio=deadzone_ratio, max_pan_ratio=max_pan_ratio, sport_profile=sport_profile, ball_tracking=ball_tracking, overlay_composite=overlay_composite, panel_mode=panel_mode, progress_cb=_progress_cb)
        except Exception as exc:
            ok, msg = False, str(exc)
        if ok:
            st.session_state.reframed_path = out_path; progress_bar.progress(1.0, text="Vertical master complete"); st.success("Vertical master created.")
        else:
            st.error(msg)
    if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
        st.video(st.session_state.reframed_path)
    if st.button("Start VOD \u2192 Live push", disabled=not (cf_cfg and st.session_state.reframed_path)):
        if st.session_state.live_session:
            try: backend.stop_live_session(cf_cfg, st.session_state.live_session)
            except Exception: pass
        try:
            session = backend.start_vod_to_live_push(cf_cfg, st.session_state.reframed_path, uploaded_name, loop_input=loop_file)
            st.session_state.live_session = session; st.success("VOD \u2192 Live push started.")
        except Exception as exc:
            st.error(f"Failed: {exc}")
else:
    st.markdown("### 3B) Delayed realtime")
    st.caption("Horizontal file / RTMP / SRT \u2192 frame-by-frame vertical output with configurable delay. Sends placeholder frames immediately so Cloudflare playback starts quickly.")
    if st.button("Start delayed realtime push", disabled=not (cf_cfg and source_value)):
        if st.session_state.live_session:
            try: backend.stop_live_session(cf_cfg, st.session_state.live_session)
            except Exception: pass
        try:
            session = backend.start_realtime_delayed_vertical_push(cf_cfg, str(source_value), uploaded_name if uploaded_name != "source" else (source_value or "source"), target_w=target_w, target_h=target_h, delay_seconds=float(delay_seconds), smooth_strength=float(smooth_strength), analysis_stride=int(analysis_stride), deadzone_ratio=float(deadzone_ratio), max_pan_ratio=float(max_pan_ratio), loop_file=(loop_file and source_kind in ("Upload file", "Local path / arbitrary URL")), pace_input=(source_kind in ("Upload file", "Local path / arbitrary URL")), sport_profile=sport_profile, ball_tracking=ball_tracking, overlay_composite=overlay_composite, panel_mode=panel_mode)
            st.session_state.live_session = session; st.success("Delayed realtime worker started.")
        except Exception as exc:
            st.error(f"Failed: {exc}")

la, ra = st.columns([1, 1])
with la:
    if st.button("Stop current live push", disabled=not bool(st.session_state.live_session)):
        if cf_cfg and st.session_state.live_session:
            try: backend.stop_live_session(cf_cfg, st.session_state.live_session)
            except Exception: pass
        st.session_state.live_session = None; st.info("Live session stopped.")
with ra:
    auto_refresh = st.checkbox("Auto-refresh session status", value=True)

if st.session_state.live_session:
    session = st.session_state.live_session
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Cloudflare playback section**"); st.caption(f"Live input UID: {getattr(session, 'uid', '-')}")
    nhls = (getattr(session, 'hls_url', '') or '').split('?')[0]
    st.text_input("Normal HLS playback URL", value=nhls, key="hls_url_box")
    st.caption("Use this URL in any HLS player if LL-HLS is unstable.")
    st.text_input("Cloudflare iframe player URL", value=getattr(session, 'iframe_url', ''), key="iframe_url_box")
    stats = getattr(session, 'stats', {}) or {}; ss = getattr(session, 'status', 'unknown'); se = getattr(session, 'error', '')
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pipeline status", ss); s2.metric("Working resolution", stats.get("working_resolution", "-"))
    s3.metric("Delay frames", stats.get("delay_frames", "-")); s4.metric("Frames out", stats.get("frames_out", 0))
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Ball confidence", stats.get("ball_confidence", "-"))
    a2.metric("Overlay top", "\u2705" if stats.get("overlay_top") else "\u274c")
    a3.metric("Overlay bottom", "\u2705" if stats.get("overlay_bottom") else "\u274c")
    a4.metric("Panel active", "\u2705" if stats.get("panel_active") else ("\u23f8 off" if not panel_mode else "\u274c"))
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Frames in", stats.get("frames_in", 0)); b2.metric("Buffer length", stats.get("buffer_len", 0))
    b3.metric("Placeholder frames", stats.get("placeholder_frames", 0)); b4.metric("Source stalls", stats.get("source_stalls", 0))
    if se: st.error(se)
    if ss in {"priming_output", "connecting_source", "buffering"}:
        st.warning("Output started, Cloudflare being primed. Wait a few seconds.")
    elif ss == "streaming": st.success("Pipeline actively pushing frames.")
    elif ss in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}:
        st.error("Worker error or source ended. Check log below.")
    with st.expander("FFmpeg push log", expanded=ss in {"ffmpeg_pipe_broken", "worker_error", "ffmpeg_start_failed", "source_ended"}):
        lp = getattr(session, 'log_path', '')
        st.code(backend.read_log_tail(lp) if lp else "(no log)", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("4) In-app playback")
    ifu = getattr(session, 'iframe_url', '')
    if ifu:
        components.html(f'<div style="position:relative;padding-top:177.78%;max-width:360px;margin:0 auto;"><iframe src="{ifu}" style="border:none;position:absolute;top:0;left:0;height:100%;width:100%;border-radius:16px;overflow:hidden;" allow="accelerometer;gyroscope;autoplay;encrypted-media;picture-in-picture;" allowfullscreen="true"></iframe></div>', height=760, scrolling=False)
    if auto_refresh: time.sleep(3); st.rerun()

st.divider()
st.markdown("### Features included")
st.markdown("- **Workflow switch** \u2014 VOD \u2192 Live or Delayed realtime\n- **Source type switch** \u2014 Upload, RTMP, SRT, URL\n- **Sport profiles** \u2014 soccer, basketball, cricket, auto\n- **Ball tracking** \u2014 multi-method (HoughCircles + contour + color blob)\n- **Overlay composite** \u2014 preserves scorecards & lower-thirds\n- **Panel discussion mode** \u2014 split-screen for 2\u20134 people with anti-jitter\n- **Scene-change detection** \u2014 gentle reset on camera cuts\n- **Cloudflare startup priming** \u2014 avoids \u2018stream not started\u2019 issue\n- **Stale session protection** for Streamlit redeploys\n- **Shared playback section** \u2014 HLS + iframe + full diagnostics")
