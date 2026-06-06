
from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(page_title="VideoForge Studio", page_icon="▶️", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
      .vf-card {border: 1px solid rgba(148,163,184,.24); border-radius: 18px; padding: 18px; background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color: #e5e7eb; box-shadow: 0 14px 30px rgba(2,6,23,.18);}
      .vf-chip {display:inline-block; padding: 6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right: 6px;}
      .vf-hero {border: 1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background: radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color: white;}
      .vf-title {font-size: 1.85rem; font-weight: 800; margin-bottom: 4px;}
      .vf-subtitle {font-size: .98rem; color: #cbd5e1;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_ENHANCE = {
    "denoise": False,
    "denoise_strength": 5,
    "sharpen": False,
    "sharpen_amount": 0.5,
    "sharpen_threshold": 5,
    "upscale": False,
    "upscale_algo": "lanczos",
    "upscale_width": 0,
    "upscale_height": 0,
    "hdr_convert": False,
    "tonemap_algo": "hable",
    "color_enhance": False,
    "vibrance": 0.15,
    "contrast": 1.0,
    "deblock": False,
    "deblock_strength": 5,
    "frame_interp": False,
    "target_fps": 60,
}

DEFAULTS = {
    "workflow_mode": "Restream → HLS",
    "inp": None,
    "name": "",
    "meta": None,
    "sz": 0.0,
    "results": [],
    "result_logs": {},
    "loudness": None,
    "enhance_settings": DEFAULT_ENHANCE.copy(),
    "restream_input_path": None,
    "restream_upload_name": None,
    "restream_meta": None,
    "restream_manifest": None,
    "restream_output_dir": None,
    "restream_zip": None,
    "restream_ladder": [],
    "restream_job": None,
    "restream_manifest_url": None,
    "restream_public_asset_url": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def render_header():
    st.markdown(
        """
        <div class='vf-hero'>
          <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
            <div>
              <div class='vf-title'>VideoForge Studio</div>
              <div class='vf-subtitle'>Streamlit UI triggers worker/FFmpeg job → HLS written to public/static origin → UI receives public master.m3u8 → embedded HLS.js playback + player analytics</div>
            </div>
            <div>
              <span class='vf-chip'>Public/static origin</span>
              <span class='vf-chip'>ABR capped to 360p / 540p</span>
              <span class='vf-chip'>Embedded HLS.js</span>
              <span class='vf-chip'>Player event analytics</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_info(meta: dict, size_mb: float):
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown("**📊 Source Media Info**")
    c1, c2 = st.columns(2)
    c1.metric("Duration", f"{meta['duration']:.1f}s")
    c2.metric("Resolution", f"{meta['width']}×{meta['height']}")
    c1.metric("Frame Rate", f"{meta['fps']} fps")
    c2.metric("Codec", meta['vcodec'].upper())
    c1.metric("Bitrate", f"{meta['vbitrate_kbps']} kbps" if meta['vbitrate_kbps'] else "—")
    c2.metric("File Size", f"{size_mb:.2f} MB")
    if meta.get("has_audio"):
        st.markdown("---")
        a1, a2 = st.columns(2)
        a1.metric("Audio codec", backend.format_audio_codec(meta['acodec']))
        a2.metric("Channels", backend.format_channels(meta['channels']))
        a1.metric("Sample rate", backend.format_sample_rate(meta['sample_rate']))
        a2.metric("Audio bitrate", f"{meta['abitrate_kbps']} kbps" if meta['abitrate_kbps'] > 0 else "Variable")
    st.markdown("</div>", unsafe_allow_html=True)


def render_abr_ladder_table(ladder: list[dict]):
    if not ladder:
        st.caption("Single rendition mode selected")
        return
    df = pd.DataFrame([
        {"Variant": v["name"], "Resolution": f"{v['width']}×{v['height']}", "Video bitrate": v["video_bitrate"], "Max rate": v["maxrate"], "Audio bitrate": v["audio_bitrate"], "Bandwidth": v["bandwidth"]}
        for v in ladder
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_manifest_playback(manifest_url: str | None, title: str):
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown("**▶ Public HLS Playback Endpoint**")
    if not manifest_url:
        st.caption("Public master.m3u8 URL will appear here after the job starts.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    st.code(manifest_url)
    st.caption("Embedded player uses the public master.m3u8 URL instead of localhost, so browser reachability is no longer tied to the runtime host.")
    components.html(backend.build_hlsjs_player_html(manifest_url, title=title, autoplay=True, muted=True, low_latency=True), height=970, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_job_status(job: dict | None):
    if not job:
        return
    proc = job.get("proc")
    running = proc is not None and proc.poll() is None
    st.markdown(f"**Status:** {'🟢 Running / writing HLS to origin' if running else '⚪ Finished / stopped'}")
    st.caption(f"Master manifest file: {job.get('master_manifest', '—')}")
    st.caption(f"Public playback URL: {job.get('manifest_url', '—')}")
    if os.path.exists(job.get("log_path", "")):
        with st.expander("FFmpeg log"):
            with open(job["log_path"], "r", encoding="utf-8", errors="ignore") as fp:
                st.code(fp.read()[-12000:] or "(empty)", language="bash")


def restream_page():
    st.subheader("📡 Restream → HLS")
    st.info("Requested flow implemented: user upload → Streamlit triggers worker/FFmpeg → manifest + segments are written to a public/static origin → UI gets the public master.m3u8 URL → embedded HLS.js plays that URL → analytics panel reads real player events.")

    st.markdown("### 1) Public/static origin configuration")
    default_origin_dir = os.getenv("PUBLIC_HLS_ORIGIN_DIR", "./public_hls")
    default_origin_url = os.getenv("PUBLIC_HLS_ORIGIN_BASE_URL", "")
    o1, o2 = st.columns([2, 3])
    origin_dir = o1.text_input("Local/static origin directory", value=default_origin_dir, help="Directory on disk that is already exposed by your web server / CDN / static file origin.")
    origin_url = o2.text_input("Public origin base URL", value=default_origin_url, placeholder="https://cdn.example.com/hls", help="Public URL mapped to the same directory above.")

    try:
        origin_cfg = backend.resolve_origin_config(origin_dir, origin_url) if origin_dir and origin_url else None
        if origin_cfg:
            st.success("Origin configuration looks valid. New jobs will write into that origin path and return a public URL.")
        else:
            st.warning("Provide both origin directory and public base URL before starting a job.")
    except Exception as exc:
        origin_cfg = None
        st.error(str(exc))

    st.markdown("### 2) Transcode / packaging settings")
    c1, c2, c3, c4 = st.columns(4)
    aspect = c1.selectbox("Output layout", list(backend.ASPECT_PRESETS.keys()), index=list(backend.ASPECT_PRESETS.keys()).index("16:9 Landscape"))
    target_fps = c2.selectbox("Output FPS", ["Source", 24, 25, 30, 50, 60], index=0)
    segment_seconds = c3.slider("HLS segment (s)", 1, 6, 2)
    preset = c4.selectbox("x264 preset", ["ultrafast", "superfast", "veryfast", "faster", "fast"], index=1)
    abr_enabled = st.checkbox("Enable ABR ladder", value=True, help="Capped to 360p and 540p for faster startup and lower origin load.")
    preview_ladder = backend.build_abr_ladder(aspect_label=aspect)
    with st.expander("Preview ladder", expanded=True):
        render_abr_ladder_table(preview_ladder if abr_enabled else [])
    fps_value = None if target_fps == "Source" else int(target_fps)

    mode = st.radio("Job mode", ["Upload file → public HLS", "Live ingest → public HLS"], horizontal=True)

    if mode == "Upload file → public HLS":
        upl = st.file_uploader(f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)", type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"], key="restream_upload")
        if not upl:
            st.caption("Upload a file to start a public-origin HLS job.")
            return
        if getattr(upl, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {upl.size / (1024 * 1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.")
            return
        if st.session_state.restream_upload_name != f"{upl.name}:{getattr(upl, 'size', 0)}":
            suffix = os.path.splitext(upl.name)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upl.read())
                st.session_state.restream_input_path = tmp.name
            st.session_state.restream_upload_name = f"{upl.name}:{getattr(upl, 'size', 0)}"
            st.session_state.restream_meta = backend.probe(st.session_state.restream_input_path)
        src_path = st.session_state.restream_input_path
        meta = st.session_state.restream_meta
        st.caption(f"Source: {meta['width']}×{meta['height']} @ {meta['fps']} fps · {meta['duration']:.1f}s · {meta['vcodec'].upper()}")

        current_job = st.session_state.get("restream_job")
        active = bool(current_job and current_job.get("proc") and current_job["proc"].poll() is None)
        if st.button("🎬 Start public-origin HLS job", type="primary", disabled=not origin_cfg or active):
            result = backend.start_vod_public_preview_job(input_source=src_path, asset_name=upl.name, aspect_label=aspect, preset=preset, fps=fps_value, segment_seconds=segment_seconds, abr_enabled=abr_enabled, origin=origin_cfg, src_meta=meta)
            st.session_state.restream_output_dir = result.output_dir
            st.session_state.restream_manifest = result.master_manifest
            st.session_state.restream_manifest_url = result.manifest_url
            st.session_state.restream_public_asset_url = result.public_asset_url
            st.session_state.restream_ladder = result.ladder
            st.session_state.restream_job = result.job
            st.session_state.restream_zip = None
            st.success("Worker/FFmpeg job started. HLS is being written into the configured public/static origin.")

        current_job = st.session_state.get("restream_job")
        job_done = bool(current_job and current_job.get("proc") and current_job["proc"].poll() is not None)
        if job_done and not st.session_state.get("restream_zip"):
            outputs = backend.collect_job_outputs(current_job)
            st.session_state.restream_zip = outputs.get("zip_bytes")

        left, right = st.columns([2, 3], gap="large")
        with left:
            render_source_info(meta, os.path.getsize(src_path) / (1024 * 1024))
            render_job_status(current_job)
            if st.session_state.get("restream_public_asset_url"):
                st.caption(f"Public asset path: {st.session_state.restream_public_asset_url}")
            if st.session_state.get("restream_ladder"):
                st.markdown("#### Active ladder")
                render_abr_ladder_table(st.session_state.restream_ladder)
            if st.session_state.get("restream_zip"):
                st.download_button("⬇ Download output bundle (.zip)", data=st.session_state.restream_zip, file_name="public_origin_hls_bundle.zip", mime="application/zip")
        with right:
            render_manifest_playback(st.session_state.get("restream_manifest_url"), title=f"{aspect} public-origin playback")
    else:
        ingest_url = st.text_input("Ingest URL", placeholder="rtmp://host/app/streamKey or srt://host:port?mode=caller&latency=120")
        current_job = st.session_state.get("restream_job")
        active = bool(current_job and current_job.get("proc") and current_job["proc"].poll() is None)
        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("▶ Start live public-origin HLS", type="primary", disabled=not ingest_url or not origin_cfg or active):
                job, ladder, public_asset_url = backend.start_live_hls_job(input_source=ingest_url.strip(), aspect_label=aspect, preset=preset, fps=fps_value, segment_seconds=segment_seconds, abr_enabled=abr_enabled, origin=origin_cfg)
                st.session_state.restream_job = job
                st.session_state.restream_ladder = ladder
                st.session_state.restream_manifest_url = job.get("manifest_url")
                st.session_state.restream_public_asset_url = public_asset_url
                st.session_state.restream_zip = None
                st.success("Live worker/FFmpeg job started. HLS is being written into the configured public/static origin.")
        with stop_col:
            if st.button("⏹ Stop live job", disabled=not active):
                backend.stop_live_job(current_job)
                st.success("Live job stopped.")
        left, right = st.columns([2, 3], gap="large")
        with left:
            render_job_status(st.session_state.get("restream_job"))
            if st.session_state.get("restream_public_asset_url"):
                st.caption(f"Public asset path: {st.session_state.restream_public_asset_url}")
            if st.session_state.get("restream_ladder"):
                st.markdown("#### Active ladder")
                render_abr_ladder_table(st.session_state.restream_ladder)
        with right:
            render_manifest_playback(st.session_state.get("restream_manifest_url"), title=f"{aspect} live public-origin playback")

    st.divider()
    st.markdown("### Flow implemented in code")
    st.markdown(
        "- User uploads video.\n"
        "- Streamlit UI triggers worker/FFmpeg job.\n"
        "- Worker writes manifest + segments into the configured public/static origin directory.\n"
        "- UI receives the public `master.m3u8` URL derived from the configured public base URL.\n"
        "- Embedded HLS.js plays that public URL.\n"
        "- Analytics panel tracks real player events (attach, parse, switch, buffer, drops, latency)."
    )
    st.markdown("### Deployment note")
    st.markdown(
        "You must map the **local/static origin directory** to a **real public URL** using your web server, object storage website hosting, CDN, or reverse proxy. Example: local path `/var/www/hls` ↔ public URL `https://cdn.example.com/hls`."
    )


def main():
    render_header()
    if not backend.ffmpeg_ok():
        st.error("FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.")
        st.stop()
    workflow = st.radio("Workflow", ["Restream → HLS", "Test Player"], horizontal=True)
    if workflow == "Restream → HLS":
        restream_page()
    else:
        test_player = st.text_input("Public master.m3u8 URL", placeholder="https://cdn.example.com/hls/asset/master.m3u8")
        if test_player:
            render_manifest_playback(test_player, title="External public URL playback")
        else:
            st.info("Paste a public master.m3u8 URL to test playback.")
    st.markdown("---")
    st.caption("VideoForge Studio · public-origin HLS workflow · worker/FFmpeg job writes to static origin · UI plays public master.m3u8 URL")


if __name__ == "__main__":
    main()
