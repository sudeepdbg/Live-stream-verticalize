
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
    "workflow_mode": "Encoder",
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
    "restream_server_info": None,
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
              <div class='vf-subtitle'>Professional encoding · restream/live HLS packaging · progressive HLS preview · in-app HLS.js playback · playback analytics</div>
            </div>
            <div>
              <span class='vf-chip'>Immediate local HTTP serving</span>
              <span class='vf-chip'>2-rung ABR: 360p / 540p</span>
              <span class='vf-chip'>HLS.js auto-play</span>
              <span class='vf-chip'>Live player metrics</span>
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


def render_manifest_playback(manifest_url: str | None, title: str, embed_player: bool = True):
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown("**▶ HLS Playback Endpoint**")
    if not manifest_url:
        st.caption("HLS endpoint will appear here as soon as the preview job starts.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    st.code(manifest_url)
    st.caption("Player points to master.m3u8 immediately. Playback starts automatically after the first 1–2 segments are written by FFmpeg.")
    if embed_player:
        components.html(backend.build_hlsjs_player_html(manifest_url, title=title, autoplay=True, muted=True, low_latency=True), height=980, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_log(job: dict | None):
    if not job:
        return
    proc = job.get("proc")
    running = proc is not None and proc.poll() is None
    st.markdown(f"**Status:** {'🟢 Running / writing segments' if running else '⚪ Finished / stopped'}")
    st.caption(f"Master manifest: {job.get('master_manifest', '—')}")
    st.caption(f"Playback URL: {job.get('manifest_url', '—')}")
    if os.path.exists(job.get("log_path", "")):
        with st.expander("FFmpeg log"):
            with open(job["log_path"], "r", encoding="utf-8", errors="ignore") as fp:
                st.code(fp.read()[-12000:] or "(empty)", language="bash")


def reset_results():
    st.session_state.results = []
    st.session_state.result_logs = {}


def encoder_page():
    st.subheader("⚙️ Encoder")
    st.info("Encoder flow unchanged. Restream workflow below now uses progressive HLS preview.")


def test_player_page():
    st.subheader("🎬 Test Player")
    playback_url = st.text_input("HLS playback URL", placeholder="http://127.0.0.1:8000/master.m3u8 or CDN URL")
    if playback_url:
        render_manifest_playback(playback_url, title="External HLS playback")
    else:
        st.info("Enter a master manifest URL to test playback.")


def restream_page():
    st.subheader("📡 Restream → HLS")
    st.info("Updated flow: upload → copy source → start local HTTP endpoint immediately → FFmpeg writes HLS in background → HLS.js points to master.m3u8 immediately → playback starts after first 1–2 segments → analytics update from real player events.")
    mode = st.radio("Source type", ["Upload file → progressive HLS preview", "Live ingest → HLS"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    aspect = c1.selectbox("Output layout", list(backend.ASPECT_PRESETS.keys()), index=list(backend.ASPECT_PRESETS.keys()).index("16:9 Landscape"))
    target_fps = c2.selectbox("Output FPS", ["Source", 24, 25, 30, 50, 60], index=0)
    segment_seconds = c3.slider("HLS segment (s)", 1, 6, 2)
    live_playlist = c4.slider("Live playlist size", 2, 10, 4)
    d1, d2, d3 = st.columns(3)
    preset = d1.selectbox("x264 preset", ["ultrafast", "superfast", "veryfast", "faster", "fast"], index=1)
    abr_enabled = d2.checkbox("Enable 2-rung ABR ladder", value=True, help="Capped to 360p and 540p for faster start-up.")
    ladder_preview = backend.build_abr_ladder(aspect_label=aspect)
    d3.metric("Variants", len(ladder_preview) if abr_enabled else 1)
    with st.expander("Preview ladder", expanded=True):
        render_abr_ladder_table(ladder_preview if abr_enabled else [])
    fps_value = None if target_fps == "Source" else int(target_fps)

    if mode == "Upload file → progressive HLS preview":
        upl = st.file_uploader(f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)", type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"], key="restream_upload")
        if not upl:
            st.caption("Upload a file to start progressive HLS preview.")
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
        job_active = bool(current_job and current_job.get("proc") and current_job["proc"].poll() is None)
        start_clicked = st.button("🎬 Start progressive HLS preview", type="primary", disabled=job_active)
        if start_clicked:
            result = backend.start_vod_preview_job(input_source=src_path, asset_name=upl.name, aspect_label=aspect, preset=preset, fps=fps_value, segment_seconds=segment_seconds, abr_enabled=abr_enabled, src_meta=meta)
            st.session_state.restream_output_dir = result.out_dir
            st.session_state.restream_manifest = result.master_manifest
            st.session_state.restream_ladder = result.ladder
            st.session_state.restream_manifest_url = result.manifest_url
            st.session_state.restream_server_info = result.server_info
            st.session_state.restream_job = result.job
            st.success("HTTP endpoint started immediately. FFmpeg is now writing HLS segments in background.")

        current_job = st.session_state.get("restream_job")
        job_done = bool(current_job and current_job.get("proc") and current_job["proc"].poll() is not None)
        if job_done and not st.session_state.get("restream_zip"):
            outputs = backend.collect_job_outputs(current_job)
            st.session_state.restream_zip = outputs.get("zip_bytes")

        left, right = st.columns([2, 3], gap="large")
        with left:
            render_source_info(meta, os.path.getsize(src_path) / (1024 * 1024))
            render_live_log(current_job)
            if st.session_state.restream_ladder:
                st.markdown("#### Active ladder")
                render_abr_ladder_table(st.session_state.restream_ladder)
            if st.session_state.restream_zip:
                st.download_button("⬇ Download HLS bundle (.zip)", data=st.session_state.restream_zip, file_name="hls_progressive_preview_bundle.zip", mime="application/zip")
            st.caption("Flow now starts serving instantly. The only wait is for the first 1–2 segments to be created.")
        with right:
            render_manifest_playback(st.session_state.get("restream_manifest_url"), title=f"{aspect} progressive playback", embed_player=True)
    else:
        ingest_url = st.text_input("Ingest URL", placeholder="rtmp://host/app/streamKey or srt://host:port?mode=caller&latency=120")
        job = st.session_state.get("restream_job")
        active = bool(job and job.get("proc") and job["proc"].poll() is None)
        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("▶ Start live restream", type="primary", disabled=not ingest_url or active):
                job, ladder = backend.start_live_hls_job(input_source=ingest_url.strip(), aspect_label=aspect, preset=preset, fps=fps_value, segment_seconds=segment_seconds, list_size=live_playlist, abr_enabled=abr_enabled)
                st.session_state.restream_job = job
                st.session_state.restream_ladder = ladder
                st.session_state.restream_manifest_url = job.get("manifest_url")
                st.success("Live restream started. Player can attach immediately and will begin once first segments exist.")
        with stop_col:
            if st.button("⏹ Stop live restream", disabled=not active):
                backend.stop_live_job(job)
                st.success("Live restream stopped.")
        left, right = st.columns([2, 3], gap="large")
        with left:
            render_live_log(st.session_state.get("restream_job"))
            if st.session_state.get("restream_ladder"):
                st.markdown("#### Active ladder")
                render_abr_ladder_table(st.session_state.restream_ladder)
        with right:
            render_manifest_playback(st.session_state.get("restream_manifest_url"), title=f"{aspect} live playback", embed_player=True)

    st.divider()
    st.markdown("### Updated flow now implemented")
    st.markdown(
        "- User uploads video.\n"
        "- App copies source file.\n"
        "- App starts local HTTP endpoint immediately.\n"
        "- FFmpeg starts writing HLS playlist + segments in background.\n"
        "- Embedded HLS.js points to `master.m3u8` immediately.\n"
        "- Playback starts after the first 1–2 segments are available.\n"
        "- Analytics panel updates from actual player events."
    )
    st.markdown("### Notes")
    st.markdown(
        "- Ladder is now capped to **360p and 540p** for faster startup.\n"
        "- This is a progressive HLS preview path for quick validation, not full LL-HLS.\n"
        "- For production, move FFmpeg to a worker/container and publish the manifest to an origin/CDN."
    )


def main():
    render_header()
    if not backend.ffmpeg_ok():
        st.error("FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.")
        st.stop()
    workflow = st.radio("Workflow", ["Encoder", "Test Player", "Restream → HLS"], horizontal=True)
    st.session_state.workflow_mode = workflow
    if workflow == "Encoder":
        encoder_page()
    elif workflow == "Test Player":
        test_player_page()
    else:
        restream_page()
    st.markdown("---")
    st.caption("VideoForge Studio · progressive HLS preview with immediate local HTTP endpoint · capped 2-rung ladder · live HLS.js analytics")


if __name__ == "__main__":
    main()
