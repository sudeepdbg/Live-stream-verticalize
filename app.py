
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

DEFAULTS = {
    "restream_input_path": None,
    "restream_upload_name": None,
    "restream_meta": None,
    "restream_result": None,
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
              <div class='vf-subtitle'>Free-prototype architecture: Streamlit UI triggers FFmpeg → HLS packaged locally → deployed to Cloudflare Pages Direct Upload → public master.m3u8 played in HLS.js</div>
            </div>
            <div>
              <span class='vf-chip'>Cloudflare Pages Direct Upload</span>
              <span class='vf-chip'>ABR 360p / 540p</span>
              <span class='vf-chip'>HLS.js embedded player</span>
              <span class='vf-chip'>Player-side analytics</span>
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


def render_ladder(ladder: list[dict]):
    df = pd.DataFrame([
        {"Variant": x["name"], "Resolution": f"{x['width']}×{x['height']}", "Video bitrate": x["video_bitrate"], "Max rate": x["maxrate"], "Audio bitrate": x["audio_bitrate"], "Bandwidth": x["bandwidth"]}
        for x in ladder
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_player(manifest_url: str | None, title: str):
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown("**▶ Public HLS.js Playback**")
    if not manifest_url:
        st.caption("Public `master.m3u8` URL appears here after Cloudflare Pages deployment finishes.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    st.code(manifest_url)
    components.html(backend.build_hlsjs_player_html(manifest_url, title=title, autoplay=True, muted=True, low_latency=True), height=980, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def restream_pages_workflow():
    st.subheader("📡 Upload file → Cloudflare Pages origin → HLS.js playback")
    st.info("This final prototype path is optimized for a free demo: Streamlit handles UI + FFmpeg processing, but Cloudflare Pages is the public origin for `master.m3u8` and segments.")

    st.markdown("### 1) Cloudflare Pages Direct Upload settings")
    c1, c2 = st.columns(2)
    with c1:
        project_name = st.text_input("Pages project name", value=os.getenv("CLOUDFLARE_PAGES_PROJECT_NAME", ""), help="Direct Upload project in Cloudflare Pages. Production URL is <project>.pages.dev.")
        branch_prefix = st.text_input("Preview branch prefix", value=os.getenv("CLOUDFLARE_PAGES_BRANCH_PREFIX", "preview"), help="Each upload can be pushed to a preview branch alias for a unique URL.")
    with c2:
        account_id = st.text_input("Cloudflare account ID", value=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))
        api_token = st.text_input("Cloudflare API token", value=os.getenv("CLOUDFLARE_API_TOKEN", ""), type="password")
    use_production_branch = st.checkbox("Deploy to production branch / root pages.dev URL", value=False, help="Unchecked is safer for prototypes because each upload creates a preview alias URL instead of overwriting the root deployment.")

    cf_cfg = None
    if project_name and account_id and api_token:
        try:
            cf_cfg = backend.cloudflare_config_from_inputs(project_name, account_id, api_token, branch_prefix, use_production_branch)
            st.success("Cloudflare configuration looks valid.")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.warning("Fill in Cloudflare Pages project name, account ID, and API token.")

    st.markdown("### 2) HLS packaging settings")
    s1, s2, s3, s4 = st.columns(4)
    aspect = s1.selectbox("Output layout", list(backend.ASPECT_PRESETS.keys()), index=list(backend.ASPECT_PRESETS.keys()).index("16:9 Landscape"))
    target_fps = s2.selectbox("Output FPS", ["Source", 24, 25, 30, 50, 60], index=0)
    segment_seconds = s3.slider("HLS segment (s)", 1, 6, backend.DEFAULT_SEGMENT_SECONDS)
    preset = s4.selectbox("x264 preset", ["ultrafast", "superfast", "veryfast", "faster", "fast"], index=1)
    ladder = backend.build_abr_ladder(aspect_label=aspect)
    with st.expander("ABR ladder (capped for free prototype)", expanded=True):
        render_ladder(ladder)

    uploaded = st.file_uploader(f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)", type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"], key="upload_pages_hls")
    if not uploaded:
        st.caption("Upload a file to generate HLS and deploy it to Cloudflare Pages.")
        return
    if getattr(uploaded, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is {uploaded.size / (1024 * 1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.")
        return
    if st.session_state.restream_upload_name != f"{uploaded.name}:{getattr(uploaded, 'size', 0)}":
        suffix = os.path.splitext(uploaded.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.restream_input_path = tmp.name
        st.session_state.restream_upload_name = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        st.session_state.restream_meta = backend.probe(st.session_state.restream_input_path)
        st.session_state.restream_result = None

    src_path = st.session_state.restream_input_path
    meta = st.session_state.restream_meta
    size_mb = os.path.getsize(src_path) / (1024 * 1024)
    st.caption(f"Source: {meta['width']}×{meta['height']} @ {meta['fps']} fps · {meta['duration']:.1f}s · {meta['vcodec'].upper()}")

    run_disabled = (cf_cfg is None) or (not backend.ffmpeg_ok()) or (not backend.wrangler_ok())
    if st.button("🎬 Generate HLS + Deploy to Cloudflare Pages", type="primary", disabled=run_disabled):
        with st.spinner("FFmpeg is generating HLS and Wrangler is deploying the folder to Cloudflare Pages…"):
            fps_value = None if target_fps == "Source" else int(target_fps)
            result = backend.package_and_deploy_vod_to_pages(
                input_source=src_path,
                asset_name=uploaded.name,
                aspect_label=aspect,
                preset=preset,
                fps=fps_value,
                segment_seconds=segment_seconds,
                cf=cf_cfg,
                src_meta=meta,
            )
            st.session_state.restream_result = result

    result = st.session_state.restream_result

    left, right = st.columns([2, 3], gap="large")
    with left:
        render_source_info(meta, size_mb)
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        st.markdown("**🧾 Job status**")
        st.metric("FFmpeg available", "Yes" if backend.ffmpeg_ok() else "No")
        st.metric("Wrangler available", "Yes" if backend.wrangler_ok() else "No")
        if result:
            st.metric("Deployment status", "Success" if result.ok else "Failed")
            if result.site_url:
                st.caption(f"Site URL: {result.site_url}")
            if result.branch_alias:
                st.caption(f"Preview branch alias: {result.branch_alias}")
            with st.expander("FFmpeg log"):
                st.code(result.ffmpeg_log or "(empty)", language="bash")
            with st.expander("Wrangler / Cloudflare Pages deploy log"):
                st.code(result.deploy_log or "(empty)", language="bash")
            if result.zip_bytes:
                st.download_button("⬇ Download generated HLS bundle (.zip)", data=result.zip_bytes, file_name="cloudflare_pages_hls_bundle.zip", mime="application/zip")
        else:
            st.caption("No deployment has run yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        if result and result.ladder:
            st.markdown("### Active ladder")
            render_ladder(result.ladder)
    with right:
        render_player(result.manifest_url if result else None, title=f"{aspect} Cloudflare Pages playback")
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        st.markdown("**📡 Playback analytics companion**")
        components.html(backend.build_player_analytics_html(meta, source_label="Source-side analytics companion"), height=360, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### How this prototype works")
    st.markdown(
        "- **Streamlit UI triggers the job** after upload.\n"
        "- **FFmpeg** generates ABR HLS (360p + 540p) locally.\n"
        "- **Wrangler Pages Direct Upload** pushes the generated folder to Cloudflare Pages.\n"
        "- The UI receives a **public `master.m3u8` URL** (`<branch>.<project>.pages.dev/master.m3u8` for preview, or `<project>.pages.dev/master.m3u8` for production).\n"
        "- Embedded **HLS.js** plays that public URL.\n"
        "- The analytics widget reads **actual player events** from the embedded player."
    )
    st.markdown("### Important notes for free prototype")
    st.markdown(
        "- This implementation is best for **prototype / demo VOD packaging** rather than true continuous livestream origin updates.\n"
        "- Cloudflare Pages Direct Upload is static deployment oriented, so a real always-on low-latency live origin is not the target for Pages.\n"
        "- Use the **preview branch alias** option for safer experiments; deploy to production only when you want to overwrite the root Pages URL."
    )


def main():
    render_header()
    if not backend.ffmpeg_ok():
        st.error("FFmpeg / ffprobe not found. Make sure ffmpeg is installed in your runtime.")
        st.stop()
    restream_pages_workflow()
    st.markdown("---")
    st.caption("VideoForge Studio · Streamlit UI + processing · Cloudflare Pages origin · embedded HLS.js player")


if __name__ == "__main__":
    main()
