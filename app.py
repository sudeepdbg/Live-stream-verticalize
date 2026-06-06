
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend


# =============================
# Page setup / style
# =============================
st.set_page_config(
    page_title="VideoForge Studio",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .vf-card {
        border: 1px solid rgba(148,163,184,.24);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92));
        color: #e5e7eb;
        box-shadow: 0 14px 30px rgba(2,6,23,.18);
      }
      .vf-muted {color:#94a3b8; font-size: .9rem;}
      .vf-chip {
        display:inline-block; padding: 6px 12px; border-radius:999px; font-size:.78rem;
        border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right: 6px;
      }
      .vf-hero {
        border: 1px solid rgba(59,130,246,.18); border-radius: 22px; padding: 20px 22px; margin-bottom: 16px;
        background: radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%),
                    linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: white;
      }
      .vf-title {font-size: 1.85rem; font-weight: 800; margin-bottom: 4px;}
      .vf-subtitle {font-size: .98rem; color: #cbd5e1;}
      .vf-kpi {background:#0f172a; border:1px solid rgba(148,163,184,.18); border-radius:14px; padding:14px;}
      .stTabs [data-baseweb="tab-list"] {gap: 8px;}
      .stTabs [data-baseweb="tab"] {height: 42px; border-radius: 10px; padding: 0 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================
# Session state
# =============================
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
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =============================
# Utility renderers
# =============================
def vmaf_display(v):
    if v is None:
        return "—"
    if v >= 93:
        return f"{v:.1f} · Excellent"
    if v >= 80:
        return f"{v:.1f} · Good"
    if v >= 60:
        return f"{v:.1f} · Fair"
    return f"{v:.1f} · Poor"


def ssim_display(v):
    if v is None:
        return "—"
    label = "Excellent" if v >= 0.98 else "Good" if v >= 0.95 else "Fair" if v >= 0.90 else "Poor"
    return f"{v:.5f} · {label}"


def psnr_display(v):
    if v is None:
        return "—"
    tag = "Excellent" if v >= 50 else "Good" if v >= 40 else "Acceptable" if v >= 30 else "Poor"
    return f"{v:.2f} dB · {tag}"


def reset_results():
    st.session_state.results = []
    st.session_state.result_logs = {}


def render_header():
    st.markdown(
        """
        <div class='vf-hero'>
          <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
            <div>
              <div class='vf-title'>VideoForge Studio</div>
              <div class='vf-subtitle'>Professional encoding · restream/live HLS packaging · playback analytics · ABR ladder generation</div>
            </div>
            <div>
              <span class='vf-chip'>H.264 / HEVC / AV1</span>
              <span class='vf-chip'>VMAF / PSNR / SSIM</span>
              <span class='vf-chip'>HLS ABR ladder</span>
              <span class='vf-chip'>16:9 · 9:16 · 1:1 · 4:5</span>
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
        sync_diff = abs((meta.get("audio_duration") or 0) - (meta.get("duration") or 0))
        st.caption(f"A/V sync: {'✓ Synced' if sync_diff < 0.1 else f'⚠ {sync_diff:.2f}s drift'}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_player_analytics(meta: dict, label: str):
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown("**📡 Playback & ABR Analytics**")
    st.caption("Analytics shown inline so operator can monitor playback and delivery behavior from the same screen.")
    components.html(backend.build_player_analytics_html(meta, source_label=label), height=560, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_abr_ladder_table(ladder: list[dict]):
    if not ladder:
        return
    df = pd.DataFrame(
        [{
            "Variant": v["name"],
            "Resolution": f"{v['width']}×{v['height']}",
            "Video bitrate": v["video_bitrate"],
            "Max rate": v["maxrate"],
            "Audio bitrate": v["audio_bitrate"],
            "Bandwidth": v["bandwidth"],
        } for v in ladder]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_live_log(job: dict | None):
    if not job:
        return
    proc = job.get("proc")
    running = proc is not None and proc.poll() is None
    st.markdown(f"**Status:** {'🟢 Running' if running else '⚪ Stopped'}")
    st.caption(f"Master manifest: {job.get('master_manifest', '—')}")
    if os.path.exists(job.get("log_path", "")):
        with st.expander("FFmpeg live log"):
            with open(job["log_path"], "r", encoding="utf-8", errors="ignore") as fp:
                st.code(fp.read()[-12000:] or "(empty)", language="bash")


# =============================
# Encoder workflow
# =============================
def encoder_page():
    st.subheader("⚙️ Encoder")
    uploaded = st.file_uploader(
        "Drop a video or click to browse",
        type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"],
        key="encoder_upload",
    )
    if not uploaded:
        st.info("Upload a source file to unlock preview, analytics, enhancement controls, and encoding.")
        return

    if getattr(uploaded, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"This file is {uploaded.size / (1024 * 1024):.1f} MB. Current UI target is ≤ {backend.MAX_UPLOAD_MB} MB.")
        return

    suffix = os.path.splitext(uploaded.name)[-1].lower()
    if st.session_state.name != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.inp = tmp.name
        st.session_state.name = uploaded.name
        st.session_state.meta = backend.probe(st.session_state.inp)
        st.session_state.sz = os.path.getsize(st.session_state.inp) / (1024 * 1024)
        st.session_state.loudness = None
        meta = st.session_state.meta
        st.session_state.enhance_settings["upscale_width"] = meta["width"] * 2
        st.session_state.enhance_settings["upscale_height"] = meta["height"] * 2
        reset_results()

    meta = st.session_state.meta
    size_mb = st.session_state.sz
    src = st.session_state.inp

    left, mid, right = st.columns([3, 2, 2], gap="large")
    with left:
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        st.markdown("**▶ Preview**")
        st.video(src)
        st.markdown("</div>", unsafe_allow_html=True)
    with mid:
        render_source_info(meta, size_mb)
        if st.button("🔊 Measure Loudness"):
            with st.spinner("Measuring loudness…"):
                st.session_state.loudness = backend.probe_loudness(src)
        if st.session_state.loudness:
            ld = st.session_state.loudness
            st.info(f"Mean: {ld.get('mean_volume')} dBFS · Peak: {ld.get('max_volume')} dBFS")
    with right:
        render_player_analytics(meta, label="Source playback")

    st.divider()
    es = st.session_state.enhance_settings
    st.markdown("### ✨ AI Video Enhancement")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.checkbox("Enable temporal denoise", key="denoise", value=es["denoise"])
        es["denoise"] = st.session_state.denoise
        if es["denoise"]:
            es["denoise_strength"] = st.slider("Denoise strength", 1, 10, es["denoise_strength"])

        st.checkbox("Enable adaptive sharpening", key="sharpen", value=es["sharpen"])
        es["sharpen"] = st.session_state.sharpen
        if es["sharpen"]:
            es["sharpen_amount"] = st.slider("Sharpen amount", -1.5, 1.5, es["sharpen_amount"], 0.1)
            es["sharpen_threshold"] = st.slider("Chroma softness", 0, 50, es["sharpen_threshold"])

        st.checkbox("Enable upscaling", key="upscale", value=es["upscale"])
        es["upscale"] = st.session_state.upscale
        if es["upscale"]:
            es["upscale_algo"] = st.selectbox("Upscale algorithm", ["lanczos", "spline", "bicubic"], index=["lanczos", "spline", "bicubic"].index(es["upscale_algo"]))
            es["upscale_width"] = st.number_input("Target width", min_value=meta["width"], max_value=7680, value=max(meta["width"], es["upscale_width"] or meta["width"] * 2), step=max(2, meta["width"] // 2))
            es["upscale_height"] = st.number_input("Target height", min_value=meta["height"], max_value=4320, value=max(meta["height"], es["upscale_height"] or meta["height"] * 2), step=max(2, meta["height"] // 2))

    with e2:
        is_hdr_source = meta.get("bit_depth", 8) >= 10
        es["hdr_convert"] = st.checkbox("HDR → SDR tonemap", value=es["hdr_convert"] if is_hdr_source else False, disabled=not is_hdr_source)
        if es["hdr_convert"] and is_hdr_source:
            es["tonemap_algo"] = st.selectbox("Tonemap algorithm", ["hable", "reinhard", "mobius", "linear"], index=["hable", "reinhard", "mobius", "linear"].index(es["tonemap_algo"]))

        es["color_enhance"] = st.checkbox("Color enhancement", value=es["color_enhance"])
        if es["color_enhance"]:
            es["vibrance"] = st.slider("Vibrance", -0.5, 0.5, es["vibrance"], 0.05)
            es["contrast"] = st.slider("Contrast", 0.5, 2.0, es["contrast"], 0.05)

        es["deblock"] = st.checkbox("Artifact reduction / deblock", value=es["deblock"])
        if es["deblock"]:
            es["deblock_strength"] = st.slider("Deblock strength", 1, 10, es["deblock_strength"])

        es["frame_interp"] = st.checkbox("Frame interpolation", value=es["frame_interp"])
        if es["frame_interp"]:
            es["target_fps"] = st.selectbox("Target FPS", [30, 48, 60, 120], index=[30, 48, 60, 120].index(es.get("target_fps", 60)) if es.get("target_fps", 60) in [30, 48, 60, 120] else 2)

    active_enhancements = sum(bool(es.get(k)) for k in ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"])
    if active_enhancements:
        st.info(f"Estimated processing time: {backend.estimate_processing_time(meta, es)} · {active_enhancements} enhancement(s) active")

    st.divider()
    s1, s2, s3 = st.columns([2, 1.5, 1.5])
    codec = s1.selectbox("Video codec", ["AVC (H.264)", "HEVC (H.265)", "AV1"])
    crf = s2.slider("CRF", 0, 51, 23)
    do_vmaf = s3.checkbox("VMAF", value=backend.vmaf_ok(), disabled=not backend.vmaf_ok())
    do_psnr = s3.checkbox("PSNR/SSIM", value=True)

    b1, b2, _ = st.columns([1.3, 1.3, 4])
    preview = b1.button("🔍 Preview impact", use_container_width=True)
    go = b2.button("✨ Enhance + Encode", type="primary", use_container_width=True)

    if preview:
        est_meta = meta.copy()
        if es.get("upscale"):
            est_meta["width"] = int(es["upscale_width"])
            est_meta["height"] = int(es["upscale_height"])
        if es.get("frame_interp"):
            est_meta["fps"] = es["target_fps"]
        c1, c2 = st.columns(2)
        c1.metric("Source", f"{meta['width']}×{meta['height']}", f"@ {meta['fps']} fps")
        c2.metric("Estimated output", f"{est_meta['width']}×{est_meta['height']}", f"@ {est_meta['fps']} fps")

    if go:
        codec_short = codec.split()[0].lower()
        enh_tag = "enh_" if active_enhancements > 0 else ""
        out_path = src.replace(suffix, f"_{enh_tag}{codec_short}_crf{crf}.mp4")
        bar = st.progress(0.0, text=f"Initializing {codec}…")
        with st.spinner("Encoding in progress…"):
            ok, msg, ffmpeg_log, enc_t = backend.encode(
                src,
                out_path,
                codec,
                crf,
                es,
                meta,
                progress_cb=lambda p: bar.progress(p, text=f"Processing… {p * 100:.0f}%"),
                duration=meta["duration"],
            )
        if not ok:
            bar.empty()
            st.error(msg)
            if ffmpeg_log:
                with st.expander("FFmpeg log"):
                    st.code(ffmpeg_log, language="bash")
            return

        out_meta = backend.probe(out_path)
        out_sz = os.path.getsize(out_path) / (1024 * 1024)
        saved_pct = (1 - out_sz / size_mb) * 100 if size_mb > 0 else 0.0
        qual = {"psnr": None, "ssim": None, "vmaf": None}
        if do_psnr or do_vmaf:
            with st.spinner("Computing quality metrics…"):
                qual = backend.quality_metrics(src, out_path, do_vmaf)

        result = {
            "codec": codec,
            "crf": crf,
            "size_mb": out_sz,
            "bitrate": out_meta.get("vbitrate_kbps", 0),
            "enc_time": enc_t,
            "saved": saved_pct,
            "cr": size_mb / out_sz if out_sz > 0 else 0.0,
            "psnr": qual["psnr"],
            "ssim": qual["ssim"],
            "vmaf": qual["vmaf"],
            "path": out_path,
            "acodec": meta["acodec"],
            "abitrate": meta["abitrate_kbps"],
            "sample_rate": meta["sample_rate"],
            "channels": meta["channels"],
            "enhancements": {k: v for k, v in es.items() if k in ["denoise","sharpen","upscale","hdr_convert","color_enhance","deblock","frame_interp"] and bool(v)},
            "out_res": f"{out_meta.get('width', meta['width'])}×{out_meta.get('height', meta['height'])}",
            "out_fps": out_meta.get("fps", meta["fps"]),
        }
        st.session_state.result_logs[len(st.session_state.results)] = ffmpeg_log
        st.session_state.results.append(result)
        bar.empty()
        st.success(f"Completed: {codec} CRF {crf} · {out_sz:.2f} MB · saved {saved_pct:.1f}% · {enc_t:.1f}s")

    if not st.session_state.results:
        st.caption("Results will appear here after encoding completes.")
        return

    st.divider()
    st.markdown("### 📈 Analytics Dashboard")
    tab_tbl, tab_chart, tab_dl, tab_logs = st.tabs(["📋 Comparison", "📊 Charts", "⬇ Downloads", "🪵 Logs"])
    results = st.session_state.results

    with tab_tbl:
        rows = []
        for r in results:
            rows.append({
                "Codec": r["codec"],
                "CRF": r["crf"],
                "Size": f"{r['size_mb']:.2f} MB",
                "Bitrate": f"{r['bitrate']} kbps",
                "Ratio": f"{r['cr']:.2f}x",
                "Saved": f"{r['saved']:.1f}%",
                "Time": f"{r['enc_time']:.1f}s",
                "VMAF": vmaf_display(r["vmaf"]),
                "PSNR": psnr_display(r["psnr"]),
                "SSIM": ssim_display(r["ssim"]),
                "Resolution": r["out_res"],
                "Audio": f"{backend.format_audio_codec(r['acodec'])} · {backend.format_channels(r['channels'])}" if r.get("acodec") and r["acodec"] != "unknown" else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Export as CSV",
            data=backend.results_to_csv(results, meta, size_mb),
            file_name="videoforge_results.csv",
            mime="text/csv",
        )

    with tab_chart:
        df = pd.DataFrame([
            {
                "Codec": r["codec"] + (" ✨" if r.get("enhancements") else "") + f" CRF{r['crf']}",
                "File Size (MB)": round(r["size_mb"], 3),
                "Bitrate (kbps)": r["bitrate"],
                "Encode Time (s)": round(r["enc_time"], 2),
                "Space Saved (%)": round(r["saved"], 1),
                "VMAF": r["vmaf"],
                "PSNR (dB)": round(r["psnr"], 2) if r["psnr"] else None,
                "SSIM": round(r["ssim"], 4) if r["ssim"] else None,
            } for r in results
        ])
        c1, c2 = st.columns(2)
        with c1:
            size_df = pd.DataFrame([{"Codec": "Original", "File Size (MB)": round(size_mb, 3)}] + [{"Codec": row["Codec"], "File Size (MB)": row["File Size (MB)"]} for _, row in df.iterrows()]).set_index("Codec")
            st.bar_chart(size_df, use_container_width=True)
            st.bar_chart(df.set_index("Codec")[["Encode Time (s)"]], use_container_width=True)
        with c2:
            brate_df = pd.DataFrame([{"Codec": "Original", "Bitrate (kbps)": meta["vbitrate_kbps"]}] + [{"Codec": row["Codec"], "Bitrate (kbps)": row["Bitrate (kbps)"]} for _, row in df.iterrows()]).set_index("Codec")
            st.bar_chart(brate_df, use_container_width=True)
            st.bar_chart(df.set_index("Codec")[["Space Saved (%)"]], use_container_width=True)
        q_cols = [c for c in ["VMAF", "PSNR (dB)", "SSIM"] if df[c].notna().any()]
        if q_cols:
            st.bar_chart(df.set_index("Codec")[q_cols].dropna(how="all"), use_container_width=True)

    with tab_dl:
        for i, r in enumerate(results):
            left, right = st.columns([1, 3])
            with left:
                fname = f"videoforge_{r['codec'].split()[0].lower()}_crf{r['crf']}.mp4"
                try:
                    with open(r["path"], "rb") as fp:
                        st.download_button(
                            label=f"⬇ {r['codec'].split()[0]} CRF {r['crf']}",
                            data=fp,
                            file_name=fname,
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"dl_{i}",
                        )
                except FileNotFoundError:
                    st.caption("Temp file expired — re-run to download.")
            with right:
                st.caption(f"{r['size_mb']:.2f} MB · {r['bitrate']} kbps · {r['out_res']} · VMAF {r['vmaf'] if r['vmaf'] is not None else '—'}")

    with tab_logs:
        for idx, log_text in st.session_state.result_logs.items():
            r = results[idx]
            with st.expander(f"{r['codec'].split()[0]} CRF {r['crf']} — {r['size_mb']:.2f} MB · {r['enc_time']:.1f}s"):
                st.code(log_text or "(empty)", language="bash")


# =============================
# Test player workflow
# =============================
def test_player_page():
    st.subheader("🎬 Test Player")
    uploaded = st.file_uploader(
        "Upload a source video for playback and analytics",
        type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"],
        key="player_upload",
    )
    playback_url = st.text_input(
        "Optional external playback URL",
        placeholder="https://example.com/master.m3u8",
        help="Use this when you want to verify the final manifest URL from CDN/origin on the same page.",
    )

    if not uploaded and not playback_url:
        st.info("Upload a video and/or provide the HLS playback URL you want to validate.")
        return

    meta = {"width": 1920, "height": 1080, "fps": 30.0, "vbitrate_kbps": 4000}
    src_path = None
    if uploaded:
        suffix = os.path.splitext(uploaded.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            src_path = tmp.name
        meta = backend.probe(src_path)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
        st.markdown("**▶ Playback**")
        if src_path:
            st.video(src_path)
        elif playback_url:
            st.caption("No direct HLS player widget is embedded here because Streamlit does not natively play arbitrary HLS URLs everywhere. Use the analytics panel + your device/browser player for URL validation.")
            st.code(playback_url)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        render_player_analytics(meta, label="Playback analytics")

    if src_path:
        st.divider()
        render_source_info(meta, os.path.getsize(src_path) / (1024 * 1024))


# =============================
# Restream / HLS workflow
# =============================
def restream_page():
    st.subheader("📡 Restream → HLS")
    st.info("Use this workflow for VOD upload → HLS package or live ingest → rolling HLS. The page also exposes the ABR ladder so you can verify what gets generated.")

    mode = st.radio(
        "Source type",
        ["Upload file → HLS (VOD)", "Live ingest → HLS (RTMP / SRT / UDP / HTTP)"],
        horizontal=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    aspect = c1.selectbox("Output layout", list(backend.ASPECT_PRESETS.keys()), index=list(backend.ASPECT_PRESETS.keys()).index("16:9 Landscape"))
    target_fps = c2.selectbox("Output FPS", ["Source", 24, 25, 30, 50, 60], index=0)
    segment_seconds = c3.slider("HLS segment (s)", 2, 10, 4)
    live_playlist = c4.slider("Live playlist size", 3, 20, 6)

    d1, d2, d3 = st.columns(3)
    preset = d1.selectbox("x264 preset", ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"], index=2)
    abr_enabled = d2.checkbox("Enable ABR ladder + master.m3u8", value=True, help="Generates 360p / 540p / 720p / 1080p variants with a master playlist.")
    ladder_preview = backend.build_abr_ladder(aspect_label=aspect)
    d3.metric("Variants", len(ladder_preview) if abr_enabled else 1)

    with st.expander("ABR ladder preview", expanded=True):
        if abr_enabled:
            render_abr_ladder_table(ladder_preview)
        else:
            st.caption("Single rendition mode selected.")

    fps_value = None if target_fps == "Source" else int(target_fps)

    if mode == "Upload file → HLS (VOD)":
        upl = st.file_uploader(
            f"Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)",
            type=["avi", "mp4", "mkv", "mov", "webm", "flv", "ts", "m4v", "mxf"],
            key="restream_upload",
        )
        if not upl:
            st.caption("Upload a file to build the HLS package.")
            return
        if getattr(upl, "size", 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is {upl.size / (1024 * 1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB or increase your app/server limit.")
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
        if src_path and os.path.exists(src_path):
            st.caption(f"Source: {meta['width']}×{meta['height']} @ {meta['fps']} fps · {meta['duration']:.1f}s · {meta['vcodec'].upper()}")
            if st.button("🎬 Build HLS package", type="primary"):
                with st.spinner("Packaging HLS…"):
                    result = backend.build_vod_hls_package(
                        input_source=src_path,
                        asset_name=upl.name,
                        aspect_label=aspect,
                        preset=preset,
                        fps=fps_value,
                        segment_seconds=segment_seconds,
                        abr_enabled=abr_enabled,
                        src_meta=meta,
                    )
                if not result.ok:
                    st.error(result.message)
                    with st.expander("FFmpeg log"):
                        st.code(result.ffmpeg_log or "(empty)", language="bash")
                else:
                    st.session_state.restream_output_dir = result.out_dir
                    st.session_state.restream_manifest = result.master_manifest
                    st.session_state.restream_zip = result.zip_bytes
                    st.session_state.restream_ladder = result.ladder
                    st.success(f"HLS package created. Master manifest: {result.master_manifest}")

        if st.session_state.restream_manifest and st.session_state.restream_zip:
            st.download_button(
                "⬇ Download HLS bundle (.zip)",
                data=st.session_state.restream_zip,
                file_name="hls_output_bundle.zip",
                mime="application/zip",
            )
            st.code(st.session_state.restream_manifest)
            if st.session_state.restream_ladder:
                st.markdown("#### Generated ladder")
                render_abr_ladder_table(st.session_state.restream_ladder)

    else:
        ingest_url = st.text_input("Ingest URL", placeholder="rtmp://host/app/streamKey or srt://host:port?mode=caller&latency=120")
        job = st.session_state.get("restream_job")
        active = bool(job and job.get("proc") and job["proc"].poll() is None)
        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("▶ Start live restream", type="primary", disabled=not ingest_url or active):
                job, ladder = backend.start_live_hls_job(
                    input_source=ingest_url.strip(),
                    aspect_label=aspect,
                    preset=preset,
                    fps=fps_value,
                    segment_seconds=segment_seconds,
                    list_size=live_playlist,
                    abr_enabled=abr_enabled,
                )
                st.session_state.restream_job = job
                st.session_state.restream_ladder = ladder
                st.success("Live restream started.")
        with stop_col:
            if st.button("⏹ Stop live restream", disabled=not active):
                backend.stop_live_job(job)
                st.success("Live restream stopped.")

        render_live_log(st.session_state.get("restream_job"))
        if st.session_state.get("restream_ladder"):
            st.markdown("#### Active live ladder")
            render_abr_ladder_table(st.session_state.restream_ladder)

    st.divider()
    st.markdown("### Suggested production pattern")
    st.markdown("""
- Keep **Streamlit as control-plane UI** and move long-running FFmpeg jobs to a worker/container service.
- Publish the generated **master.m3u8** to a web-accessible origin/CDN so the playback URL can be validated from the same page.
- Add **health telemetry** (segment age, manifest freshness, encoder FPS, CPU/RAM, drop-frame counters) for operations readiness.
- Optionally persist job history + HLS artifacts in object storage for audit/debug and re-download.
""")


# =============================
# Main
# =============================
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
    st.caption("VideoForge Studio · Streamlit UI separated from backend logic · HLS ABR ladder + master playlist ready")


if __name__ == "__main__":
    main()
