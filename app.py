from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(page_title='VideoForge Studio', page_icon='▶️', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.vf-card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
.vf-chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right:6px;}
.vf-hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg,#0b1220 0%,#0f172a 100%); color:white;}
.vf-title {font-size:1.85rem; font-weight:800; margin-bottom:4px;}
.vf-subtitle {font-size:.98rem; color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

for key, value in {
    'input_path': None,
    'upload_name': None,
    'meta': None,
    'result': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<div class='vf-hero'>
  <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
    <div>
      <div class='vf-title'>VideoForge Studio</div>
      <div class='vf-subtitle'>Streamlit + GitHub publication layer + Cloudflare Pages Git integration. Streamlit generates HLS, publishes files into a GitHub repo path via the GitHub API, and Cloudflare Pages auto-deploys the repo to a public HLS URL.</div>
    </div>
    <div>
      <span class='vf-chip'>No Wrangler</span>
      <span class='vf-chip'>Cloudflare Pages Git integration</span>
      <span class='vf-chip'>GitHub Contents API</span>
      <span class='vf-chip'>HLS.js embedded player</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not backend.ffmpeg_ok():
    st.error('FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.')
    st.stop()

st.subheader('📡 Upload file → GitHub repo path → Cloudflare Pages → HLS.js playback')
st.info('This version works inside Streamlit Cloud because it uses pure Python HTTPS calls to the GitHub API. Cloudflare Pages must already be connected to the same repository via Git integration.')

st.markdown('### 1) GitHub + Cloudflare Pages settings')
g1, g2 = st.columns(2)
with g1:
    owner = st.text_input('GitHub owner / org', value=os.getenv('GITHUB_OWNER', ''))
    repo = st.text_input('GitHub repo', value=os.getenv('GITHUB_REPO', ''))
    token = st.text_input('GitHub token (PAT)', value=os.getenv('GITHUB_TOKEN', ''), type='password')
    pages_base_url = st.text_input('Cloudflare Pages public base URL', value=os.getenv('PAGES_BASE_URL', ''), placeholder='https://av1-software-player.pages.dev')
with g2:
    target_branch = st.text_input('Target branch to publish into', value=os.getenv('GITHUB_TARGET_BRANCH', 'main'))
    default_branch = st.text_input('Default branch (for branch creation fallback)', value=os.getenv('GITHUB_DEFAULT_BRANCH', 'main'))
    folder_prefix = st.text_input('Repository folder prefix', value=os.getenv('GITHUB_FOLDER_PREFIX', 'public/hls'))
    deploy_hook_url = st.text_input('Optional Cloudflare Deploy Hook URL', value=os.getenv('CF_PAGES_DEPLOY_HOOK', ''), placeholder='https://api.cloudflare.com/client/v4/pages/webhooks/deploy_hooks/...')

gh_cfg = None
if owner and repo and token and target_branch and pages_base_url:
    try:
        gh_cfg = backend.github_config_from_inputs(owner, repo, token, target_branch, pages_base_url, folder_prefix, default_branch, deploy_hook_url)
        st.success('GitHub + Pages configuration looks valid.')
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning('Fill in GitHub owner, repo, token, target branch, and Cloudflare Pages base URL.')

st.markdown('### 2) HLS packaging settings')
s1, s2, s3, s4 = st.columns(4)
aspect = s1.selectbox('Output layout', list(backend.ASPECT_PRESETS.keys()), index=list(backend.ASPECT_PRESETS.keys()).index('16:9 Landscape'))
target_fps = s2.selectbox('Output FPS', ['Source', 24, 25, 30, 50, 60], index=0)
segment_seconds = s3.slider('HLS segment (s)', 2, 10, backend.DEFAULT_SEGMENT_SECONDS)
preset = s4.selectbox('x264 preset', ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast'], index=1)
ladder = backend.build_abr_ladder(aspect_label=aspect)
with st.expander('ABR ladder (capped for prototype)', expanded=True):
    st.dataframe(pd.DataFrame([{ 'Variant': x['name'], 'Resolution': f"{x['width']}×{x['height']}", 'Video bitrate': x['video_bitrate'], 'Max rate': x['maxrate'], 'Audio bitrate': x['audio_bitrate'], 'Bandwidth': x['bandwidth']} for x in ladder]), use_container_width=True, hide_index=True)

uploaded = st.file_uploader(f'Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)', type=['avi','mp4','mkv','mov','webm','flv','ts','m4v','mxf'])
if uploaded:
    if getattr(uploaded, 'size', 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f'File is {uploaded.size / (1024 * 1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.')
        st.stop()
    if st.session_state.upload_name != f"{uploaded.name}:{getattr(uploaded, 'size', 0)}":
        suffix = os.path.splitext(uploaded.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            st.session_state.input_path = tmp.name
        st.session_state.upload_name = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        st.session_state.meta = backend.probe(st.session_state.input_path)
        st.session_state.result = None
else:
    st.caption('Upload a file to generate HLS and publish it into GitHub for Cloudflare Pages to serve.')
    st.stop()

meta = st.session_state.meta
src_path = st.session_state.input_path
size_mb = os.path.getsize(src_path) / (1024 * 1024)
st.caption(f"Source: {meta['width']}×{meta['height']} @ {meta['fps']} fps · {meta['duration']:.1f}s · {meta['vcodec'].upper()}")

disabled = (gh_cfg is None)
if st.button('🎬 Generate HLS + Publish via GitHub', type='primary', disabled=disabled):
    with st.spinner('FFmpeg is generating HLS and the app is publishing the files into GitHub...'):
        fps_value = None if target_fps == 'Source' else int(target_fps)
        st.session_state.result = backend.package_and_publish_via_github(src_path, uploaded.name, aspect, preset, fps_value, segment_seconds, gh_cfg, meta)

result = st.session_state.result
left, right = st.columns([2, 3], gap='large')
with left:
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown('**📊 Source Media Info**')
    c1, c2 = st.columns(2)
    c1.metric('Duration', f"{meta['duration']:.1f}s")
    c2.metric('Resolution', f"{meta['width']}×{meta['height']}")
    c1.metric('Frame Rate', f"{meta['fps']} fps")
    c2.metric('Codec', meta['vcodec'].upper())
    c1.metric('Bitrate', f"{meta['vbitrate_kbps']} kbps" if meta['vbitrate_kbps'] else '—')
    c2.metric('File Size', f"{size_mb:.2f} MB")
    if meta.get('has_audio'):
        st.markdown('---')
        a1, a2 = st.columns(2)
        a1.metric('Audio codec', backend.format_audio_codec(meta['acodec']))
        a2.metric('Channels', backend.format_channels(meta['channels']))
        a1.metric('Sample rate', backend.format_sample_rate(meta['sample_rate']))
        a2.metric('Audio bitrate', f"{meta['abitrate_kbps']} kbps" if meta['abitrate_kbps'] > 0 else 'Variable')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown('**🧾 Publish status**')
    st.metric('FFmpeg available', 'Yes' if backend.ffmpeg_ok() else 'No')
    if result:
        st.metric('GitHub publish status', 'Success' if result.ok else 'Failed')
        if result.repo_path_prefix:
            st.caption(f"Repository path: {result.repo_path_prefix}")
        with st.expander('FFmpeg log'):
            st.code(result.ffmpeg_log or '(empty)', language='bash')
        with st.expander('GitHub publish log'):
            st.code(result.github_log or '(empty)', language='bash')
        with st.expander('Cloudflare Deploy Hook log'):
            st.code(result.deploy_hook_log or '(empty)', language='bash')
        if result.zip_bytes:
            st.download_button('⬇ Download generated HLS bundle (.zip)', data=result.zip_bytes, file_name='github_pages_hls_bundle.zip', mime='application/zip')
        if result.ladder:
            st.markdown('### Active ladder')
            st.dataframe(pd.DataFrame([{ 'Variant': x['name'], 'Resolution': f"{x['width']}×{x['height']}", 'Video bitrate': x['video_bitrate'], 'Max rate': x['maxrate'], 'Audio bitrate': x['audio_bitrate'], 'Bandwidth': x['bandwidth']} for x in result.ladder]), use_container_width=True, hide_index=True)
    else:
        st.caption('No publish has run yet.')
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown('**▶ Public HLS.js Playback**')
    if result and result.manifest_url:
        st.code(result.manifest_url)
        st.caption('Note: if Cloudflare Pages has not finished rebuilding yet, wait a short while and refresh the page.')
        components.html(backend.build_hlsjs_player_html(result.manifest_url, title=f'{aspect} Pages playback', autoplay=True, muted=True, low_latency=True), height=980, scrolling=True)
    else:
        st.caption('Public `master.m3u8` URL appears here after GitHub publication completes.')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div class='vf-card'>", unsafe_allow_html=True)
    st.markdown('**📡 Playback analytics companion**')
    components.html(backend.build_player_analytics_html(meta, source_label='Source-side analytics companion'), height=220, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.markdown('### How this version works')
st.markdown('- FFmpeg generates ABR HLS locally inside Streamlit Cloud.\n- The app publishes the generated files into your GitHub repository using the GitHub REST API.\n- Cloudflare Pages Git integration automatically deploys the repo content for the target branch.\n- The player uses your configured Pages base URL plus the published repository folder path to build the `master.m3u8` URL.\n- Optional: if you provide a Cloudflare Deploy Hook, the app also triggers a rebuild explicitly after publication.')
