from __future__ import annotations
import os, tempfile
import streamlit as st
import streamlit.components.v1 as components
import backend

st.set_page_config(page_title='Smooth Vertical Live → Cloudflare Stream', page_icon='📱', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
.hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;}
.chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

for key, value in {'input_path': None, 'meta': None, 'reframed_path': None, 'live_session': None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<div class='hero'>
  <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
    <div>
      <div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>Smooth Vertical Live → Cloudflare Stream</div>
      <div style='font-size:.98rem; color:#cbd5e1;'>Primary goal optimized: very smooth vertical output with minimal jitter, and reliable in-app playback using the built-in Cloudflare iframe player.</div>
    </div>
    <div>
      <span class='chip'>Simple smooth vertical crop</span>
      <span class='chip'>Minimal jitter</span>
      <span class='chip'>Normal HLS test URL</span>
      <span class='chip'>Cloudflare iframe in app</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not backend.ffmpeg_ok():
    st.error('FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.')
    st.stop()

st.subheader('1) Cloudflare Stream Live settings')
left_cf, right_cf = st.columns(2)
with left_cf:
    account_id = st.text_input('Cloudflare account ID', value=os.getenv('CLOUDFLARE_ACCOUNT_ID', ''))
    api_token = st.text_input('Cloudflare Stream API token', value=os.getenv('CLOUDFLARE_STREAM_API_TOKEN', ''), type='password')
with right_cf:
    customer_code = st.text_input('Cloudflare Stream customer code', value=os.getenv('CLOUDFLARE_STREAM_CUSTOMER_CODE', ''), help='Enter only the code, not the full domain.')
    prefer_low_latency = st.checkbox('Prefer LL-HLS where available', value=False, help='OFF by default. Normal HLS is simpler and more stable for this workflow.')

cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(account_id, api_token, customer_code, prefer_low_latency)
        st.success('Cloudflare Stream configuration looks valid.')
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning('Fill in Account ID, Stream API token, and customer code.')

st.subheader('2) Smooth vertical master settings')
upl = st.file_uploader(f'Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)', type=['avi','mp4','mkv','mov','webm','flv','ts','m4v','mxf'])
_, controls = st.columns([2, 1], gap='large')
with controls:
    smooth_strength = st.slider('Smoothness', 0.85, 0.99, 0.96, 0.01)
    analysis_stride = st.slider('Update every N frames', 3, 10, 6, 1)
    deadzone_ratio = st.slider('Deadzone ratio', 0.02, 0.10, 0.06, 0.01)
    max_pan_ratio = st.slider('Max pan ratio', 0.005, 0.05, 0.02, 0.005)
    target_w = st.selectbox('Vertical output width', [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))
    loop_input = st.checkbox('Loop uploaded clip as pseudo-live source', value=True)

if upl:
    if getattr(upl, 'size', 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f'File is {upl.size / (1024*1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.')
        st.stop()
    suffix = os.path.splitext(upl.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upl.read())
        st.session_state.input_path = tmp.name
    st.session_state.meta = backend.probe(st.session_state.input_path)

if st.session_state.meta:
    meta = st.session_state.meta
    a, b, c = st.columns(3)
    a.metric('Source resolution', f"{meta['width']}×{meta['height']}")
    b.metric('FPS', f"{meta['fps']}")
    c.metric('Duration', f"{meta['duration']:.1f}s")

progress_bar = st.progress(0.0, text='Waiting')
if st.button('Analyse + create smooth vertical master', disabled=not bool(st.session_state.input_path)):
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    def _cb(pct, msg):
        progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=msg)
    ok, msg = backend.smart_reframe_vertical_smooth(
        st.session_state.input_path, out_path,
        target_w=target_w, target_h=target_h,
        smooth_strength=smooth_strength,
        analysis_stride=analysis_stride,
        deadzone_ratio=deadzone_ratio,
        max_pan_ratio=max_pan_ratio,
        progress_cb=_cb,
    )
    if ok:
        st.session_state.reframed_path = out_path
        progress_bar.progress(1.0, text='Smooth vertical master complete')
        st.success('Smooth vertical master created.')
    else:
        st.error(msg)

if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Smooth vertical master preview**')
    st.video(st.session_state.reframed_path)
    st.caption('This version intentionally prioritizes smoothness over aggressive tracking.')
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader('3) Push to Cloudflare Stream Live input')
if st.button('Start Cloudflare live push', disabled=not (cf_cfg and st.session_state.reframed_path)):
    if st.session_state.live_session:
        backend.stop_cloudflare_live_push(cf_cfg, st.session_state.live_session)
    session = backend.start_cloudflare_live_push(cf_cfg, st.session_state.reframed_path, upl.name if upl else 'vertical_live_demo', loop_input=loop_input)
    st.session_state.live_session = session
    st.success('Cloudflare live input created and FFmpeg push started with conservative smooth settings.')

if st.button('Stop Cloudflare live push', disabled=not bool(st.session_state.live_session)):
    backend.stop_cloudflare_live_push(cf_cfg, st.session_state.live_session)
    st.session_state.live_session = None
    st.info('Cloudflare live push stopped and input disabled.')

if st.session_state.live_session:
    session = st.session_state.live_session
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Cloudflare Stream live session**')
    st.caption(f'Live input UID: {session.uid}')
    normal_hls = f'https://customer-{cf_cfg.customer_code}.cloudflarestream.com/{session.uid}/manifest/video.m3u8' if cf_cfg else session.hls_url
    st.text_input('Normal HLS test playback URL', value=normal_hls, key='normal_hls_test_url')
    st.caption('Use this URL in any open-source HLS player (hls.js demo, VLC, ffplay).')
    st.text_input('Cloudflare iframe player URL (used inside this app)', value=session.iframe_url, key='iframe_player_url')
    with st.expander('FFmpeg push log'):
        if os.path.exists(session.log_path):
            with open(session.log_path, 'r', encoding='utf-8', errors='ignore') as fp:
                st.code(fp.read()[-12000:] or '(empty)', language='bash')
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader('4) In-app playback (Cloudflare built-in player)')
    iframe_html = (
        '<div style="position:relative;padding-top:177.78%;max-width:360px;margin:0 auto;">'
        f'<iframe src="{session.iframe_url}" '
        'style="border:none;position:absolute;top:0;left:0;height:100%;width:100%;border-radius:16px;overflow:hidden;" '
        'allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;" allowfullscreen="true"></iframe>'
        '</div>'
    )
    components.html(iframe_html, height=760, scrolling=False)

st.divider()
st.markdown('### What changed in this version')
st.markdown(
    '- Replaced aggressive sports tracking with a **simple, very smooth vertical crop**.\n'
    '- Crop movement update is intentionally conservative: fewer target updates, larger deadzone, and hard pan-speed cap.\n'
    '- Cloudflare ingest is tuned to **30 fps** normal HLS with a conservative bitrate for smoother motion preview.\n'
    '- In-app playback now uses the **Cloudflare built-in iframe player** instead of a custom HLS.js block, because the iframe player is more reliable inside Streamlit.\n'
    '- The **normal HLS test URL** is still shown so you can test in any open-source HLS player.'
)
