from __future__ import annotations

import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(page_title='TikTok Live Verticalizer → Cloudflare Stream (Sports)', page_icon='📱', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
.hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;}
.chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

for key, value in {
    'input_path': None,
    'meta': None,
    'reframed_path': None,
    'live_session': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<div class='hero'>
  <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
    <div>
      <div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>TikTok Live Verticalizer → Cloudflare Stream (Sports)</div>
      <div style='font-size:.98rem; color:#cbd5e1;'>Updated version: sports-aware smart reframe, motion-aware crop smoothing, multi-focus detection (ball + players), and normal HLS playback URL for any open-source HLS player.</div>
    </div>
    <div>
      <span class='chip'>Sports-aware smart reframe</span>
      <span class='chip'>Motion-aware crop smoothing</span>
      <span class='chip'>Ball + players focus</span>
      <span class='chip'>Normal HLS test URL</span>
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
    customer_code = st.text_input('Cloudflare Stream customer code', value=os.getenv('CLOUDFLARE_STREAM_CUSTOMER_CODE', ''), help='Enter only the code, not the full domain. Example: p2urnuq01pg24ltd')
    prefer_low_latency = st.checkbox('Prefer LL-HLS where available', value=False, help='Keep OFF for now to test normal HLS and reduce jitter/debug complexity.')

cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(account_id, api_token, customer_code, prefer_low_latency)
        st.success('Cloudflare Stream configuration looks valid.')
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning('Fill in Account ID, Stream API token, and customer code.')

st.subheader('2) Sports smart reframe settings')
upl = st.file_uploader(f'Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)', type=['avi', 'mp4', 'mkv', 'mov', 'webm', 'flv', 'ts', 'm4v', 'mxf'])
reframe_left, reframe_right = st.columns([2, 1], gap='large')
with reframe_right:
    smooth_strength = st.slider('Motion-aware smoothing strength', 0.60, 0.98, 0.90, 0.01)
    lead_room = st.slider('Lead-room', 0.0, 0.50, 0.20, 0.01)
    target_w = st.selectbox('Vertical output width', [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))
    sports_mode = st.checkbox('Sports-aware mode', value=True)
    detect_ball = st.checkbox('Detect ball focus', value=True)
    detect_players = st.checkbox('Detect player focus', value=True)
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
if st.button('Analyse + create sports-aware vertical master', disabled=not bool(st.session_state.input_path)):
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    def _cb(pct, msg):
        progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=msg)
    ok, msg = backend.smart_reframe_vertical(
        st.session_state.input_path,
        out_path,
        target_w=target_w,
        target_h=target_h,
        smooth_strength=smooth_strength,
        lead_room=lead_room,
        sports_mode=sports_mode,
        detect_ball=detect_ball,
        detect_players=detect_players,
        progress_cb=_cb,
    )
    if ok:
        st.session_state.reframed_path = out_path
        progress_bar.progress(1.0, text='Sports-aware smart reframe complete')
        st.success('Sports-aware vertical master created.')
    else:
        st.error(msg)

if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Sports-aware vertical smart-reframed master preview**')
    st.video(st.session_state.reframed_path)
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader('3) Push to Cloudflare Stream Live input')
if st.button('Start Cloudflare live push', disabled=not (cf_cfg and st.session_state.reframed_path)):
    if st.session_state.live_session:
        backend.stop_cloudflare_live_push(cf_cfg, st.session_state.live_session)
    session = backend.start_cloudflare_live_push(
        cfg=cf_cfg,
        reframed_mp4=st.session_state.reframed_path,
        asset_name=upl.name if upl else 'vertical_live_demo',
        fps=30,
        loop_input=loop_input,
    )
    st.session_state.live_session = session
    st.success('Cloudflare live input created and FFmpeg push started with sports-tuned normal HLS settings.')

if st.button('Stop Cloudflare live push', disabled=not bool(st.session_state.live_session)):
    backend.stop_cloudflare_live_push(cf_cfg, st.session_state.live_session)
    st.session_state.live_session = None
    st.info('Cloudflare live push stopped and input disabled.')

if st.session_state.live_session:
    session = st.session_state.live_session
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Cloudflare Stream live session**')
    st.caption(f'Live input UID: {session.uid}')
    st.code(session.hls_url)
    st.caption('Use this NORMAL HLS test URL in any open-source HLS player (hls.js demo, VLC, etc.).')
    test_url = f'https://customer-{cf_cfg.customer_code}.cloudflarestream.com/{session.uid}/manifest/video.m3u8' if cf_cfg else session.hls_url
    st.text_input('Normal HLS test playback URL', value=test_url, key='normal_hls_test_url')
    with st.expander('FFmpeg push log'):
        if os.path.exists(session.log_path):
            with open(session.log_path, 'r', encoding='utf-8', errors='ignore') as fp:
                st.code(fp.read()[-12000:] or '(empty)', language='bash')
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader('4) Public Cloudflare playback with LIVE UX preserved')
    components.html(backend.build_cloudflare_live_player_html(test_url, title='TikTok-style vertical live on Cloudflare (sports-tuned normal HLS)', autoplay=True, muted=True), height=980, scrolling=True)

st.divider()
st.markdown('### Notes')
st.markdown(
    '- This update adds **sports-aware smart reframe** using player detection + ball heuristics + motion saliency.\n'
    '- Crop smoothing is now **motion-aware**, with deadzone and max-pan limits to reduce jitter.\n'
    '- Multi-focus combines **ball + player cluster + motion** instead of using a single center.\n'
    '- The **normal HLS test URL** is shown explicitly so you can paste it into any open-source HLS player.\n'
    '- LL-HLS is OFF by default here because you asked to proceed with normal HLS for easier testing.'
)
