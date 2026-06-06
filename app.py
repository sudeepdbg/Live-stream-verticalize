
from __future__ import annotations

import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

import backend

st.set_page_config(page_title='TikTok Live Verticalizer', page_icon='📱', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);}
.hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;}
.chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

if 'input_path' not in st.session_state:
    st.session_state.input_path = None
if 'meta' not in st.session_state:
    st.session_state.meta = None
if 'reframed_path' not in st.session_state:
    st.session_state.reframed_path = None
if 'live_job' not in st.session_state:
    st.session_state.live_job = None
if 'status_text' not in st.session_state:
    st.session_state.status_text = ''

st.markdown("""
<div class='hero'>
  <div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'>
    <div>
      <div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>TikTok Live Verticalizer</div>
      <div style='font-size:.98rem; color:#cbd5e1;'>Smart reframe MVP + live sliding HLS packaging + LIVE player UX. This version is built to demonstrate the vertical live-stream problem statement, not VOD packaging.</div>
    </div>
    <div>
      <span class='chip'>Subject/action tracking</span>
      <span class='chip'>Crop-to-fill 9:16</span>
      <span class='chip'>Smoothing + lead-room</span>
      <span class='chip'>Live playlist</span>
      <span class='chip'>LIVE badge + Go live</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not backend.ffmpeg_ok():
    st.error('FFmpeg / ffprobe not found. Add ffmpeg to your runtime and redeploy.')
    st.stop()

st.subheader('A) Smart reframe mode (true TikTok-style vertical fill)')
left, right = st.columns([2, 1], gap='large')
with left:
    upl = st.file_uploader(f'Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)', type=['avi', 'mp4', 'mkv', 'mov', 'webm', 'flv', 'ts', 'm4v', 'mxf'])
    if upl:
        if getattr(upl, 'size', 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f'File is {upl.size / (1024*1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.')
            st.stop()
        suffix = os.path.splitext(upl.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upl.read())
            st.session_state.input_path = tmp.name
        st.session_state.meta = backend.probe(st.session_state.input_path)
with right:
    smooth_strength = st.slider('Smoothing strength', 0.50, 0.98, 0.88, 0.01, help='Higher = steadier crop window, lower = more reactive movement.')
    lead_room = st.slider('Lead-room', 0.0, 0.50, 0.18, 0.01, help='Bias crop slightly in the direction of motion, like short-form vertical apps.')
    target_w = st.selectbox('Vertical output width', [360, 540, 720], index=1)
    target_h = int(round(target_w * 16 / 9))

if st.session_state.meta:
    meta = st.session_state.meta
    a, b, c = st.columns(3)
    a.metric('Source resolution', f"{meta['width']}×{meta['height']}")
    b.metric('FPS', f"{meta['fps']}")
    c.metric('Duration', f"{meta['duration']:.1f}s")

progress_bar = st.progress(0.0, text='Waiting')
if st.button('1️⃣ Analyse + create TikTok-style reframed vertical master') and st.session_state.input_path:
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    def _cb(pct, msg):
        progress_bar.progress(min(max(float(pct),0.0),1.0), text=msg)
    ok, msg = backend.smart_reframe_vertical(
        st.session_state.input_path,
        out_path,
        target_w=target_w,
        target_h=target_h,
        smooth_strength=smooth_strength,
        lead_room=lead_room,
        progress_cb=_cb,
    )
    if ok:
        st.session_state.reframed_path = out_path
        progress_bar.progress(1.0, text='Smart reframe complete')
        st.success('TikTok-style vertical master created.')
    else:
        st.error(msg)

if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Reframed vertical master ready**')
    st.video(st.session_state.reframed_path)
    st.caption('This preview is the dynamically cropped 9:16 master, not a fit-and-pad canvas.')
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader('B) Live mode packaging (real live window, not VOD)')
l1, l2, l3 = st.columns(3)
segment_seconds = l1.slider('Live segment seconds', 1, 6, backend.DEFAULT_SEGMENT_SECONDS)
live_list_size = l2.slider('Sliding playlist size', 2, 10, backend.DEFAULT_LIVE_LIST_SIZE)
loop_input = l3.checkbox('Loop uploaded clip as pseudo-live source', value=True, help='Useful to demonstrate live behavior from a short clip. For real live ingest, replace with RTMP/SRT input in the backend pipeline.')

if st.button('2️⃣ Start true live vertical HLS origin', disabled=not bool(st.session_state.reframed_path)):
    if st.session_state.live_job:
        backend.stop_live_job(st.session_state.live_job)
    job = backend.start_live_job_from_reframed_file(
        st.session_state.reframed_path,
        asset_name='tiktok_live_demo',
        segment_seconds=segment_seconds,
        live_list_size=live_list_size,
        fps=int(round(st.session_state.meta.get('fps') or 30)) if st.session_state.meta else 30,
        loop_input=loop_input,
    )
    st.session_state.live_job = job
    st.success('Live HLS origin started. This uses a sliding playlist and deletes old segments.')

if st.button('⏹ Stop live origin', disabled=not bool(st.session_state.live_job)):
    backend.stop_live_job(st.session_state.live_job)
    st.session_state.live_job = None
    st.info('Live origin stopped.')

if st.session_state.live_job:
    job = st.session_state.live_job
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Live origin status**')
    st.code(job.manifest_url)
    with st.expander('FFmpeg live log'):
        if os.path.exists(job.log_path):
            with open(job.log_path, 'r', encoding='utf-8', errors='ignore') as fp:
                st.code(fp.read()[-12000:] or '(empty)', language='bash')
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader('C) Live player UX')
    components.html(backend.build_live_player_html(job.manifest_url, title='TikTok-style vertical live preview', autoplay=True, muted=True), height=980, scrolling=True)

st.divider()
st.markdown('### What this version changes compared with the old one')
st.markdown(
    '- **Smart reframe**: subject/action tracking using faces + saliency + motion, then crop-to-fill 9:16 with smoothing and lead-room.\n'
    '- **Live packaging**: no VOD playlist type, short sliding playlist, old segment deletion, and a continuously updated live manifest.\n'
    '- **Live player UX**: LIVE badge, Go live button, and live-window metrics instead of clip-style playback behavior.\n'
    '- **Important**: this is a practical local/VM MVP. For production low-latency scale, swap the local origin for a real live media origin / CDN or managed service.'
)
