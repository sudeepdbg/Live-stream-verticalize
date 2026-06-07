from __future__ import annotations
import os, tempfile
import streamlit as st
import streamlit.components.v1 as components
import backend

st.set_page_config(page_title='Dual Flow Vertical Live → Cloudflare Stream', page_icon='📱', layout='wide', initial_sidebar_state='expanded')
st.markdown("<style>.block-container {padding-top: 1rem; padding-bottom: 2rem;} .card {border:1px solid rgba(148,163,184,.24); border-radius:18px; padding:18px; background:linear-gradient(180deg, rgba(15,23,42,.98), rgba(15,23,42,.92)); color:#e5e7eb; box-shadow:0 14px 30px rgba(2,6,23,.18);} .hero {border:1px solid rgba(59,130,246,.18); border-radius:22px; padding:20px 22px; margin-bottom:16px; background:radial-gradient(circle at top right, rgba(37,99,235,.18), transparent 28%), linear-gradient(180deg, #0b1220 0%, #0f172a 100%); color:white;} .chip {display:inline-block; padding:6px 12px; border-radius:999px; font-size:.78rem; border:1px solid rgba(148,163,184,.25); background:#0f172a; color:#cbd5e1; margin-right:6px;}" , unsafe_allow_html=True)
for key, value in {'input_path': None, 'meta': None, 'reframed_path': None, 'live_session': None}.items():
    if key not in st.session_state:
        st.session_state[key] = value
st.markdown("<div class='hero'><div style='display:flex; justify-content:space-between; gap:12px; align-items:flex-start; flex-wrap:wrap;'><div><div style='font-size:1.85rem; font-weight:800; margin-bottom:4px;'>Dual Flow Vertical Live → Cloudflare Stream</div><div style='font-size:.98rem; color:#cbd5e1;'>Single merged package with both workflows: (A) VOD → Live and (B) delayed realtime. Includes source-type switch and shared Cloudflare playback section.</div></div><div><span class='chip'>workflow switch</span><span class='chip'>source-type switch</span><span class='chip'>VOD → Live</span><span class='chip'>delayed realtime</span><span class='chip'>shared playback</span></div></div></div>", unsafe_allow_html=True)
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
    prefer_low_latency = st.checkbox('Prefer LL-HLS where available', value=False)
cf_cfg = None
if account_id and api_token and customer_code:
    try:
        cf_cfg = backend.cfstream_config_from_inputs(account_id, api_token, customer_code, prefer_low_latency)
        st.success('Cloudflare Stream configuration looks valid.')
    except Exception as exc:
        st.error(str(exc))
else:
    st.warning('Fill in Account ID, Stream API token, and customer code.')
st.subheader('2) Select workflow and source')
workflow = st.radio('Workflow', ['VOD → Live (full-file verticalize first)', 'Delayed realtime (frame-by-frame then delay buffer)'], horizontal=True)
source_kind = st.radio('Source type', ['Upload file', 'RTMP URL', 'SRT URL', 'Local path / arbitrary URL'], horizontal=True)
source_value = None
uploaded_name = 'source'
if source_kind == 'Upload file':
    upl = st.file_uploader(f'Upload source video (recommended max {backend.MAX_UPLOAD_MB} MB)', type=['avi','mp4','mkv','mov','webm','flv','ts','m4v','mxf'])
    if upl:
        if getattr(upl, 'size', 0) > backend.MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f'File is {upl.size / (1024*1024):.1f} MB. Keep it ≤ {backend.MAX_UPLOAD_MB} MB.')
            st.stop()
        suffix = os.path.splitext(upl.name)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upl.read())
            st.session_state.input_path = tmp.name
        uploaded_name = upl.name
        source_value = st.session_state.input_path
elif source_kind == 'RTMP URL':
    source_value = st.text_input('RTMP ingest URL', placeholder='rtmp://...')
elif source_kind == 'SRT URL':
    source_value = st.text_input('SRT ingest URL', placeholder='srt://host:port?mode=caller')
else:
    source_value = st.text_input('Local file path or URL', placeholder='/mount/src/... or https://...')
smooth_strength = st.slider('Smoothness', 0.90, 0.995, 0.97, 0.005)
analysis_stride = st.slider('Analysis stride', 3, 10, 6, 1)
deadzone_ratio = st.slider('Deadzone ratio', 0.02, 0.10, 0.06, 0.01)
max_pan_ratio = st.slider('Max pan ratio', 0.005, 0.03, 0.015, 0.005)
target_w = st.selectbox('Vertical output width', [360, 540, 720], index=1)
target_h = int(round(target_w * 16 / 9))
delay_seconds = st.slider('Output delay (seconds)', 5, 30, 20, 1)
loop_file = st.checkbox('Loop file source when it ends', value=True)
if source_value and source_kind in ('Upload file', 'Local path / arbitrary URL') and os.path.exists(str(source_value)):
    st.session_state.meta = backend.probe_source(str(source_value))
elif source_value and isinstance(source_value, str) and source_value.lower().startswith(('rtmp://', 'rtmps://', 'srt://', 'http://', 'https://')):
    st.info('For URL-based ingest, probe metadata may be unavailable until runtime.')
if st.session_state.meta:
    meta = st.session_state.meta
    a, b, c = st.columns(3)
    a.metric('Source resolution', f"{meta['width']}×{meta['height']}")
    b.metric('FPS', f"{meta['fps']}")
    c.metric('Duration', f"{meta['duration']:.1f}s")
progress_bar = st.progress(0.0, text='Waiting')
if workflow == 'VOD → Live (full-file verticalize first)':
    st.markdown('### 3A) VOD → Live')
    if source_kind in ('RTMP URL', 'SRT URL'):
        st.warning('VOD → Live is intended for file-like sources. Switch to Delayed realtime for RTMP/SRT.')
    if st.button('Create vertical master', disabled=not bool(source_value) or source_kind in ('RTMP URL', 'SRT URL')):
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        def _cb(pct, msg):
            progress_bar.progress(min(max(float(pct), 0.0), 1.0), text=msg)
        ok, msg = backend.create_vertical_master(str(source_value), out_path, target_w, target_h, smooth_strength, analysis_stride, deadzone_ratio, max_pan_ratio, _cb)
        if ok:
            st.session_state.reframed_path = out_path
            progress_bar.progress(1.0, text='Vertical master complete')
            st.success('Vertical master created.')
        else:
            st.error(msg)
    if st.session_state.reframed_path and os.path.exists(st.session_state.reframed_path):
        st.video(st.session_state.reframed_path)
    if st.button('Start VOD → Live push', disabled=not (cf_cfg and st.session_state.reframed_path)):
        if st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        session = backend.start_vod_to_live_push(cf_cfg, st.session_state.reframed_path, uploaded_name, loop_input=loop_file)
        st.session_state.live_session = session
        st.success('VOD → Live push started.')
else:
    st.markdown('### 3B) Delayed realtime')
    st.caption('Use this for horizontal file / RTMP / SRT source → frame-by-frame vertical output with ~20–25s delay.')
    if st.button('Start delayed realtime push', disabled=not (cf_cfg and source_value)):
        if st.session_state.live_session:
            backend.stop_live_session(cf_cfg, st.session_state.live_session)
        session = backend.start_realtime_delayed_vertical_push(cf_cfg, str(source_value), uploaded_name, target_w=target_w, target_h=target_h, delay_seconds=float(delay_seconds), smooth_strength=float(smooth_strength), analysis_stride=int(analysis_stride), deadzone_ratio=float(deadzone_ratio), max_pan_ratio=float(max_pan_ratio), loop_file=(loop_file and source_kind in ('Upload file', 'Local path / arbitrary URL')), pace_input=(source_kind in ('Upload file', 'Local path / arbitrary URL')))
        st.session_state.live_session = session
        st.success('Delayed realtime worker started.')
if st.button('Stop current live push', disabled=not bool(st.session_state.live_session)):
    backend.stop_live_session(cf_cfg, st.session_state.live_session)
    st.session_state.live_session = None
    st.info('Current live session stopped and Cloudflare input disabled.')
if st.session_state.live_session:
    session = st.session_state.live_session
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('**Shared Cloudflare playback section**')
    st.caption(f'Live input UID: {session.uid}')
    normal_hls = session.hls_url.split('?')[0]
    st.text_input('Normal HLS test playback URL', value=normal_hls, key='normal_hls_test_url')
    st.caption('Use this URL in any HLS player. This is the public playback URL.')
    st.text_input('Cloudflare iframe player URL', value=session.iframe_url, key='iframe_player_url')
    if session.stats:
        s1, s2, s3 = st.columns(3)
        s1.metric('Pipeline status', session.status)
        s2.metric('Frame delay', session.stats.get('delay_frames', '-'))
        s3.metric('Frames out', session.stats.get('frames_out', 0))
    with st.expander('FFmpeg push log'):
        if os.path.exists(session.log_path):
            with open(session.log_path, 'r', encoding='utf-8', errors='ignore') as fp:
                st.code(fp.read()[-12000:] or '(empty)', language='bash')
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader('4) In-app playback')
    iframe_html = ('<div style="position:relative;padding-top:177.78%;max-width:360px;margin:0 auto;">' + f'<iframe src="{session.iframe_url}" ' + 'style="border:none;position:absolute;top:0;left:0;height:100%;width:100%;border-radius:16px;overflow:hidden;" ' + 'allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;" allowfullscreen="true"></iframe>' + '</div>')
    components.html(iframe_html, height=760, scrolling=False)
st.divider()
st.markdown('### Included in this final merged package')
st.markdown('- workflow switch\n- source-type switch\n- VOD → Live mode\n- delayed realtime mode\n- shared Cloudflare playback section')
