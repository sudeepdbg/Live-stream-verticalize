from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_UPLOAD_MB = 300
DEFAULT_SEGMENT_SECONDS = 2
ABR_RUNGS = (360, 540)

ASPECT_PRESETS = {
    "16:9 Landscape": (16, 9),
    "9:16 Vertical": (9, 16),
    "1:1 Square": (1, 1),
    "4:5 Portrait": (4, 5),
    "3:4 Portrait": (3, 4),
    "Source / Passthrough": None,
}

ABR_RUNG_SETTINGS = {
    360: {"video_bitrate": "800k", "maxrate": "856k", "bufsize": "1200k", "audio_bitrate": "96k"},
    540: {"video_bitrate": "1400k", "maxrate": "1498k", "bufsize": "2100k", "audio_bitrate": "96k"},
}


@dataclass
class CloudflareConfig:
    project_name: str
    account_id: str
    api_token: str
    branch_prefix: str = "preview"
    use_production_branch: bool = False


@dataclass
class DeployResult:
    ok: bool
    message: str
    out_dir: str
    manifest_path: str
    manifest_url: str | None
    site_url: str | None
    zip_bytes: bytes | None
    ffmpeg_log: str
    deploy_log: str
    ladder: list[dict]
    branch_alias: str | None


def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def wrangler_ok() -> bool:
    try:
        subprocess.run(["npx", "wrangler", "--version"], capture_output=True, check=True, timeout=20)
        return True
    except Exception:
        try:
            subprocess.run(["wrangler", "--version"], capture_output=True, check=True, timeout=20)
            return True
        except Exception:
            return False


def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "stream"


def safe_branch(value: str) -> str:
    value = value or "preview"
    value = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-").lower()
    return value[:28] or "preview"


def ensure_clean_dir(path: str | os.PathLike[str]) -> None:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            else:
                os.unlink(fp)
        except Exception:
            pass


def zip_dir_bytes(dir_path: str | os.PathLike[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for file_name in files:
                fp = os.path.join(root, file_name)
                arc = os.path.relpath(fp, dir_path)
                zf.write(fp, arc)
    buf.seek(0)
    return buf.getvalue()


def _to_even(v: int) -> int:
    v = int(v)
    return v if v % 2 == 0 else v - 1


def _gop_for_fps(fps: Optional[int]) -> int:
    return max((fps or 24) * 2, 48)


def format_audio_codec(codec: str) -> str:
    mapping = {"aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis", "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC", "pcm_s16le": "PCM 16-bit", "alac": "ALAC"}
    return mapping.get(codec, (codec or "unknown").upper())


def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"


def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")


def probe(path: str) -> dict:
    res = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "vcodec": "unknown",
        "vbitrate_kbps": 0,
        "acodec": "unknown",
        "abitrate_kbps": 0,
        "sample_rate": 0,
        "channels": 0,
        "has_audio": False,
    }
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        data = json.loads(out)
        fmt = data.get("format", {})
        res["duration"] = float(fmt.get("duration", 0) or 0)
        res["vbitrate_kbps"] = int(fmt.get("bit_rate", 0) or 0) // 1000
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video" and res["width"] == 0:
                res["width"] = int(stream.get("width", 0) or 0)
                res["height"] = int(stream.get("height", 0) or 0)
                res["vcodec"] = stream.get("codec_name", "unknown")
                try:
                    n, d = map(int, str(stream.get("r_frame_rate", "0/1")).split("/"))
                    res["fps"] = round(n / d, 3) if d else 0.0
                except Exception:
                    pass
            elif stream.get("codec_type") == "audio" and not res["has_audio"]:
                res["has_audio"] = True
                res["acodec"] = stream.get("codec_name", "unknown")
                res["abitrate_kbps"] = int(stream.get("bit_rate", 0) or 0) // 1000
                res["sample_rate"] = int(stream.get("sample_rate", 0) or 0)
                res["channels"] = int(stream.get("channels", 0) or 0)
    except Exception:
        pass
    return res


def ladder_dimensions(aspect_label: str, rung: int, src_meta: Optional[dict] = None) -> tuple[int, int]:
    if aspect_label == "Source / Passthrough" and src_meta and src_meta.get("width") and src_meta.get("height"):
        sw, sh = int(src_meta["width"]), int(src_meta["height"])
        if sw >= sh:
            h = rung
            w = _to_even(round(h * sw / sh))
            return w, _to_even(h)
        w = rung
        h = _to_even(round(w * sh / sw))
        return _to_even(w), h
    per_aspect = {
        "16:9 Landscape": {360: (640, 360), 540: (960, 540)},
        "9:16 Vertical": {360: (360, 640), 540: (540, 960)},
        "1:1 Square": {360: (360, 360), 540: (540, 540)},
        "4:5 Portrait": {360: (288, 360), 540: (432, 540)},
        "3:4 Portrait": {360: (270, 360), 540: (406, 540)},
        "Source / Passthrough": {360: (640, 360), 540: (960, 540)},
    }
    return per_aspect.get(aspect_label, per_aspect["16:9 Landscape"])[rung]


def build_abr_ladder(aspect_label: str, src_meta: Optional[dict] = None) -> list[dict]:
    ladder = []
    for rung in ABR_RUNGS:
        width, height = ladder_dimensions(aspect_label, rung, src_meta=src_meta)
        s = ABR_RUNG_SETTINGS[rung]
        vbps = int(re.sub(r"[^0-9]", "", s["video_bitrate"])) * 1000
        abps = int(re.sub(r"[^0-9]", "", s["audio_bitrate"])) * 1000
        ladder.append({
            "name": f"{rung}p",
            "rung": rung,
            "width": width,
            "height": height,
            **s,
            "bandwidth": vbps + abps,
            "avg_bandwidth": vbps,
        })
    return ladder


def variant_video_filter(width: int, height: int) -> str:
    return f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"


def build_multi_variant_vod_hls_cmd(input_source: str, out_dir: str, aspect_label: str = "16:9 Landscape", preset: str = "superfast", fps: Optional[int] = None, segment_seconds: int = DEFAULT_SEGMENT_SECONDS, src_meta: Optional[dict] = None) -> tuple[list[str], list[dict], str]:
    ladder = build_abr_ladder(aspect_label=aspect_label, src_meta=src_meta)
    split_labels = [f"v{i}" for i in range(len(ladder))]
    out_labels = [f"v{i}out" for i in range(len(ladder))]
    filter_parts = [f"[0:v]split={len(ladder)}" + "".join(f"[{x}]" for x in split_labels)]
    for idx, variant in enumerate(ladder):
        vf = variant_video_filter(variant["width"], variant["height"])
        if fps:
            vf += f",fps={fps}"
        filter_parts.append(f"[{split_labels[idx]}]{vf}[{out_labels[idx]}]")
    cmd = ["ffmpeg", "-y", "-i", input_source, "-filter_complex", ";".join(filter_parts)]
    for idx in range(len(ladder)):
        cmd += ["-map", f"[{out_labels[idx]}]", "-map", "0:a:0?"]
    for idx, variant in enumerate(ladder):
        cmd += [
            f"-c:v:{idx}", "libx264",
            f"-preset:v:{idx}", preset,
            f"-pix_fmt:v:{idx}", "yuv420p",
            f"-profile:v:{idx}", "main",
            f"-b:v:{idx}", variant["video_bitrate"],
            f"-maxrate:v:{idx}", variant["maxrate"],
            f"-bufsize:v:{idx}", variant["bufsize"],
            f"-g:v:{idx}", str(_gop_for_fps(fps)),
            f"-keyint_min:v:{idx}", str(_gop_for_fps(fps)),
            f"-sc_threshold:v:{idx}", "0",
            f"-c:a:{idx}", "aac",
            f"-b:a:{idx}", variant["audio_bitrate"],
            f"-ar:a:{idx}", "48000",
            f"-ac:a:{idx}", "2",
        ]
    master = os.path.join(out_dir, "master.m3u8")
    segment_pattern = os.path.join(out_dir, "%v_%06d.ts")
    playlist_pattern = os.path.join(out_dir, "%v.m3u8")
    var_stream_map = " ".join(f"v:{idx},a:{idx},name:{variant['name']}" for idx, variant in enumerate(ladder))
    cmd += [
        "-master_pl_name", "master.m3u8",
        "-f", "hls",
        "-hls_time", str(segment_seconds),
        "-hls_playlist_type", "vod",
        "-hls_list_size", "0",
        "-hls_flags", "independent_segments",
        "-hls_segment_filename", segment_pattern,
        "-var_stream_map", var_stream_map,
        playlist_pattern,
    ]
    return cmd, ladder, master


def run_ffmpeg(cmd: list[str], timeout: Optional[int] = None) -> tuple[bool, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        log = ((proc.stderr or "") + ("\n" + proc.stdout if proc.stdout else "")).strip()
        return ok, ("Done" if ok else f"FFmpeg exited with code {proc.returncode}"), log
    except subprocess.TimeoutExpired as exc:
        return False, "Timed out", str(exc)
    except Exception as exc:
        return False, str(exc), ""


def run_wrangler_pages_deploy(out_dir: str, cf: CloudflareConfig, branch_name: str) -> tuple[bool, str, str, str | None]:
    env = os.environ.copy()
    env["CLOUDFLARE_API_TOKEN"] = cf.api_token
    env["CLOUDFLARE_ACCOUNT_ID"] = cf.account_id
    cmd = ["npx", "wrangler", "pages", "deploy", out_dir, "--project-name", cf.project_name]
    branch = None
    if cf.use_production_branch:
        site_url = f"https://{cf.project_name}.pages.dev"
    else:
        branch = safe_branch(branch_name)
        cmd += ["--branch", branch]
        site_url = f"https://{branch}.{cf.project_name}.pages.dev"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    log = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    if proc.returncode != 0:
        return False, f"Wrangler exited with code {proc.returncode}", log, None
    return True, "Cloudflare Pages deployment succeeded", log, site_url


def cloudflare_config_from_inputs(project_name: str, account_id: str, api_token: str, branch_prefix: str, use_production_branch: bool) -> CloudflareConfig:
    project_name = safe_token(project_name).replace("_", "-").lower()
    if not project_name:
        raise ValueError("Cloudflare Pages project name is required.")
    if not account_id:
        raise ValueError("Cloudflare account ID is required.")
    if not api_token:
        raise ValueError("Cloudflare API token is required.")
    return CloudflareConfig(project_name=project_name, account_id=account_id.strip(), api_token=api_token.strip(), branch_prefix=safe_branch(branch_prefix), use_production_branch=bool(use_production_branch))


def build_branch_name(cf: CloudflareConfig, asset_name: str) -> str:
    base = safe_branch(Path(asset_name).stem)
    return safe_branch(f"{cf.branch_prefix}-{base}-{int(time.time())}")


def build_player_analytics_html(meta: dict, source_label: str = "Source") -> str:
    width = int(meta.get("width", 1920) or 1920)
    height = int(meta.get("height", 1080) or 1080)
    fps = float(meta.get("fps", 30.0) or 30.0)
    bitrate = int(meta.get("vbitrate_kbps", 4000) or 4000)
    html = f"""
<div style='font-family: Inter, Segoe UI, Arial, sans-serif; border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#0f172a; color:#e2e8f0;'>
  <div style='font-size:12px; color:#93c5fd; text-transform:uppercase; letter-spacing:.08em;'>Playback Analytics</div>
  <div style='font-size:20px; font-weight:700; margin-bottom:8px;'>{source_label}</div>
  <div style='font-size:12px; color:#94a3b8;'>{width}×{height} · {fps:.2f} fps · bitrate ~{bitrate} kbps</div>
</div>
"""
    return html


def build_hlsjs_player_html(manifest_url: str, title: str = "HLS playback", autoplay: bool = True, muted: bool = True, low_latency: bool = True) -> str:
    autoplay_str = "true" if autoplay else "false"
    low_latency_str = "true" if low_latency else "false"
    muted_attr = "muted" if muted else ""
    return f"""
<div style='font-family: Inter, Segoe UI, Arial, sans-serif; border:1px solid #e5e7eb; border-radius:16px; padding:16px; background:#0f172a; color:#e2e8f0;'>
  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
    <div>
      <div style='font-size:12px; color:#93c5fd; text-transform:uppercase; letter-spacing:.08em;'>Embedded HLS.js Playback</div>
      <div style='font-size:20px; font-weight:700;'>{title}</div>
      <div style='font-size:12px; color:#94a3b8; word-break:break-all;'>{manifest_url}</div>
    </div>
  </div>
  <video id='video' controls playsinline {muted_attr} style='width:100%; max-height:560px; background:#000; border-radius:12px;'></video>
  <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:14px;'>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>State</div><div id='state' style='font-size:22px; font-weight:700;'>booting</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Current level</div><div id='level' style='font-size:22px; font-weight:700;'>-</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Buffer ahead</div><div id='buffer' style='font-size:22px; font-weight:700;'>0.0 s</div></div>
    <div style='background:#111827; border-radius:12px; padding:12px;'><div style='font-size:12px; color:#94a3b8;'>Dropped frames</div><div id='drops' style='font-size:22px; font-weight:700;'>0</div></div>
  </div>
  <div style='background:#111827; border-radius:12px; padding:12px; margin-top:14px;'>
    <div style='font-size:13px; font-weight:600; margin-bottom:8px;'>Event log</div>
    <div id='log' style='font-size:12px; line-height:1.6; color:#cbd5e1; max-height:180px; overflow:auto;'>Waiting for manifest…</div>
  </div>
</div>
<script src='https://cdn.jsdelivr.net/npm/hls.js@latest'></script>
<script>
(function() {{
  const manifestUrl = {json.dumps(manifest_url)};
  const video = document.getElementById('video');
  const elState = document.getElementById('state');
  const elLevel = document.getElementById('level');
  const elBuffer = document.getElementById('buffer');
  const elDrops = document.getElementById('drops');
  const elLog = document.getElementById('log');
  function pushLog(msg) {{
    const now = new Date().toLocaleTimeString();
    elLog.innerHTML = '[' + now + '] ' + msg + '<br/>' + elLog.innerHTML.split('<br/>').slice(0,10).join('<br/>');
  }}
  function updateMetrics() {{
    let bufferAhead = 0;
    if (video.buffered && video.buffered.length) {{
      for (let i = 0; i < video.buffered.length; i++) {{
        if (video.buffered.start(i) <= video.currentTime && video.currentTime <= video.buffered.end(i)) {{
          bufferAhead = video.buffered.end(i) - video.currentTime;
          break;
        }}
      }}
    }}
    elBuffer.textContent = bufferAhead.toFixed(1) + ' s';
    try {{
      const q = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
      if (q && typeof q.droppedVideoFrames === 'number') elDrops.textContent = String(q.droppedVideoFrames);
    }} catch (e) {{}}
  }}
  if (Hls.isSupported()) {{
    const hls = new Hls({{ lowLatencyMode: {low_latency_str}, enableWorker: true }});
    hls.loadSource(manifestUrl);
    hls.attachMedia(video);
    hls.on(Hls.Events.MEDIA_ATTACHED, function() {{ elState.textContent = 'media attached'; pushLog('Media attached'); }});
    hls.on(Hls.Events.MANIFEST_PARSED, function(event, data) {{ elState.textContent = 'manifest parsed'; pushLog('Manifest parsed with ' + data.levels.length + ' level(s)'); if ({autoplay_str}) video.play().catch(() => {{}}); }});
    hls.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{ const level = hls.levels[data.level]; elLevel.textContent = level ? ((level.height || '?') + 'p') : String(data.level); pushLog('Level switched'); }});
    hls.on(Hls.Events.ERROR, function(event, data) {{ pushLog('HLS error: ' + data.type + ' | ' + data.details); elState.textContent = data.fatal ? 'fatal error' : 'recoverable error'; if (data.fatal) {{ if (data.type === Hls.ErrorTypes.NETWORK_ERROR) hls.startLoad(); else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError(); }} }});
  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
    video.src = manifestUrl;
    elState.textContent = 'native HLS';
    if ({autoplay_str}) video.play().catch(() => {{}});
  }} else {{
    elState.textContent = 'unsupported';
    pushLog('Browser does not support HLS playback');
  }}
  ['play','pause','waiting','playing','seeking','stalled','ended','loadedmetadata','canplay'].forEach(function(evt) {{
    video.addEventListener(evt, function() {{ elState.textContent = evt; pushLog('Video event: ' + evt); updateMetrics(); }});
  }});
  setInterval(updateMetrics, 1000);
}})();
</script>
"""


def package_and_deploy_vod_to_pages(input_source: str, asset_name: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, cf: CloudflareConfig, src_meta: Optional[dict] = None) -> DeployResult:
    out_dir = tempfile.mkdtemp(prefix="videoforge_pages_hls_")
    ensure_clean_dir(out_dir)
    cmd, ladder, manifest_path = build_multi_variant_vod_hls_cmd(input_source=input_source, out_dir=out_dir, aspect_label=aspect_label, preset=preset, fps=fps, segment_seconds=segment_seconds, src_meta=src_meta)
    ffmpeg_ok, ffmpeg_msg, ffmpeg_log = run_ffmpeg(cmd)
    if not ffmpeg_ok:
        return DeployResult(False, ffmpeg_msg, out_dir, manifest_path, None, None, None, ffmpeg_log, "", ladder, None)
    branch_alias = None if cf.use_production_branch else build_branch_name(cf, asset_name)
    deploy_ok, deploy_msg, deploy_log, site_url = run_wrangler_pages_deploy(out_dir, cf, branch_alias or "")
    manifest_url = f"{site_url}/master.m3u8" if site_url else None
    zbytes = zip_dir_bytes(out_dir) if deploy_ok else None
    return DeployResult(deploy_ok, deploy_msg, out_dir, manifest_path, manifest_url, site_url, zbytes, ffmpeg_log, deploy_log, ladder, branch_alias)
