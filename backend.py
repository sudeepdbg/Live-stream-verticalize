
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_UPLOAD_MB = 300
DEFAULT_SEGMENT_SECONDS = 4  # slightly larger segments to reduce GitHub file/update volume
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
class GitHubConfig:
    owner: str
    repo: str
    token: str
    target_branch: str
    pages_base_url: str
    folder_prefix: str = "public/hls"
    default_branch: str = "main"
    deploy_hook_url: str | None = None

@dataclass
class PublishResult:
    ok: bool
    message: str
    out_dir: str
    manifest_path: str | None
    manifest_url: str | None
    repo_path_prefix: str | None
    zip_bytes: bytes | None
    ffmpeg_log: str
    github_log: str
    deploy_hook_log: str
    ladder: list[dict]

# -----------------------------
# Local media helpers
# -----------------------------
def ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def safe_token(value: str) -> str:
    value = value or "stream"
    return re.sub(r"[^A-Za-z0-9._/-]+", "_", value).strip("._-/") or "stream"


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
        out = subprocess.check_output([
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path
        ], text=True, stderr=subprocess.DEVNULL, timeout=30)
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


def format_audio_codec(codec: str) -> str:
    mapping = {"aac": "AAC", "mp3": "MP3", "opus": "Opus", "vorbis": "Vorbis", "ac3": "AC-3", "eac3": "E-AC-3", "flac": "FLAC", "pcm_s16le": "PCM 16-bit", "alac": "ALAC"}
    return mapping.get(codec, (codec or "unknown").upper())


def format_sample_rate(sr: int) -> str:
    return f"{sr // 1000} kHz" if sr >= 1000 else f"{sr} Hz"


def format_channels(ch: int) -> str:
    return {1: "Mono", 2: "Stereo", 6: "5.1", 8: "7.1"}.get(ch, f"{ch} ch")

# -----------------------------
# HLS packaging helpers
# -----------------------------
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
    segment_pattern = os.path.join(out_dir, "%v_%05d.ts")
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

# -----------------------------
# GitHub publication layer
# -----------------------------
def github_config_from_inputs(owner: str, repo: str, token: str, target_branch: str, pages_base_url: str, folder_prefix: str, default_branch: str, deploy_hook_url: str) -> GitHubConfig:
    if not owner:
        raise ValueError("GitHub owner is required.")
    if not repo:
        raise ValueError("GitHub repo is required.")
    if not token:
        raise ValueError("GitHub token is required.")
    if not target_branch:
        raise ValueError("Target branch is required.")
    if not pages_base_url:
        raise ValueError("Cloudflare Pages base URL is required.")
    return GitHubConfig(
        owner=owner.strip(),
        repo=repo.strip(),
        token=token.strip(),
        target_branch=target_branch.strip(),
        pages_base_url=pages_base_url.rstrip('/'),
        folder_prefix=safe_token(folder_prefix or 'public/hls').strip('/'),
        default_branch=(default_branch or 'main').strip(),
        deploy_hook_url=(deploy_hook_url or '').strip() or None,
    )


def _github_request(cfg: GitHubConfig, method: str, path: str, payload: Optional[dict] = None) -> tuple[int, dict | bytes | None, dict]:
    url = f"https://api.github.com{path}"
    data = None
    headers = {
        'Authorization': f'Bearer {cfg.token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'VideoForge-Streamlit-Publisher',
    }
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            content_type = resp.headers.get('Content-Type', '')
            parsed = json.loads(raw.decode('utf-8')) if 'application/json' in content_type and raw else None
            return resp.status, parsed, dict(resp.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        parsed = None
        try:
            parsed = json.loads(raw.decode('utf-8'))
        except Exception:
            parsed = raw
        return e.code, parsed, dict(e.headers)


def branch_exists(cfg: GitHubConfig, branch: str) -> bool:
    status, _, _ = _github_request(cfg, 'GET', f"/repos/{cfg.owner}/{cfg.repo}/git/ref/heads/{urllib.parse.quote(branch, safe='')}")
    return status == 200


def get_branch_head_sha(cfg: GitHubConfig, branch: str) -> str:
    status, payload, _ = _github_request(cfg, 'GET', f"/repos/{cfg.owner}/{cfg.repo}/git/ref/heads/{urllib.parse.quote(branch, safe='')}")
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"Failed to read branch ref for {branch}: {payload}")
    return payload['object']['sha']


def create_branch_from_default(cfg: GitHubConfig, branch: str) -> str:
    base_sha = get_branch_head_sha(cfg, cfg.default_branch)
    payload = {'ref': f'refs/heads/{branch}', 'sha': base_sha}
    status, data, _ = _github_request(cfg, 'POST', f"/repos/{cfg.owner}/{cfg.repo}/git/refs", payload)
    if status not in (201, 200):
        raise RuntimeError(f"Failed to create branch {branch}: {data}")
    return base_sha


def ensure_branch(cfg: GitHubConfig) -> None:
    if branch_exists(cfg, cfg.target_branch):
        return
    create_branch_from_default(cfg, cfg.target_branch)


def get_file_sha_if_exists(cfg: GitHubConfig, repo_path: str) -> str | None:
    encoded_path = '/'.join(urllib.parse.quote(part, safe='') for part in repo_path.split('/'))
    status, payload, _ = _github_request(cfg, 'GET', f"/repos/{cfg.owner}/{cfg.repo}/contents/{encoded_path}?ref={urllib.parse.quote(cfg.target_branch, safe='')}")
    if status == 200 and isinstance(payload, dict) and payload.get('sha'):
        return payload['sha']
    return None


def upsert_file_contents(cfg: GitHubConfig, repo_path: str, content_bytes: bytes, commit_message: str) -> str:
    encoded_path = '/'.join(urllib.parse.quote(part, safe='') for part in repo_path.split('/'))
    existing_sha = get_file_sha_if_exists(cfg, repo_path)
    payload = {
        'message': commit_message,
        'content': base64.b64encode(content_bytes).decode('ascii'),
        'branch': cfg.target_branch,
    }
    if existing_sha:
        payload['sha'] = existing_sha
    status, data, _ = _github_request(cfg, 'PUT', f"/repos/{cfg.owner}/{cfg.repo}/contents/{encoded_path}", payload)
    if status not in (200, 201):
        raise RuntimeError(f"Failed to write {repo_path}: {data}")
    return data.get('content', {}).get('sha', '') if isinstance(data, dict) else ''


def publish_directory_to_github(cfg: GitHubConfig, local_dir: str, repo_path_prefix: str) -> str:
    ensure_branch(cfg)
    logs = []
    for root, _, files in os.walk(local_dir):
        files.sort()
        for file_name in files:
            local_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(local_path, local_dir).replace('\\', '/')
            repo_path = f"{repo_path_prefix}/{rel_path}" if repo_path_prefix else rel_path
            with open(local_path, 'rb') as fh:
                content = fh.read()
            sha = upsert_file_contents(cfg, repo_path, content, f"Update HLS asset {repo_path}")
            logs.append(f"upserted {repo_path} sha={sha[:10] if sha else 'n/a'}")
    return '\n'.join(logs)


def trigger_deploy_hook(cfg: GitHubConfig) -> str:
    if not cfg.deploy_hook_url:
        return 'No deploy hook configured; relying on Cloudflare Pages automatic git deployment.'
    req = urllib.request.Request(cfg.deploy_hook_url, data=b'', method='POST', headers={'User-Agent': 'VideoForge-Streamlit-Publisher'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode('utf-8', errors='ignore')
        return f"Deploy hook POST returned HTTP {resp.status}. Response: {body[:1000]}"


def build_public_manifest_url(cfg: GitHubConfig, repo_path_prefix: str) -> str:
    # Pages publishes repo contents as-is relative to site root.
    return f"{cfg.pages_base_url}/{repo_path_prefix}/master.m3u8"

# -----------------------------
# Player HTML helpers
# -----------------------------
def build_player_analytics_html(meta: dict, source_label: str = 'Source') -> str:
    width = int(meta.get('width', 1920) or 1920)
    height = int(meta.get('height', 1080) or 1080)
    fps = float(meta.get('fps', 30.0) or 30.0)
    bitrate = int(meta.get('vbitrate_kbps', 4000) or 4000)
    return f"<div style='font-family:Inter,Segoe UI,Arial,sans-serif;border:1px solid #e5e7eb;border-radius:16px;padding:16px;background:#0f172a;color:#e2e8f0;'><div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em;'>Playback Analytics</div><div style='font-size:20px;font-weight:700;margin-bottom:8px;'>{source_label}</div><div style='font-size:12px;color:#94a3b8;'>{width}×{height} · {fps:.2f} fps · bitrate ~{bitrate} kbps</div></div>"


def build_hlsjs_player_html(manifest_url: str, title: str = 'HLS playback', autoplay: bool = True, muted: bool = True, low_latency: bool = True) -> str:
    autoplay_str = 'true' if autoplay else 'false'
    muted_attr = 'muted' if muted else ''
    low_latency_str = 'true' if low_latency else 'false'
    return f"""
<div style='font-family:Inter,Segoe UI,Arial,sans-serif;border:1px solid #e5e7eb;border-radius:16px;padding:16px;background:#0f172a;color:#e2e8f0;'>
  <div style='margin-bottom:12px;'><div style='font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.08em;'>Embedded HLS.js Playback</div><div style='font-size:20px;font-weight:700;'>{title}</div><div style='font-size:12px;color:#94a3b8;word-break:break-all;'>{manifest_url}</div></div>
  <video id='video' controls playsinline {muted_attr} style='width:100%;max-height:560px;background:#000;border-radius:12px;'></video>
  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;'><div style='background:#111827;border-radius:12px;padding:12px;'><div style='font-size:12px;color:#94a3b8;'>State</div><div id='state' style='font-size:22px;font-weight:700;'>booting</div></div><div style='background:#111827;border-radius:12px;padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Current level</div><div id='level' style='font-size:22px;font-weight:700;'>-</div></div><div style='background:#111827;border-radius:12px;padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Buffer ahead</div><div id='buffer' style='font-size:22px;font-weight:700;'>0.0 s</div></div><div style='background:#111827;border-radius:12px;padding:12px;'><div style='font-size:12px;color:#94a3b8;'>Dropped frames</div><div id='drops' style='font-size:22px;font-weight:700;'>0</div></div></div>
  <div style='background:#111827;border-radius:12px;padding:12px;margin-top:14px;'><div style='font-size:13px;font-weight:600;margin-bottom:8px;'>Event log</div><div id='log' style='font-size:12px;line-height:1.6;color:#cbd5e1;max-height:180px;overflow:auto;'>Waiting for manifest…</div></div>
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
    try {{ const q = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null; if (q && typeof q.droppedVideoFrames === 'number') elDrops.textContent = String(q.droppedVideoFrames); }} catch (e) {{}}
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
    video.src = manifestUrl; elState.textContent = 'native HLS'; if ({autoplay_str}) video.play().catch(() => {{}});
  }} else {{ elState.textContent = 'unsupported'; pushLog('Browser does not support HLS playback'); }}
  ['play','pause','waiting','playing','seeking','stalled','ended','loadedmetadata','canplay'].forEach(function(evt) {{ video.addEventListener(evt, function() {{ elState.textContent = evt; pushLog('Video event: ' + evt); updateMetrics(); }}); }});
  setInterval(updateMetrics, 1000);
}})();
</script>
"""

# -----------------------------
# End-to-end publish flow
# -----------------------------
def package_and_publish_via_github(input_source: str, asset_name: str, aspect_label: str, preset: str, fps: Optional[int], segment_seconds: int, cfg: GitHubConfig, src_meta: Optional[dict] = None) -> PublishResult:
    out_dir = tempfile.mkdtemp(prefix='videoforge_pages_git_hls_')
    ensure_clean_dir(out_dir)
    cmd, ladder, manifest_path = build_multi_variant_vod_hls_cmd(input_source, out_dir, aspect_label, preset, fps, segment_seconds, src_meta)
    ff_ok, ff_msg, ff_log = run_ffmpeg(cmd)
    if not ff_ok:
        return PublishResult(False, ff_msg, out_dir, manifest_path, None, None, None, ff_log, '', '', ladder)

    object_prefix = f"{cfg.folder_prefix}/{safe_token(Path(asset_name).stem)}_{int(time.time())}"
    github_log = ''
    deploy_hook_log = ''
    try:
        github_log = publish_directory_to_github(cfg, out_dir, object_prefix)
        if cfg.deploy_hook_url:
            deploy_hook_log = trigger_deploy_hook(cfg)
        else:
            deploy_hook_log = 'No deploy hook configured; relying on Cloudflare Pages automatic git deployment.'
        manifest_url = build_public_manifest_url(cfg, object_prefix)
        zbytes = zip_dir_bytes(out_dir)
        return PublishResult(True, 'GitHub publication succeeded', out_dir, manifest_path, manifest_url, object_prefix, zbytes, ff_log, github_log, deploy_hook_log, ladder)
    except Exception as exc:
        return PublishResult(False, f'GitHub publication failed: {exc}', out_dir, manifest_path, None, object_prefix, None, ff_log, github_log or str(exc), deploy_hook_log, ladder)
