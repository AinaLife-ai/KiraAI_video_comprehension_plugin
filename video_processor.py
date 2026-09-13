"""
视频处理器 —— 压缩、抽帧、标注、拼图

策略（三级叠加，确保信息密度最大）：
  第一级：场景变换帧（scene change detection）— 捕捉镜头切换点
  第二级：I帧（关键帧）— 编码层面最清晰的帧，补充场景变化未覆盖的
  第三级：均匀抽帧 — 时序兜底，保证时间全覆盖

标注：每帧左上角标注 时间戳 + 帧类型[I/P/B] + 帧序号
拼图：按时间顺序排列，每张不超过 max_frames_per_grid 帧
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    from PIL import __version__ as _pil_version
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

# ── 字体加载 ──────────────────────────────────────────────
_FONT_CANDIDATES = [
    "/usr/share/fonts/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/NotoSansCJK-Regular.ttc",
    "/system/fonts/NotoSansCJK-Bold.ttc",
]


def _load_font(size: int = 14):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── 工具函数 ───────────────────────────────────────────────

def _ensure_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用"""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _format_ts(seconds: float) -> str:
    """将秒数转为 MM:SS.mmm 格式"""
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def _get_video_info(video_path: str) -> dict:
    """用 ffprobe 读取视频基本信息"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return {"duration": 0, "width": 0, "height": 0, "size": 0}
    info = json.loads(r.stdout)
    duration = 0
    width, height = 0, 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            duration = float(s.get("duration", 0) or info.get("format", {}).get("duration", 0))
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
            break
    file_size = int(info.get("format", {}).get("size", 0))
    return {"duration": duration, "width": width, "height": height, "size": file_size}


# ── 视频下载 ───────────────────────────────────────────────

async def download_video(url: str, dest: str, timeout: int = 120) -> str:
    """异步下载视频文件，返回本地路径"""
    loop = asyncio.get_event_loop()

    def _dl():
        urllib.request.urlretrieve(url, dest)
        return dest

    return await asyncio.wait_for(
        loop.run_in_executor(None, _dl),
        timeout=timeout,
    )


# ── 视频压缩 ───────────────────────────────────────────────

async def compress_video(
    input_path: str,
    output_path: str,
    max_width: int = 720,
    crf: int = 28,
    audio_bitrate: str = "32k",
) -> str:
    """硬件加速压缩：缩放 720p + H.264 CRF28 + 低音频

    保持原时长不变，仅减小体积。
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale='min({max_width},iw)':-2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", str(crf),
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_path,
    ]

    def _run():
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"压缩失败: {r.stderr[:500]}")
        return output_path

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── 三级抽帧 ───────────────────────────────────────────────

def _scan_all_frames(video_path: str) -> list:
    """ffprobe 扫描所有帧 → [(timestamp, frame_type, 0.0), ...]

    ⚠️ ffmpeg 5+ 已移除 pkt_pts_time：用它会得到空列（如 "I,"），
    导致每一帧解析失败、抽帧整体失效。这里按
    best_effort_timestamp_time → pts_time → pkt_pts_time 依次回退。
    """
    entries_candidates = (
        "best_effort_timestamp_time,pict_type",
        "pts_time,pict_type",
        "pkt_pts_time,pict_type",
    )
    for entries in entries_candidates:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", f"frame={entries}",
            "-of", "csv=p=0",
            video_path,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception:
            continue
        if r.returncode != 0:
            continue
        frames = []
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            try:
                ts = float(parts[0])
            except (ValueError, IndexError):
                continue
            ft = parts[1].strip().upper() if len(parts) >= 2 and parts[1].strip() else "P"
            frames.append((ts, ft, 0.0))
        if frames:
            return frames
    return []


def _calc_scene_scores(frames, span: float, scene_threshold: float):
    """启发式场景检测：I 帧且与前帧间隔异常（远超平均间隔）→ 判为镜头切换。

    scene_threshold 越大越敏感（越容易判定为场景变化）。
    span 为该批帧覆盖的时长（用于估算平均帧间隔）。
    """
    scored = []
    for i, (ts, ft, _) in enumerate(frames):
        if i == 0:
            scored.append((ts, ft, 0.0))
            continue
        gap = ts - frames[i - 1][0]
        avg_gap = max(span / len(frames), 0.02)
        # 敏感度 0.1 → 阈值 2.8×平均间隔；0.8 → 阈值 1.4×平均间隔
        factor = 1.0 + 2.0 * (1.0 - scene_threshold)
        if ft == "I" and gap > avg_gap * factor:
            scene_score = min(gap / avg_gap * 0.1, 1.0)
        else:
            scene_score = 0.0
        scored.append((ts, ft, scene_score))
    return scored


def _extract_range(video_path: str, out_dir: str, all_frames: list,
                   seg_start: float, seg_end: float, n_frames: int,
                   scene_threshold: float, seg_idx: int = 0) -> list:
    """在 [seg_start, seg_end] 内做三级抽帧；时间戳保留原时间轴（绝对秒）。

    三级策略：场景变化帧 → I 帧补充 → 区间内均匀补帧。
    """
    seg_frames = [f for f in all_frames if seg_start <= f[0] <= seg_end]
    if not seg_frames:
        # 区间内扫不到帧（段极短/扫描异常）→ 用区间首尾兜底
        seg_frames = [(seg_start, "P", 0.0), (seg_end, "P", 0.0)]
    seg_span = max(seg_end - seg_start, 0.001)
    scored_frames = _calc_scene_scores(seg_frames, seg_span, scene_threshold)

    selected = set()
    result = []
    prefix = f"s{seg_idx}"
    # 去重阈值随段长/目标帧数自适应（短段用更小间隔，否则帧被过滤得太少）
    min_gap = max(0.05, min(0.3, seg_span / max(n_frames * 2, 1)))

    def _extract_one(ts: float, ft: str, label: str, idx: int):
        fname = f"{prefix}_{label}_{idx:04d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ts),
            "-i", video_path,
            "-vframes", "1",
            "-qscale:v", "3",
            "-an",
            fpath,
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        if not os.path.exists(fpath):
            return None
        return {"path": fpath, "timestamp": ts, "frame_type": ft,
                "frame_idx": idx, "segment": seg_idx}

    def _safe_extract(ts: float, ft: str, label: str, idx: int):
        for e in selected:
            if abs(e - ts) < min_gap:
                return None
        got = _extract_one(ts, ft, label, idx)
        if got:
            selected.add(round(ts, 2))
        return got

    # 第一级：场景变换帧（最多 1/3 目标帧数）
    scene_ts_list = [(ts, ft) for ts, ft, sc in scored_frames if sc > 0.001]
    max_scene = max(1, n_frames // 3)
    picked = 0
    for ts, ft in scene_ts_list[:max_scene]:
        r = _safe_extract(ts, ft, "sc", picked)
        if r:
            r["_source"] = "scene"
            result.append(r)
            picked += 1

    # 第二级：I 帧补充
    i_frame_ts = [ts for i, (ts, ft, sc) in enumerate(scored_frames)
                  if ft == "I" and i % 12 == 0]
    need = n_frames // 2 - len(result)
    step = max(1, len(i_frame_ts) // max(need, 1)) if i_frame_ts else 1
    for j in range(0, len(i_frame_ts), step):
        if len(result) >= n_frames * 2 // 3:
            break
        r = _safe_extract(i_frame_ts[j], "I", "i", len(result))
        if r:
            r["_source"] = "ifr"
            result.append(r)

    # 第三级：区间内均匀补帧
    need_uniform = n_frames - len(result)
    if need_uniform > 0:
        interval = seg_span / (need_uniform + 1)
        for k in range(1, need_uniform + 1):
            ts = seg_start + k * interval
            best_ts, best_gap = ts, 999.0
            for fts, fft, _ in scored_frames:
                gap = abs(fts - ts)
                if gap < best_gap:
                    best_gap, best_ts = gap, fts
            r = _safe_extract(best_ts, "P", "u", len(result))
            if r:
                r["_source"] = "uniform"
                result.append(r)

    result.sort(key=lambda x: x["timestamp"])
    # 超出 n_frames 时优先丢弃均匀补帧
    while len(result) > n_frames:
        discard_idx = None
        for i in range(1, len(result) - 1):
            if result[i].get("_source") == "uniform":
                discard_idx = i
                break
        result.pop(discard_idx if discard_idx is not None else len(result) // 2)

    for r in result:
        r["source"] = r.pop("_source", "")
    return result


def segment_frame_budget(seg_len: float, n_segments: int, target_frames: int) -> int:
    """按段长自适应帧数：每 ~2 秒 1 帧，最少 6 帧，不超过每段配额。

    多段时每段配额 = target_frames // 段数（总帧数不会失控）。
    """
    per_cap = target_frames if n_segments <= 1 else max(6, target_frames // n_segments)
    return int(max(6, min(per_cap, math.ceil(max(seg_len, 0.1) / 2))))


def normalize_segments(segments, duration: float) -> list:
    """校验并归一化区间：[[s,e],...] → [(s,e),...]，截断到 [0, duration]，丢弃空段。"""
    segs = []
    if not segments:
        return [(0.0, float(duration))]
    for item in segments:
        try:
            s, e = float(item[0]), float(item[1])
        except (TypeError, IndexError, ValueError):
            continue
        s = max(0.0, s)
        e = min(float(duration), e)
        if e - s >= 0.05:
            segs.append((s, e))
    return segs or [(0.0, float(duration))]


async def extract_frames(
    video_path: str,
    target_frames: int = 40,
    scene_threshold: float = 0.3,
    segments=None,
) -> list:
    """三级抽帧策略，返回 [{path, timestamp, frame_type, frame_idx, segment}, ...]

    segments: None/[] = 全片；[[start_sec, end_sec], ...] = 只抽这些区间。
              每段帧数按段长自适应（见 segment_frame_budget）。

    关键改进：先用 ffprobe 算出所有帧的精确时间戳，再逐个用 -ss 精确提取，
    避免 select filter 的帧索引漂移问题；时间戳始终为视频绝对时间。
    """
    out_dir = tempfile.mkdtemp(prefix="vcf_")
    info = _get_video_info(video_path)
    duration = info.get("duration", 0)
    if duration <= 0:
        cmd = ["ffmpeg", "-i", video_path, "-f", "null", "-"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
        if m:
            duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if duration <= 0:
        return []

    loop = asyncio.get_event_loop()
    all_frames = await loop.run_in_executor(None, _scan_all_frames, video_path)
    if not all_frames:
        return [
            {"path": "", "timestamp": 0, "frame_type": "I", "frame_idx": 0, "segment": 0},
            {"path": "", "timestamp": duration, "frame_type": "P", "frame_idx": 1, "segment": 0},
        ]

    segs = normalize_segments(segments, duration)
    n_seg = len(segs)
    result = []
    for si, (s, e) in enumerate(segs):
        n = segment_frame_budget(e - s, n_seg, target_frames)
        result.extend(_extract_range(video_path, out_dir, all_frames,
                                     s, e, n, scene_threshold, seg_idx=si))
    return result


async def clip_video(src: str, dst: str, start: float, end: float) -> tuple:
    """秒切视频片段（-c copy，起点对齐最近关键帧）。

    返回 (dst, 实际时长, 实际起点估算)。不重编码所以极快；
    起点可能提前 0~2 秒（对齐到关键帧），用时长差值反推估算实际起点。
    """
    start = max(0.0, float(start))
    req_dur = max(0.1, float(end) - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", src,
        "-t", str(req_dur),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        dst,
    ]

    def _run():
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            raise RuntimeError(f"裁剪失败: {(r.stderr or '')[:300]}")
        info = _get_video_info(dst)
        real_dur = float(info.get("duration") or req_dur)
        # 秒切会从最近的关键帧开始 → 实际时长 > 请求时长，差值即起点提前量
        actual_start = max(0.0, start - max(0.0, real_dur - req_dur))
        return dst, real_dur, actual_start

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── 时间戳标注 ─────────────────────────────────────────────

def stamp_frame(frame_img: Image.Image, timestamp: float,
                frame_type: str = "", frame_idx: int = 0) -> Image.Image:
    """在帧图像左上角叠加时间戳 + 帧类型 + 帧序号

    半透明黑底 + 黄色文字，清晰可辨。
    """
    draw = ImageDraw.Draw(frame_img)
    w = frame_img.width

    text = f"{_format_ts(timestamp)}"
    if frame_type:
        text = f"{_format_ts(timestamp)} [{frame_type}]"
    if frame_idx > 0:
        text += f" #{frame_idx}"

    font = _load_font(max(12, w // 24))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 10
    th = bbox[3] - bbox[1] + 6

    # 半透明黑底（RGB 无 alpha，用实色底 + 黄字保证可读）
    draw.rectangle([(0, 0), (tw, th)], fill=(0, 0, 0))

    # 黄色文字
    draw.text((5, 3), text, fill=(255, 255, 0), font=font)

    return frame_img


# ── 拼图合成 ───────────────────────────────────────────────

async def composite_grid(
    frames: list[dict],
    cols: int = 5,
    max_per_grid: int = 20,
    cell_width: int = 320,
    cell_ratio: str = "16:9",
    duration: float = 0,
    video_width: int = 0,
    video_height: int = 0,
    scene_count: int = 0,
    seg_label: str = "",
) -> list[Image.Image]:
    """将帧列表合成一张或多张拼图

    返回 Image.Image 列表（帧数超 max_per_grid 自动分片）
    """
    if not frames:
        return []

    # 计算 cell 尺寸
    if cell_ratio == "16:9":
        cell_h = int(cell_width * 9 / 16)
    elif cell_ratio == "4:3":
        cell_h = int(cell_width * 3 / 4)
    elif cell_ratio == "1:1":
        cell_h = cell_width
    else:  # auto — 按第一帧比例
        try:
            first = Image.open(frames[0]["path"])
            r = first.height / first.width
            cell_h = int(cell_width * r)
        except Exception:
            cell_h = int(cell_width * 9 / 16)

    # 分片
    chunks = [frames[i:i + max_per_grid] for i in range(0, len(frames), max_per_grid)]
    results = []

    loop = asyncio.get_event_loop()

    for chunk_idx, chunk in enumerate(chunks):
        total = len(chunk)
        rows = math.ceil(total / cols)
        # 实际列数：如果最后一行不满，用实际数量
        actual_cols = min(cols, total)

        grid_w = actual_cols * cell_width
        grid_h = rows * cell_h + 24  # 多24px底部信息条

        grid_img = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

        for i, f in enumerate(chunk):
            # 打开帧
            try:
                img = Image.open(f["path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (cell_width, cell_h), (60, 60, 60))

            # 缩放 + 裁剪到目标比例
            img_w, img_h = img.size
            target_ratio = cell_width / cell_h
            src_ratio = img_w / img_h

            if src_ratio > target_ratio:
                new_w = int(img_h * target_ratio)
                offset = (img_w - new_w) // 2
                img = img.crop((offset, 0, offset + new_w, img_h))
            elif src_ratio < target_ratio:
                new_h = int(img_w / target_ratio)
                offset = (img_h - new_h) // 2
                img = img.crop((0, offset, img_w, offset + new_h))

            img = img.resize((cell_width, cell_h), Image.LANCZOS)

            # 标注时间戳
            stamp_frame(
                img,
                timestamp=f.get("timestamp", 0),
                frame_type=f.get("frame_type", ""),
                frame_idx=f.get("frame_idx", 0),
            )

            # 粘贴到 grid 对应位置
            r, c = divmod(i, cols)
            x = c * cell_width
            y = r * cell_h
            grid_img.paste(img, (x, y))

        # 底部信息条
        if duration > 0:
            draw_bot = ImageDraw.Draw(grid_img)
            info_font = _load_font(11)
            info_text = (
                (seg_label or "") +
                f"拼图 {chunk_idx + 1}/{len(chunks)} | "
                f"帧: {total} | "
                f"时间: {_format_ts(chunk[0]['timestamp'])} ~ {_format_ts(chunk[-1]['timestamp'])}"
            )
            if video_width and video_height:
                info_text += f" | 原分辨率: {video_width}×{video_height}"
            if scene_count > 0:
                info_text += f" | 场景变化: {scene_count}"
            draw_bot.text((6, grid_h - 20), info_text, fill=(180, 180, 180), font=info_font)

        results.append(grid_img)

    return results


# ── 保存拼图为 base64 ─────────────────────────────────────

def grid_to_base64(grid: Image.Image, quality: int = 85) -> str:
    """拼图 → base64 Data URL"""
    buf = BytesIO()
    grid.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── 完整流水线 ─────────────────────────────────────────────

async def process_video(
    video_url: str,
    work_dir: str = None,
    max_file_mb: int = 200,
    max_duration_sec: int = 600,
    download_timeout: int = 120,
    # 压缩参数
    compress_max_width: int = 720,
    compress_crf: int = 28,
    # 抽帧参数
    target_frames: int = 40,
    scene_threshold: float = 0.3,
    # 拼图参数
    max_per_grid: int = 20,
    grid_cols: int = 5,
    cell_width: int = 320,
    cell_ratio: str = "16:9",
    # 区间参数：None = 全片；[[start_sec, end_sec], ...] = 只处理这些区间
    segments=None,
    # 只处理片段时可跳过全片压缩（省时），抽帧直接读原片
    skip_compress: bool = False,
    # 文件小于该值(MB)时也跳过压缩（上传模式：小文件保画质直接传）
    skip_compress_if_under_mb: float = 0,
) -> dict:
    """完整流水线：下载→压缩→抽帧→标注→拼图

    返回结构：
        {
            "status": "ok" | "rejected" | "error",
            "error": str,           # 仅错误时
            "duration": float,      # 视频总时长
            "width": int, "height": int,
            "file_size_mb": float,
            "compressed_path": str,     # 压缩后视频路径（原生模式用）
            "compressed_size_mb": float,
            "grids_base64": list[str],  # 拼图 base64 列表（帧模式用）
            "grid_count": int,
            "scene_count": int,
            "total_frames": int,
            "timestamps": list[float],  # 所有帧的时间戳
        }
    """
    if not _ensure_ffmpeg():
        return {"status": "error", "error": "ffmpeg 不可用，请安装 ffmpeg"}

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="vc_")
    else:
        os.makedirs(work_dir, exist_ok=True)

    raw_path = os.path.join(work_dir, "raw.mp4")
    compressed_path = os.path.join(work_dir, "compressed.mp4")
    is_local = os.path.isfile(video_url)

    try:
        # 1. 下载（仅当是 URL 时）
        start = time.time()
        if is_local:
            raw_path = video_url
        else:
            await download_video(video_url, raw_path, timeout=download_timeout)

        # 2. 检查大小
        file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
        if file_size_mb > max_file_mb:
            os.remove(raw_path)
            return {
                "status": "rejected",
                "error": f"视频文件过大 ({file_size_mb:.1f}MB > {max_file_mb}MB 限制)",
                "file_size_mb": file_size_mb,
            }

        # 3. 获取基本信息
        info = _get_video_info(raw_path)
        duration = info.get("duration", 0)
        width = info.get("width", 0)
        height = info.get("height", 0)

        if duration > max_duration_sec:
            os.remove(raw_path)
            return {
                "status": "rejected",
                "error": f"视频时长过长 ({duration:.0f}s > {max_duration_sec}s 限制)",
                "file_size_mb": file_size_mb,
                "duration": duration,
            }

        # 4. 压缩（保持时长，缩小体积）
        #    skip_compress → 跳过；skip_compress_if_under_mb>0 且文件更小 → 也跳过
        do_compress = not skip_compress
        if do_compress and skip_compress_if_under_mb > 0 and file_size_mb <= skip_compress_if_under_mb:
            do_compress = False
        if do_compress:
            try:
                await compress_video(raw_path, compressed_path,
                                     max_width=compress_max_width,
                                     crf=compress_crf)
            except Exception as e:
                # 压缩失败，改用原文件
                compressed_path = raw_path
        else:
            compressed_path = raw_path

        compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
        compress_ratio = compressed_size_mb / file_size_mb if file_size_mb > 0 else 1

        # 5. 抽帧（可指定区间）
        frames = await extract_frames(
            compressed_path if os.path.exists(compressed_path) else raw_path,
            target_frames=target_frames,
            scene_threshold=scene_threshold,
            segments=segments,
        )

        # 统计真实场景变化帧数（此前误用 I 帧数量）
        scene_count = sum(1 for f in frames if f.get("source") == "scene")

        # 6. 拼图：按段分别拼（段数 > 1 时在底部标注段号与区间）
        segs = normalize_segments(segments, duration)
        multi = len(segs) > 1
        grids = []
        for si, (seg_s, seg_e) in enumerate(segs):
            seg_frames = [f for f in frames if f.get("segment", 0) == si]
            if not seg_frames:
                continue
            label = (f"段{si + 1}/{len(segs)} {_format_ts(seg_s)}~{_format_ts(seg_e)} | "
                     if multi else "")
            grids.extend(await composite_grid(
                seg_frames,
                cols=grid_cols,
                max_per_grid=max_per_grid,
                cell_width=cell_width,
                cell_ratio=cell_ratio,
                duration=duration,
                video_width=width,
                video_height=height,
                scene_count=scene_count,
                seg_label=label,
            ))

        # 7. 转 base64
        grids_b64 = [grid_to_base64(g) for g in grids]

        elapsed = time.time() - start

        return {
            "status": "ok",
            "duration": duration,
            "width": width,
            "height": height,
            "file_size_mb": file_size_mb,
            "compressed_path": compressed_path,
            "compressed_size_mb": compressed_size_mb,
            "compress_ratio": compress_ratio,
            "grids_base64": grids_b64,
            "grid_count": len(grids),
            "scene_count": scene_count,
            "total_frames": len(frames),
            "timestamps": [f["timestamp"] for f in frames],
            "segments": [[s, e] for s, e in segs],
            "elapsed_sec": elapsed,
        }

    except asyncio.TimeoutError:
        return {"status": "error", "error": f"视频下载超时 ({download_timeout}s)"}
    except Exception as e:
        return {"status": "error", "error": f"视频处理异常: {type(e).__name__}: {str(e)[:200]}"}