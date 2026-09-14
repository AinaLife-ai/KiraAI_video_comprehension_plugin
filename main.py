"""
Video Comprehension 插件 — QQ/B站视频理解

功能：
  - send_video 工具：下载B站视频+压缩+发送到QQ
  - search_bili_video 工具：搜索B站视频
  - analyze_video 工具（默认关）：视频内容分析
  - 自动链接检测钩子（默认关）
  - 缓存双向：B站 → files/video_cache/，其他 → files/video_analysis_cache/
"""
from __future__ import annotations

# ── 导入区 ──

import asyncio
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from core.plugin import BasePlugin, logger, on, Priority, register
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.provider import LLMRequest
from core.utils.path_utils import get_data_path

from video_processor import (process_video, compress_video, clip_video, download_video,
                             extract_audio, detect_speech_ranges, merge_ranges, slice_audio)
from asr import (transcribe as asr_transcribe, build_timeline_doc, ASRError,
                  normalize_segments_by_duration)
from video_host import upload_to_any, UploadError, DEFAULT_HOSTS as DEFAULT_UPLOAD_HOSTS
from llm_proxy import (ModelProfile, select_model, build_meta, analyze_frames,
                       analyze_native, NATIVE_MAX_MB, _as_dict)
from bili_dl import (search_bili, get_bili_info, get_ai_summary, download_bili_video,
                     extract_bvid, BiliError, get_bilibili_subtitle,
                     get_bili_direct_url)

BILI_RE = re.compile(r"(BV[0-9A-Za-z]{10}|b23\.tv/[^\s]+|bilibili\.com/(?:video/|BV))", re.I)


def _collect_chain_text(chain) -> str:
    """收集消息链里所有可能带链接的文本。

    ⚠️ 不能只看 Text 元素：QQ 小程序卡片（app=com.tencent.miniapp_01）之类的
    元素不是 Text 类型，但它们的字段里带着 qqdocurl（B站短链）。
    这里把每个元素的常见字符串字段（含 dict）都展开收集，只用于「找链接」，
    不影响其它逻辑。
    """
    parts: list = []
    try:
        items = list(chain or [])
    except Exception:
        return ""
    for ele in items:
        try:
            if isinstance(ele, Text):
                t = getattr(ele, "text", "") or ""
                if t:
                    parts.append(t)
                continue
            # 非 Text 元素：展开它的属性（含 __dict__），只收字符串与 dict
            attrs = {}
            try:
                attrs = dict(vars(ele))
            except Exception:
                attrs = {}
            for key in ("text", "data", "json", "raw", "url", "content", "summary",
                        "qqdocurl", "prompt", "desc", "title"):
                v = getattr(ele, key, None)
                if v is not None and key not in attrs:
                    attrs[key] = v
            for v in attrs.values():
                if isinstance(v, str):
                    if v:
                        parts.append(v)
                elif isinstance(v, dict):
                    try:
                        parts.append(json.dumps(v, ensure_ascii=False))
                    except Exception:
                        pass
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, str) and x:
                            parts.append(x)
        except Exception:
            continue
    return "\n".join(parts)
BVID_RE = re.compile(r"BV[0-9A-Za-z]{10}")

# 时间段分析限制
MAX_SEGMENTS = 5          # 一次最多几段
MAX_SEGMENT_SEC = 300     # 单段最长秒数


class VideoSession:
    __slots__ = (
        "session_id", "sid", "source", "source_url", "title",
        "compressed_path", "duration", "width", "height",
        "grids_base64", "scene_count", "total_frames", "timestamps",
        "file_size_mb", "compressed_size_mb",
        "analysis", "analysis_model", "analysis_mode",
        "history", "last_interact", "bili_ai_summary",
        "host_url", "transcript_doc", "model_tag",
    )
    def __init__(self, session_id, sid, source, source_url):
        self.session_id = session_id
        self.sid = sid
        self.source = source
        self.source_url = source_url
        self.title = ""
        self.compressed_path = ""
        self.duration = 0.0
        self.width = self.height = 0
        self.grids_base64 = []
        self.scene_count = 0
        self.total_frames = 0
        self.timestamps = []
        self.file_size_mb = self.compressed_size_mb = 0.0
        self.analysis = ""
        self.analysis_model = ""
        self.analysis_mode = ""
        self.history = []
        self.last_interact = time.time()
        self.bili_ai_summary = None
        self.host_url = ""          # 上传到文件中转后的公开直链（若有）
        self.transcript_doc = ""    # 语音转写时间轴文档（若有）
        self.model_tag = ""         # 该会话首次分析用的模型组（追问默认沿用，避免"换模型"）

    def is_stale(self, ttl: int) -> bool:
        return time.time() - self.last_interact > ttl * 60

    def add_turn(self, q: str, a: str):
        self.history.append({"role": "user", "text": q})
        self.history.append({"role": "bot", "text": a})
        self.last_interact = time.time()


class VideoComprehensionPlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        # 运行状态（热重载配置时保留）
        self._pending: dict[str, dict] = {}
        self._sessions: dict[str, VideoSession] = {}
        self._sid_sessions: dict[str, list[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cleanup: Optional[asyncio.Task] = None
        self._auto_sent: dict[str, dict] = {}  # sid → {bvid, title, file_path, text}
        self._ffmpeg_ok = False
        self._stream_unsupported = False
        self._upload_cache: dict[str, str] = {}   # 本地路径 → 已上传的公开 URL
        self._cached_videos: dict[str, list] = {}   # sid → [{orig_name, path, rel, size_mb, ts}, ...]
        self._video_failures: dict[str, dict] = {}  # sid → {原文件名: 失败原因}（供消息改写）
        self._asr_tasks: dict[str, asyncio.Task] = {}   # 转写缓存 key → 进行中的任务
        self._load_cfg(cfg)

    def _load_cfg(self, cfg: dict):
        """读取/热重载配置（不影响会话与后台任务状态）"""
        basic = cfg.get("section_basic", {}) or {}
        self.enabled = basic.get("enabled", True)
        self.video_analysis_enabled = basic.get("video_analysis_enabled", False)
        self.auto_select = basic.get("auto_select", True)
        self.default_model = str(basic.get("default_model", "auto"))
        self.allowed_adapters = basic.get("allowed_adapters", [])
        self.max_session_per_user = int(basic.get("max_session_per_user", 5))
        self.auto_cache_video = bool(basic.get("auto_cache_video", True))

        # 双缓存
        cs = cfg.get("section_cache", {}) or {}
        self.bili_cache_dir = str(Path(get_data_path()) / cs.get("bili_cache_dir", "files/video_cache").lstrip("/"))
        self.bili_max_cache = int(cs.get("bili_max_cache_files", 100))
        self.bili_cleanup = int(cs.get("bili_cleanup_count", 20))
        self.other_cache_dir = str(Path(get_data_path()) / cs.get("other_cache_dir", "files/video_analysis_cache").lstrip("/"))
        self.other_max_cache = int(cs.get("other_max_cache_files", 200))
        self.other_cleanup = int(cs.get("other_cleanup_count", 30))
        self.cache_max_file_mb = int(cs.get("cache_max_file_mb", 20))
        self.cache_ttl_hours = float(cs.get("cache_ttl_hours", 24))
        self.cache_max_total_mb = int(cs.get("cache_max_total_mb", 2048))

        self._profiles: list[ModelProfile] = []
        for g in range(1, 5):
            p = ModelProfile.from_cfg(cfg, g)
            if p: self._profiles.append(p)

        fs = cfg.get("section_frame", {}) or {}
        self.target_frames = int(fs.get("target_frames", 40))
        self.max_per_grid = int(fs.get("max_frames_per_grid", 20))
        self.grid_cols = int(fs.get("grid_cols", 5))
        self.scene_threshold = float(fs.get("scene_threshold", 0.3))
        self.cell_width = int(fs.get("frame_width", 320))
        self.cell_ratio = fs.get("frame_ratio", "16:9")

        lm = cfg.get("section_limits", {}) or {}
        self.max_file_mb = int(lm.get("max_file_size_mb", 200))
        _md = int(lm.get("max_duration_sec", 0) or 0)
        if _md <= 0:
            # 0 = 自动跟随：取所有已启用模型组里最大的时长上限。
            # 避免"组里配了 1800 秒，却被这里 600 秒的硬上限先拦死"的矛盾。
            _md = max([p.max_video_sec for p in self._profiles] or [600])
        self.max_duration = _md
        self.max_duration_auto = int(lm.get("max_duration_sec", 0) or 0) <= 0
        self.dl_timeout = int(lm.get("download_timeout_sec", 120))

        bs = cfg.get("section_bili", {}) or {}
        self.bili_enabled = bs.get("bili_enabled", True)
        self.bili_cookie = bs.get("bili_cookie", "")
        self.bili_use_ai = bs.get("bili_use_ai_summary", True)
        self.bili_search_n = int(bs.get("bili_search_count", 5))
        self.bili_max_dl = int(bs.get("bili_max_download_sec", 600))
        self.auto_send_link = bs.get("auto_send_link", False)
        # NapCat 的 upload_file_stream（第三方扩展 action，分块上传）—— 默认关闭。
        # 官方 NapCat 没有它，同类软件（AstrBot 等）也不用它；直接用本地路径发送
        # 对 NapCat / SnowLuma 都够用。开启后才尝试分块上传（失败仍会自动降级）。
        self.napcat_stream = bool(bs.get("napcat_stream_upload", False))
        self.auto_send_allowed_sid = [str(s).strip() for s in bs.get("auto_send_allowed_sid", []) if str(s).strip()]
        self.search_show_desc = bs.get("search_show_desc", True)
        self.search_desc_max_chars = int(bs.get("search_desc_max_chars", 100))
        self.bili_download_quality = bs.get("bili_download_quality", "low")
        self.bili_compress_quality = bs.get("bili_compress_quality", "original")

        ss = cfg.get("section_session", {}) or {}
        self.session_ttl = int(ss.get("session_ttl_minutes", 30))
        self.send_video_quality = ss.get("send_video_quality", "low")
        self.default_prompt = ss.get("default_prompt",
            "你刚刚收到一个视频。请分析其内容，包括：\n1. 视频整体描述\n2. 关键事件时间线\n3. 值得注意的细节\n4. 语音/对话内容")

        us = cfg.get("section_upload", {}) or {}
        self.upload_enabled = bool(us.get("upload_enabled", True))
        # 多源：upload_hosts（list）优先；兼容旧的 upload_host（string）
        hosts = us.get("upload_hosts")
        if hosts is None:
            old = us.get("upload_host")
            hosts = [old] if old else []
        if isinstance(hosts, str):
            hosts = [hosts]
        self.upload_hosts = [str(h).strip() for h in (hosts or []) if str(h).strip()] \
            or list(DEFAULT_UPLOAD_HOSTS)
        self.upload_host = self.upload_hosts[0]   # 兼容旧引用

        # ── 语音转写（给模型补上"声音"信息） ──
        au = cfg.get("section_audio", {}) or {}
        self.audio_enabled = bool(au.get("audio_transcribe_enabled", True))
        self.audio_base_url = str(au.get("audio_stt_base_url", "") or "").strip()
        self.audio_api_key = str(au.get("audio_stt_api_key", "") or "").strip()
        self.audio_model = str(au.get("audio_stt_model", "") or "").strip()
        self.audio_wait_sec = float(au.get("audio_wait_sec", 60) or 60)
        self.audio_max_sec = float(au.get("audio_max_sec", 0) or 0)
        self.audio_language = str(au.get("audio_language", "") or "").strip()
        self.audio_timeout = float(au.get("audio_timeout_sec", 300) or 300)
        self.audio_use_proxy = bool(au.get("audio_use_proxy", False))
        self.audio_block_sec = float(au.get("audio_block_sec", 30) or 30)
        self.audio_gap_sec = float(au.get("audio_gap_sec", 2.5) or 2.5)
        self.bili_use_subtitle = bool(au.get("bili_use_subtitle", True))
        self.bili_direct_url = bool(bs.get("bili_direct_url", True))
        self.audio_extra_headers = _as_dict(au.get("audio_stt_extra_headers"))
        self.audio_extra_body = _as_dict(au.get("audio_stt_extra_body"))
        self.cache_scope = str(cs.get("cache_scope", "mentioned") or "mentioned").strip().lower()
        if self.cache_scope not in ("mentioned", "batch", "all"):
            self.cache_scope = "mentioned"
        self.audio_max_blocks = int(au.get("audio_max_blocks", 80) or 80)
        self.audio_concurrency = max(1, min(16, int(au.get("audio_concurrency", 5) or 5)))
        self.audio_silence_db = float(au.get("audio_silence_db", -35) or -35)
        self.upload_max_mb = int(us.get("upload_max_mb", 200))
        self.upload_compress_over_mb = int(us.get("upload_compress_over_mb", 20))
        self.upload_timeout = int(us.get("upload_timeout_sec", 300))
        self.upload_keep_name = bool(us.get("upload_keep_name", True))
        self.upload_use_proxy = bool(us.get("upload_use_proxy", False))


    async def _ensure_ffmpeg_async(self):
        """后台异步确保 ffmpeg 可用

        策略：shutil.which() 检查 PATH → 有就用
              → 没有就下载静态 ffmpeg 到缓存目录
              → 加到 os.environ['PATH'] 全局生效
        """
        if self._ffmpeg_ok:
            return True

        # 1) 检查 PATH（ffmpeg + ffprobe 都需要）
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            if not shutil.which("ffprobe"):
                logger.warning("[VC] ⚠️ ffmpeg 存在但 ffprobe 缺失，抽帧/时长探测会失败")
            self._ffmpeg_ok = True
            logger.info("[VC] ✅ ffmpeg 已可用: %s", ffmpeg_path)
            return True

        import platform as _pf
        system = _pf.system().lower()
        logger.info("[VC] ffmpeg 不在 PATH 中，系统=%s，准备下载静态版本...", system)

        # 2) 尝试系统包管理器（仅 Linux）
        if system == "linux":
            for pm, cmd in [("apk", ["apk", "add", "ffmpeg"]),
                            ("apt-get", ["apt-get", "install", "-y", "ffmpeg"])]:
                try:
                    which_pm = shutil.which(pm)
                    if not which_pm: continue
                    proc = await asyncio.create_subprocess_exec(
                        *cmd, stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    if await proc.wait() == 0 and shutil.which("ffmpeg"):
                        self._ffmpeg_ok = True
                        logger.info("[VC] ✅ ffmpeg 通过 %s 安装成功", pm)
                        return True
                except: continue

        # 3) 下载静态 ffmpeg（Win/Linux 通用兜底）
        static_dir = os.path.join(self.bili_cache_dir, ".ffmpeg")
        os.makedirs(static_dir, exist_ok=True)
        exe_name = "ffmpeg.exe" if system == "windows" else "ffmpeg"
        static_bin = os.path.join(static_dir, exe_name)

        # 检查之前是否已下载
        if os.path.isfile(static_bin):
            os.environ["PATH"] = static_dir + os.pathsep + os.environ.get("PATH", "")
            self._ffmpeg_ok = True
            logger.info("[VC] ✅ 使用已下载的静态 ffmpeg: %s", static_bin)
            return True

        logger.info("[VC] ⬇️ 下载静态 ffmpeg → %s ...", static_bin)
        try:
            import httpx as _hx
            arch = _pf.machine().lower()

            if system == "windows":
                # Windows: gyan.dev 提供的 zip
                dl_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
                async with _hx.AsyncClient(follow_redirects=True, timeout=180) as c:
                    resp = await c.get(dl_url)
                    if resp.status_code == 200:
                        import zipfile, io
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            got = False
                            for name in zf.namelist():
                                if name.endswith(("ffmpeg.exe", "ffprobe.exe")):
                                    zf.extract(name, static_dir)
                                    exe_path = os.path.join(static_dir, name)
                                    shutil.move(exe_path, os.path.join(static_dir, os.path.basename(name)))
                                    got = True
                            if got:
                                # 清理多余文件
                                for d in os.listdir(static_dir):
                                    dp = os.path.join(static_dir, d)
                                    if os.path.isdir(dp) and d.startswith("ffmpeg"):
                                        shutil.rmtree(dp, ignore_errors=True)
                                os.environ["PATH"] = static_dir + os.pathsep + os.environ.get("PATH", "")
                                self._ffmpeg_ok = True
                                logger.info("[VC] ✅ Windows 静态 ffmpeg+ffprobe 下载完成")
                                return True
            else:
                # Linux: johnvansickle 提供的 tar.xz（同时包含 ffmpeg 与 ffprobe）
                dl_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz" if arch in ("aarch64", "arm64") else "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
                async with _hx.AsyncClient(follow_redirects=True, timeout=180) as c:
                    resp = await c.get(dl_url)
                    if resp.status_code == 200:
                        import tarfile, io
                        with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
                            got = False
                            for m in tar.getmembers():
                                base = os.path.basename(m.name)
                                if base in ("ffmpeg", "ffprobe"):
                                    tar.extract(m, path=static_dir)
                                    exe_path = os.path.join(static_dir, m.name)
                                    shutil.move(exe_path, os.path.join(static_dir, base))
                                    os.chmod(os.path.join(static_dir, base), 0o755)
                                    got = True
                            if got:
                                # 清理
                                for d in os.listdir(static_dir):
                                    dp = os.path.join(static_dir, d)
                                    if os.path.isdir(dp) and d.startswith("ffmpeg"):
                                        shutil.rmtree(dp, ignore_errors=True)
                                os.environ["PATH"] = static_dir + os.pathsep + os.environ.get("PATH", "")
                                self._ffmpeg_ok = True
                                logger.info("[VC] ✅ Linux 静态 ffmpeg+ffprobe 下载完成")
                                return True
        except Exception as e:
            logger.warning("[VC] 静态 ffmpeg 下载失败: %s", e)

        logger.warning("[VC] ❌ 无法获取 ffmpeg，视频压缩/抽帧/音视频合并功能不可用")
        return False

    async def initialize(self):
        if not self.enabled: return
        # 后台异步安装 ffmpeg（首条日志提示用户）
        logger.info("[VC] 🔍 检查 ffmpeg...（若缺失将后台自动安装，视频压缩/抽帧需要它）")
        asyncio.create_task(self._ensure_ffmpeg_async())
        os.makedirs(self.bili_cache_dir, exist_ok=True)
        os.makedirs(self.other_cache_dir, exist_ok=True)
        await self._do_cleanup(self.bili_cache_dir, self.bili_max_cache, self.bili_cleanup, "B站")
        await self._do_cleanup(self.other_cache_dir, self.other_max_cache, self.other_cleanup, "其他")
        self._cleanup = asyncio.create_task(self._cleanup_loop())
        logger.info("[VC] 分析=%s B站=%s | B站缓存=%s | 其他缓存=%s",
                     self.video_analysis_enabled, self.bili_enabled,
                     self.bili_cache_dir, self.other_cache_dir)

    async def terminate(self):
        if self._cleanup and not self._cleanup.done():
            self._cleanup.cancel()
            try: await self._cleanup
            except asyncio.CancelledError: pass
        self._pending.clear(); self._sessions.clear(); self._sid_sessions.clear()
        self._cached_videos.clear()
        self._video_failures.clear()
        for t in list(self._asr_tasks.values()):
            if not t.done():
                t.cancel()
        self._asr_tasks.clear()

    # ── 缓存清理（通用） ──

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
        return total

    def _scan_cache(self, cache_dir: str, max_files: int, cleanup_n: int, label: str = "",
                    ttl_hours: float = 0, max_total_mb: float = 0):
        """返回待删除条目名列表。

        三个维度叠加：① 超龄(TTL) → ② 超总容量 → ③ 超条数。
        文件与「分析子目录」都算条目（子目录按其中所有文件的总大小计）。
        """
        try:
            if not os.path.isdir(cache_dir): return []
            entries = []
            for name in os.listdir(cache_dir):
                if name.startswith("."):
                    continue  # 跳过 .ffmpeg 等隐藏目录
                p = os.path.join(cache_dir, name)
                try:
                    st = os.stat(p)
                except Exception:
                    continue
                is_dir = os.path.isdir(p)
                entries.append({
                    "name": name, "mtime": st.st_mtime, "is_dir": is_dir,
                    "size": st.st_size if not is_dir else self._dir_size(p),
                })
            if not entries:
                return []
            entries.sort(key=lambda e: e["mtime"])   # 最旧优先
            to_del = []
            keep = entries
            # ① TTL：超龄一律删
            if ttl_hours and ttl_hours > 0:
                cutoff = time.time() - ttl_hours * 3600
                to_del = [e for e in entries if e["mtime"] < cutoff]
                keep = [e for e in entries if e["mtime"] >= cutoff]
            # ② 总容量超限：从最旧的删起
            if max_total_mb and max_total_mb > 0:
                limit = max_total_mb * 1024 * 1024
                total = sum(e["size"] for e in keep)
                i = 0
                while total > limit and i < len(keep):
                    to_del.append(keep[i]); total -= keep[i]["size"]; i += 1
                keep = keep[i:]
            # ③ 条数超限：删最旧的若干（不超过实际超出的数量）
            if max_files and max_files > 0 and len(keep) > max_files:
                n = min(cleanup_n, len(keep) - max_files)
                to_del.extend(keep[:n])
            return [e["name"] for e in to_del]
        except Exception as e:
            logger.warning("[VC] 缓存[%s]扫描异常: %s", label or cache_dir, e)
            return []

    async def _do_cleanup(self, cache_dir: str, max_files: int, cleanup_n: int, label: str = "",
                          ttl_hours: float = 0, max_total_mb: float = 0):
        to_del = self._scan_cache(cache_dir, max_files, cleanup_n, label,
                                  ttl_hours=ttl_hours, max_total_mb=max_total_mb)
        if not to_del: return
        deleted = 0
        for name in to_del:
            p = os.path.join(cache_dir, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
                deleted += 1
            except: pass
            await asyncio.sleep(0)
        try:
            remain = len([x for x in os.listdir(cache_dir) if not x.startswith(".")])
        except Exception:
            remain = 0
        logger.info("[VC] 缓存清理[%s]: 删%d个余%d个", label or cache_dir, deleted, remain)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(300)
            await self._do_cleanup(self.bili_cache_dir, self.bili_max_cache, self.bili_cleanup, "B站")
            await self._do_cleanup(self.other_cache_dir, self.other_max_cache, self.other_cleanup, "其他",
                                   ttl_hours=self.cache_ttl_hours,
                                   max_total_mb=self.cache_max_total_mb)
            stale = [k for k, v in self._sessions.items() if v.is_stale(self.session_ttl)]
            for k in stale:
                self._sessions.pop(k, None)
                for sl in self._sid_sessions.values():
                    if k in sl: sl.remove(k)
            self._sid_sessions = {k: v for k, v in self._sid_sessions.items() if v}
            # 未被告知 bot 的缓存记录（30 分钟未用则丢弃）
            _now = time.time()
            self._cached_videos = {k: [c for c in v if _now - c.get("ts", 0) < 1800]
                                   for k, v in self._cached_videos.items() if v}

    def _sid(self, event) -> str:
        return getattr(event.session, "sid", None) or getattr(event, "sid", "") or ""

    def _is_qq(self, event) -> bool:
        """判断是否 QQ 平台（platform 来自适配器 manifest.name，内置为 "QQ"，大小写不敏感）"""
        return str(getattr(event.adapter, "platform", "") or "").strip().lower() == "qq"

    def _ok(self, event) -> bool:
        if not self.allowed_adapters: return True
        allow = {str(a).strip().lower() for a in self.allowed_adapters}
        n = str(getattr(event.adapter, "name", "") or "").strip().lower()
        p = str(getattr(event.adapter, "platform", "") or "").strip().lower()
        return n in allow or p in allow

    def _register_session(self, sess, sid):
        self._sessions[sess.session_id] = sess
        if sid not in self._sid_sessions: self._sid_sessions[sid] = []
        lst = self._sid_sessions[sid]
        if sess.session_id in lst: lst.remove(sess.session_id)
        lst.insert(0, sess.session_id)
        if len(lst) > self.max_session_per_user:
            old = lst.pop(); self._sessions.pop(old, None)

    def _get_by_session_id(self, sid):
        return self._sessions.get(sid)

    def _list_sessions(self, sid):
        return [self._sessions[s] for s in self._sid_sessions.get(sid, []) if s in self._sessions]

    # ── 自动发送B站链接（对标音频条 auto_send_link，0 token） ──

    @on.im_message(priority=Priority.HIGH)
    async def _auto_send_hook(self, event: KiraMessageEvent, *_):
        if not self.enabled or not self.auto_send_link:
            return
        if not self.bili_enabled: return
        if self.auto_send_allowed_sid and event.session.sid not in self.auto_send_allowed_sid:
            return
        if not self._is_qq(event):
            return
        sid = event.session.sid or ""
        if not sid:
            return

        bvid = ""

        # 收集整条消息链里所有可能带链接的文本。
        # ⚠️ 不能只看 Text 元素：QQ 小程序卡片（com.tencent.miniapp_01）等
        #    元素不是 Text 类型，但它们的字段里带着 qqdocurl（B站短链）。
        text = _collect_chain_text(event.message.chain)

        # 1) 精确搜 BV 号（BVID_RE 就是 BV[0-9A-Za-z]{10}，不会脏匹配）
        m = BVID_RE.search(text)
        if m:
            bvid = m.group(0)

        # 2) 没有 BV 号但有 b23 短链 → 解析成 BV 号
        if not bvid:
            m = re.search(r'b23\.tv/([0-9A-Za-z]+)', text)
            if m:
                try: bvid = await extract_bvid(f"https://b23.tv/{m.group(1)}", self.dl_timeout)
                except Exception:
                    logger.info("[VC] b23 短链解析失败: %s", m.group(1))

        # 3) raw_message JSON（小程序卡片/app分享的 qqdocurl 里藏 b23）
        if not bvid:
            import json as _json
            raw = getattr(event, "raw_message", None)
            if raw is None and hasattr(event, "message"):
                raw = getattr(event.message, "raw_message", None)
            if raw is None and hasattr(event, "message") and hasattr(event.message, "source_message"):
                raw = getattr(event.message, "source_message", None)
            if raw is None:
                raw = str(event)
            if isinstance(raw, dict):
                try: raw = _json.dumps(raw)
                except: raw = ""
            if isinstance(raw, str):
                m = re.search(r'https?://b23\.tv/[0-9A-Za-z]+', raw)
                if m:
                    try: bvid = await extract_bvid(m.group(0), self.dl_timeout)
                    except: pass
                if not bvid:
                    m = re.search(r'BV[0-9A-Za-z]{10}', raw)
                    if m: bvid = m.group(0)

        if not bvid: return
        logger.info("[VC] auto_send 检测到B站视频: %s", bvid)
        asyncio.create_task(self._auto_send_do(bvid, event.adapter.name, sid))

    async def _auto_send_do(self, bvid: str, adapter_name: str, sid: str):
        """异步后台发送，成功后记录 auto_sent 用于 LLM 上下文标注"""
        try:
            reply = await self._send_video_by_bvid(None, bvid, sid=sid, adapter_name=adapter_name)
            if reply and reply.startswith("✅"):
                # 成功 → 记录 auto_sent，不 discard，消息继续自然流转
                title = bvid
                for line in reply.split("\n"):
                    if "已发送：" in line:
                        title = line.split("已发送：")[-1].strip()
                self._auto_sent[sid] = {
                    "bvid": bvid,
                    "title": title,
                    "file_path": reply.split("本地路径:")[-1].strip() if "本地路径:" in reply else "",
                }
            elif reply:
                # 失败 → 补发文字提示
                await self.ctx.message_processor.send_message_chain(
                    sid, MessageChain([Text(reply)]))
        except Exception as e:
            logger.warning("[VC] auto_send 失败: %s", e)
            try: await self.ctx.message_processor.send_message_chain(sid, MessageChain([Text(f"❌ 发送B站视频失败: {e}")]))
            except: pass

    # ── 批次阶段缓存（避免群里与 bot 无关的视频也被下载+转写） ──

    @on.im_batch_message(priority=Priority.LOW)
    async def _on_batch_cache(self, event, *_):
        """消息合并成批次（确定要送给 bot）后，才按 cache_scope 决定要不要缓存+转写。

        - mentioned：只处理「被 @ / 引用 / 唤醒」的消息里的视频（默认，最省）
        - batch    ：处理批次里所有消息的视频
        - all      ：已在 _detect 阶段逐条缓存，这里不重复做

        优先级 LOW：批次若被更早的插件 stop 掉，本钩子根本不会执行。
        """
        if not self.enabled or not self.auto_cache_video: return
        if self.cache_scope == "all": return
        if not self._ok(event): return
        sid = self._sid(event)
        if not sid: return
        try:
            msgs = list(getattr(event, "messages", None) or [])
        except Exception:
            return
        if self.cache_scope == "mentioned":
            msgs = [m for m in msgs if getattr(m, "is_mentioned", False)]
        for m in msgs:
            try:
                for ele in self._iter_videos(getattr(m, "chain", None)):
                    f = str(getattr(ele, "file", "") or "")
                    if f.startswith(("http://", "https://")):
                        nm = str(getattr(ele, "name", "") or "")
                        asyncio.create_task(self._cache_then_transcribe(sid, f, nm))
            except Exception:
                continue

    # ── 已直发的 LLM 上下文标注（对齐音频条 inject_auto_sent_note） ──

    @on.llm_request(priority=Priority.LOW)
    async def _inject_auto_sent_note(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.enabled: return
        sid = getattr(event.session, "sid", None)
        if not sid: return
        sent = self._auto_sent.pop(sid, None)
        if not sent: return
        note = (
            f"\n[系统提示：该B站视频（《{sent['title']}》"
            f"BV:{sent['bvid']}）已自动发送压缩版视频（{sent.get('file_path','')}）]"
        )
        bvid = sent["bvid"]
        # 按顺序遍历 messages ↔ user_prompt，定位原始消息追加 note
        prompt_idx = 0
        for msg in event.messages:
            while prompt_idx < len(req.user_prompt) and not (
                    req.user_prompt[prompt_idx].name == "message"
                    and req.user_prompt[prompt_idx].source == "system"
            ):
                prompt_idx += 1
            if prompt_idx >= len(req.user_prompt):
                break
            p = req.user_prompt[prompt_idx]
            prompt_idx += 1
            msg_text = "".join(e.text for e in msg.chain if isinstance(e, Text))
            if bvid in msg_text:
                p.content += note
                break

    # ── 改写消息里的视频占位（让 bot 拿到可用的路径，而不是"没缓存"） ──

    _VIDEO_PLACEHOLDER_RE = re.compile(
        r"\[Video name:\s*([^\]\(]+?)\s*\(Video size over 10MB, not cached\)\]")

    def _rewrite_video_notes(self, event, req, sid: str) -> None:
        """把框架渲染的 `[Video name: X (Video size over 10MB, not cached)]`
        替换成我们能提供的最佳信息：

          - 已缓存     → [视频已缓存: <相对路径>]
          - 缓存失败   → [视频未缓存: <原因>]
          - 未预缓存   → [视频 X（未预缓存，可用 analyze_video 分析）]

        最后一种很关键：即使因为缓存策略没提前下，也要让 bot 知道
        「工具其实能处理」，而不是像以前那样回答"我看不到内容"。
        """
        cached = self._cached_videos.get(sid) or []
        failed = self._video_failures.get(sid) or {}
        if not cached and not failed and sid not in self._pending:
            return
        repl = {}
        for c in cached:
            n = str(c.get("orig_name") or "").strip().lower()
            if n:
                repl[n] = f"[视频已缓存: {c.get('rel','')}]"
        for n, reason in failed.items():
            repl.setdefault(n.lower(), f"[视频未缓存: {reason}]")

        prompt_idx = 0
        for msg in (getattr(event, "messages", None) or []):
            while prompt_idx < len(req.user_prompt) and not (
                    req.user_prompt[prompt_idx].name == "message"
                    and req.user_prompt[prompt_idx].source == "system"):
                prompt_idx += 1
            if prompt_idx >= len(req.user_prompt):
                break
            p = req.user_prompt[prompt_idx]; prompt_idx += 1
            if not isinstance(p.content, str) or "[Video name:" not in p.content:
                continue

            def _sub(m, _repl=repl):
                name = (m.group(1) or "").strip()
                hit = _repl.get(name.lower())
                if hit:
                    return hit
                return f"[视频 {name}（未预缓存，可用 analyze_video 分析）]"

            p.content = self._VIDEO_PLACEHOLDER_RE.sub(_sub, p.content)

    # ── Prompt ──

    @on.llm_request(priority=Priority.LOW)
    async def _inject(self, event, req: LLMRequest, *_):
        if not self.enabled: return
        sid = self._sid(event)
        if not sid: return
        has_pending = sid in self._pending
        cached_list = self._cached_videos.get(sid) or []
        # pending 有时效：超过 10 分钟没被用掉就不再提示（避免每轮都刷）
        if has_pending:
            if time.time() - self._pending[sid].get("ts", 0) > 600:
                self._pending.pop(sid, None)
                has_pending = False
        # ① 先把消息里框架写的「(Video size over 10MB, not cached)」改写成可用信息
        #    （每条 prompt 都是当轮新渲染的，所以每轮都要改；改写内容会随消息进历史）
        try:
            self._rewrite_video_notes(event, req, sid)
        except Exception as e:
            logger.debug("[VC] 改写视频占位失败: %s", e)

        if (not has_pending and not cached_list
                and not self._sid_sessions.get(sid)
                and not self._video_failures.get(sid)):
            return
        hint = ""
        if cached_list:
            # 已缓存 → 在 system prompt 里给「怎么用」的提示（路径已在消息里）
            first = cached_list[-1]
            hint = (f"\n【视频】已缓存的视频用 analyze_video(local_path=\"{first.get('rel','')}\") "
                    f"分析内容（画面 + 语音转写一起）")
        elif self.video_analysis_enabled and has_pending:
            url = self._pending[sid].get("url", "")
            if BILI_RE.search(url):
                hint = "\n【B站视频】analyze_video / send_video / search_bili_video"
            else:
                hint = "\n【视频】analyze_video 分析内容"
        if hint:
            for p in req.system_prompt:
                if p.name and "tool" in p.name.lower():
                    p.content += hint; break
            else:
                if req.system_prompt: req.system_prompt[-1].content += hint

    # ── 视频检测 ──

    @staticmethod
    def _is_video_ele(ele) -> bool:
        """判断是否为视频元素（真实类名是 Video；兼容子类与包装类）"""
        n = type(ele).__name__
        if n == "Video" or n.endswith("Video"):
            return True
        t = getattr(ele, "type", None)
        return str(getattr(t, "name", t) or "").lower() == "video" and hasattr(ele, "file")

    def _iter_videos(self, chain, depth: int = 0):
        """递归遍历消息链找 Video 元素。

        ⚠️ 只扫顶层是不够的，视频可能藏在：
          - `Reply.chain`（引用消息，单数 MessageChain）
          - `Forward.chains`（合并转发，复数 list[MessageChain]）
        框架对这类视频常因拿不到 file_size 而放弃缓存，所以这里必须递归接管。
        """
        if depth > 4:
            return
        for ele in chain or []:
            try:
                if self._is_video_ele(ele):
                    yield ele
                # 引用消息：单条内层链
                inner = getattr(ele, "chain", None)
                if inner:
                    yield from self._iter_videos(inner, depth + 1)
                # 合并转发：内层链列表
                for c in (getattr(ele, "chains", None) or []):
                    yield from self._iter_videos(c, depth + 1)
            except Exception:
                continue

    @staticmethod
    def _reply_message_ids(chain) -> list:
        """取出「内层链为空」的引用元素 ID。

        内层链有内容说明适配器已完整解析过，本地找过没有就是真没有；
        只有链为空（没解析出来）才值得按 ID 主动拉一次，避免白调接口。
        """
        ids = []
        for ele in chain or []:
            try:
                if type(ele).__name__ == "Reply":
                    if getattr(ele, "chain", None):
                        continue
                    mid = getattr(ele, "message_id", None)
                    if mid:
                        ids.append(str(mid))
            except Exception:
                continue
        return ids

    def _find_cached(self, safe_name: str) -> str:
        """在缓存目录里找同一个原始文件名已缓存过的文件（避免同一视频反复下载）"""
        try:
            suffix = "_" + safe_name
            for fn in os.listdir(self.other_cache_dir):
                if fn.endswith(suffix) or fn == safe_name:
                    p = os.path.join(self.other_cache_dir, fn)
                    if os.path.isfile(p) and os.path.getsize(p) > 0:
                        return p
        except Exception:
            pass
        return ""

    def _remember_cached(self, sid: str, safe_name: str, path: str):
        """登记到该会话的已缓存列表（列表结构，支持一条消息里多个视频）"""
        try:
            rel = os.path.relpath(path, get_data_path()).replace("\\", "/")
        except Exception:
            rel = path
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
        except Exception:
            size_mb = 0.0
        lst = self._cached_videos.setdefault(sid, [])
        for c in lst:
            if c.get("path") == path:
                c["ts"] = time.time()
                return
        lst.append({"orig_name": safe_name, "path": path, "rel": rel,
                    "size_mb": size_mb, "ts": time.time()})

    @staticmethod
    def _explain_failure(e: Exception) -> str:
        msg = str(e)
        if "超过上限" in msg or type(e).__name__ == "DownloadTooLarge":
            return "超过单文件大小上限"
        return f"下载失败（{type(e).__name__}）"

    async def _cache_incoming_video(self, sid: str, url: str, name: str = "") -> str:
        """把收到的视频下载到本地缓存目录，供 bot 直接使用（返回本地路径）

        框架在 file_size 缺失时不会缓存视频（且文案会误导成"超过 10MB"），
        这里自己下下来并记住路径，下一轮对话告诉 bot。
        """
        try:
            os.makedirs(self.other_cache_dir, exist_ok=True)
            safe = re.sub(r"[^\w.\-]+", "_", (name or "").strip()) or "video.mp4"
            if not re.search(r"\.(mp4|mov|mkv|webm|avi|flv|m4v|ts|wmv)$", safe, re.I):
                safe += ".mp4"
            # 已有同名缓存 → 直接复用（同一视频常被反复引用，不必重复下载/转写）
            existing = self._find_cached(safe)
            if existing:
                self._remember_cached(sid, safe, existing)
                logger.info("[VC] 视频已在缓存中，复用: %s", os.path.basename(existing))
                return existing
            path = os.path.join(self.other_cache_dir, f"v{int(time.time())}_{safe}")
            max_bytes = int(self.cache_max_file_mb * 1024 * 1024) if self.cache_max_file_mb > 0 else 0
            if not (os.path.exists(path) and os.path.getsize(path) > 0):
                try:
                    await download_video(url, path, timeout=self.dl_timeout, max_bytes=max_bytes)
                except Exception as de:
                    self._video_failures.setdefault(sid, {})[safe] = self._explain_failure(de)
                    raise
            if os.path.exists(path) and os.path.getsize(path) > 0:
                self._remember_cached(sid, safe, path)
                logger.info("[VC] 视频已缓存(%d): %s (%.2fMB)",
                            len(self._cached_videos.get(sid) or []), os.path.basename(path),
                            os.path.getsize(path) / (1024 * 1024))
                return path
            logger.warning("[VC] 视频缓存后文件为空: %s", url)
        except Exception as e:
            logger.warning("[VC] 视频缓存失败: %s", e)
        return ""

    def _any_group_needs_stt(self) -> bool:
        """是否还有模型组需要 ASR 转写（即：不是所有启用的组都自带音视频理解）"""
        if not self._profiles:
            return True
        return any(not p.native_audio for p in self._profiles)

    async def _cache_then_transcribe(self, sid: str, url: str, name: str = ""):
        """缓存视频后立刻并行启动转写（这样 bot 真要分析时通常已算好）

        若当前启用的模型组**全部**自带音视频理解（不需要 STT），就不再跑 ASR。
        """
        path = await self._cache_incoming_video(sid, url, name)
        if path:
            self._start_transcript_task(path, skip_asr=not self._any_group_needs_stt())

    # ── 语音转写（把"声音"变成模型读得到的文字） ──

    def _asr_ready(self) -> bool:
        return bool(self.audio_enabled and self.audio_base_url and self.audio_model)

    def _transcript_key(self, video_path: str) -> str:
        try:
            st = os.stat(video_path)
            raw = f"{os.path.abspath(video_path)}:{st.st_size}:{int(st.st_mtime)}"
        except Exception:
            raw = str(video_path)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _transcript_path(self, key: str) -> str:
        d = os.path.join(self.other_cache_dir, ".transcripts")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key}.json")

    def _load_transcript(self, key: str):
        try:
            p = self._transcript_path(key)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_transcript(self, key: str, data: dict):
        try:
            with open(self._transcript_path(key), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.debug("[VC] 转写缓存写入失败: %s", e)

    async def _segment_transcribe(self, wav: str, speech_ranges, work: str):
        """兜底：ASR 不返回时间戳时，用本地 VAD 切块逐块识别来造时间轴。

        只在「服务只给纯文本」时才会走到这里（例如硅基流动 SenseVoice）。
        """
        blocks = merge_ranges(speech_ranges, gap=self.audio_gap_sec,
                              max_len=self.audio_block_sec,
                              max_count=self.audio_max_blocks)
        if not blocks:
            return []
        sem = asyncio.Semaphore(self.audio_concurrency)

        async def _one(i: int, s: float, e: float):
            async with sem:
                try:
                    seg_path = os.path.join(work, f"blk_{i:03d}.wav")
                    await slice_audio(wav, seg_path, s, e)
                    r = await asr_transcribe(seg_path, self.audio_base_url,
                                             self.audio_api_key, self.audio_model,
                                             timeout=self.audio_timeout,
                                             language=self.audio_language,
                                             use_proxy=self.audio_use_proxy,
                                             extra_headers=self.audio_extra_headers,
                                             extra_body=self.audio_extra_body)
                    txt = (r.get("text") or "").strip()
                    return {"start": s, "end": e, "text": txt} if txt else None
                except Exception as ex:
                    logger.debug("[VC] 切块转写失败 [%.1f-%.1f]: %s", s, e, ex)
                    return None

        results = await asyncio.gather(*(_one(i, s, e) for i, (s, e) in enumerate(blocks)))
        logger.info("[VC] 切块转写：%d 块 → %d 段有文字（ASR 无原生时间戳）",
                    len(blocks), sum(1 for x in results if x))
        return [x for x in results if x]

    async def _build_transcript(self, video_path: str, bvid: str = "", cid: int = 0,
                                skip_asr: bool = False) -> dict:
        """完整转写流程。

        ① B 站视频优先用**官方字幕**（精确时间轴、免费、不用抽音轨）
        ② 没有字幕再走 ASR：抽音轨 → 静音分析 → 转写 → (必要时切块补轴)
        """
        want_sub = bool(self.bili_use_subtitle and bvid and cid)
        if not self.audio_enabled or (not want_sub and not self._asr_ready()):
            return {}
        key = self._transcript_key(video_path)
        cached = self._load_transcript(key)
        if cached:
            return cached
        t0 = time.time()

        # ① B 站官方字幕优先
        if want_sub:
            try:
                segs, lan_doc, n_tracks = await get_bilibili_subtitle(
                    bvid, cid, self.bili_cookie, prefer_lan=self.audio_language,
                    timeout=self.dl_timeout)
                if segs:
                    doc = build_timeline_doc(
                        segs, [], 0,
                        header=(f"【视频字幕（B站 {lan_doc or 'CC'}，共{n_tracks}条轨）"
                                f"· 时间轴与画面帧口径一致】"))
                    data = {"doc": doc, "segments": segs, "speech": [],
                            "total": round(float(segs[-1].get("end") or 0), 3),
                            "native_axis": True, "asr_source": "bilibili_subtitle",
                            "elapsed": round(time.time() - t0, 2)}
                    self._save_transcript(key, data)
                    logger.info("[VC] 用上 B 站官方字幕：%d 条（%s，用时 %.1fs）",
                                len(segs), lan_doc or "?", time.time() - t0)
                    return data
                logger.info("[VC] 该 B 站视频无可用字幕，转音频识别")
            except Exception as e:
                logger.info("[VC] B 站字幕获取失败，转音频识别: %s", e)
        # ② 音频识别（自带音视频理解的模型不需要）
        if skip_asr:
            logger.info("[VC] 该模型组自带音视频理解，跳过语音识别")
            return {}
        if not self._asr_ready():
            return {}
        work = os.path.join(self.other_cache_dir, f".asr_{key}")
        os.makedirs(work, exist_ok=True)
        try:
            wav = os.path.join(work, "audio.wav")
            await extract_audio(video_path, wav)
            speech, total = await detect_speech_ranges(wav, noise_db=self.audio_silence_db)
            if self.audio_max_sec > 0 and total > self.audio_max_sec:
                logger.info("[VC] 音频 %.0fs 超过转写上限 %.0fs，跳过", total, self.audio_max_sec)
                return {}
            result = await asr_transcribe(wav, self.audio_base_url, self.audio_api_key,
                                          self.audio_model, timeout=self.audio_timeout,
                                          language=self.audio_language,
                                          use_proxy=self.audio_use_proxy,
                                          extra_headers=self.audio_extra_headers,
                                          extra_body=self.audio_extra_body)
            segs = result.get("segments") or []
            native = bool(segs)
            if segs:
                # 兜底校正：有些服务给毫秒却用秒的字段名
                segs = normalize_segments_by_duration(segs, total)
            if not segs and (result.get("text") or "").strip() and speech:
                segs = await self._segment_transcribe(wav, speech, work)
            if result.get("silent") and not segs:
                logger.info("[VC] 音频识别完成：这段 %.1fs 音频里没有可转写的语音", total)
            # 即使没有语音，也把「有声但无人声」的片段标出来（L2）
            doc = build_timeline_doc(segs, speech, total)
            data = {
                "doc": doc, "segments": segs,
                "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
                "total": round(total, 3),
                "native_axis": native,
                "asr_source": result.get("source", ""),
                "elapsed": round(time.time() - t0, 2),
            }
            self._save_transcript(key, data)
            logger.info("[VC] 语音转写完成：%.1fs，%d 段%s（用时 %.1fs）",
                        total, len(segs), "，原生时间轴" if native else "，本地切块补轴",
                        time.time() - t0)
            return data
        except ASRError as e:
            logger.warning("[VC] 语音转写失败（不影响视频分析）: %s", e)
        except Exception as e:
            logger.warning("[VC] 语音转写异常（不影响视频分析）: %s", e)
        return {}

    def _start_transcript_task(self, video_path: str, bvid: str = "", cid: int = 0,
                               skip_asr: bool = False):
        """后台启动转写（与视频缓存并行，拿到就缓存好，分析时零等待）"""
        want_sub = bool(self.bili_use_subtitle and bvid and cid)
        if not self.audio_enabled or (not want_sub and not self._asr_ready()):
            return
        if skip_asr and not want_sub:
            return
        if not video_path or not os.path.isfile(video_path):
            return
        key = self._transcript_key(video_path)
        if key in self._asr_tasks and not self._asr_tasks[key].done():
            return
        if self._load_transcript(key):
            return
        task = asyncio.create_task(self._build_transcript(video_path, bvid=bvid, cid=cid,
                                                          skip_asr=skip_asr))
        self._asr_tasks[key] = task

        def _cleanup(_t, k=key):
            if self._asr_tasks.get(k) is _t:
                self._asr_tasks.pop(k, None)
        task.add_done_callback(_cleanup)

    async def _get_transcript(self, video_path: str, wait: float,
                              bvid: str = "", cid: int = 0,
                              skip_asr: bool = False) -> dict:
        """取转写结果：缓存命中→秒用；有进行中任务→最多等 wait 秒；否则现场跑"""
        want_sub = bool(self.bili_use_subtitle and bvid and cid)
        if not self.audio_enabled or (not want_sub and not self._asr_ready()):
            return {}
        if not video_path or not os.path.isfile(video_path):
            return {}
        key = self._transcript_key(video_path)
        cached = self._load_transcript(key)
        if cached:
            return cached
        task = self._asr_tasks.get(key)
        if task is None or task.done():
            self._start_transcript_task(video_path, bvid=bvid, cid=cid, skip_asr=skip_asr)
            task = self._asr_tasks.get(key)
        if task is None:
            return {}
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=max(0.1, wait))
        except asyncio.TimeoutError:
            logger.info("[VC] 语音转写未在 %.0fs 内完成，本次分析先不带转写", wait)
            return {}
        except Exception as e:
            logger.warning("[VC] 取转写失败: %s", e)
            return {}

    @on.im_message(priority=Priority.HIGH)
    async def _detect(self, event: KiraMessageEvent, *_):
        if not self.enabled or not self._ok(event):
            return
        sid = self._sid(event)
        url = None
        vname = ""
        chain_top = getattr(event.message, "chain", None)
        saw_video_ele = False      # 链里出现过 Video（即便 file 为空）
        has_video_seg = False      # raw_message 里有 video 段

        # 1) 消息链里的 Video 元素（含引用 chain / 合并转发 chains）
        try:
            for ele in self._iter_videos(chain_top):
                saw_video_ele = True
                f = getattr(ele, "file", "") or ""
                if f:
                    url = str(f)
                    vname = str(getattr(ele, "name", "") or "")
                    break
        except Exception:
            pass

        # 2) raw_message（OneBot 原始结构）里找 video 段
        raw = None
        if not url:
            for a in ("raw_message", "source_message", "original_message"):
                v = getattr(event.message, a, None) or getattr(event, a, None)
                if v:
                    if isinstance(v, str):
                        try: raw = json.loads(v)
                        except: continue
                    elif isinstance(v, (dict, list)): raw = v
                    if raw: break
            segs = None
            if isinstance(raw, dict):
                segs = raw.get("message")
            elif isinstance(raw, list):
                segs = raw
            if isinstance(segs, list):
                for s in segs:
                    if isinstance(s, dict) and s.get("type") == "video":
                        has_video_seg = True
                        url = (s.get("data") or {}).get("url") or ""
                        break

        # 3) 兜底：主动调 OneBot 接口拉（当前消息 → 引用消息）
        #    只在「确实有视频迹象」时才调，避免每条纯文字消息都白跑一次 API
        reply_ids = self._reply_message_ids(chain_top)
        # 主动调 get_msg 是 OneBot 的能力，其他平台只靠消息链/raw_message
        if not url and self._is_qq(event) and (saw_video_ele or has_video_seg or reply_ids):
            try:
                ad = self.ctx.adapter_mgr.get_adapter(event.adapter.name)
                cl = ad.get_client()

                async def _scan_get_msg(mid) -> str:
                    """取一条消息，返回其中的视频 URL"""
                    if not mid:
                        return ""
                    rm = await cl.send_action("get_msg", {"id": mid}, timeout=15)
                    if isinstance(rm, dict):
                        for s in (rm.get("message") or []):
                            if isinstance(s, dict) and s.get("type") == "video":
                                return (s.get("data") or {}).get("url", "") or ""
                    return ""

                # 3a) 当前消息
                mid = getattr(event.message, "message_id", None) or getattr(event, "message_id", None)
                try:
                    url = await _scan_get_msg(mid)
                except Exception:
                    url = ""
                # 3b) 引用消息（适配器若没解析出内层链，这里按被引用消息 ID 主动拉）
                if not url:
                    for rid in reply_ids:
                        try:
                            url = await _scan_get_msg(rid)
                        except Exception:
                            url = ""
                        if url:
                            logger.info("[VC] 通过引用消息 ID 主动拉到视频: %s", rid)
                            break
            except Exception:
                pass
        if url:
            self._pending[sid] = {"url": url, "source": "onebot", "ts": time.time()}
            # 缓存策略（cache_scope）：
            #   all       → 每条消息就缓存（旧行为，最耗）
            #   batch     → 等进入 bot 批次后再缓存
            #   mentioned → 只在被 @ / 引用 / 唤醒时才缓存（默认，最省）
            # 注意：URL 始终记录在 _pending，所以 bot 主动调工具时永远有源可用。
            if self.auto_cache_video:
                if str(url).startswith(("http://", "https://")):
                    if self.cache_scope == "all":
                        asyncio.create_task(self._cache_then_transcribe(sid, url, vname))
                elif os.path.isfile(url):
                    self._remember_cached(sid, os.path.basename(url), url)

    # ────────────── 工具1：search_bili_video ──────────────

    @register.tool(
        name="search_bili_video",
        description="搜索B站视频，返回结果列表（含标题/UP主/时长/播放量/简介）。用户要找B站视频时调用。",
        params={
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "搜索关键词"}},
            "required": ["keyword"],
        },
    )
    async def _tool_search(self, event, keyword: str) -> str:
        if not self.bili_enabled: return "B站功能未启用"
        try: rs = await search_bili(keyword, self.bili_search_n, self.bili_cookie)
        except Exception as e: return f"⚠️ 搜索失败：{e}"
        if not rs: return "未找到相关视频"
        lines = [f"🔍 搜索「{keyword}」结果："]
        for i, r in enumerate(rs, 1):
            d = r.get("duration", 0); desc = r.get("desc", "")
            lines.append(f"{i}. {r.get('title','')}\n   👤 {r.get('author','')} | ⏱ {d//60}:{d%60:02d} | 👁 {r.get('play',0)}\n   BV: {r.get('bvid','')}")
            if desc and self.search_show_desc: lines.append(f"   📝 {desc[:self.search_desc_max_chars]}")
        lines.append("\n→ send_video(bvid=...) 直接发送\n→ analyze_video(bvid=...) 分析")
        return "\n".join(lines)

    # ────────────── 工具2：send_video ──────────────

    @register.tool(
        name="send_video",
        description="下载B站视频并发送到QQ（可压缩），也支持本地视频路径发送。用户要求下载/发B站视频时调用。传关键词返回候选列表。",
        params={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "B站链接/BV号/搜索关键词"},
                "bvid": {"type": "string", "description": "已知BV号（优先）"},
                "local_path": {"type": "string", "description": "本地视频路径（绝对路径或相对 data/ 目录的相对路径）"},
                "quality": {"type": "string", "description": "质量: low|medium|original", "default": ""},
            },
        },
    )
    async def _tool_send_video(self, event, target: str = "", bvid: str = "", local_path: str = "", quality: str = "") -> str:
        if not self._is_qq(event): return "当前不是QQ"
        sid = self._sid(event)
        if local_path:
            # 本地文件（绝对路径直用，相对路径基于 get_data_path() 解析）
            lp = local_path.strip()
            if os.path.isabs(lp) and os.path.isfile(lp): pass
            else:
                # 相对路径 → 以 get_data_path() 为基准
                resolved = os.path.join(get_data_path(), lp)
                if os.path.isfile(resolved): lp = resolved
                else: return f"⚠️ 找不到文件: {local_path}（相对路径以 data/ 为基准）"
            q = quality or self.send_video_quality
            # 压缩
            os.makedirs(self.other_cache_dir, exist_ok=True)
            out_path = os.path.join(self.other_cache_dir, f"send_local_{Path(lp).stem}.mp4")
            try:
                if q == "low": await compress_video(lp, out_path, max_width=360, crf=32)
                elif q == "medium": await compress_video(lp, out_path, max_width=720, crf=28)
                elif q == "original": out_path = lp
                else: await compress_video(lp, out_path, max_width=720, crf=28)
            except Exception as e:
                logger.warning("[VC] 本地视频压缩失败，改发原文件: %s", e)
                out_path = lp
            try:
                ad = self.ctx.adapter_mgr.get_adapter(event.adapter.name)
                cl = ad.get_client()
                if "gm:" in sid:
                    await cl.send_action("send_group_msg", {"group_id": int(sid.split(":")[-1]), "message": [{"type": "video", "data": {"file": out_path}}]})
                elif "dm:" in sid:
                    await cl.send_action("send_private_msg", {"user_id": int(sid.split(":")[-1]), "message": [{"type": "video", "data": {"file": out_path}}]})
                else: return "无法判断群聊/私聊"
            except Exception as e: return f"⚠️ 发送失败：{e}"
            return f"✅ 已发送本地视频：{Path(lp).name} | 质量: {q}"
        if bvid:
            bv = bvid.strip()
            if not bv.startswith("BV"): bv = (await extract_bvid(bv)) or ""
            if bv: return await self._send_video_by_bvid(event, bv, quality)
        if target:
            bv = await extract_bvid(target, self.dl_timeout)
            if bv: return await self._send_video_by_bvid(event, bv, quality)
        keyword = (target or "").strip()
        if not keyword: return "请提供B站链接/BV号/搜索关键词"
        try: rs = await search_bili(keyword, self.bili_search_n, self.bili_cookie)
        except Exception as e: return f"⚠️ 搜索失败：{e}"
        if not rs: return f"未找到「{keyword}」相关视频"
        lines = [f"🔍 搜索「{keyword}」结果："]
        for i, r in enumerate(rs, 1):
            d = r.get("duration", 0); desc = r.get("desc", "")
            lines.append(f"{i}. {r.get('title','')}\n   👤 {r.get('author','')} | ⏱ {d//60}:{d%60:02d} | 👁 {r.get('play',0)}")
            if desc and self.search_show_desc: lines.append(f"   📝 {desc[:self.search_desc_max_chars]}")
            lines.append(f"   BV: {r.get('bvid','')}")
        lines.append("\n→ send_video(bvid=BVxxx)")
        return "\n".join(lines)

    async def _send_video_by_bvid(self, event, bvid: str, quality: str = "", sid: str = "", adapter_name: str = "") -> str:
        """发送B站视频到QQ（内置 NapCat 分块上传防断连）
        event 可为 None（auto_send 钩子 discard 后用 adapter_name 参数代替）
        """
        try: info = await get_bili_info(bvid, self.bili_cookie)
        except Exception as e: return f"⚠️ 获取信息失败：{e}"
        d = info.get("duration", 0); title = info.get("title", bvid)
        if self.bili_max_dl and d > self.bili_max_dl:
            return f"⏱ 「{title}」时长{d}s超上限，不下发"
        os.makedirs(self.bili_cache_dir, exist_ok=True)
        q = quality or self.bili_download_quality
        try:
            path, _ = await download_bili_video(bvid, self.bili_cache_dir, info=info,
                cookie=self.bili_cookie, timeout=self.dl_timeout, max_seconds=self.bili_max_dl,
                quality=q)
        except Exception as e: return f"⚠️ 下载失败：{e}"
        # 下载后是否再压缩：只在 compress_quality 比下载质量更低时才有意义
        _rank = {"low": 1, "medium": 2, "original": 3}
        cq = self.bili_compress_quality
        if cq and cq != "original" and _rank.get(cq, 3) < _rank.get(q, 3):
            try:
                compressed = os.path.join(self.bili_cache_dir, f"send_{bvid}_compressed.mp4")
                if cq == "low": await compress_video(path, compressed, max_width=360, crf=32)
                elif cq == "medium": await compress_video(path, compressed, max_width=720, crf=28)
                out_path = compressed
            except Exception as e:
                logger.warning("[VC] B站视频压缩失败，改发原文件: %s", e)
                out_path = path
        else:
            out_path = path

        send_sid = sid or (self._sid(event) if event else "")
        if not send_sid: return "⚠️ 无法获取会话ID"
        ad_name = adapter_name or (event.adapter.name if event and hasattr(event, 'adapter') else "")
        if not ad_name: return "⚠️ 无法获取 adapter"
        try:
            ad = self.ctx.adapter_mgr.get_adapter(ad_name)
            cl = ad.get_client()
            is_group = "gm:" in send_sid
            target_id = int(send_sid.split(":")[-1])

            # ── NapCat 分块上传（第三方扩展 action，**默认关闭**，配置可开） ──
            # 直接用本地路径发送是标准做法（AstrBot 等同类软件都这么做），
            # 对 NapCat / SnowLuma 都能工作。这里只在用户显式开启时才尝试。
            file_ref = out_path  # 兜底/默认：直接发本地路径
            try:
                file_size = Path(out_path).stat().st_size
            except Exception:
                file_size = 0
            if self.napcat_stream and file_size > 1024 * 1024 and not self._stream_unsupported:
                try:
                    filename = f"{bvid}.mp4"
                    chunk_size = 512 * 1024
                    total_chunks = max(1, math.ceil(file_size / chunk_size))
                    stream_id = uuid.uuid4().hex
                    digest = hashlib.sha256()
                    with open(out_path, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            digest.update(chunk)
                    sha256 = digest.hexdigest()
                    retention = 600000
                    with open(out_path, "rb") as f:
                        for ci in range(total_chunks):
                            chunk = f.read(chunk_size)
                            if not chunk: break
                            _raise_for_stream(await cl.send_action("upload_file_stream", {
                                "stream_id": stream_id, "chunk_index": ci,
                                "total_chunks": total_chunks, "file_size": file_size,
                                "filename": filename, "expected_sha256": sha256,
                                "file_retention": retention,
                                "chunk_data": base64.b64encode(chunk).decode("ascii"),
                            }, timeout=120))
                    resp = await cl.send_action("upload_file_stream", {
                        "stream_id": stream_id, "is_complete": True,
                        "total_chunks": total_chunks, "file_size": file_size,
                        "filename": filename, "expected_sha256": sha256,
                        "file_retention": retention,
                    }, timeout=120)
                    _raise_for_stream(resp)
                    napcat_path = _extract_stream_path(resp)
                    if napcat_path:
                        file_ref = napcat_path
                except Exception as e:
                    msg = str(e).lower()
                    if "不支持" in msg or "unsupported" in msg or "unknown action" in msg or "not found" in msg:
                        self._stream_unsupported = True  # 记住：本次运行不再尝试
                    logger.info("[VC] stream 上传不可用，降级直接发路径: %s", e)

            # 发送视频（file_ref 是 NapCat 引用路径或本地路径）
            if is_group:
                await cl.send_action("send_group_msg", {
                    "group_id": target_id,
                    "message": [{"type": "video", "data": {"file": file_ref, "name": f"{bvid}.mp4"}}],
                })
            else:
                await cl.send_action("send_private_msg", {
                    "user_id": target_id,
                    "message": [{"type": "video", "data": {"file": file_ref, "name": f"{bvid}.mp4"}}],
                })
        except Exception as e:
            return f"⚠️ 发送失败：{e}"
        return f"✅ 已发送：{title}\nBV: {bvid} | ⏱ {d//60}:{d%60:02d} | 质量: {q}\n📁 本地路径: {out_path}"
    # ────────────── 工具3：analyze_video（分析开关控制） ──────────────

    @register.tool(
        name="analyze_video",
        description=("分析视频内容。支持QQ视频/B站视频/本地路径。首次返回session_id，追问传回。"
                     "只看某一段时间就传 start_sec/end_sec（数字秒）；一次看多段传 segments=[[起,止],...]（最多5段、每段≤300秒）。"
                     "时间段分析会复用已下载的视频，不会重新下载。"
                     "用户指定了模型（如「用 Agnes 分析」）就传 model=那个名字；"
                     "**同一次对话里后续的每次调用（包括追问）都要继续带上同一个 model**，"
                     "否则会退回默认模型组、可能答非所问。"
                     "不确定有哪些可选就先不传，传错时返回值会列出全部模型组。"),
        params={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "留空=完整分析"},
                "bvid": {"type": "string", "description": "B站BV号"},
                "deep_analysis": {"type": "boolean", "description": "深度视觉分析", "default": False},
                "local_path": {"type": "string", "description": "本地视频路径"},
                "session_id": {"type": "string", "description": "追问用session_id"},
                "model": {"type": "string", "description": "指定用哪个模型组（填别名/模型名/组号，如 \"Agnes\"）；不填则按优先级自动选"},
                "start_sec": {"type": "number", "description": "只分析从第几秒开始（数字秒）", "default": 0},
                "end_sec": {"type": "number", "description": "分析到第几秒结束；传 0 = 到视频结尾", "default": 0},
                "segments": {
                    "type": "array",
                    "description": "多段分析：[[起秒,止秒],[起秒,止秒]]，最多5段、每段≤300秒",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
    )
    async def _tool_analyze(self, event, question: str = "", bvid: str = "",
                             deep_analysis: bool = False, local_path: str = "",
                             session_id: str = "",
                             start_sec: float = 0, end_sec: float = 0,
                             segments=None, model: str = "") -> str:
        if not self.video_analysis_enabled:
            return "⚠️ 视频分析功能已关闭，可在 WebUI 启用"

        profile_spec = None
        if model.strip():
            profile_spec = self._find_profile(model)
            if profile_spec is None:
                return (f"⚠️ 找不到名为「{model}」的模型组。"
                        f"当前可用：{self._list_profiles_text()}")
        segs, err = self._parse_segments(start_sec, end_sec, segments)
        if err: return f"⚠️ {err}"

        sid = self._sid(event)
        if not sid: return "无法获取会话ID"

        if session_id:
            sess = self._get_by_session_id(session_id)
            if not sess: return f"⚠️ session_id={session_id} 不存在"
            self._register_session(sess, sid)
            if segs: return await self._segment_analyze(sess, question, segs, profile_spec)
            if question: return await self._followup(sess, question, profile_spec)
            return (f"📌 session_id={session_id}\n🤖 {sess.analysis_model}\n"
                    f"{self._link_line(sess.host_url)}━━━\n{sess.analysis[:500]}\n━━━\n"
                    f"追问用 session_id=\"{session_id}\"")

        source_url = ""; source_type = ""

        if bvid:
            bvid = bvid.strip()
            if not bvid.startswith("BV"):
                bvid = (await extract_bvid(bvid)) or ""
            if not bvid:
                return ("⚠️ 没从 bvid 参数里认出有效的 BV 号。"
                        "请传 BV 号（如 BV1xx411c7mD）或完整链接；"
                        "要分析本地文件请用 local_path 参数。")
            source_type = "bilibili"; source_url = f"https://www.bilibili.com/video/{bvid}"
        elif local_path:
            lp = local_path.strip()
            if os.path.isabs(lp) and os.path.isfile(lp): source_url = lp
            else:
                resolved = os.path.join(get_data_path(), lp)
                if os.path.isfile(resolved): source_url = resolved
                else: return f"⚠️ 找不到文件: {local_path}（相对路径以 data/ 为基准）"
            source_type = "local"
        else:
            pend = self._pending.pop(sid, None)
            if pend:
                source_url = pend["url"]; source_type = "onebot"
                m = BVID_RE.search(source_url)
                if m: bvid = m.group(0); source_type = "bilibili"
            else:
                # 兜底：用该会话最近缓存的视频
                # （插件 reload 后 _pending 会被清空，但缓存文件还在）
                cached = self._cached_videos.get(sid) or []
                if cached:
                    latest = cached[-1]
                    if os.path.isfile(latest.get("path", "")):
                        source_url = latest["path"]; source_type = "local"
                if not source_url:
                    olds = self._list_sessions(sid)
                    if olds:
                        cur = olds[0]
                        if question: return await self._followup(cur, question, profile_spec)
                        return (f"🔁 已有{len(olds)}个历史，最新session_id={cur.session_id}\n"
                                f"🤖 {cur.analysis_model}\n{self._link_line(cur.host_url)}━━━\n"
                                f"{cur.analysis[:300]}\n━━━\n追问用 session_id=\"{cur.session_id}\"")
                    return "当前无视频"

        sess_id = hashlib.md5(source_url.encode()).hexdigest()[:12]
        if sess_id in self._sessions:
            cur = self._sessions[sess_id]
            self._register_session(cur, sid)
            if segs: return await self._segment_analyze(cur, question, segs, profile_spec)
            if question: return await self._followup(cur, question, profile_spec)
            if deep_analysis and cur.analysis_mode == "AI_summary" and not cur.grids_base64:
                return await self._deep(cur)
            return (f"🔁 已有分析\n📌 session_id={sess_id}\n🤖 {cur.analysis_model}\n"
                    f"{self._link_line(cur.host_url)}━━━\n{cur.analysis[:400]}\n━━━\n"
                    f"追问用 session_id=\"{sess_id}\"")

        if source_type == "bilibili" and self.bili_use_ai and not deep_analysis and not segs:
            try:
                info = await get_bili_info(bvid, self.bili_cookie)
                ai = await get_ai_summary(bvid, info["cid"], info.get("up_mid", 0), self.bili_cookie)
                if ai.get("has_summary"): return self._build_ai_result(sess_id, sid, source_url, info, ai)
            except Exception as e: logger.info("[VC] B站AI降级: %s", e)

        return await self._vision(sess_id, sid, source_type, source_url, bvid,
                                   question or "请完整分析这段视频", segments=segs,
                                   model_spec=profile_spec)

    # ── 模型组选择（支持按别名/模型名/组号指定） ──

    def _find_profile(self, spec):
        """按「别名 → 模型名 → 组号」找模型组；找不到返回 None。

        让 bot 能听懂「用 Agnes 抽帧分析这个」这类指令。
        spec 既可以是字符串（别名/模型名/组号），也可以已经是 ModelProfile 对象。
        """
        if spec is None:
            return None
        if isinstance(spec, ModelProfile):
            return spec
        s = str(spec).strip()
        if not s:
            return None
        if s.isdigit():
            for p in self._profiles:
                if str(p.group) == s:
                    return p
        low = s.lower()
        for p in self._profiles:                       # 别名精确
            if p.label and p.label.lower() == low:
                return p
        for p in self._profiles:                       # 模型名精确
            if p.name and p.name.lower() == low:
                return p
        for p in self._profiles:                       # 别名包含
            if p.label and low in p.label.lower():
                return p
        for p in self._profiles:                       # 模型名包含
            if p.name and low in p.name.lower():
                return p
        return None

    def _list_profiles_text(self) -> str:
        if not self._profiles:
            return "（没有启用任何模型组）"
        parts = []
        for p in sorted(self._profiles, key=lambda x: x.priority):
            tag = p.label or p.name or f"组{p.group}"
            parts.append(f"{tag}(组{p.group}/{p.mode})")
        return "、".join(parts)

    # ── 时间段参数解析 ──

    def _parse_segments(self, start_sec, end_sec, segments):
        """归一化时间段参数 → ([(s,e), ...] | None, 错误信息)

        segments 优先于 start_sec/end_sec；end_sec=0 表示到视频结尾。
        未指定时返回 (None, "")，由下游按全片处理。
        """
        raw = []
        if segments:
            items = segments
            if isinstance(segments, str):
                # 容错："10-30,100-130" / "10~30；100~130"
                items = []
                for part in re.split(r"[,;，；]", segments):
                    m = re.match(r"\s*([\d.]+)\s*[-~到至]\s*([\d.]+)\s*$", part)
                    if m: items.append((m.group(1), m.group(2)))
            if isinstance(items, (list, tuple)):
                for item in items:
                    try:
                        if isinstance(item, dict):
                            s = item.get("start", item.get("start_sec", item.get("from")))
                            e = item.get("end", item.get("end_sec", item.get("to")))
                        else:
                            s, e = item[0], item[1]
                        raw.append((float(s), float(e)))
                    except Exception:
                        continue
        elif start_sec or end_sec:
            try:
                raw.append((float(start_sec or 0), float(end_sec or 0)))
            except Exception:
                return None, "时间段参数格式不对（应为数字秒）"

        if not raw:
            return None, ""

        norm = []
        for s, e in raw:
            s = max(0.0, s)
            if e <= 0: e = 1e9        # 到结尾，稍后按视频时长截断
            if e <= s:
                return None, f"时间段非法：{s:.0f}~{e:.0f}（结束必须大于开始）"
            if e < 1e9 and (e - s) > MAX_SEGMENT_SEC:
                return None, (f"单段最长 {MAX_SEGMENT_SEC} 秒，请把 {s:.0f}~{e:.0f} "
                              f"拆成多段传 segments")
            norm.append((s, e))
        if len(norm) > MAX_SEGMENTS:
            return None, f"最多 {MAX_SEGMENTS} 段，当前 {len(norm)} 段"
        return norm, ""

    # ── B站AI总结 ──

    def _build_ai_result(self, sess_id, sid, url, info, ai):
        sess = VideoSession(sess_id, sid, "bilibili", url)
        sess.title = info.get("title", ""); sess.duration = info.get("duration", 0)
        sess.bili_ai_summary = ai; self._register_session(sess, sid)
        s = ai.get("summary", "")
        outline = "\n".join(
            f"  [{_ts(o.get('timestamp',0))}] {o.get('title','')}\n" +
            "\n".join(f"    → [{_ts(p.get('timestamp',0))}] {p.get('content','')}"
                      for p in o.get("part_outline", []))
            for o in ai.get("outline", []))
        result = (f"🎬 {info.get('title','')}\n📌 session_id={sess_id}\n━━━\n"
                  f"⏱ {info.get('duration',0)}s | 🤖 B站AI总结\n━━━\n{s}\n")
        if outline: result += f"\n📑 大纲\n{outline}\n"
        result += f"\n━━━\n💡 追问用 session_id=\"{sess_id}\""
        sess.analysis = result; sess.analysis_model = "B站AI总结"; sess.analysis_mode = "AI_summary"
        return result

    # ── 视觉分析（非B站视频存到 other_cache_dir） ──

    async def _vision(self, sess_id, sid, stype, surl, bvid, question,
                      segments=None, model_spec=None):
        if not self._profiles: return "未配置模型"

        # 选模型用的时长：指定片段时用片段总长，否则用视频真实时长
        duration_hint = self.max_duration
        if segments:
            duration_hint = sum(e - s for s, e in segments)
        elif stype == "bilibili" and bvid:
            try:
                _info = await get_bili_info(bvid, self.bili_cookie)
                duration_hint = float(_info.get("duration") or 0) or self.max_duration
            except Exception:
                pass
        elif stype == "local" and os.path.isfile(surl):
            try:
                from video_processor import _get_video_info
                duration_hint = float(_get_video_info(surl).get("duration") or 0) or self.max_duration
            except Exception:
                pass
        profile = model_spec or select_model(self._profiles, duration_hint, self.default_model)
        if not profile: return "无合适模型"

        info = {}          # B 站分支会填上 cid 等（供字幕获取用）
        if stype == "bilibili" and bvid:
            # B站: 源文件存在 bili_cache_dir
            os.makedirs(self.bili_cache_dir, exist_ok=True)
            try:
                raw_path, info = await download_bili_video(bvid, self.bili_cache_dir, cookie=self.bili_cookie,
                    timeout=self.dl_timeout, max_seconds=self.max_duration,
                    quality=self.bili_download_quality)
            except Exception as e: return f"⚠️ 下载失败：{e}"
            # 分析工作也放 bili_cache_dir
            work = os.path.join(self.bili_cache_dir, f"analysis_{sess_id}")
        else:
            # 非B站（QQ/本地）：存到 other_cache_dir
            os.makedirs(self.other_cache_dir, exist_ok=True)
            raw_path = surl
            work = os.path.join(self.other_cache_dir, f"analysis_{sess_id}")
            if stype == "local":
                raw_path = surl

        os.makedirs(work, exist_ok=True)
        # native + 上传模式：小文件不预压缩（保画质）；阈值 0 = 永不压缩（用极大值让所有文件都跳过）
        native_upload = bool(self.upload_enabled and profile.mode == "native")
        under_mb = 0
        if native_upload:
            under_mb = self.upload_compress_over_mb if self.upload_compress_over_mb > 0 else 10 ** 6
        result = await process_video(raw_path, work_dir=work,
            max_file_mb=self.max_file_mb, max_duration_sec=self.max_duration,
            download_timeout=self.dl_timeout,
            target_frames=self.target_frames, scene_threshold=self.scene_threshold,
            max_per_grid=self.max_per_grid, grid_cols=self.grid_cols,
            cell_width=self.cell_width, cell_ratio=self.cell_ratio,
            segments=[[s, e] for s, e in segments] if segments else None,
            skip_compress=bool(segments),
            skip_compress_if_under_mb=under_mb)
        if result["status"] == "rejected": return f"⚠️ {result['error']}"
        if result["status"] == "error": return f"⚠️ 处理失败：{result['error']}"

        sess = VideoSession(sess_id, sid, stype, surl) if sess_id not in self._sessions else self._sessions[sess_id]
        sess.compressed_path = result.get("compressed_path", "")
        sess.duration = result.get("duration", 0)
        sess.grids_base64 = result.get("grids_base64", [])
        sess.scene_count = result.get("scene_count", 0); sess.total_frames = result.get("total_frames", 0)
        sess.timestamps = result.get("timestamps", [])
        sess.file_size_mb = result.get("file_size_mb", 0); sess.compressed_size_mb = result.get("compressed_size_mb", 0)
        self._register_session(sess, sid)

        real_segs = [(s, e) for s, e in (result.get("segments") or [])]
        # 取转写/字幕（最多等 audio_wait_sec 秒；超时就不带，绝不卡住分析）
        # B 站视频优先用官方字幕，其他视频走音频识别
        tr = await self._get_transcript(raw_path, self.audio_wait_sec,
                                       bvid=bvid if stype == "bilibili" else "",
                                       cid=int((info or {}).get("cid") or 0)
                                       if stype == "bilibili" else 0,
                                       skip_asr=profile.native_audio)
        tdoc = tr.get("doc", "") or ""
        analysis, label, downgrade_note, host_url = await self._analyze_result(
            profile, result, real_segs, question, work, transcript_doc=tdoc,
            bili_bvid=bvid if stype == "bilibili" else "",
            bili_cid=int((info or {}).get("cid") or 0) if stype == "bilibili" else 0)
        sess.analysis = analysis; sess.analysis_model = label; sess.analysis_mode = profile.mode
        sess.model_tag = profile.label or str(profile.group)   # 记住，供追问沿用
        if host_url:
            sess.host_url = host_url
        if tdoc:
            sess.transcript_doc = tdoc

        range_line = self._range_line(real_segs, result.get("duration", 0))
        link_line = self._link_line(host_url)
        return (f"🎬 视频分析完成\n📌 session_id={sess_id}\n━━━\n"
                f"{range_line}"
                f"⏱ {result['duration']:.1f}s | 📐 {result.get('width','?')}×{result.get('height','?')}\n"
                f"📦 {result['file_size_mb']:.1f}MB→{result['compressed_size_mb']:.1f}MB\n"
                f"🖼 {result['total_frames']}帧/{result['grid_count']}张/{result['scene_count']}场景\n"
                f"🤖 {label}\n{link_line}━━━\n{analysis}{downgrade_note}\n━━━\n"
                f"💡 追问用 session_id=\"{sess_id}\"")

    @staticmethod
    def _direct_ttl_text(url: str) -> str:
        """从 B 站直链里解析 deadline 签名，算出剩余有效期"""
        m = re.search(r"[?&]deadline=(\d+)", url or "")
        if not m:
            return "带签名"
        try:
            remain = int(m.group(1)) - int(time.time())
        except Exception:
            return "带签名"
        if remain <= 0:
            return "⚠️ 可能已失效"
        if remain >= 3600:
            return f"约 {remain // 3600} 小时后失效"
        if remain >= 60:
            return f"约 {remain // 60} 分钟后失效"
        return f"约 {remain} 秒后失效"

    def _link_line(self, host_url: str) -> str:
        """给 bot 的链接提示。

        - 上传到文件中转的链接：可分享、可长期引用
        - B 站 temporary 直链：**带签名的时效链接**，只能现拉现用；
          访问多少次都无法延长，想要长期分享得改用 send_video（发视频到QQ）。
        """
        if not host_url:
            return ""
        if "bilivideo" in host_url or "deadline=" in host_url:
            return (f"🔗 视频直链(B站临时): {host_url}\n"
                    f"   ⚠️ 带签名，{self._direct_ttl_text(host_url)}；"
                    f"只能现取现用（反复访问**不会**延长有效期）。"
                    f"需要长期分享请改用 send_video 把视频发到QQ，"
                    f"或重新调用 analyze_video 取一条新链\n")
        return f"🔗 视频直链: {host_url}（临时公开链接，可直接分享或后续引用）\n"

    # ── 时间段分析（复用已下载的视频） ──

    def _clip_transcript(self, tr: dict, segs) -> str:
        """只保留指定时间段的转写（时间段分析用，避免整片字幕干扰）"""
        if not tr:
            return ""
        doc = tr.get("doc", "") or ""
        if not doc or not segs:
            return doc
        s0 = min(float(s) for s, e in segs)
        e0 = max(float(e) for s, e in segs)
        subs = [g for g in (tr.get("segments") or [])
                if float(g.get("end") or 0) > s0 and float(g.get("start") or 0) < e0]
        speech = [r for r in (tr.get("speech") or [])
                  if len(r) >= 2 and float(r[1]) > s0 and float(r[0]) < e0]
        if not subs and not speech:
            return ""
        return build_timeline_doc(subs, speech, float(tr.get("total") or 0))

    def _range_line(self, segs, duration: float) -> str:
        """结果头部的分析范围标注"""
        if not segs or (len(segs) == 1 and segs[0][0] <= 0.05 and segs[0][1] >= duration - 0.5):
            return ""
        if len(segs) == 1:
            return f"🔍 分析范围: {_ts(segs[0][0])} - {_ts(segs[0][1])}\n"
        return ("🔍 分析范围: " + str(len(segs)) + " 段 " +
                " ".join(f"[{_ts(s)}-{_ts(e)}]" for s, e in segs) + "\n")

    async def _analyze_result(self, profile, result, segs, question, work,
                              transcript_doc: str = "", bili_bvid: str = "",
                              bili_cid: int = 0):
        """按 模式 + 段数 选择分析路径，返回 (analysis, label, note, host_url)。

        - native + 单段 → 秒切该段片段（含音频）传给模型
        - 其他（frames / native 多段）→ 拼图帧模式
        - native 失败自动降级 frames
        - transcript_doc：语音转写时间轴，两种模式都会带上
        """
        meta = build_meta(result["duration"], result["total_frames"],
                          result["grid_count"], result["scene_count"],
                          result.get("timestamps", []))
        # 转写拼在提示词之后、问题之前：模型既看到画面，也知道"说了什么、什么时候说的"
        ask_prompt = self.default_prompt
        if transcript_doc:
            ask_prompt = f"{self.default_prompt}\n\n{transcript_doc}"
        multi = len(segs) > 1
        src_for_native = result.get("compressed_path") or ""
        # 全片（未指定时间段）还是指定区间？全片直接用压缩后的整段视频，不做秒切
        is_full = (not segs) or (len(segs) == 1
                                 and segs[0][0] <= 0.5
                                 and segs[0][1] >= result.get("duration", 0) - 0.5)

        async def _native_payload():
            """返回 (视频路径, 附加说明)。全片=压缩后的完整视频；单段=秒切片段。"""
            if is_full:
                return src_for_native, ""
            s, e = segs[0]
            clip_path = os.path.join(work, "clip.mp4")
            clip_path, real_dur, actual_start = await clip_video(src_for_native, clip_path, s, e)
            clip_note = (f"\n（秒切对齐关键帧，实际片段约 {_ts(actual_start)} - "
                         f"{_ts(actual_start + real_dur)}）") if abs(actual_start - s) > 0.3 else ""
            return clip_path, clip_note

        async def _upload_or_none(path: str):
            """尝试把视频换成公开链接；未启用/超限/失败都返回 None（回退 base64）"""
            if not self.upload_enabled:
                return None
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > self.upload_max_mb:
                    logger.info("[VC] 文件 %.1fMB 超过上传上限 %dMB，改用 base64",
                                size_mb, self.upload_max_mb)
                    return None
                key = f"{path}:{os.path.getmtime(path):.0f}"
                if key in self._upload_cache:
                    return self._upload_cache[key]
                t0 = time.time()
                url, used_host = await upload_to_any(path, self.upload_hosts,
                                                     timeout=self.upload_timeout,
                                                     keep_name=self.upload_keep_name,
                                                     use_proxy=self.upload_use_proxy)
                self._upload_cache[key] = url
                logger.info("[VC] 视频已上传（%.1fMB, %.1fs, %s）→ %s",
                            size_mb, time.time() - t0, used_host, url)
                return url
            except UploadError as e:
                logger.warning("[VC] 上传失败，回退 base64: %s", e)
                return None
            except Exception as e:
                logger.warning("[VC] 上传异常，回退 base64: %s", e)
                return None

        async def _native_call():
            vpath, clip_note = await _native_payload()
            size_mb = os.path.getsize(vpath) / (1024 * 1024)
            # 超过阈值 → 上传前先压（体积可控）；≤阈值 → 直接传，保住画质
            if self.upload_enabled and self.upload_compress_over_mb > 0 \
                    and size_mb > self.upload_compress_over_mb:
                try:
                    small = os.path.join(work, "upload_small.mp4")
                    t0 = time.time()
                    await compress_video(vpath, small, max_width=720, crf=28)
                    new_mb = os.path.getsize(small) / (1024 * 1024)
                    logger.info("[VC] 上传前压缩 %.1fMB → %.1fMB（%.1fs）",
                                size_mb, new_mb, time.time() - t0)
                    vpath, size_mb = small, new_mb
                except Exception as e:
                    logger.warning("[VC] 上传前压缩失败，按原文件上传: %s", e)
            url = ""
            # ★ B 站视频 + 开启直传 → 直接用 html5 MP4 直链交给模型，省一次上传
            #   仅限全片：时间段分析用的是裁剪片段，没有对应直链
            if self.bili_direct_url and bili_bvid and is_full:
                try:
                    direct, _q, _ms = await get_bili_direct_url(
                        bili_bvid, bili_cid, self.bili_cookie, self.dl_timeout)
                    url = direct
                    logger.info("[VC] B站视频走直链交给模型（免上传）")
                except Exception as e:
                    logger.info("[VC] B站直链不可用，回退上传: %s", e)
            if not url:
                url = await _upload_or_none(vpath)
            if not url and size_mb > NATIVE_MAX_MB:
                # 没上传成功且超过 base64 上限 → 压到能内联
                try:
                    small = os.path.join(work, "native_small.mp4")
                    await compress_video(vpath, small, max_width=720, crf=30)
                    vpath = small
                except Exception as e:
                    logger.warning("[VC] base64 回退压缩失败: %s", e)
            ans = await analyze_native(profile, vpath, question, ask_prompt,
                                       video_url=url)
            is_direct = bool(url) and ("bilivideo" in url or "deadline=" in url)
            tag = "native B站直链" if is_direct else ("native URL" if url else "native base64")
            if not is_full:
                tag += " 片段"
            return ans, clip_note, tag, url

        # native（全片或单段都走；多段走帧模式）
        if profile.mode == "native" and not multi:
            try:
                analysis, clip_note, tag, url = await _native_call()
                return analysis, f"{profile.name} ({tag})", clip_note, url
            except Exception as e:
                logger.warning("[VC] native 模式失败，降级帧模式: %s", e)
                if not result.get("grids_base64"):
                    return f"⚠️ AI分析失败（{type(e).__name__}）", f"{profile.name} (native)", "", None
                try:
                    analysis = await analyze_frames(profile, result["grids_base64"], meta,
                                                    question, ask_prompt)
                    return (analysis, f"{profile.name} (native→frames)",
                            f"\n（原生视频模式失败，已降级为帧模式：{str(e)[:120]}）", None)
                except Exception as e2:
                    logger.error("[VC] LLM失败: %s", e2)
                    return (f"⚠️ AI分析失败（native: {type(e).__name__} / frames: {type(e2).__name__}）",
                            f"{profile.name} (native)", "", None)

        # frames 路径（含 native 多段：一次请求覆盖所有段）
        try:
            analysis = await analyze_frames(profile, result["grids_base64"], meta,
                                            question, ask_prompt)
            label = f"{profile.name} (frames" + (" 多段)" if multi else ")")
            note = "\n（多段分析走帧模式：一次请求覆盖所有段）" if (multi and profile.mode == "native") else ""
            return analysis, label, note, None
        except Exception as e:
            logger.error("[VC] LLM失败: %s", e)
            return f"⚠️ AI分析失败（{type(e).__name__}）", f"{profile.name} (frames)", "", None

    async def _segment_analyze(self, sess, question, segs, model_spec=None):
        """对已有 session 的视频做指定时间段分析（用本地文件，不重新下载）"""
        path = sess.compressed_path
        if not path or not os.path.isfile(path):
            return ("⚠️ 本地视频文件已失效（可能被缓存清理），请重新发送或重新分析该视频")
        if not self._profiles:
            return "未配置模型"

        # 按视频实际时长截断
        real = []
        for s, e in segs:
            e2 = min(e, sess.duration) if sess.duration else e
            if e2 - s >= 0.05:
                real.append((s, e2))
        if not real:
            return (f"⚠️ 时间段超出视频时长（视频共 {sess.duration:.1f}s）")
        total = sum(e - s for s, e in real)
        spec = model_spec or sess.model_tag               # 同样沿用会话的模型
        profile = (self._find_profile(spec) if spec else None) \
            or select_model(self._profiles, total, self.default_model) \
            or self._profiles[0]
        if model_spec:
            sess.model_tag = profile.label or str(profile.group)

        work = os.path.join(os.path.dirname(path), f"seg_{int(time.time()*1000) % 10**9}")
        os.makedirs(work, exist_ok=True)
        result = await process_video(path, work_dir=work,
            max_file_mb=max(self.max_file_mb, 4096),
            max_duration_sec=max(self.max_duration, int(sess.duration) + 1),
            download_timeout=self.dl_timeout,
            target_frames=self.target_frames, scene_threshold=self.scene_threshold,
            max_per_grid=self.max_per_grid, grid_cols=self.grid_cols,
            cell_width=self.cell_width, cell_ratio=self.cell_ratio,
            segments=[[s, e] for s, e in real], skip_compress=True)
        if result["status"] != "ok":
            return f"⚠️ 处理失败：{result.get('error', result['status'])}"

        # 转写：优先用缓存/进行中的任务，超时就不带（不卡住）
        tr = await self._get_transcript(path, self.audio_wait_sec,
                                        skip_asr=profile.native_audio)
        analysis, label, note, host_url = await self._analyze_result(
            profile, result, real, question, work,
            transcript_doc=self._clip_transcript(tr, real))
        sess.add_turn(question or "(时间段分析)", analysis)
        if host_url:
            sess.host_url = host_url
        range_line = self._range_line(real, sess.duration)
        link_line = self._link_line(host_url)
        return (f"🎬 时间段分析完成\n📌 session_id={sess.session_id}\n━━━\n"
                f"{range_line}"
                f"🖼 {result['total_frames']}帧/{result['grid_count']}张\n"
                f"🤖 {label}\n{link_line}━━━\n{analysis}{note}\n━━━\n"
                f"💡 继续追问用 session_id=\"{sess.session_id}\"")

    async def _deep(self, sess):
        bvid = BVID_RE.search(sess.source_url)
        return await self._vision(sess.session_id, sess.sid, "bilibili", sess.source_url,
                                   bvid.group(0) if bvid else "",
                                   "对B站AI总结做补充，深入分析画面")

    async def _followup(self, sess, question, model_spec=None):
        if not question:
            return (f"当前 session={sess.session_id}\n{self._link_line(sess.host_url)}"
                    f"{sess.analysis[:300]}\n追问用 session_id=\"{sess.session_id}\"")
        if not sess.grids_base64:
            return (f"只有{sess.analysis_mode}结果，深度分析后可追问画面。\n"
                    f"{self._link_line(sess.host_url)}已有: {sess.analysis[:200]}")
        # 模型粘性：追问没指定 model 时，沿用该会话首次分析用的那一组，
        # 而不是退回默认优先级（否则「同一个视频前后换了模型」）
        spec = model_spec or sess.model_tag
        profile = (self._find_profile(spec) if spec else None) \
            or select_model(self._profiles, sess.duration, self.default_model) \
            or (self._profiles[0] if self._profiles else None)
        if not profile: return "无可用模型"
        if model_spec:                                   # 本次显式换了模型 → 更新会话
            sess.model_tag = profile.label or str(profile.group)
        meta = build_meta(sess.duration, sess.total_frames, len(sess.grids_base64),
                          sess.scene_count, sess.timestamps)
        ctx = f"之前: {sess.analysis[:500]}\n\n追问: {question}\n\n基于帧回答指出时间。"
        try:
            ans = await analyze_frames(profile, sess.grids_base64, meta, ctx,
                                       sess.transcript_doc or "")
        except Exception as e: ans = f"⚠️ 追问失败: {type(e).__name__}: {e}"
        sess.add_turn(question, ans)
        return (f"🤖 {profile.name} | session={sess.session_id}\n"
                f"{self._link_line(sess.host_url)}━━━\n{ans}")

    def reload_cfg(self, cfg: dict):
        """热重载配置：只重读配置项，保留会话、缓存任务与已探测状态"""
        try:
            self._load_cfg(cfg)
        except Exception as e:
            logger.error("[VC] 配置热重载失败: %s", e)


def _ts(s: float) -> str:
    return f"{int(s//60):02d}:{s - int(s//60)*60:06.3f}"


# ── NapCat stream 工具函数（模块级，供类内方法调用） ──────────

def _raise_for_stream(resp):
    if not isinstance(resp, dict):
        raise RuntimeError(f"stream response: {resp!r}")
    if resp.get("status") == "ok":
        return
    msg = (str(resp.get("data", {})) if isinstance(resp.get("data"), dict) else str(resp))[:200]
    if "unsupported" in msg.lower() or "not found" in msg.lower() or "unknown action" in msg.lower():
        raise RuntimeError(f"NapCat 不支持 upload_file_stream: {msg}")
    raise RuntimeError(f"stream 上传失败: {msg}")


def _extract_stream_path(resp) -> Optional[str]:
    data = resp.get("data")
    if isinstance(data, dict):
        for k in ("file_path", "path", "file"):
            v = data.get(k)
            if v: return str(v)
    for k in ("file_path", "path", "file"):
        v = resp.get(k)
        if v: return str(v)
    return None