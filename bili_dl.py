"""
Bilibili API 封装 — 搜索/信息/AI总结/视频下载
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

import httpx

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
REFERER = "https://www.bilibili.com"
HEADERS = {"User-Agent": UA, "Referer": REFERER}
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
SHORT_RE = re.compile(r"(?:https?://)?b23\.tv/[0-9A-Za-z]+")
API_BASE = "https://api.bilibili.com"


class BiliError(Exception): pass
class BiliNetError(BiliError): pass


def _client(cookie="", timeout=60.0, use_proxy=False):
    h = dict(HEADERS)
    if cookie: h["Cookie"] = cookie
    return httpx.AsyncClient(headers=h, follow_redirects=True, timeout=timeout, trust_env=use_proxy)


async def _get_json(client, url, params=None):
    try:
        resp = await client.get(url, params=params)
    except httpx.HTTPError as e:
        raise BiliNetError(f"网络请求失败: {e}") from e
    try: return resp.json()
    except ValueError:
        raise BiliError(f"B站接口返回异常 (HTTP {resp.status_code})") from None


class _Requester:
    def __init__(self, cookie="", timeout=60.0, proxy_mode="auto"):
        self.cookie = cookie; self.timeout = timeout; self.proxy_mode = proxy_mode
    async def _get_once(self, url, params, use_proxy):
        async with _client(self.cookie, self.timeout, use_proxy) as c:
            return await _get_json(c, url, params)
    async def get_json(self, url, params=None):
        try: return await self._get_once(url, params, False)
        except BiliNetError:
            if self.proxy_mode != "auto": raise
            return await self._get_once(url, params, True)
    async def stream(self, url, fpath):
        for use_proxy in (False, True):
            try:
                async with _client(self.cookie, self.timeout, use_proxy) as c:
                    async with c.stream("GET", url) as resp:
                        if resp.status_code != 200: raise BiliError(f"下载失败 HTTP {resp.status_code}")
                        with open(fpath, "wb") as f:
                            async for chunk in resp.aiter_bytes(8192): f.write(chunk)
                return
            except (BiliNetError, httpx.HTTPError):
                if self.proxy_mode != "auto" or use_proxy: raise


# WBI 签名固定重排表（B站前端 mixinKeyEncTab）
_MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52,
]


def _mixin_key(orig: str) -> str:
    return "".join(orig[i] for i in _MIXIN_KEY_ENC_TAB)[:32]


async def _wbi_sign(req, params):
    """B站 WBI 签名。

    算法：img_key + sub_key → 按 mixinKeyEncTab 重排取前 32 位 = mixin_key；
    参数转字符串 + 追加 wts → 按 key 排序 urlencode → 去掉 !'()* 字符 →
    md5(query + mixin_key) = w_rid。

    旧实现用 sorted([img, sub]) 拼串、不做 urlencode，签名恒不通过（接口返回 -403）。
    """
    nav = await req.get_json(f"{API_BASE}/x/web-interface/nav")
    wbi = nav.get("data", {}).get("wbi_img", {}) or {}
    img_key = (wbi.get("img_url") or "").rsplit("/", 1)[-1].split(".")[0]
    sub_key = (wbi.get("sub_url") or "").rsplit("/", 1)[-1].split(".")[0]
    if not img_key or not sub_key: raise BiliError("WBI签名初始化失败")
    mixin_key = _mixin_key(img_key + sub_key)
    p = {k: str(v) for k, v in params.items()}
    p["wts"] = int(time.time())
    query = urllib.parse.urlencode(sorted(p.items()))
    query = re.sub(r"[!'()*]", "", query)
    p["w_rid"] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return p


# ══════════════════════════════════════════════════════
#  搜索 — duration 可能是 "4:24" 字符串
# ══════════════════════════════════════════════════════

def _parse_dur(dur) -> int:
    if isinstance(dur, (int, float)): return int(dur)
    if isinstance(dur, str):
        dur = dur.strip()
        if ":" in dur:
            parts = dur.split(":")
            if len(parts) == 2:
                try: return int(parts[0])*60 + int(parts[1])
                except: pass
        try: return int(dur)
        except: pass
    return 0

def _parse_num(val) -> int:
    if isinstance(val, (int, float)): return int(val)
    if isinstance(val, str):
        try: return int(val.replace(",", "").strip())
        except: return 0
    return 0


async def search_bili(keyword, count=5, cookie="", proxy_mode="auto"):
    req = _Requester(cookie, proxy_mode=proxy_mode)
    d = await req.get_json(f"{API_BASE}/x/web-interface/search/all/v2", {"keyword": keyword, "page": 1})
    if d.get("code") != 0: raise BiliError(f"搜索失败 (code={d.get('code')})")
    results = []
    for item in d.get("data", {}).get("result", []):
        if item.get("result_type") != "video": continue
        for v in item.get("data", [])[:count]:
            results.append({
                "bvid": v.get("bvid", ""),
                "title": re.sub(r"<[^>]+>", "", v.get("title", "")),
                "author": v.get("author", ""),
                "duration": _parse_dur(v.get("duration", 0)),
                "play": _parse_num(v.get("play", 0)),
                "desc": re.sub(r"<[^>]+>", "", v.get("description", "") or ""),
            })
        break
    return results[:count]


# ══════════════════════════════════════════════════════
#  视频信息
# ══════════════════════════════════════════════════════

async def get_bili_info(bvid, cookie="", timeout=60.0, proxy_mode="auto"):
    req = _Requester(cookie, timeout, proxy_mode)
    d = await req.get_json(f"{API_BASE}/x/web-interface/view", {"bvid": bvid})
    if d.get("code") != 0: raise BiliError(f"获取信息失败 (code={d.get('code')})")
    data = d["data"]
    return {
        "bvid": bvid, "cid": data.get("cid", 0),
        "title": data.get("title", bvid),
        "duration": int(data.get("duration", 0)),
        "desc": data.get("desc", ""),
        "up_mid": data.get("owner", {}).get("mid", 0),
        "pic": data.get("pic", ""),
    }


# ══════════════════════════════════════════════════════
#  AI 总结
# ══════════════════════════════════════════════════════

async def get_ai_summary(bvid, cid, up_mid=0, cookie="", proxy_mode="auto"):
    req = _Requester(cookie, proxy_mode=proxy_mode)
    params = {"bvid": bvid, "cid": cid}
    if up_mid: params["up_mid"] = up_mid
    signed = await _wbi_sign(req, params)
    d = await req.get_json(f"{API_BASE}/x/web-interface/view/conclusion/get", signed)
    if d.get("code") != 0: raise BiliError(f"AI总结接口失败 (code={d.get('code')})")
    data = d.get("data", {}); code = data.get("code", -1)
    if code == -1: return {"has_summary": False, "reason": "不支持AI摘要"}
    if code == 1: return {"has_summary": False, "reason": "无AI摘要（正在生成或无语音）"}
    model = data.get("model_result", {})
    if model.get("result_type", 0) == 0: return {"has_summary": False, "reason": "AI摘要为空"}
    return {
        "has_summary": True,
        "summary": model.get("summary", ""),
        "outline": model.get("outline", []),
        "subtitle": model.get("subtitle", []),
    }


# ══════════════════════════════════════════════════════
#  下载视频（音视频合并）
# ══════════════════════════════════════════════════════

async def _get_video_urls(req, bvid, cid, quality_hint: int = 0):
    """获取视频流直链，quality_hint 按高度过滤：0=最高, 360=低, 720=中, 1080=高"""
    params = {"cid": cid, "bvid": bvid, "fnval": 4048, "fnver": 0, "fourk": 1, "platform": "web"}
    d = await req.get_json(f"{API_BASE}/x/player/playurl", params)
    if d.get("code") != 0:
        signed = await _wbi_sign(req, params)
        d = await req.get_json(f"{API_BASE}/x/player/wbi/playurl", signed)
    if d.get("code") != 0: raise BiliError(f"获取视频流失败 (code={d.get('code')})")
    data = d.get("data", {})
    dash = data.get("dash", {})
    video = dash.get("video", [])
    audio = dash.get("audio", [])
    audio_url = audio[0]["baseUrl"] if audio else None

    # 按质量选择视频流（DASH 默认从高到低排）
    video_url = None
    if quality_hint > 0:
        # 取第一个不超过 quality_hint 高度的流（越小越省流量）
        for v in video:
            h = int(v.get("height", 0) or 0)
            if 0 < h <= quality_hint:
                video_url = v["baseUrl"]
                break
    if not video_url and video:
        video_url = video[-1]["baseUrl"]  # 兜底：最低质量
    # mp4 兜底
    durl = data.get("durl", [])
    return {"video_url": video_url or (durl[0]["url"] if durl else None), "audio_url": audio_url}


async def download_bili_video(bvid, out_dir, info=None, cookie="",
                               timeout=120.0, max_seconds=0, proxy_mode="auto",
                               quality: str = "") -> tuple[str, dict]:
    """下载 B 站视频（按 quality 选对应档，不下原画再压缩）
    quality: "low"(360p), "medium"(720p), "original"(最高), 空=low
    """
    quality_hint = {"low": 360, "medium": 720, "original": 0}.get(quality, 360)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    req = _Requester(cookie, timeout, proxy_mode)
    info = info or await get_bili_info(bvid, cookie, timeout, proxy_mode)
    if max_seconds and info["duration"] > max_seconds:
        raise BiliError(f"视频时长 {info['duration']}s 超过上限 {max_seconds}s")
    fname = f"{bvid}.mp4"
    fpath = out_dir / fname
    if fpath.exists() and fpath.stat().st_size > 0:
        return str(fpath), info

    urls = await _get_video_urls(req, bvid, info["cid"], quality_hint=quality_hint)
    if not urls["video_url"]: raise BiliError("无法获取视频下载链接")

    vpath = out_dir / f"{bvid}_v.mp4"
    apath = out_dir / f"{bvid}_a.m4a"

    await req.stream(urls["video_url"], vpath)
    if not vpath.exists() or vpath.stat().st_size == 0:
        raise BiliError("视频流下载文件为空")

    has_audio = bool(urls.get("audio_url"))
    if has_audio:
        try:
            await req.stream(urls["audio_url"], apath)
            if not apath.exists() or apath.stat().st_size == 0:
                has_audio = False
        except: has_audio = False

    if has_audio:
        try:
            r = subprocess.run([
                "ffmpeg", "-y", "-i", str(vpath), "-i", str(apath),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(fpath),
            ], capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                try: vpath.unlink()
                except: pass
                try: apath.unlink()
                except: pass
            else:
                raise BiliError(f"音视频合并失败: {r.stderr[:200]}")
        except FileNotFoundError:
            # ffmpeg 不可用（Windows），降级：仅视频流无音轨
            import shutil; shutil.move(str(vpath), str(fpath))
        except Exception as e:
            raise BiliError(f"音视频合并异常: {e}")
    else:
        import shutil; shutil.move(str(vpath), str(fpath))

    if not fpath.exists() or fpath.stat().st_size == 0:
        raise BiliError("最终视频文件为空")
    return str(fpath), info


# ══════════════════════════════════════════════════════
#  BV 号提取
# ══════════════════════════════════════════════════════

async def extract_bvid(text, timeout=15.0, proxy_mode="auto"):
    m = BV_RE.search(text)
    if m: return m.group(0)
    m2 = SHORT_RE.search(text)
    if m2:
        url = m2.group(0)
        if not url.startswith("http"):
            url = "https://" + url
        url = url.split("?")[0]
        # 直连和代理都试
        for use_proxy in (False, True):
            try:
                async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=timeout, trust_env=use_proxy) as c:
                    r = await c.get(url)
                    final_url = str(r.url)
                m3 = BV_RE.search(final_url)
                if m3: return m3.group(0)
            except Exception:
                continue
    return ""