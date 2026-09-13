# Video Comprehension 视频理解插件

KiraAI 视频理解插件：自动检测 QQ 视频消息，压缩 + 三级抽帧 + 拼图分析；支持 B 站链接下载/发送/搜索/AI 总结；支持本地视频。

## 核心能力

1. **自动检测视频消息** — 消息链 `Video` 元素 → `raw_message` → `get_msg` 三级兜底
2. **压缩管线** — ffmpeg 缩放 + CRF 压缩，保持时长不变
3. **三级抽帧** — 场景变化帧 → I 帧 → 均匀补帧，信息密度最大化
4. **拼图 + 时间戳标注** — 每帧标注 `MM:SS.mmm [I/P] #N`，多帧合成拼图，按时间顺序排列
5. **多模型路由** — 4 组独立配置，按优先级 + 时长自动选择；`native`（传视频）或 `frames`（传拼图）
6. **会话管理** — 首轮分析 + 后续追问（`session_id`），新视频自动切换会话
7. **时间段分析** — bot 可指定只看第 N~M 秒，支持一次多段（最多 5 段、每段 ≤300 秒）
8. **B 站能力** — 链接自动下载发送（0 token）、搜索、AI 总结优先、按质量下载

## 自定义请求头 / 请求体（每组模型独立）

每个模型组都有两个 JSON 配置项，对应 OpenAI SDK 的 `extra_headers` / `extra_body`：

| 配置项 | 作用 | 示例 |
|--------|------|------|
| `extra_headers_N` | 合并进每次 API 请求头 | `{"X-Api-Version": "2024-01"}` |
| `extra_body_N` | 合并进请求体（厂商私有参数） | `{"enable_thinking": true, "thinking_budget": 4096}` |

常见用法：

- **Qwen3-VL 开思考模式**：`extra_body` = `{"enable_thinking": true, "thinking_budget": 8192}`
- **GLM 开深度思考**：`extra_body` = `{"thinking": {"type": "enabled"}}`
- **自定义网关鉴权**：`extra_headers` = `{"X-Api-Key": "xxx"}` 或 `{"Authorization": "Bearer xxx"}`
- **厂商私有采样参数**：`extra_body` = `{"top_k": 5, "seed": 42}`

行为说明：

- 字段为空 `{}` 或非法 JSON → 视作未配置（**不会因此报错**，只是不带）
- 直接写 dict 或写 JSON 字符串都能识别
- 这两个参数**不会覆盖** `model` / `messages` / `timeout` / `max_tokens` 这几个插件自己设置的字段（同名键以 extra_body 为准，所以别往这里写它们）
- `frames` 与 `native` 两条调用路径都已带上

## 上传换公开链接（给 GLM 这类只吃 URL 的模型）

智谱 GLM 官方示例里的 `video_url` 是**公网 URL**形式，base64 没有官方保证。开启后，native 模式会先把视频上传到文件中转服务，把返回的链接传给模型：

```
本地视频 → (单段则秒切) → 上传 → https://x0.at/xxxx.mp4 → 模型自行拉流
```

- **配置**：设置 → 视频上传 → `upload_enabled`（**默认关闭**）、`upload_hosts`（默认 `litterbox` → `x0.at`，**前一个失败自动换下一个**）
- **压缩策略**（`upload_compress_over_mb`，默认 **20MB**）：≤该值**不压缩**直接上传（保住画质、少一次转码）；超过才压到 720p 再传。设 `0` = 全程不压缩

### 上传源对比（实测）

| 服务 | 上限 | 保留 | 匿名上传 | 国内可达（上传端） | HEAD |
|------|------|------|---------|-----------------|------|
| **litterbox.catbox.moe**（默认） | 1 GB | 1h / 12h / **24h** / 72h | ✅ | ✅ 46/49 | ✅ |
| **x0.at**（备用） | 1024 MiB | 3~100 天 | ✅ | ✅ 33/33 | ❌ 404 |
| catbox.moe | 200 MB | **永久** | ❌ `Invalid uploader` | ❌ 0/47 | — |
| 自建 filehost2 | 看配置 | 看配置 | ✅ | ✅ | 看配置 |

- **服务类型自动识别**（`detect_kind`）：`litterbox.catbox.moe` → litterbox 协议；`catbox.moe` → catbox 协议；其余按 filehost2 协议（x0.at / 0x0.st / 自建）
- ⚠️ catbox 主站**国内上传端全不通**（匿名上传直接被拒），所以没做默认；想用要自己有账号且网络能通
- litterbox 的链接是**临时**的（默认 24h 后失效），比 catbox 的永久存储更适合"传上去给模型看一眼就丢"
- **失败自动回退**：上传未启用/超限/失败 → 自动改用 base64 内联，不会因此失败
- **重复不重传**：同一文件（路径 + mtime）在本次运行内缓存 URL；x0.at 服务端还会按文件哈希去重
- **绕过体积限制**：base64 模式受 20MB 上限（超了降级帧模式），有 URL 时不受此限
- **隐私**：⚠️ 上传后该视频可通过链接被**任何人**访问（链接随机但公开）。链接与文件保留 3~100 天（越大越短）。介意就不要开，或换成自建的 filehost2

实测：x0.at 国内电信 33 节点全部 200；litterbox 文件域 33/34、上传域 46/49；两者的**拉取方 UA 均无限制**（空 UA/各种客户端都 200）、都支持 Range、`content-type: video/mp4` 正确。

## 时间段分析

让 bot 只看视频的某一段，不用把整片都喂给模型：

```
analyze_video(session_id="abc123", question="这里在说什么", start_sec=30, end_sec=45)
analyze_video(bvid="BV1xx...", start_sec=0, end_sec=20)            # 首次调用也能限定
analyze_video(session_id="abc123", segments=[[10,30],[100,130]])   # 一次多段
```

- **参数**：`start_sec` / `end_sec` 为数字秒（`end_sec=0` 表示到视频结尾）；多段用 `segments=[[起,止],...]`
- **帧密度自适应**：每 ~2 秒 1 帧，最少 6 帧；多段时每段配额 = `target_frames / 段数`，总帧数不会失控
- **复用已下载的视频**：对已有 session 做时间段分析不会重新下载；同时跳过全片压缩（只处理片段）
- **native 单段**：用 `-c copy` 秒切该片段（含音频）传给模型 —— 能听到这段的语音；起点会对齐关键帧，结果里标注实际范围
- **多段**：统一走帧模式，一次请求覆盖所有段，拼图按段分开并在底部标注「段N/M 起~止」
- **限制**：单段 ≤300 秒、最多 5 段，超出会提示分段；`end_sec` 超出视频时长会自动截断

## 模型接口能力（重要）

| 厂商 | OpenAI 兼容 `video_url` | 建议模式 | 说明 |
|------|------------------------|----------|------|
| 智谱 GLM | 官方示例为 **URL** 形式 | `native` 或 `frames` | 官方文档仅演示公网 URL；base64 视频未在文档中保证，失败会自动降级 `frames` |
| 阿里 Qwen (百炼) | ✅ 支持，另有 `fps` 参数 | `native` | 本地文件支持 Base64 上传 |
| Google Gemini | ❌ **兼容层只有 image_url/audio** | **`frames`** | 要用原生视频需走 Gemini 原生 API，本插件的 OpenAI 通道不支持 |
| 其他 OpenAI 兼容 | 视厂商而定 | `frames` 最稳 | `native` 失败会自动降级 |

**自动降级**：`native` 模式请求失败（厂商不支持 / 视频超过 20MB / 请求体过大）时，自动改用拼图帧模式重试，并在结果中标注 `native→frames`。

**体积限制**：`native` 会把压缩后的视频整体 base64（体积 ×1.33）塞进请求。超过 20MB 会直接走帧模式，避免请求被拒或超时。

## 配置

### 质量配置（三个独立开关）

| 配置项 | 默认 | 作用 |
|--------|------|------|
| `bili_download_quality` | `low` | **下载质量**：从 B 站 DASH 直接选对应档流（360p/720p/原画），不下原画再压 |
| `bili_compress_quality` | `original` | **下载后再压**：只在比下载质量更低时才执行（避免把小视频放大） |
| `send_video_quality` | `low` | 本地视频（`local_path`）发送时的压缩质量 |

### 安全限制
- `max_file_size_mb`：最大视频文件（默认 200MB）
- `max_duration_sec`：最大视频时长（默认 600 秒）
- `download_timeout_sec`：下载超时（默认 120 秒）

### B 站配置
- `bili_cookie`：**AI 总结必需**（B 站接口要求登录，无 cookie 返回 `-101 账号未登录`）；下载高清/大会员视频也需要
- `bili_use_ai_summary`：优先用 B 站官方 AI 总结（免费、不消耗模型 token），失败自动降级视觉分析
- `auto_send_link`：检测到消息里的 B 站链接自动下载发送（0 token）

### 抽帧参数
- `target_frames`：总目标帧数（默认 40）
- `max_frames_per_grid`：每张拼图最多帧数（默认 20）
- `grid_cols`：拼图列数（默认 5）
- `scene_threshold`：场景变化敏感度（默认 0.3，越大越敏感）
- `frame_width` / `frame_ratio`：单帧宽度与比例

## 依赖

- **ffmpeg + ffprobe**（系统级；插件启动时会自动检测，缺失则尝试 `apk`/`apt` 安装或下载静态包，静态包会同时提取 ffmpeg 与 ffprobe）
- Pillow（拼图合成）
- httpx / openai（网络与模型调用）

## 已知限制

- 场景变化检测为启发式（基于 I 帧与相邻帧间隔突变），不是逐帧图像差异比对
- B 站搜索接口在无 cookie 时可能被风控限制
- NapCat 官方**没有** `upload_file_stream` action（那是第三方扩展）；插件会尝试一次，失败后记住并直接发送本地路径

## 版本

- 1.7.0 — 每个模型组新增 `extra_headers` / `extra_body`（JSON），可自定义请求头与厂商私有请求体参数（如 Qwen 的 enable_thinking、GLM 的 thinking）；frames 与 native 两条路径都生效
- 1.6.0 — 上传源支持多个（默认 litterbox → x0.at，前者失败自动换下一个）+ 自动识别服务协议（litterbox/catbox/filehost2）；实测 catbox 匿名上传国内不可用故未做默认
- 1.5.0 — 上传前的压缩改为按体积自动（`upload_compress_over_mb`，默认 20MB）：≤阈值不压缩直传保住画质，超过才压；阈值 0 = 全程不压缩。修正此前"全片压缩上传 / 单段原片直传"的不一致
- 1.4.0 — 新增「上传换公开链接」：native 模式可把视频上传到 x0.at / filehost2 类服务换取公网 URL（给智谱 GLM 这类只接受 URL 的模型用），失败自动回退 base64、同文件不重复上传；顺带修正全片 native 被误当"单段"去秒切的问题
- 1.3.0 — 新增时间段分析：`start_sec`/`end_sec` 单段 + `segments` 多段（≤5 段、每段 ≤300 秒），帧数按段长自适应，native 单段用 `-c copy` 秒切片段（含音频），复用已下载文件不重复下载
- 1.2.0 — 修复类定义被截断（`analyze_video` 未注册）、模型组配置读不到（嵌套结构）、平台判定大小写、ffprobe 字段（`pkt_pts_time` 已废弃）、拼图 `cell_w` 未定义、WBI 签名算法、重复发送、子目录不清理等 10+ 个问题；新增 native→frames 自动降级与体积保护
- 1.1.0 — 下载质量与压缩质量拆分
- 1.0.0 — 初版
