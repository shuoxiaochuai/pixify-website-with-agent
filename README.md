# AI 修图智能体

一个由 LLM Agent 驱动的通用 AI 图片编辑器。用户用一句自然语言描述需求，Agent 负责理解、拆解并调用工具完成生成、修图、局部编辑、拆层与尺寸适配，结果实时回流到前端画布。

- 前端：React 19 + TypeScript + Konva
- 后端：FastAPI + LangChain / LangGraph + ARQ Worker
- 数据：PostgreSQL（业务数据）+ Redis（队列与事件）+ S3 兼容存储（图片）

## 功能

### 对话式修图

- 指令分流：修图动作、无关提问、两者混合的一句话分别走不同通道，互不丢步。
- 多步规划：Agent 一次输出完整工具调用序列，服务端做存在性、参数与依赖校验后再执行。
- 过程可见：阶段、步骤与正文增量通过 SSE 推送，支持流式输出与伪思考展示。
- 任务控制：可取消进行中的智能修图，可删除单条对话消息。

### 画布编辑器

- 基于 react-konva 的画布，支持缩放、平移、图层文档、撤销重做与前后对比。
- 图层可单独缩放、旋转、移动、调色、改透明度、改显隐、调整层序。
- 工具栏与 Agent 共用同一套工具定义，界面点按和自然语言调用行为一致。

### 编辑工具（24 个）

| 分组 | 工具 |
| --- | --- |
| 生成 | `generate_image` 文生图、`generate_marketing` 衍生图、`prepare_delivery_sizes` 尺寸适配 |
| 一键处理 | `smart_edit` 自然语言整图编辑、`replace_background` 换背景、`remove_background` 去背景、`expand_canvas` 扩图、`upscale_image` 超分、`adjust_image` 调色 |
| 区域编辑 | `replace_region` 局部替换、`erase_region` 局部消除（均依赖选区） |
| 图层与画布 | `split_layers` 拆层、`promote_object_to_layer` 提升为图层、`crop_canvas` 裁剪，以及翻转、缩放、旋转、移动、层序、透明度、显隐、文字层共 11 个操作 |
| 批量 | `batch_process` 多图同套处理 |
| 领域能力 | `segment_lesion` 眼底病灶分割 |

### 选区与局部编辑

- 点选分割（SAM）与笔刷两种选区方式，点选时后续点击只跑 decoder。
- 局部消除用 inpainting 填补背景，局部替换只改选区内容，其余像素保持不变。

### 图层

- 一键把图片拆成主体层与背景层，可选拆出 OCR 文字层。
- 任意物体可提升为独立图层，背景自动修复。
- 图层文档以 JSONB 持久化，切换图片再切回不丢结构。

### 生成与交付

- 文生图一次生成多张候选，四宫格挑选后进入编辑器。
- 衍生图支持纯净背景、场景氛围图、人物出镜、文字海报四类，结果只进图片墙，不动当前画布。
- 尺寸适配按 1:1、4:5、9:16 等比例完整放入画布，不裁切主体。
- 批量处理对多张图执行同一套操作，进度实时回传。
- 导出打包为 zip，附 `manifest.json`，可只导出当前画布或图片墙产物。

### 病灶分割

内置 CAUNet 推理模块，对眼底图标注出血点、硬性渗出、微血管瘤、软性渗出。无 GPU、无 torch 或无权重时直接返回可展示的不可用原因，不影响其他功能启动。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 前端 | React 19、TypeScript、Vite、react-router-dom、zustand、TanStack Query、konva / react-konva、Tailwind CSS |
| 后端 | Python ≥ 3.13、FastAPI、Pydantic Settings、SQLAlchemy 2（async）、Alembic |
| Agent | LangChain Core、LangGraph、OpenAI 兼容接口（DashScope 千问） |
| 任务与消息 | ARQ、Redis（Pub/Sub + Stream）、SSE |
| 图像处理 | Pillow、NumPy、OpenCV、rembg / ONNX Runtime、RapidOCR |
| 存储 | PostgreSQL、S3 / MinIO |
| 模型推理 | PyTorch、CAUNet（病灶分割，仅 Linux 部署机安装） |

## 架构

后端按路由层、服务层、工具执行层分层，像素类工具通过 ARQ 异步队列提交到独立 Worker 执行，进度经 Redis Pub/Sub 与 Stream 推送，再由 `/events` 的 SSE 端点转给浏览器。只改图层文档的工具当场执行，不占用队列。

一次修图请求的链路：

1. 用户提交一句话，写入 `AgentRun` 并派发后台任务。
2. 分流判断通道；混合句先拆成修图段与提问段。
3. 修图段交给规划模型，输出工具调用计划。
4. 服务端校验工具名、参数、依赖并补全顺序，失败时重规划一次。
5. 执行外壳按步落 `ToolRun`，像素工具入队，Worker 执行并上报进度。
6. 画布修订号递增、结果落库，过程与结果通过 SSE 推给前端。
7. 前端更新画布、图片墙、计划卡片与对话内容。

## 目录结构

```
backend/
  app/
    agent/       分流、拆分、规划、LangGraph 编排、LLM 接入
    edits/       图层文档操作、像素管线、蒙版、分割、拆层、OCR、渲染
    eval/        离线评测：用例、任务、指标、运行器
    lesion/      CAUNet 病灶分割：模型、推理、掩码对齐
    models/      SQLAlchemy 模型：用户、素材、会话、编辑历史、工具运行、Agent 轮次
    providers/   图像生成/编辑供应层：mock 与 dashscope
    routers/     health、auth、assets、sessions、runs、batches、lesion、events
    schemas/     请求/响应模型
    services/    业务服务：会话、素材、生成、工具执行、导出、选区、批量
    tasks/       ARQ 任务：工具执行、连通性检查
    tools/       24 个工具的注册表与实现
    config.py    环境变量与路径
    db.py        异步引擎与会话工厂
    events.py    Redis 发布/订阅/Stream
    layers.py    图层文档模型与解析
    queue.py     入队与中止
    worker.py    ARQ Worker 入口
    main.py      FastAPI 装配与前端同源托管
  migrations/    Alembic 迁移
  tests/         pytest（mock 隔离模型与第三方调用）
frontend/
  src/
    api/         各资源的 HTTP 客户端与 SSE
    components/  落地页、编辑器、通用 UI
    hooks/       数据请求、选区、任务、画布缩放等
    layouts/     登录守卫与工作台布局
    lib/         工具定义、图层操作、批处理步骤等
    pages/       落地页、创作、候选、编辑器、衍生图、批量、认证
    stores/      zustand：画布视口、编辑器 UI、提示
deploy/          裸机部署脚本与 supervisor 配置
deploy.ps1       Windows 侧推送脚本
Dockerfile       多阶段构建，前端产物打进后端镜像
docker-compose.yml  中间件与 deploy profile 全栈
```

## 核心代码说明

### Agent 编排

`app/agent/graph.py` 是核心。状态图从分流开始：`triage.classify` 用关键词命中加一次模型调用判定通道，`triage.split` 在句子同时包含提问与修图动作时按标点与连接词拆分，两条通道各自产出结果再合并。规划阶段把画布摘要与工具签名交给模型，`plan.validate` 负责工具存在性、参数合法与依赖检查；模型只回了散文式调用或漏掉动作时，会做一次兜底解析与重规划。执行阶段逐条派发工具，并把阶段事件通过 `publish_turn` 写入 Redis Stream，晚连上的订阅者也能补齐开头。

`app/agent/llm.py` 统一构造规划、分流、闲聊三类模型客户端；`providers/` 则负责真正调用图像生成与图像编辑模型，`mock` 可在无 API Key 时跑通全链路。

### 工具注册表

`app/tools/base.py` 的 `ToolSpec` 同时描述界面与 Agent 所需的全部信息：名称、展示标签、说明、Pydantic 参数模型、执行函数、是否入队、是否隐藏参数。`app/tools/__init__.py` 汇总全部工具；新增工具只要在这里登记，界面工具栏、Agent 函数签名与服务端校验就会同步生效，不会出现两套定义漂移。

### 图层文档与渲染

`app/layers.py` 定义 `LayerDocument`：画布宽高加图层数组，图层含类型、变换、透明度、文字属性。`app/edits/document.py` 里的每个函数都返回新文档（写时复制），保证一次编辑对应一条历史记录、可整体撤销。`app/edits/render.py` 按图层顺序合成像素，`edits/pixels.py` 提供缩放、letterbox、编码等底层工具。

### 选区与像素管线

选区存在 Redis 中，按会话与修订号隔离，`services/selections.py` 负责读写，`routers/sessions.py` 暴露接口。分割由 `edits/segment.py` 完成，点选与笔刷共用同一套蒙版格式；`edits/mask.py` 负责蒙版与原图的对齐、羽化与合成，`edits/split.py` 负责拆层与物体提升，`edits/ocr.py` 在安装 RapidOCR 时提供文字层识别。

### 异步任务与进度

`queue.py` 以 run id 作为任务 ID，重复投递不会执行两次；中止信号写入 ARQ 的中止集合，Worker 轮询到后掐断任务。`worker.py` 是 ARQ 入口，`tasks/tools.py` 是统一执行外壳：落状态、调工具、上报进度、写结果。`events.py` 提供 run 的 Pub/Sub 与 turn 的 Stream 两种通道，`routers/events.py` 把两者包装成 SSE，空闲时发心跳。

### 会话、资产与导出

`models/` 中用户、素材、会话、图片墙关联、编辑历史、工具运行、Agent 轮次各自独立；`services/sessions.py` 维护画布修订号与当前图切换，`services/assets.py` 负责落盘与同源文件代理，`services/exports.py` 生成 zip 与清单。图片不依赖对象存储的公开读，统一走 `/api/assets/{id}/file`。

### 评估

`app/eval` 是可离线运行的评测器：按目录读取用例与素材，逐任务执行图像操作，计算像素与指标并输出报告。默认自带一组示例用例，可通过 `python -m app.eval <数据目录>` 指定自己的评测集。

### 病灶分割

`app/lesion/engine.py` 对齐训练与验证时的预处理链（短边缩放、中心裁剪、CLAHE、绿通道），模型输出概率后按阈值二值化并着色；`lesion/masks.py` 处理掩码回贴与 Dice 计算。`lesion/caunet.py` 与 `layers.py` 逐字取自训练仓库，因此不参与格式化。

### 配置与安全

`config.py` 用 Pydantic Settings 读取仓库根目录 `.env`，模型名、供应方、存储与数据库地址都可切换。认证使用 httpOnly Cookie + JWT，密码经 bcrypt 哈希；所有资源查询都带用户条件，跨用户访问统一返回 404。

## 本地运行

前置条件：Docker（含 Compose）、Python ≥ 3.13、Node.js ≥ 18，以及可选的 DashScope API Key。

```bash
# 中间件：PostgreSQL 7311、Redis 7312、MinIO 7313
docker compose up -d
```

```bash
# 后端
cd backend
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows 用 powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --all-extras
cp .env.example .env                               # IMAGE_PROVIDER=mock 可零额度跑通
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 7302
```

```bash
# Worker（另开终端）
cd backend
uv run arq app.worker.WorkerSettings
```

```bash
# 前端（另开终端）
cd frontend
npm install
npm run dev        # http://127.0.0.1:7301，/api 与 /events 代理到 7302
```

健康检查：`GET http://localhost:7302/api/health`，字段全部为 `ok` 即正常。

### 测试与检查

```bash
cd backend
uv run pytest
uv run ruff check .
```

```bash
cd frontend
npm run build      # tsc -b + vite build
npm run lint       # oxlint
```

测试通过 mock 隔离大模型与第三方接口，不消耗 API 额度。

### 关键环境变量

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `IMAGE_PROVIDER` | `mock` 或 `dashscope` | `mock` |
| `DASHSCOPE_API_KEY` | 百炼 API Key | 空 |
| `TEXT_TO_IMAGE_MODEL` / `IMAGE_EDIT_MODEL` | 文生图与图像编辑模型 | `qwen-image-3.0-pro` / `qwen-image-edit-max` |
| `PLANNER_MODEL` / `INTENT_MODEL` / `CHAT_MODEL` | 规划、分流、闲聊模型 | `qwen-plus` / `qwen-flash` / `qwen-plus` |
| `DATABASE_URL` / `REDIS_URL` | 数据库与 Redis 地址 | 本地 7311 / 7312 |
| `S3_ENDPOINT` / `S3_BUCKET` | 对象存储 | 本地 7313 / `retouch` |
| `JWT_SECRET` | 会话签名密钥，生产必改 | 开发占位值 |
| `MATTING_PROVIDER` | `auto` 有 rembg 用之，否则四角抠图 | `auto` |
| `OCR_PROVIDER` | `auto` / `none` | `auto` |
| `LESION_MODEL_*` / `LESION_DEVICE` / `LESION_THRESHOLD` | 病灶分割权重与推理设置 | 空 / `auto` / `0.5` |

## 部署

`Dockerfile` 为多阶段构建，前端编译产物由后端同源托管，对外只需暴露 **7302** 一个端口：

```bash
cp .env.example .env
# 生产环境至少修改 JWT_SECRET
docker compose --profile deploy up -d --build
docker compose --profile deploy exec app alembic upgrade head
```

没有 docker daemon 的算力平台实例使用 `deploy/` 下的裸机方案：`bootstrap.sh` 重建环境、supervisor 配置与数据库迁移，`retouch` 命令负责日常状态、重启、日志与备份，Windows 侧用 `deploy.ps1` 推送。改后端代码后 `app` 与 `worker` 必须同时重启。

注意：不要将 MinIO（7313）暴露到公网；`/api/auth/register` 是开放注册，公网演示前应控制访问范围。