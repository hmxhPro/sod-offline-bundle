# 视频目标检测系统 · 部署文档

基于 **FastAPI + YOLOE + ByteTrack + React** 的视频目标检测系统。本文档讲**如何安装、配置、启动与迁移**；系统的功能、架构与 API 说明见 [`docs/系统功能与技术.md`](docs/系统功能与技术.md)。后端启动后，交互式 API 文档在 `http://localhost:8000/docs`。

> 架构审计与修复路线图见 [`技术评审与修复方案.md`](技术评审与修复方案.md)。

---

## 目录

- [环境要求](#环境要求)
- [后端部署](#后端部署)
  - [1. 安装依赖](#1-安装依赖)
  - [2. 模型与额外文件](#2-模型与额外文件)
  - [3. PostgreSQL 建库](#postgresql-建库)
  - [4. 配置 `.env`](#4-配置-env)
  - [5. 启动后端](#5-启动后端)
- [前端部署](#前端部署)
- [大模型 (VLM) 配置](#大模型-vlm-配置)
- [迁移到新设备](#迁移到新设备)
- [故障排查](#故障排查)
- [横向扩展](#横向扩展)

---

## 环境要求

| 依赖 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.10 | 后端运行时 |
| PyTorch | 2.1.0 | 推理 / 训练 |
| CUDA | 11.8 | GPU 加速 |
| FastAPI | 0.111.0 | 后端框架 |
| **PostgreSQL** | **13+** | 任务历史数据库 |
| **SQLAlchemy** | **2.0** | 异步 ORM |
| **asyncpg** | **0.29** | PostgreSQL 异步驱动 |
| **ultralytics** | **8.3.0** | YOLOE 开放词汇推理 + 自定义 YOLO11 训练 |
| **sahi** | **0.11.18** | 小目标切片推理 |
| **httpx** | **0.27.0** | 调用 VLM（本地或远端 OpenAI 兼容 API） |
| Node.js | 18+ | 前端构建（Vite） |
| transformers | 4.40.0 | Grounding DINO / Florence-2（可选） |

运行环境：Linux + NVIDIA GPU (CUDA)，推荐 RTX 3090 / A100 或更高。

---

## 后端部署

### 1. 安装依赖

```bash
cd backend
python -m venv venv && source venv/bin/activate   # 或使用 conda 环境（Python 3.10）
pip install -r requirements.txt
```

`requirements.txt` 已含 YOLOE 默认后端所需的 `ultralytics>=8.3.0` 与 `sahi>=0.11.18`。

> torch / torchvision 通常由 GPU 容器自带，不经 pip 安装。

#### 较难安装的依赖（含 C / CUDA 扩展，需单独处理）

以下几个包带本地编译扩展，**`pip install -r requirements.txt` 不会自动装好**。它们对应的第三方库已随仓库 **vendored** 在 `backend/ByteTrack`、`backend/GroundingDINO`、`backend/YOLOWorld`，无需另行 `git clone`，只要补齐各自的编译依赖即可。

**① ByteTrack —— 视频多目标跟踪（强烈建议安装）**

ByteTrack 为每个目标分配跨帧稳定的 `track_id`，并支撑“每 N 帧检测 + 中间帧 Kalman 预测”的 ~5× 加速（原理见 [系统功能与技术 · 检测 + 跟踪流程](docs/系统功能与技术.md#检测--跟踪流程)）。**它是默认视频管线的核心，与是否使用备选检测后端无关。** 缺失时不会报错，但会静默降级为 *passthrough* 跟踪器——每帧给每个框分配自增的新 ID，没有任何跨帧持久性（框颜色逐帧乱跳、目标计数灌水、时序确认与 VLM 复核失效）。

`backend/ByteTrack` 已附带，只差一个需编译的 `cython_bbox`（`lap` 已在 `requirements.txt` 内）。二选一：

```bash
pip install cython_bbox     # 路线A（推荐）：复用 backend/ByteTrack，tracker.py 会自动加载它
pip install bytetracker     # 路线B：独立 pip 包，自带依赖，无需 backend/ByteTrack
```

> `cython_bbox` 需现场编译，要求系统装有 `gcc` / `g++`（Ubuntu：`sudo apt-get install build-essential`）；编译失败时改用路线 B。
>
> **验证**：重启后端，日志出现 `ByteTrack initialized` 即生效；若仍是 `ByteTrack not found. Falling back to passthrough tracker` 则未装上。

**② Grounding DINO —— 可选检测后端（仅 `DETECTION_MODEL=grounding_dino` 时需要）**

`backend/GroundingDINO` 已附带，但它要**编译 CUDA C++ 算子**（`_C` 扩展），是最容易装失败的一个：需本机有与 torch 版本匹配的 **CUDA 工具链（`nvcc`）** 并设置 `CUDA_HOME`。

```bash
export CUDA_HOME=/usr/local/cuda     # 指向含 bin/nvcc 的 CUDA 目录
pip install -e GroundingDINO         # 就地编译 backend/GroundingDINO（接上面 venv，在 backend/ 下执行）
# 无 nvcc / 想省事：pip install groundingdino-py
```

> 常见报错 `NVCC not found` / `CUDA_HOME environment variable is not set` → 先装 CUDA toolkit 再导出 `CUDA_HOME`。默认后端是 YOLOE，不切到 Grounding DINO 就**无需安装这个**。

> **Florence-2** 备选后端（`DETECTION_MODEL=florence2`）无需编译，`transformers` 会在首次运行自动下载约 1.5GB 权重（见 [配置 `.env`](#4-配置-env)）。

### 2. 模型与额外文件

本项目**强制离线模式**（`YOLO_OFFLINE=1` / `ULTRALYTICS_OFFLINE=1`），运行时**不会自动联网下载**，必须先手动放好以下文件（均被 `.gitignore` 忽略，不随仓库分发）：

| 文件 | 大小 | 目标路径（相对 `SOD/` 根） | 必需性 |
|---|---|---|---|
| `yoloe-11l-seg.pt` | ~68 MB | `backend/models/yolo/yoloe-11l-seg.pt` | **必需**：检测与训练共用的基座权重 |
| `mobileclip_blt.ts` | ~572 MB | `backend/mobileclip_blt.ts` | **文本提示必需**：缺失则开放词汇检测报错 |
| `groundingdino_swint_ogc.pth` | ~662 MB | `backend/models/groundingdino/weights/` | 可选：仅切到 Grounding DINO 后端时需要 |

**下载方式一（推荐：有网机器让 Ultralytics 自己拉）**

```bash
cd backend
YOLO_OFFLINE=0 ULTRALYTICS_OFFLINE=0 python - <<'PY'
from ultralytics import YOLOE
m = YOLOE("yoloe-11l-seg.pt")                # 下载基座权重到当前目录
names = ["person", "car"]
m.set_classes(names, m.get_text_pe(names))   # 触发下载 mobileclip_blt.ts 到当前目录
print("downloaded ok")
PY
mkdir -p models/yolo && mv -f yoloe-11l-seg.pt models/yolo/   # 权重归位；mobileclip_blt.ts 留在 backend/
```

**下载方式二（手动直链：内网 / 离线分发）**

```bash
cd backend && mkdir -p models/yolo
wget -O models/yolo/yoloe-11l-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt
wget -O mobileclip_blt.ts            https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts
# 可选 Grounding DINO：
mkdir -p models/groundingdino/weights
wget -O models/groundingdino/weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

> 换规格只需替换文件名：`yoloe-11s-seg.pt`（最小最快）/ `yoloe-11m-seg.pt`（均衡）/ `yoloe-11l-seg.pt`（精度最好，默认）。

**`mobileclip_blt.ts` 放哪（易踩坑）**：Ultralytics 相对**进程 CWD** 解析该文件，而 `backend/start.sh` 会先 `cd backend` 再起服务，所以放在 **`backend/mobileclip_blt.ts`** 最稳妥。若仓库里 `backend/weights/mobileclip_blt.ts` 是指向旧机器的断链软链，删掉它：`rm -f backend/weights/mobileclip_blt.ts`。

**放置自检**：

```bash
cd backend
python -c "from ultralytics import YOLOE; print('YOLOE import OK')"
ls -lh models/yolo/yoloe-11l-seg.pt mobileclip_blt.ts
```

### PostgreSQL 建库

表结构由后端启动时 `init_db()` 自动创建（`Base.metadata.create_all`），**无需手写 DDL**——但要先准备好数据库和角色。表清单与设计见 [系统功能与技术 · 任务历史持久化](docs/系统功能与技术.md#任务历史持久化设计)。

```bash
# 1. 安装并确认 PostgreSQL(≥13) 在运行
sudo apt-get install -y postgresql && pg_isready

# 2. 一键初始化（建角色 sod_app + 库 sod + 授权 + 写回 DATABASE_URL 到 backend/.env）
bash backend/scripts/init_postgres.sh           # 需 sudo（用 postgres 系统用户执行 DDL）
# 可改名：SOD_DB_NAME=mydb SOD_DB_USER=myrole bash backend/scripts/init_postgres.sh
```

脚本会：建角色（随机 24 位密码或复用 `.env` 已有密码）→ 建库（owner=sod_app）→ 授权 `public` schema → TCP 密码登录验证 → 把 `DATABASE_URL` 写回 `backend/.env` 并 `chmod 600`。

<details><summary>不用脚本时的手动等价做法</summary>

```bash
sudo -u postgres psql -c "CREATE ROLE sod_app LOGIN PASSWORD '改成你的密码';"
sudo -u postgres psql -c "CREATE DATABASE sod OWNER sod_app;"
sudo -u postgres psql -d sod -c "GRANT ALL ON SCHEMA public TO sod_app;"
# backend/.env（驱动后缀必须是 +asyncpg）：
# DATABASE_URL=postgresql+asyncpg://sod_app:改成你的密码@localhost:5432/sod
```
</details>

> `create_all` 只建缺失的表，不改已存在的表；Schema 演进需接入 Alembic。验证：`sudo -u postgres psql -d sod -c '\dt'`。

### 4. 配置 `.env`

```bash
cp backend/.env.example backend/.env
```

关键配置项：

```env
# ── 检测模型（默认 YOLOE 开放词表 + SAHI 小目标切片）──
DETECTION_MODEL=yoloe
DEVICE=cuda:0                                   # 无 GPU 改 cpu
YOLOE_BASE_MODEL=/abs/path/to/SOD/backend/models/yolo/yoloe-11l-seg.pt

# ── 可选 SAHI 切片（大图小目标显著提升召回）──
SAHI_SLICE_HEIGHT=640
SAHI_SLICE_WIDTH=640

# ── 日志（目录解析为绝对路径，按天轮转、保留 7 天；uvicorn/标准库日志一并汇入）──
LOG_DIR=./logs
LOG_LEVEL=INFO                                  # 控制台级别；文件始终 DEBUG。DEBUG=true 时 500 才返回原始错误

# ── 数据库（建议由 init_postgres.sh 自动写入）──
DATABASE_URL=postgresql+asyncpg://sod_app:<password>@localhost:5432/sod
DATABASE_ECHO=false

# ── 大模型 VLM（见下方“大模型 (VLM) 配置”）──
VLM_ENABLED=true
VLM_API_BASE=http://localhost:8010/v1
VLM_MODEL_NAME=MiniCPM-V-4_5
```

切换其它检测后端：`DETECTION_MODEL=grounding_dino`（配 `GDINO_CHECKPOINT_PATH`）或 `florence2`（配 `FLORENCE2_MODEL_ID=microsoft/Florence-2-large`，首次运行自动下载约 1.5GB）。

### 5. 启动后端

```bash
cd backend && bash start.sh
# 等价于：uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-graceful-shutdown 30
# 日志出现 “DB schema ensured” 即建表成功；访问 http://localhost:8000/docs
```

`start.sh` 额外设置了 HuggingFace 离线开关与 FFmpeg 静默，并给优雅关闭留 30s 排空在途任务、收割训练子进程。健康探针：`/livez`（存活）、`/readyz`（DB + 模型 + GPU 就绪）。

---

## 前端部署

```bash
cd frontend
npm install
```

**开发模式**（Vite 已配置把 `/api` 代理到 `localhost:8000`，本地无需额外配置）：

```bash
npm run dev          # 默认 http://localhost:5173
```

**生产构建**（产出静态文件到 `frontend/dist/`）：

```bash
npm run build
```

`dist/` 可用任意静态服务器托管，或由 FastAPI / Nginx 提供。前端用 **HashRouter**（路由在 `#` 片段里），纯静态托管即可，**无需服务端 SPA 回退配置**。生产环境若前后端不同源，用 Nginx 把 `/api` 反代到后端：

```nginx
location /api/ { proxy_pass http://127.0.0.1:8000; }
location /     { root /path/to/frontend/dist; try_files $uri /index.html; }
```

---

## 大模型 (VLM) 配置

VLM 用于 **Prompt 标准化** 与 **检测结果语义复核**（原理见 [系统功能与技术](docs/系统功能与技术.md#大模型-vlm-工作原理)）。两个模块统一走 **OpenAI 兼容 Chat Completions 协议**，改 `.env` 两个字段即可在本地 / 远端间切换。

**选项 A：本地大模型（默认）** — 指向本机 vLLM 的 MiniCPM-V：

```env
VLM_ENABLED=true
VLM_API_BASE=http://localhost:8010/v1
VLM_MODEL_NAME=MiniCPM-V-4_5
```

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model openbmb/MiniCPM-V-2_6 --served-model-name MiniCPM-V-4_5 \
    --port 8010 --trust-remote-code --dtype auto
```

> 注意端口：vLLM 用 `8010`，避免与后端 `8000` 冲突，并与 `VLM_API_BASE` 一致。也可用 Ollama / LM Studio / Xinference / SGLang 等任意 OpenAI 兼容框架。

**选项 B：远端 API** — 把 `VLM_API_BASE` / `VLM_MODEL_NAME` 改成对应服务：

| 提供商 | `VLM_API_BASE` | `VLM_MODEL_NAME` 示例 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` / `gpt-4o-mini` |
| 阿里云通义 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v` |

> 需鉴权的远端 API：当前 `httpx.Client` 未注入 `Authorization`，需在 `app/services/vlm_service.py` / `prompt_normalizer.py` 的 `httpx.Client(...)` 加 `headers={"Authorization": f"Bearer {settings.VLM_API_KEY}"}` 并在 `config.py` 增 `VLM_API_KEY`。

**关闭 VLM（纯检测模式）**：`.env` 设 `VLM_ENABLED=false`，或单次请求 `POST /api/detect` 带 `"enable_vlm": false`（覆盖全局）。

调优阈值（`VLM_WEIGHT`/`DINO_WEIGHT`/`CONFIRM_THRESHOLD` 等）见 `app/core/config.py`。

---

## 迁移到新设备

> 前提：`.gitignore` 忽略了 `*.pt / *.pth / mobileclip*.ts / weights/`、本地 `.env`、依赖环境及运行数据；TypeScript 的 `.ts` 源码正常保留。`git clone` **只带来代码**，权重 / 数据集 / 训练产物都要手动搬，运行环境按部署步骤重新安装。

完成检查清单：

- [ ] `git clone` 代码，`pip install -r backend/requirements.txt`
- [ ] `yoloe-11l-seg.pt` → `backend/models/yolo/`；`mobileclip_blt.ts` → `backend/`（见 [模型与额外文件](#2-模型与额外文件)）
- [ ] 删除 / 重建 `backend/weights/mobileclip_blt.ts` 断链
- [ ] 改 `backend/.env`：`YOLOE_BASE_MODEL` 绝对路径、`DEVICE`、`DETECTION_MODEL=yoloe`
- [ ] `bash backend/scripts/init_postgres.sh` 建库建角色（写好 `DATABASE_URL`）
- [ ] （可选）拷 `backend/datasets/ annotations/ runs/`，或 `pg_dump` 旧库连数据迁
- [ ] `bash backend/start.sh` → 看到 “DB schema ensured” 且检测能跑通
- [ ] 前端 `cd frontend && npm install && npm run dev`

> ⚠️ **绝对路径警告**：数据库里 `yoloe_dataset_images.stored_path`、`yoloe_training_jobs.dataset_yaml`、`yoloe_trained_models.weights_path` 存的是**绝对路径**。新机若项目根路径与旧机不同，这些历史记录会指向错误位置——最省事是把项目放到**与旧机相同的绝对路径**，否则迁移后需用 SQL 批量改写这些列。
>
> ⚠️ **凭据安全**：`backend/.env` 含真实 DB 口令。若仓库为公开，请尽快轮换口令、`git rm --cached backend/.env`、加入 `.gitignore`、清洗历史并转为私有仓库。

连数据一起迁（可选）：

```bash
pg_dump -U sod_app -h localhost sod > sod_backup.sql        # 旧机导出
psql   -U sod_app -h localhost -d sod < sod_backup.sql       # 新机（先建好空库 sod 再导入）
```

---

## 故障排查

| 现象 | 排查 |
|------|------|
| `Port 8000 already in use` | 已有后端在跑：`ss -ltnp 'sport = :8000'` 查并 kill。 |
| 后端启动即退出 | 多为依赖缺失或 venv/conda 未激活；看终端日志。 |
| `/readyz` 返回 503 | 模型仍在加载，或 DB / GPU 不可用；`/livez` 仍应为 200。 |
| 历史接口 503 / 看不到历史 | PostgreSQL 未运行或 `DATABASE_URL` 错（注意 `+asyncpg` 后缀）；检测主路径不受影响。 |
| "需要 mobileclip_blt.ts 文本编码器" | 文件没放到 `backend/`；见 [模型与额外文件](#2-模型与额外文件)。 |
| 跟踪框 ID 每帧跳变 / 无持久跟踪 | ByteTrack 未装好，已降级为 passthrough；日志若见 `Falling back to passthrough tracker`，按 [安装依赖 · ByteTrack](#1-安装依赖) 装 `cython_bbox`（或 `bytetracker`）后重启。 |
| VLM 调用失败 / 复核不生效 | 确认 vLLM 在 `:8010`、`VLM_API_BASE` 一致；或暂设 `VLM_ENABLED=false`。 |
| 前端有界面但拿不到数据 | 后端未起，或生产环境 `/api` 未反代到后端；开发态确认 Vite 代理。 |
| CUDA OOM | 调小 `batch` / `imgsz`、关闭 SAHI、降低 `VLM_MAX_CONCURRENT`。 |
| 某功能报错 / 想定位问题 | 后端统一返回 `{code,message,detail,request_id}`，响应头带 `X-Request-ID`；用该 request_id `grep backend/logs/app_<日期>.log` 可查完整堆栈（详见 [系统功能与技术 · 错误处理与日志](docs/系统功能与技术.md#错误处理与日志)）。 |

---

## 横向扩展

多 GPU 部署、吞吐优化、Redis / Celery 任务队列等方案见 [系统功能与技术 · 横向扩展](docs/系统功能与技术.md#横向扩展)。最简多 GPU 方案：每张卡起一个 worker，用 Nginx 负载均衡。

```bash
DEVICE=cuda:0 uvicorn app.main:app --port 8000 &
DEVICE=cuda:1 uvicorn app.main:app --port 8001 &
```
