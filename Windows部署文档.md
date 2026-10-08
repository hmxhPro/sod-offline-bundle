# SOD Windows 部署文档

面向**没有配置过任何环境**的 Windows 机器。首次安装需联网，脚本会自动下载并就地部署
便携版 **Node.js + PostgreSQL + Miniconda(Python)**，安装 **CUDA 版 PyTorch** 及全部依赖，
初始化本地数据库、构建前端。装完即可断网离线使用，日常靠双击 `.bat` 一键起停。

> 所有组件都装在 `windows\runtime\` 下，**不写系统 PATH、不建服务、不需要管理员**，
> 卸载只需删掉整个文件夹。

---

## 一、前置要求

| 项 | 要求 |
|---|---|
| 系统 | Windows 10 / 11 64 位 |
| 显卡 | NVIDIA 显卡 + 驱动 **≥ 527.41**（对应 CUDA 12.1 运行时）。无显卡见「常见问题①」 |
| 联网 | **仅首次安装需要**（下载组件与依赖，约 4~6 GB） |
| 磁盘 | 建议预留 **15 GB** 以上 |
| 路径 | **必须是纯英文、无空格、尽量短的路径**，推荐直接放到 `C:\sod` |

> ⚠️ **路径很关键**：把整个 `sod-offline-bundle` 文件夹放到形如 `C:\sod` 的位置。
> 若放在 `C:\Users\张三\下载\...` 这种含中文/空格/超长的路径下，Miniconda 与 PyTorch
> 会因 Windows 260 字符路径上限而安装失败。安装脚本会提前检测并拦截。

---

## 二、快速开始（三步）

1. 把整个 `sod-offline-bundle` 文件夹复制到 **`C:\sod`**（或其它纯英文短路径）。
2. 双击 **`安装.bat`** —— 首次联网安装，全程自动，约 15~40 分钟（取决于网速）。
   - 期间会依次部署 Node、PostgreSQL、Miniconda，安装 PyTorch(CUDA) 与后端依赖，
     初始化数据库，构建前端。
   - 结束时若看到绿色「安装完成」即成功；若出现红色 `[X]`，见「常见问题」。
3. 双击 **`启动.bat`** —— 一键起 数据库 + 后端 + 前端，并自动打开浏览器。

> 首次「启动」时后端要预加载检测模型，约 **30~90 秒** 才就绪，请稍候。

---

## 三、四个 .bat 分别做什么

| 双击文件 | 作用 |
|---|---|
| `安装.bat` | 一次性安装/配置全部环境（首次必跑，可重复运行，已装的步骤会跳过） |
| `启动.bat` | **生产模式**：数据库 + 后端(:8000) + 前端(:8080，后端托管已构建页面) + 自动开浏览器 |
| `启动-开发模式.bat` | **开发模式**：数据库 + 后端(:8000) + Vite 热更新前端(:5173)，改前端代码即时生效 |
| `停止.bat` | 停止前端、后端与数据库 |

两种前端模式区别：

- **生产模式（`启动.bat`）**：访问 `http://localhost:8080/`。用 `serve_frontend.py` 托管
  `frontend\dist` 的静态页面并把 `/api` 反代到后端，前后端同源、稳定，日常使用选它。
- **开发模式（`启动-开发模式.bat`）**：访问 `http://localhost:5173/`。用 Vite 开发服务器，
  改 `frontend\src` 下代码浏览器自动热更新，适合二次开发。此模式后端端口须保持 **8000**
  （Vite 代理写死转发到 8000，见 `frontend\vite.config.js`）。

---

## 四、目录与端口

```
C:\sod\                     ← 安装根目录（SOD_HOME）
├─ 安装.bat / 启动.bat / 启动-开发模式.bat / 停止.bat
├─ backend\                 后端代码（.env 由安装脚本生成）
├─ frontend\                前端代码 + dist（安装时构建）
├─ serve_frontend.py        生产模式的前端托管+反代
├─ .config\Ultralytics\     YOLO 字体与配置（中文标注字体）
└─ windows\
   ├─ common.ps1            共享库（路径/版本/端口/函数）
   ├─ install.ps1           安装主脚本
   ├─ start.ps1 / start-dev.ps1 / stop.ps1
   ├─ runtime\              便携组件：node\ pgsql\ miniconda\
   ├─ pgdata\               PostgreSQL 数据目录
   ├─ downloads\            安装包缓存（装完可删，重装可省下载）
   ├─ logs\                 后端/前端/数据库日志
   └─ run\                  记录后端/前端 PID（供停止脚本用）
```

| 服务 | 默认端口 | 覆盖方式（启动前设环境变量） |
|---|---|---|
| 后端 API | 8000 | `SOD_BACKEND_PORT` |
| 前端（生产） | 8080 | `SOD_FRONTEND_PORT` |
| 前端（开发 Vite） | 5173 | `SOD_DEV_PORT` |
| PostgreSQL | 5432 | `SOD_PG_PORT` |

---

## 五、配置文件 `backend\.env`

安装脚本自动生成，关键项：

- `DEVICE=cuda:0` —— 使用第一块 NVIDIA 显卡。无显卡请改成 `DEVICE=cpu`。
- `DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/sod` —— 指向本地便携数据库。
- `YOLOE_BASE_MODEL` / `TRAIN_BASE_MODEL` —— 已指向 `backend/models/yolo/` 下的权重。
- `LLM_API_KEY=` —— **AI 自动标注（通义千问）** 需要在此填入 DashScope API Key，
  留空则该功能提示「未开通」。获取：https://bailian.console.aliyun.com/

改完 `.env` 后重启（`停止.bat` → `启动.bat`）生效。

---

## 六、常见问题

**① 本机没有 NVIDIA 显卡 / 想用 CPU 跑？**
安装仍会装 CUDA 版 PyTorch（不影响安装成功）。安装后把 `backend\.env` 的
`DEVICE=cuda:0` 改为 `DEVICE=cpu`，再启动即可。CPU 下检测与训练明显更慢，但功能完整。

**② 安装时提示「PyTorch 无法使用 CUDA」**
多为显卡驱动过旧。请到 NVIDIA 官网更新驱动到 **≥ 527.41**（`nvidia-smi` 右上角
CUDA Version ≥ 12.1），或按①改用 CPU。若想用更高 CUDA，编辑
`windows\common.ps1` 里的 `$TorchIndexUrl`（如改成 `.../whl/cu124`）后重跑安装。

**③ 报「安装路径包含中文/空格」或装到一半报路径过长**
把整个文件夹移动到 `C:\sod` 这种纯英文、无空格、短路径下重跑 `安装.bat`。

**④ 双击 .bat 一闪而过 / PowerShell 被安全策略拦截**
`.bat` 已用 `-ExecutionPolicy Bypass` 调用，无需改系统策略。若仍被拦，
右键 `.bat` →「以管理员身份运行」试一次；或在 PowerShell 里手动执行
`powershell -ExecutionPolicy Bypass -File windows\install.ps1` 查看完整报错。

**⑤ 端口被占用（8000/8080/5432 等）**
启动脚本会提示「端口已被占用，跳过」。可先 `停止.bat`，或用环境变量换端口，例如：
在 PowerShell 里 `$env:SOD_BACKEND_PORT=8001; .\windows\start.ps1`。
（注意：开发模式的 Vite 代理写死 8000，换后端端口请用生产模式。）

**⑥ 某个下载地址失效导致安装失败**
组件地址集中在 `windows\common.ps1` 顶部（`$NodeFallbackVersion`、
`$PgVersionCandidates`、`$MinicondaUrl`、`$TorchIndexUrl`）。PostgreSQL 已内置多个
候选版本按序尝试；如都失效，改成一个当前可下载的版本号即可。已下载的包缓存在
`windows\downloads\`，重跑不会重复下载。

**⑦ `pip install lap` 失败（缺编译器）**
个别环境无 `lap` 预编译包会尝试本地编译。装上「Microsoft C++ Build Tools」后重跑，
或在 `windows\runtime\miniconda\envs\sod` 下手动 `python -m pip install lap`。

**⑧ 数据库相关**
便携 PostgreSQL 数据在 `windows\pgdata\`，仅监听本机、采用 localhost trust 认证
（本地单机使用，无需密码）。手动操作可用
`windows\runtime\pgsql\bin\psql.exe -h localhost -p 5432 -U postgres -d sod`。
不启用数据库时视频检测仍可用，但**标注、训练、历史记录**功能依赖数据库。

**⑨ 怎么彻底卸载？**
先 `停止.bat`，然后直接删除整个 `C:\sod` 文件夹即可，不残留系统改动。

---

## 七、离线分发说明

首台机器按上面装好后，`C:\sod` 已是完整可离线运行的整套环境
（含 `windows\runtime` 里的 Node/PostgreSQL/Miniconda 与 conda 环境）。理论上可整目录
拷贝到**同架构、路径一致（同为 `C:\sod`）**的另一台 Windows 直接 `启动.bat` 使用——
但 conda 环境对绝对路径较敏感，跨机拷贝若异常，最稳妥仍是在新机重跑 `安装.bat`
（有 `windows\downloads\` 缓存时会省去重复下载）。
