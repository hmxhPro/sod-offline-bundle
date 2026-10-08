#!/usr/bin/env bash
# ============================================================================
# SOD 后端启动脚本（非 systemd 方式）
# ----------------------------------------------------------------------------
# 适用于没有 root / systemd 的环境。使用 conda-pack 解包后的环境（直接调用其
# 内置二进制，不 source activate），注入离线环境变量，并从 backend/ 目录启动 uvicorn。
#
#   前台运行： bash run_backend.sh
#   后台运行： nohup bash run_backend.sh > backend.out 2>&1 &
#   换端口：   PORT=8011 bash run_backend.sh
#
# 本脚本自动以“自身所在目录”为安装根目录（SOD_HOME），无需手工替换路径。
# ============================================================================
set -euo pipefail

SOD_HOME="$(cd "$(dirname "$0")" && pwd)"   # 本脚本位于安装根目录

# ── 离线开关（必须是真实环境变量，应用不会从 .env 注入这些键）──────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export YOLO_OFFLINE=1
export ULTRALYTICS_OFFLINE=1
# ultralytics 会在 YOLO_CONFIG_DIR 下再建 Ultralytics/ 子目录，故字体实际位于
# $SOD_HOME/.config/Ultralytics/（install.sh 已据此放置）。
export YOLO_CONFIG_DIR="$SOD_HOME/.config"
# 应用 visualizer 用独立的 CJK 字体查找逻辑（不走 ultralytics），显式指向随包字体，
# 否则检测框中文标签会因找不到系统 CJK 字体而显示成 ???。
export SOD_CJK_FONT="$SOD_HOME/.config/Ultralytics/Arial.Unicode.ttf"
export OPENCV_FFMPEG_LOGLEVEL=-8
export AV_LOG_FORCE_NOCOLOR=1

# ── 使用 conda-pack 环境（直接用其内置二进制，不 source activate）────────────
# 不用 `source env/bin/activate`：该脚本在 `set -u` 下会因 CONDA_PREFIX 未定义而报错，
# 且 conda-unpack 后的环境二进制已自包含（RPATH 自带 CUDA 库），无需激活即可运行。
export CONDA_PREFIX="$SOD_HOME/env"
export PATH="$SOD_HOME/env/bin:$PATH"

# 允许通过环境变量覆盖监听端口（默认 8000）
PORT="${PORT:-8000}"

# ── 从 backend/ 启动（相对路径 ./uploads ./results 及 mobileclip_blt.ts 依赖 CWD）─
cd "$SOD_HOME/backend"

echo "[run_backend] Python : $SOD_HOME/env/bin/python"
echo "[run_backend] 工作目录: $(pwd)"
echo "[run_backend] 启动 uvicorn :$PORT ..."

exec "$SOD_HOME/env/bin/uvicorn" app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 --timeout-graceful-shutdown 30
