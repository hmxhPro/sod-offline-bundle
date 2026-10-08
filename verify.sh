#!/usr/bin/env bash
# ============================================================================
# SOD 离线部署自检脚本
# ----------------------------------------------------------------------------
# 在目标机安装完成后运行，逐项检查关键依赖是否就绪。
#   bash verify.sh
# /home/hmxh/Workspace/Project/WorkSpace/sod-offline-bundle 由 install.sh 替换；若手动运行，可 export SOD_HOME=/path 后执行。
# ============================================================================
set -uo pipefail

SOD_HOME="${SOD_HOME:-$(cd "$(dirname "$0")" && pwd)}"
PY="$SOD_HOME/env/bin/python"
PASS=0; FAIL=0
ok(){ echo -e "  \033[32m✓\033[0m $1"; PASS=$((PASS+1)); }
no(){ echo -e "  \033[31m✗\033[0m $1"; FAIL=$((FAIL+1)); }

echo "════════════════════════════════════════════════════════════"
echo " SOD 离线部署自检   (SOD_HOME=$SOD_HOME)"
echo "════════════════════════════════════════════════════════════"

echo "[1/8] Python 环境"
if [ -x "$PY" ]; then ok "conda 环境已解包: $PY"; else no "未找到 $PY（conda 环境未解包或路径不对）"; fi

echo "[2/8] 核心 Python 依赖"
"$PY" - <<'PY' 2>/tmp/sod_verify_imp.err && ok "torch / cv2 / ultralytics / fastapi / sqlalchemy / asyncpg 导入成功" || { no "依赖导入失败，详见下方"; sed 's/^/      /' /tmp/sod_verify_imp.err; }
import torch, torchvision, cv2, ultralytics, fastapi, uvicorn, sqlalchemy, asyncpg, supervision, lap, sahi
from ultralytics import YOLO
PY

echo "[3/8] GPU / CUDA"
GPU_OUT=$("$PY" - <<'PY' 2>&1
import torch
print("AVAIL", torch.cuda.is_available())
print("INFO  torch", torch.__version__, "| built for CUDA", torch.version.cuda)
if torch.cuda.is_available():
    print("DEVICE", torch.cuda.get_device_name(0))
else:
    try:
        torch.zeros(1).cuda()
    except Exception as e:
        msg = (str(e) or repr(e)).splitlines()[0]
        print("REASON", type(e).__name__, "-", msg)
PY
)
if echo "$GPU_OUT" | grep -q "AVAIL True"; then
  ok "CUDA 可用 → $(echo "$GPU_OUT" | grep '^DEVICE' | cut -d' ' -f2-) ($(echo "$GPU_OUT" | grep '^INFO'))"
else
  no "torch.cuda.is_available()=False — 真实原因如下："
  echo "$GPU_OUT" | sed 's/^/      /'
  echo "      ↳ 多数是驱动过旧：cu130 需 NVIDIA 驱动 R580+；运行 'nvidia-smi' 看顶部 'CUDA Version' 是否 ≥ 13.0"
fi

echo "[4/8] 模型权重"
for f in backend/models/yolo/yoloe-11l-seg.pt backend/models/yolo/yolo11l.pt backend/mobileclip_blt.ts; do
  if [ -s "$SOD_HOME/$f" ]; then ok "存在 $f ($(du -h "$SOD_HOME/$f"|cut -f1))"; else no "缺失 $f"; fi
done

echo "[5/8] Ultralytics 字体（离线标注必需）"
CFG="${YOLO_CONFIG_DIR:-$SOD_HOME/.config}/Ultralytics"
for f in Arial.ttf Arial.Unicode.ttf; do
  if [ -s "$CFG/$f" ]; then ok "存在字体 $CFG/$f"; else no "缺失字体 $CFG/$f（离线无法下载，标注会失败）"; fi
done

echo "[6/8] 前端构建产物"
if [ -s "$SOD_HOME/frontend/dist/index.html" ]; then ok "frontend/dist 已就绪"; else no "缺失 frontend/dist/index.html"; fi

echo "[7/8] PostgreSQL（可选 — 任务历史）"
if command -v pg_isready >/dev/null 2>&1 && pg_isready -q 2>/dev/null; then
  ok "PostgreSQL 正在监听 5432"
else
  echo -e "  \033[33m·\033[0m PostgreSQL 未运行（后端仍可检测，仅历史记录不可用）"
fi

echo "[8/8] 后端 HTTP 健康检查（若已启动）"
if command -v curl >/dev/null 2>&1; then
  H=$(curl -s -m 5 http://127.0.0.1:8000/readyz 2>/dev/null || true)
  if [ -n "$H" ]; then ok "/readyz 响应: $H"; else echo -e "  \033[33m·\033[0m 后端未响应（尚未启动？先启动服务再复测）"; fi
else
  echo -e "  \033[33m·\033[0m 未安装 curl，跳过 HTTP 检查"
fi

echo "────────────────────────────────────────────────────────────"
echo " 结果：通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] && echo " ✅ 关键项全部通过。" || echo " ⚠️  存在失败项，请按上方提示排查后再启动服务。"
echo "════════════════════════════════════════════════════════════"
exit 0
