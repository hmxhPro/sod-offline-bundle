#!/usr/bin/env bash
# ============================================================================
# SOD 离线一键安装脚本（在目标服务器上运行）
# ----------------------------------------------------------------------------
# 解包 conda 运行环境、应用代码与模型权重，修正机器相关的绝对路径，
# 并（可选）初始化 PostgreSQL、安装 nginx 站点与 systemd 服务。
#
# 用法：
#   tar -xf sod-offline-bundle.tar -C /data            # 先解开总包
#   cd /data/sod-offline-bundle
#   bash install.sh                                    # 基础安装（解包+配置）
#   bash install.sh --with-db --with-nginx --with-systemd   # 全自动（需 sudo）
#
# 选项：
#   --home DIR        安装/运行根目录（默认：本脚本所在目录）
#   --with-db         运行 init_postgres.sh 初始化任务历史数据库（需 sudo + 运行中的 PostgreSQL）
#   --with-nginx      安装 nginx 反向代理站点（需 sudo + 已安装 nginx）
#   --with-systemd    安装并启用 systemd 服务（需 sudo）
#   -h, --help        显示帮助
# ============================================================================
set -euo pipefail

# ── 解析参数 ────────────────────────────────────────────────────────────────
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
SOD_HOME="$BUNDLE_DIR"
WITH_DB=0; WITH_NGINX=0; WITH_SYSTEMD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --home) SOD_HOME="$(cd "$2" 2>/dev/null && pwd || echo "$2")"; shift 2;;
    --with-db) WITH_DB=1; shift;;
    --with-nginx) WITH_NGINX=1; shift;;
    --with-systemd) WITH_SYSTEMD=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done
RUN_USER="${SUDO_USER:-$(id -un)}"

say(){ echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok(){  echo -e "  \033[32m✓\033[0m $*"; }
warn(){ echo -e "  \033[33m!\033[0m $*"; }
die(){ echo -e "\033[31m✗ $*\033[0m" >&2; exit 1; }

echo "════════════════════════════════════════════════════════════"
echo " SOD 离线安装"
echo "   总包目录 : $BUNDLE_DIR"
echo "   安装目录 : $SOD_HOME"
echo "   运行用户 : $RUN_USER"
echo "════════════════════════════════════════════════════════════"

# ── 0. 前置检查 ─────────────────────────────────────────────────────────────
say "0. 前置检查"
[ -f "$BUNDLE_DIR/conda/SOD_env.tar.gz" ] || die "未找到 conda/SOD_env.tar.gz，请确认在总包目录内运行"
[ -f "$BUNDLE_DIR/app/SOD-app.tar.gz" ]   || die "未找到 app/SOD-app.tar.gz"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "检测到 nvidia-smi：$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1)"
  echo "    （注意：这只确认驱动命令存在，不代表 PyTorch 能用 CUDA；权威检查在第 1 步解包后进行）"
else
  warn "未检测到 nvidia-smi。本系统需要 NVIDIA GPU + 驱动 R580+（CUDA 13.0）。CPU 模式极慢且未经测试。"
fi
# 校验和（若提供）
if [ -f "$BUNDLE_DIR/checksums.sha256" ]; then
  ( cd "$BUNDLE_DIR" && sha256sum -c --quiet checksums.sha256 ) && ok "校验和通过" || warn "校验和不匹配，文件可能在传输中损坏"
fi
mkdir -p "$SOD_HOME"

# ── 1. 解包 conda 运行环境 + conda-unpack ──────────────────────────────────
say "1. 解包 Python 运行环境（conda-pack）"
if [ -x "$SOD_HOME/env/bin/python" ] && [ -f "$SOD_HOME/env/.unpacked" ]; then
  ok "env 已存在且已 unpack，跳过"
else
  mkdir -p "$SOD_HOME/env"
  tar -xzf "$BUNDLE_DIR/conda/SOD_env.tar.gz" -C "$SOD_HOME/env"
  ok "已解包到 $SOD_HOME/env"
  # conda-unpack 修正环境内所有写死的前缀路径
  set +u; source "$SOD_HOME/env/bin/activate"; set -u
  "$SOD_HOME/env/bin/conda-unpack" && touch "$SOD_HOME/env/.unpacked"
  set +u; source "$SOD_HOME/env/bin/deactivate" 2>/dev/null || true; set -u
  ok "conda-unpack 完成（路径已重定位到 $SOD_HOME/env）"
fi

# 权威 GPU 检查：真正用 torch 初始化 CUDA（与 verify.sh 一致；不阻断安装）
GPU_OK=$("$SOD_HOME/env/bin/python" -c "import torch;print(torch.cuda.is_available())" 2>/dev/null || echo False)
if [ "$GPU_OK" = "True" ]; then
  ok "PyTorch CUDA 可用：$("$SOD_HOME/env/bin/python" -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null)"
else
  warn "PyTorch 无法使用 CUDA（torch.cuda.is_available()=False）。常见原因：NVIDIA 驱动低于 R580（cu130 要求）。"
  warn "稍后用 'bash $SOD_HOME/verify.sh' 查看真实报错；对比 'nvidia-smi' 顶部 'CUDA Version' 是否 ≥ 13.0。"
fi

# ── 2. 解包应用代码 + 模型权重 ──────────────────────────────────────────────
say "2. 解包应用代码与模型权重"
tar -xzf "$BUNDLE_DIR/app/SOD-app.tar.gz" -C "$SOD_HOME"
ok "已解包 backend/ 与 frontend/ 到 $SOD_HOME"
[ -s "$SOD_HOME/backend/mobileclip_blt.ts" ] || warn "缺失 backend/mobileclip_blt.ts（YOLOE 文本编码器，离线必需）"

# 修正历史遗留的损坏软链：backend/weights/mobileclip_blt.ts → 指向真实文件
if [ -L "$SOD_HOME/backend/weights/mobileclip_blt.ts" ] || [ ! -e "$SOD_HOME/backend/weights/mobileclip_blt.ts" ]; then
  mkdir -p "$SOD_HOME/backend/weights"
  ln -sfn ../mobileclip_blt.ts "$SOD_HOME/backend/weights/mobileclip_blt.ts"
  ok "已修正 backend/weights/mobileclip_blt.ts 软链 → ../mobileclip_blt.ts"
fi

# ── 3. Ultralytics 字体 / 配置 ──────────────────────────────────────────────
say "3. 部署 Ultralytics 字体与配置"
# ultralytics 在 YOLO_CONFIG_DIR(=$SOD_HOME/.config) 下使用 Ultralytics/ 子目录，
# 字体与 settings.json 必须放在该子目录内，否则离线 check_font 找不到字体。
CFGDIR="$SOD_HOME/.config/Ultralytics"
mkdir -p "$CFGDIR"
if [ -f "$BUNDLE_DIR/ultralytics/Ultralytics.tar.gz" ]; then
  tar -xzf "$BUNDLE_DIR/ultralytics/Ultralytics.tar.gz" -C "$CFGDIR"
fi
# 离线安全：关闭遥测/同步/hub（防止任何联网尝试）
if [ -f "$CFGDIR/settings.json" ]; then
  "$SOD_HOME/env/bin/python" - "$CFGDIR/settings.json" "$SOD_HOME/backend" <<'PY' || true
import json, sys
p, backend = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception:
    d = {}
d.update({"sync": False, "hub": False, "clearml": False, "comet": False,
          "dvc": False, "mlflow": False, "neptune": False, "raytune": False,
          "datasets_dir": f"{backend}/datasets", "weights_dir": "weights", "runs_dir": "runs"})
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
PY
fi
[ -s "$CFGDIR/Arial.Unicode.ttf" ] && ok "字体就绪（含中文 Arial.Unicode.ttf）→ $CFGDIR" || warn "缺失 Arial.Unicode.ttf，中文标注会回退/失败"

# ── 4. 生成 backend/.env（替换机器相关绝对路径）────────────────────────────
say "4. 生成后端配置 backend/.env"
ENV_OUT="$SOD_HOME/backend/.env"
if [ -f "$ENV_OUT" ]; then
  warn ".env 已存在，备份为 .env.bak 后覆盖"; cp -f "$ENV_OUT" "$ENV_OUT.bak"
fi
sed "s|__SOD_HOME__|$SOD_HOME|g" "$BUNDLE_DIR/env.template" > "$ENV_OUT"
chmod 600 "$ENV_OUT"
ok "已写入 $ENV_OUT（YOLOE_BASE_MODEL / TRAIN_BASE_MODEL 绝对路径已指向本机）"

# ── 5. 模板化运行脚本 ───────────────────────────────────────────────────────
say "5. 安装运行脚本"
# 注意：默认安装目录 == 总包目录时，源文件与目标文件是同一路径，
# 直接 `sed src > dst` 会先清空再读取 → 文件变空。故经临时文件中转。
for f in run_backend.sh verify.sh; do
  tmp="$(mktemp)"
  sed "s|__SOD_HOME__|$SOD_HOME|g" "$BUNDLE_DIR/$f" > "$tmp"
  mv -f "$tmp" "$SOD_HOME/$f"
  chmod +x "$SOD_HOME/$f"
done
ok "已生成 $SOD_HOME/run_backend.sh 与 verify.sh"
# 前端服务脚本（自解析路径，无需模板替换）
if [ "$BUNDLE_DIR/serve_frontend.py" != "$SOD_HOME/serve_frontend.py" ] && [ -f "$BUNDLE_DIR/serve_frontend.py" ]; then
  cp -f "$BUNDLE_DIR/serve_frontend.py" "$SOD_HOME/serve_frontend.py"
fi
[ -f "$SOD_HOME/serve_frontend.py" ] && chmod +x "$SOD_HOME/serve_frontend.py" && ok "前端服务脚本就绪：$SOD_HOME/serve_frontend.py"

# ── 6.（可选）PostgreSQL ────────────────────────────────────────────────────
if [ "$WITH_DB" -eq 1 ]; then
  say "6. 初始化 PostgreSQL 任务历史库"
  if command -v psql >/dev/null 2>&1 && pg_isready -q 2>/dev/null; then
    ( cd "$SOD_HOME/backend" && bash scripts/init_postgres.sh ) && ok "数据库初始化完成，DATABASE_URL 已写入 .env"
  else
    warn "未检测到运行中的 PostgreSQL。请先安装并启动 PostgreSQL 后再执行： cd backend && bash scripts/init_postgres.sh"
  fi
else
  warn "跳过数据库（--with-db 可启用）。不配数据库后端仍可检测，仅历史记录不可用。"
fi

# ── 7.（可选）nginx 反向代理 ────────────────────────────────────────────────
if [ "$WITH_NGINX" -eq 1 ]; then
  say "7. 安装 nginx 站点"
  if command -v nginx >/dev/null 2>&1; then
    sed "s|__SOD_HOME__|$SOD_HOME|g" "$BUNDLE_DIR/nginx-sod.conf" | sudo tee /etc/nginx/conf.d/sod.conf >/dev/null
    sudo nginx -t && sudo systemctl reload nginx && ok "nginx 站点已生效（http://<服务器IP>/）"
  else
    warn "未安装 nginx。请离线安装 nginx 后手动： sudo cp <(sed 's|__SOD_HOME__|$SOD_HOME|g' nginx-sod.conf) /etc/nginx/conf.d/sod.conf"
  fi
fi

# ── 8.（可选）systemd 服务 ──────────────────────────────────────────────────
if [ "$WITH_SYSTEMD" -eq 1 ]; then
  say "8. 安装 systemd 服务"
  sed -e "s|__SOD_HOME__|$SOD_HOME|g" -e "s|__SOD_USER__|$RUN_USER|g" \
      "$BUNDLE_DIR/sod-backend.service" | sudo tee /etc/systemd/system/sod-backend.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now sod-backend && ok "服务已启动： systemctl status sod-backend"
fi

# ── 完成 ────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " ✅ 基础安装完成。后续："
echo "   自检：    bash $SOD_HOME/verify.sh"
[ "$WITH_SYSTEMD" -eq 0 ] && echo "   启动后端：nohup bash $SOD_HOME/run_backend.sh > $SOD_HOME/backend.out 2>&1 &"
[ "$WITH_DB" -eq 0 ]      && echo "   （可选）数据库： cd $SOD_HOME/backend && bash scripts/init_postgres.sh"
[ "$WITH_NGINX" -eq 0 ]   && echo "   启动前端：nohup $SOD_HOME/env/bin/python $SOD_HOME/serve_frontend.py > $SOD_HOME/frontend.out 2>&1 &  （默认 :8080，访问 http://<服务器IP>:8080/）"
echo "   详细说明： $SOD_HOME/部署文档.md"
echo "════════════════════════════════════════════════════════════"
