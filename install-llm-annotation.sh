#!/bin/bash
# ============================================================================
# LLM 自动标注（通义千问 Qwen）功能 —— 安装 / 自检脚本
# ----------------------------------------------------------------------------
# 本功能通过阿里云 DashScope 的「OpenAI 兼容端点」调用 Qwen 多模态模型，使用运行
# 环境中已自带的 httpx + Pillow，**无需联网安装额外的 SDK**，适配离线整包部署。
#
# 因此本脚本不再执行任何 pip 安装，只做导入自检，并提示如何配置密钥。
# ============================================================================
set -euo pipefail

SOD_HOME="$(cd "$(dirname "$0")" && pwd)"
PY="$SOD_HOME/env/bin/python"

echo "LLM 自动标注（Qwen）功能自检"
echo "================================"

if [ ! -x "$PY" ]; then
    echo "❌ 未找到运行环境 $PY，请先运行 ./install.sh"
    exit 1
fi

cd "$SOD_HOME/backend"

echo ""
echo "验证模块导入..."
"$PY" - <<'PYEOF'
import sys

try:
    import httpx, PIL  # noqa: F401
    print("✅ 依赖就绪：httpx + Pillow（无需额外 SDK）")
except Exception as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)

try:
    from app.services.llm_annotation import create_annotator, QwenAnnotator, BoundingBox  # noqa: F401
    print("✅ llm_annotation 服务导入成功")
except Exception as e:
    print(f"❌ llm_annotation 服务错误: {e}")
    sys.exit(1)

try:
    from app.api.llm_annotation import router  # noqa: F401
    print("✅ llm_annotation API 导入成功")
except Exception as e:
    print(f"❌ llm_annotation API 错误: {e}")
    sys.exit(1)

try:
    from app.db.models import LLMAnnotationTaskRecord  # noqa: F401
    print("✅ 数据库模型导入成功")
except Exception as e:
    print(f"❌ 数据库模型错误: {e}")
    sys.exit(1)

from app.core.config import settings
print("")
print(f"   当前模型 (LLM_MODEL)     : {settings.LLM_MODEL}")
print(f"   接口地址 (LLM_API_BASE)  : {settings.LLM_API_BASE}")
print(f"   API 密钥已配置           : {'是' if settings.llm_configured else '否（请在 backend/.env 设置 LLM_API_KEY）'}")
PYEOF

echo ""
echo "================================"
echo "自检完成！"
echo ""
echo "下一步："
echo "1. 在 backend/.env 中设置 LLM_API_KEY（如需切换型号，修改 LLM_MODEL）"
echo "2. 启动后端: ./run_backend.sh"
echo "3. 打开前端「LLM标注」页面开始使用"
