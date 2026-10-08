# LLM 自动标注功能使用指南（通义千问 Qwen）

## 功能简介

调用**通义千问（Qwen）多模态模型**自动标注数据集图片，生成 YOLOv11 兼容的标注格式。

提供商、接口地址、模型型号与 API 密钥**全部在后端统一配置**（固定为 Qwen，经阿里云
DashScope 的 OpenAI 兼容端点访问）。前端不选择提供商、不填写密钥 —— 使用者只需在
`backend/.env` 中维护密钥与型号即可。

## 工作流程

1. **上传图片** → 在「模型训练」页面创建类别并上传图片
2. **配置一次** → 在 `backend/.env` 中填好 `LLM_API_KEY`（首次使用）
3. **创建任务** → 在「LLM标注」页面选择类别、填写目标描述，点击「开始标注」
4. **执行标注** → 后台异步批量调用 Qwen API
5. **训练模型** → 标注完成后返回「模型训练」页面启动训练

## 依赖说明

本功能通过 DashScope 的 OpenAI 兼容端点直接发起 HTTP 请求，使用运行环境中**已自带的
`httpx` + `Pillow`**，**无需联网安装** openai / anthropic / aiohttp 等 SDK，适配离线整包
部署。可运行自检脚本确认：

```bash
./install-llm-annotation.sh
```

## 后端配置（backend/.env）

```bash
# DashScope（百炼）API Key，形如 sk-xxxxxxxx；留空则该功能在前端提示“未配置”
LLM_API_KEY=sk-your-key

# Qwen 多模态模型型号（按需修改即可切换）
#   qwen-vl-max-latest      —— 能力最强，推荐（默认）
#   qwen-vl-plus-latest     —— 更快更省，适合大批量
#   qwen2.5-vl-72b-instruct —— 指定版本
LLM_MODEL=qwen-vl-max-latest

# 以下一般无需改动
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MAX_CONCURRENT=3       # 并发请求数（控制速率，避免限流）
LLM_IMAGE_MAX_SIZE=1024    # 送入模型前图片的最长边（像素）
LLM_TIMEOUT_SECONDS=120    # 单张图片请求超时（秒）
```

> 修改 `.env` 后需**重启后端**（`./run_backend.sh`）才会生效。
>
> API Key 获取：登录 https://bailian.console.aliyun.com/ ，开通后在「API-KEY」中创建。

## 使用示例

### 前端操作

1. 打开「LLM标注」页面
2. 顶部 / 右侧会显示当前生效的模型（来自后端配置，只读）
3. 选择已创建的类别（需先在「模型训练」页面创建并上传图片）
4. 填写**目标描述提示词**，例如「小型无人机」「红色安全帽」
5. （可选）勾选「仅标注未标注的图片」
6. 点击「开始标注」，实时查看进度与结果

### API 调用示例

```python
import requests, time

BASE = "http://localhost:8000"

# 0. （可选）查看后端配置 —— 不会返回密钥
cfg = requests.get(f"{BASE}/api/annotation-config").json()
print(cfg)  # {"provider": "qwen", "model": "...", "configured": true, "max_concurrent": 3}

# 1. 创建标注任务（无需传提供商/密钥/模型）
resp = requests.post(f"{BASE}/api/annotation-tasks", json={
    "category_id": "your-category-id",
    "prompt": "小型无人机",
    "only_pending": True,
})
task_id = resp.json()["id"]

# 2. 轮询进度
while True:
    s = requests.get(f"{BASE}/api/annotation-tasks/{task_id}").json()["task"]
    print(f"进度: {s['progress']*100:.1f}%")
    if s["status"] in ("finished", "failed", "cancelled"):
        break
    time.sleep(3)

print(f"成功 {s['success_count']} / 失败 {s['failed_count']} / 框数 {s['total_boxes']}")
```

## 提示词编写技巧

### ✅ 好的提示词

- **具体清晰**: 「红色安全帽」> 「安全帽」
- **包含特征**: 「小型四旋翼无人机」> 「无人机」
- **避免歧义**: 「施工车辆（挖掘机、推土机）」> 「车辆」

### ❌ 避免的写法

- 过于模糊: 「目标物体」「异物」
- 过于复杂: 「红色或黄色的安全帽，但不包括蓝色的…」
- 包含位置: 「画面左上角的无人机」（模型需要找到所有目标）

## 常见问题

### 1. 前端提示「未配置 LLM 密钥」/ 创建任务返回 400

- 检查 `backend/.env` 是否设置了 `LLM_API_KEY`
- 修改 `.env` 后是否**重启了后端**

### 2. 任务失败，错误含 `401` / `Invalid API-key`

- 密钥不正确或已失效，确认从百炼控制台复制完整（注意首尾空格）
- 确认账号已开通 DashScope 模型服务且额度充足

### 3. 任务失败，错误含 `429` / 限流

- 调低 `LLM_MAX_CONCURRENT`（如 3 → 1）
- 稍后重试，或提升账号限额

### 4. 标注框数为 0

- 图片中可能确实不存在目标，或提示词不够清晰
- 尝试更具体的目标描述、确认图片清晰

## 标注质量优化

1. **人工审核**：标注完成后在「模型训练」页面抽查、修正、剔除差样本
2. **混合标注**：困难样本人工标，简单样本交给 Qwen，最终人工审核
3. **迭代优化**：自动标注 → 训练初版 → 找错 → 人工修正 → 重训练

## 技术原理

1. **图片预处理**：按最长边缩放到 `LLM_IMAGE_MAX_SIZE`（默认 1024px）以节省 token
2. **提示词构建**：系统提示定义任务与 JSON 输出格式；用户提示给出目标描述
3. **模型推理**：经 DashScope OpenAI 兼容端点 `/chat/completions` 调用 Qwen-VL，返回
   像素坐标的边界框 JSON
4. **坐标转换**：像素坐标 → 归一化(0-1) → YOLO 格式(cx, cy, w, h)
5. **持久化**：保存为 `annotations/<category_id>/<image_id>.txt`

## 数据库表结构

```sql
CREATE TABLE yoloe_llm_annotation_tasks (
    id VARCHAR(36) PRIMARY KEY,
    category_id VARCHAR(36) NOT NULL,
    llm_provider VARCHAR(32) NOT NULL,   -- 固定为 'qwen'
    llm_model VARCHAR(128) NOT NULL,     -- 创建任务时的 LLM_MODEL 快照
    prompt TEXT NOT NULL,
    status VARCHAR(32) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    total_images INT DEFAULT 0,
    processed_images INT DEFAULT 0,
    success_count INT DEFAULT 0,
    failed_count INT DEFAULT 0,
    total_boxes INT DEFAULT 0,
    error TEXT,
    result_summary JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);
```

## API 端点

### GET /api/annotation-config
返回后端固定配置（`provider` / `model` / `configured` / `max_concurrent`）。**绝不返回密钥**。

### POST /api/annotation-tasks
创建标注任务。**请求体**（提供商/密钥/模型由后端决定，无需传入）：

```json
{
  "category_id": "uuid",
  "prompt": "小型无人机",
  "max_concurrent": 3,
  "only_pending": true
}
```

未配置 `LLM_API_KEY` 时返回 `400`。

### GET /api/annotation-tasks/{task_id}
查询任务详情。

### GET /api/annotation-tasks?category_id=uuid
列出标注任务（可按类别筛选）。

### POST /api/annotation-tasks/{task_id}/cancel
取消运行中的任务。

## 安全提示

⚠️ **API 密钥安全**：
- 密钥仅保存在后端 `backend/.env`，**不会**下发到前端，也不写入数据库
- `.env` 已在 `.gitignore` 中，请勿提交到代码仓库
- 建议使用受限权限的密钥并定期轮换

## 未来改进方向

- [ ] 支持本地部署的开源多模态模型（LLaVA、CogVLM 等）
- [ ] 批量图片打包发送（减少 API 调用次数）
- [ ] 标注结果可视化预览
- [ ] 主动学习（模型辅助选择困难样本）
