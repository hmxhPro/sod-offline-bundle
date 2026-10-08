# LLM自动标注功能实现总结

## 功能概述

基于**通义千问（Qwen）多模态模型**的数据集自动标注系统，自动标注上传的图片并生成
YOLOv11 兼容的标注格式。提供商、接口地址、模型型号与 API 密钥**全部在后端 `.env` 中
集中配置**（硬编码为 Qwen，经阿里云 DashScope 的 OpenAI 兼容端点访问），前端不选择
提供商、不填写密钥。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                         前端界面                              │
│  /llm-annotation - LLM标注配置和任务管理                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP API
┌──────────────────────┴──────────────────────────────────────┐
│                      后端服务                                 │
├─────────────────────────────────────────────────────────────┤
│  FastAPI路由层                                               │
│  - GET  /api/annotation-config         读取后端固定配置     │
│  - POST /api/annotation-tasks          创建标注任务         │
│  - GET  /api/annotation-tasks/{id}     查询任务状态         │
│  - GET  /api/annotation-tasks          列出所有任务         │
│  - POST /api/annotation-tasks/{id}/cancel  取消任务         │
├─────────────────────────────────────────────────────────────┤
│  服务层 (app/services/llm_annotation.py)                    │
│  - LLMAnnotator基类                    统一接口             │
│  - QwenAnnotator                       Qwen适配器(httpx)    │
│  - create_annotator()                  读取settings工厂     │
│  - annotate_images_batch()             批量标注协程         │
├─────────────────────────────────────────────────────────────┤
│  数据库层                                                     │
│  - LLMAnnotationTaskRecord             任务记录表           │
│  - DatasetImageRecord                  图片状态更新         │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              通义千问 Qwen（DashScope OpenAI 兼容端点）       │
│  POST {LLM_API_BASE}/chat/completions  （httpx 直连，无SDK） │
└─────────────────────────────────────────────────────────────┘
```

## 实现文件

### 后端核心文件

1. **app/services/llm_annotation.py**
   - `QwenAnnotator`（继承 `LLMAnnotator` 基类）+ `create_annotator()` 工厂
   - 通过 httpx 直连 DashScope OpenAI 兼容端点，无第三方 SDK 依赖
   - 图片预处理、API调用、坐标转换、批量异步标注逻辑

2. **app/api/llm_annotation.py**
   - FastAPI路由端点（含 `GET /annotation-config`）
   - 后台任务执行器、进度跟踪与状态管理
   - 提供商/模型/密钥全部取自 `settings`（后端 .env）

3. **app/core/config.py** (新增)
   - `LLM_API_KEY` / `LLM_MODEL` / `LLM_API_BASE` 等配置
   - `llm_configured` 属性

4. **app/db/models.py**
   - `LLMAnnotationTaskRecord` 表定义
   - 任务状态、进度、统计信息

5. **app/main.py**
   - 注册LLM标注路由

### 前端核心文件

6. **frontend/src/pages/LLMAnnotationPage.jsx**
   - 类别选择 + 提示词输入（不含提供商/密钥选择）
   - 只读展示后端配置的模型、实时进度、任务列表

7. **frontend/src/services/api.js** (修改)
   - `getAnnotationConfig` / `createAnnotationTask` / `getAnnotationTasks` / `cancelAnnotationTask`

8. **frontend/src/routes.jsx · Sidebar.jsx**
   - `/llm-annotation` 路由与「LLM标注」导航项

### 配置和文档

9. **backend/.env · env.template · .env.example**
   - `LLM_API_KEY` / `LLM_MODEL` 等配置项

10. **docs/LLM_ANNOTATION_GUIDE.md**
    - 功能介绍、后端配置、提示词技巧、常见问题

11. **install-llm-annotation.sh**
    - 离线自检脚本（无需联网安装 SDK）

12. **backend/tests/test_llm_annotation_manual.py**
    - 手动/离线测试脚本

## 数据流

```
1. 用户上传图片 → DatasetImageRecord (annotation_status='pending')
                    ↓
2. 创建标注任务 → LLMAnnotationTaskRecord (status='pending')
                    ↓
3. 后台执行标注 → LLM API调用（批量并发）
                    ↓
4. 解析响应     → JSON → BoundingBox对象
                    ↓
5. 保存标注     → annotations/{category_id}/{image_id}.txt
                    ↓
6. 更新状态     → DatasetImageRecord (annotation_status='annotated', box_count=N)
                    ↓
7. 任务完成     → LLMAnnotationTaskRecord (status='finished', 统计信息)
```

## 关键技术点

### 1. 异步批量处理

```python
async def annotate_images_batch(
    annotator: LLMAnnotator,
    images: List[Dict],
    prompt: str,
    category_id: str,
    max_concurrent: int = 3,
):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _annotate_one(img):
        async with semaphore:
            # 控制并发数，避免API速率限制
            return await annotator.annotate_image(...)
    
    results = await asyncio.gather(*[_annotate_one(img) for img in images])
```

### 2. 坐标转换

```python
# 像素坐标 → 归一化坐标
x1_norm = x1 / image_width
y1_norm = y1 / image_height

# 归一化坐标 → YOLO格式 (cx, cy, w, h)
cx = (x1_norm + x2_norm) / 2.0
cy = (y1_norm + y2_norm) / 2.0
w = x2_norm - x1_norm
h = y2_norm - y1_norm
```

### 3. 统一LLM接口

```python
class LLMAnnotator:
    async def annotate_image(
        self, image_path, prompt, image_width, image_height
    ) -> List[BoundingBox]:
        # 子类实现具体的API调用逻辑
        raise NotImplementedError
```

### 4. 后台任务管理

```python
# 创建后台任务
asyncio_task = asyncio.create_task(_run_annotation_task(...))
_running_tasks[task_id] = asyncio_task

# 取消任务
_running_tasks[task_id].cancel()
```

### 5. 前端轮询

```javascript
useEffect(() => {
  const interval = setInterval(() => {
    if (tasks.some(t => t.status === 'running')) {
      fetchTasks(); // 每3秒刷新一次
    }
  }, 3000);
  return () => clearInterval(interval);
}, [tasks]);
```

## 使用流程

### 1. 配置后端密钥（首次）

在 `backend/.env` 中设置：

```bash
LLM_API_KEY=sk-your-dashscope-key   # 必填
LLM_MODEL=qwen-vl-max-latest        # 按需切换型号
```

可运行 `./install-llm-annotation.sh` 做导入自检（无需联网装 SDK）。

### 2. 启动服务

```bash
# 后端
./run_backend.sh

# 前端（另一个终端）
cd frontend
npm run dev
```

### 3. 创建标注任务

1. 打开「LLM标注」页面（页面会只读展示当前生效的 Qwen 模型）
2. 选择类别（需先在"模型训练"页面创建并上传图片）
3. 填写目标描述提示词，例如"小型无人机"
4. 点击"开始标注"
5. 实时查看进度

### 4. 训练模型

标注完成后，返回"模型训练"页面：
- 查看标注统计
- 启动训练任务
- 使用训练好的模型进行检测

## API端点示例

### 创建任务

提供商/模型/密钥由后端 `.env` 决定，请求体无需传入：

```bash
curl -X POST http://localhost:8000/api/annotation-tasks \
  -H "Content-Type: application/json" \
  -d '{
    "category_id": "your-category-id",
    "prompt": "小型无人机",
    "max_concurrent": 3,
    "only_pending": true
  }'
```

### 查询进度

```bash
curl http://localhost:8000/api/annotation-tasks/{task_id}
```

### 列出任务

```bash
curl http://localhost:8000/api/annotation-tasks?category_id={category_id}
```

## 数据库表结构

```sql
CREATE TABLE yoloe_llm_annotation_tasks (
    id VARCHAR(36) PRIMARY KEY,
    category_id VARCHAR(36) NOT NULL,
    llm_provider VARCHAR(32) NOT NULL,     -- 固定为 'qwen'
    llm_model VARCHAR(128) NOT NULL,       -- 创建任务时的 LLM_MODEL 快照
    prompt TEXT NOT NULL,
    status VARCHAR(32) DEFAULT 'pending',  -- pending/running/finished/failed/cancelled
    progress FLOAT DEFAULT 0.0,            -- 0.0-1.0
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

## 成本估算

| 场景 | 图片数 | 模型 | 估算成本 |
|------|--------|------|---------|
| 小规模 | 100张 | qwen-vl-plus | 约 ¥2 |
| 中规模 | 500张 | qwen-vl-max | 约 ¥25 |
| 大规模 | 2000张 | qwen-vl-plus | 约 ¥40 |

*实际成本取决于图片大小、检测目标数量、提示词长度，以及 DashScope 当时的计费标准*

## 性能优化

1. **并发控制**: `LLM_MAX_CONCURRENT`（默认3），可按 API 速率限制调整
2. **图片预处理**: 按 `LLM_IMAGE_MAX_SIZE`（默认1024px）缩放，减少token消耗
3. **进度持久化**: 任务状态保存到数据库
4. **错误恢复**: 单张失败不影响整体流程

## 扩展方向

- [ ] 支持本地部署的开源多模态模型（LLaVA、CogVLM）
- [ ] 批量图片打包发送（减少API调用次数）
- [ ] 标注结果可视化预览
- [ ] 主动学习（选择困难样本）
- [ ] 标注质量自动评估

## 安全考虑

⚠️ **API密钥安全**:
- 密钥仅保存在后端 `backend/.env`，不下发前端、不写入数据库
- `.env` 已在 `.gitignore` 中，请勿提交到仓库
- 建议使用受限权限的密钥并定期轮换

## 测试

```bash
# 离线测试（工厂/解析/边界框，无需密钥）
cd backend
echo 5 | python tests/test_llm_annotation_manual.py

# 导入自检
cd ..
./install-llm-annotation.sh
```

## 故障排除

### 问题1: 前端提示「未配置 LLM 密钥」/ 创建任务返回 400

**解决**: 在 `backend/.env` 设置 `LLM_API_KEY` 后**重启后端**

### 问题2: API密钥无效（401 / Invalid API-key）

**解决**:
- 检查密钥是否从百炼控制台完整复制（注意首尾空格）
- 确认账号已开通 DashScope 模型服务且额度充足

### 问题3: 标注框为空

**原因**: 图片中不存在目标，或提示词不够清晰

**解决**: 使用更具体的描述，如"红色安全帽"而非"帽子"

## 总结

已完整实现基于通义千问（Qwen）多模态模型的数据集自动标注功能：

✅ **后端服务** - Qwen 适配器（httpx 直连，无第三方 SDK）  
✅ **集中配置** - 提供商/模型/密钥全部在后端 `.env`，前端零选择  
✅ **API端点** - 完整的RESTful接口（含 `/annotation-config`）  
✅ **数据库模型** - 任务状态持久化  
✅ **前端界面** - 风格与其他页面统一，只读展示当前模型  
✅ **异步处理** - 批量并发标注  
✅ **进度跟踪** - 实时状态更新  
✅ **文档和测试** - 使用指南与离线测试脚本

**下一步**: 在 `backend/.env` 设置 `LLM_API_KEY`，重启后端，然后打开「LLM标注」页面开始使用！
