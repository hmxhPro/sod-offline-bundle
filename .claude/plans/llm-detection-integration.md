# LLM 多模态检测集成计划

## 需求概述

在图片检测和视频检测页面中添加 LLM 多模态检测支持，用户可以选择：
1. **YOLO 检测**（使用已训练模型）- 不支持自然语言输入
2. **自然语言检测**（使用多模态 LLM）- 可以选择模型

## 当前架构分析

### 后端现状
- **image_detect.py**: 支持两种模式
  - `model_id` - 使用训练好的 YOLO 模型
  - `class_names` - 零样本检测（GroundingDINO）
- **llm_annotation.py**: 已有完整的 LLM 标注实现
  - `QwenAnnotator` 类封装了通义千问 API 调用
  - 返回边界框 `BoundingBox(class_id, cx, cy, w, h)`

### 前端现状
- **ImageDetectPage.jsx**: 支持选择模型或自然语言输入
  - 使用 `DetectModelSelect` 组件选择模型
  - 或输入自然语言类别名
- **DetectPage.jsx**: 视频检测，类似逻辑

## 实施方案

### 方案设计

#### 1. 检测方式选择
在 UI 中添加检测方式选择器（单选）：
- **YOLO 检测** - 选择已训练模型（下拉菜单）
- **自然语言检测** - 输入自然语言描述 + 选择 LLM 模型

#### 2. UI 交互逻辑
```
┌─────────────────────────────────────────┐
│ 检测方式：                               │
│  ○ YOLO 检测                             │
│  ● 自然语言检测                          │
├─────────────────────────────────────────┤
│ 检测目标：                               │
│  [请描述要检测的目标，如：红色安全帽]   │
├─────────────────────────────────────────┤
│ LLM 模型：                               │
│  [qwen-vl-max ▼]                        │
├─────────────────────────────────────────┤
│ 置信度：0.25 [────────]                 │
└─────────────────────────────────────────┘
```

当选择 YOLO 时：
- 显示模型选择器
- 隐藏自然语言输入和 LLM 模型选择

当选择自然语言检测时：
- 隐藏模型选择器
- 显示自然语言输入框
- 显示 LLM 模型选择器

### 后端改造

#### 1. 扩展 image_detect.py API

**当前参数：**
- `model_id`: str (可选)
- `class_names`: str (可选)
- `conf`: float

**新增参数：**
- `detection_mode`: str = "yolo" | "zeroshot" | "llm"
- `llm_model`: str (可选，用于 LLM 模式)
- `llm_prompt`: str (可选，用于 LLM 模式的自然语言描述)

**检测逻辑：**
```python
if detection_mode == "llm":
    # 使用 LLM 检测
    from app.services.llm_annotation import QwenAnnotator
    annotator = QwenAnnotator(api_key, llm_model, api_base)
    boxes = await annotator.annotate_image(image_path, llm_prompt, w, h)
    # 转换为统一的 Detection 格式
    detections = convert_boxes_to_detections(boxes)
elif model_id:
    # YOLO 模型检测
    detections = await detect_with_model(...)
else:
    # 零样本检测（GroundingDINO）
    detections = await detect_zeroshot(...)
```

#### 2. 返回格式统一

LLM 返回的 `BoundingBox` 需要转换为 `Detection` 格式：
```python
Detection(
    class_id=box.class_id,
    class_name=f"target_{box.class_id}",  # LLM 不返回类名
    confidence=box.confidence,
    bbox=[box.cx - box.w/2, box.cy - box.h/2, box.w, box.h]
)
```

### 前端改造

#### 1. ImageDetectPage.jsx 改造

**新增状态：**
```javascript
const [detectionMode, setDetectionMode] = useState('yolo')  // 'yolo' | 'llm'
const [llmModel, setLlmModel] = useState(null)
const [llmPrompt, setLlmPrompt] = useState('')
const [llmModels, setLlmModels] = useState([])  // 可用的 LLM 模型列表
```

**UI 结构：**
```jsx
{/* 检测方式选择 */}
<div className="detection-mode-selector">
  <label>
    <input type="radio" value="yolo" checked={detectionMode === 'yolo'} />
    YOLO 检测
  </label>
  <label>
    <input type="radio" value="llm" checked={detectionMode === 'llm'} />
    自然语言检测
  </label>
</div>

{/* YOLO 模式 */}
{detectionMode === 'yolo' && (
  <DetectModelSelect onSelect={handleSelectModel} />
)}

{/* LLM 模式 */}
{detectionMode === 'llm' && (
  <>
    <textarea
      value={llmPrompt}
      onChange={(e) => setLlmPrompt(e.target.value)}
      placeholder="请描述要检测的目标，如：画面中的红色安全帽"
    />
    <select value={llmModel} onChange={(e) => setLlmModel(e.target.value)}>
      {llmModels.map(m => <option value={m}>{m}</option>)}
    </select>
  </>
)}
```

**API 调用修改：**
```javascript
const params = detectionMode === 'llm'
  ? { detectionMode: 'llm', llmModel, llmPrompt, conf: confidence }
  : { detectionMode: 'yolo', modelId: selectedModelId, conf: confidence }

const result = await detectImages(files, params, setUploadPct)
```

#### 2. DetectPage.jsx 改造

类似的改造应用到视频检测页面。

#### 3. API 服务改造

**修改 api.js 中的 detectImages 函数：**
```javascript
export async function detectImages(files, params, onProgress) {
  const formData = new FormData()
  for (const f of files) formData.append('files', f)
  
  const { detectionMode, modelId, classNames, llmModel, llmPrompt, conf } = params
  
  if (detectionMode) formData.append('detection_mode', detectionMode)
  if (modelId) formData.append('model_id', modelId)
  if (classNames) formData.append('class_names', classNames)
  if (llmModel) formData.append('llm_model', llmModel)
  if (llmPrompt) formData.append('llm_prompt', llmPrompt)
  if (conf != null) formData.append('conf', String(conf))
  
  // ...
}
```

## 实施步骤

### Phase 1: 后端 API 扩展
1. 修改 `image_detect.py`，添加 `detection_mode`, `llm_model`, `llm_prompt` 参数
2. 实现 LLM 检测分支，调用 `QwenAnnotator`
3. 添加 LLM 配置检查（需要配置 LLM_API_KEY）
4. 实现 `BoundingBox` 到 `Detection` 的转换
5. 测试后端 API

### Phase 2: 前端 UI 改造
1. 在 `ImageDetectPage.jsx` 添加检测方式选择器
2. 添加 LLM 模型选择和提示词输入
3. 根据检测方式显示/隐藏相应的 UI 组件
4. 加载可用的 LLM 模型列表
5. 修改 API 调用参数

### Phase 3: 视频检测集成
1. 在 `DetectPage.jsx` 应用相同的改造
2. 确保视频抽帧后的检测也支持 LLM 模式

### Phase 4: 测试与优化
1. 测试 YOLO 检测（确保不影响现有功能）
2. 测试 LLM 检测（不同模型、不同提示词）
3. 性能优化（LLM 检测可能较慢，添加进度提示）
4. 错误处理（LLM API 失败时的提示）

## 技术细节

### 1. LLM 检测的特殊处理

**类别名称：**
- LLM 检测不返回具体类别名，只返回边界框
- 可以从提示词中提取目标名称作为类别名
- 或统一使用 "detected_object" 作为类别名

**性能考虑：**
- LLM 检测比 YOLO 慢得多（每张图 2-5 秒）
- 添加批量检测进度条
- 考虑并发控制（与 LLM 标注共享并发限制）

**错误处理：**
- LLM API 调用失败（网络、配额等）
- LLM 返回格式错误（解析失败）
- 未配置 LLM_API_KEY

### 2. 向后兼容

**API 兼容性：**
- `detection_mode` 参数可选，默认为 "yolo"
- 保持现有的 `model_id` 和 `class_names` 参数
- 旧的前端调用仍然有效

**UI 兼容性：**
- 默认选择 YOLO 检测模式
- 保持现有的操作流程

## 待确认事项

1. **零样本检测（GroundingDINO）的去留**
   - 当前实现中，输入自然语言 `class_names` 会使用零样本检测
   - 选项 A: 保留三种模式（YOLO、零样本、LLM）
   - 选项 B: 用 LLM 替代零样本（推荐，LLM 效果更好）

2. **LLM 检测的类别名**
   - 选项 A: 从提示词提取（需要 NLP 处理）
   - 选项 B: 使用用户输入的提示词作为类别名
   - 选项 C: 统一使用固定名称（如 "target"）

3. **并发和配额**
   - 与 LLM 标注共享并发限制？
   - 是否需要独立的配额控制？

## 预期效果

完成后，用户可以：
1. 在图片检测页面选择"自然语言检测"
2. 输入自然语言描述，如"画面中的人员"、"红色安全帽"
3. 选择 LLM 模型（qwen-vl-max、qwen-vl-plus 等）
4. 获得基于 LLM 的检测结果
5. 视频检测同样支持此功能

## 预计工作量

- 后端改造：2-3 小时
- 前端改造：2-3 小时  
- 测试与调试：1-2 小时
- **总计：5-8 小时**
