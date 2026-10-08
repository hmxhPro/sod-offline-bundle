# 检测模式分离与数据集合并功能实施计划

## 概述

本计划实现三个主要功能：
1. 将 YOLO 检测和多模态检测分离为两个独立模式
2. YOLO 模式支持多选模型，结果分层显示
3. 训练页面添加数据集合并功能

## 一、检测界面改造

### 1.1 用户需求
- **模式分离**：YOLO 检测与多模态检测互斥，通过选项卡切换
- **YOLO 模式**：支持多选已训练模型，结果按模型分层显示
- **多模态模式**：使用自然语言描述进行检测（现有功能保持）

### 1.2 前端改造

#### 视频检测页面 (`DetectPage.jsx`)
**当前状态**：
- 单选模型 (`DetectModelSelect`)
- 模型未选时显示提示词输入 (`PromptInput`)
- 逻辑：有模型 = 用模型，无模型 = 用自然语言

**改造方案**：
1. **添加模式选择器**（选项卡）
   ```jsx
   <DetectionModeTabs mode={detectionMode} onChange={setDetectionMode} />
   ```
   - 选项：`yolo` | `multimodal`
   - 默认：`multimodal`（保持向后兼容）

2. **YOLO 模式界面**
   ```jsx
   {detectionMode === 'yolo' && (
     <YoloModelMultiSelect
       selectedModels={selectedYoloModels}
       onChange={setSelectedYoloModels}
       disabled={anyActive}
     />
   )}
   ```
   - 复选框列表，显示所有已训练模型
   - 至少选择一个模型才能开始检测
   - 显示每个模型的类别信息

3. **多模态模式界面**（保持现状）
   ```jsx
   {detectionMode === 'multimodal' && (
     <PromptInput value={prompt} onChange={setPrompt} disabled={anyActive} />
   )}
   ```

4. **开始检测按钮逻辑**
   ```javascript
   const canStart = detectionMode === 'yolo'
     ? selectedYoloModels.length > 0
     : prompt.trim().length > 0
   ```

#### 图片检测页面 (`ImageDetectPage.jsx`)
**当前状态**：
- 单选模型或输入类别名称
- 二选一验证逻辑

**改造方案**：
- 类似视频检测页面，添加模式选择器
- YOLO 模式：多选模型
- 多模态模式：输入类别名称

### 1.3 后端改造

#### API 端点调整

**视频检测** (`POST /api/detect`)
**当前请求**：
```json
{
  "file": <视频文件>,
  "prompt": "人、车",
  "model_id": "uuid或null",
  "conf": 0.25
}
```

**新请求**：
```json
{
  "file": <视频文件>,
  "detection_mode": "yolo",  // 新增
  "model_ids": ["uuid1", "uuid2"],  // 多模型支持
  "prompt": "人、车",  // multimodal 模式使用
  "conf": 0.25
}
```

**图片检测** (`POST /api/detect/images`)
- 类似调整

#### 检测逻辑改造

**文件位置**：`backend/app/services/detection_pipeline.py`

**多模型检测实现**：
```python
async def detect_with_multiple_models(
    frame: np.ndarray,
    model_ids: List[str],
    conf_threshold: float
) -> Dict[str, List[Detection]]:
    """
    使用多个模型检测，返回按模型分组的结果
    
    Returns:
        {
            "model_uuid1": [Detection(...), ...],
            "model_uuid2": [Detection(...), ...],
        }
    """
    results = {}
    for model_id in model_ids:
        model = load_model(model_id)
        detections = model.predict(frame, conf=conf_threshold)
        results[model_id] = detections
    return results
```

**结果格式**：
```json
{
  "task_id": "...",
  "frames": [
    {
      "frame_number": 1,
      "detections_by_model": {
        "model_uuid1": [
          {"class": "person", "conf": 0.95, "bbox": [...]},
        ],
        "model_uuid2": [
          {"class": "person", "conf": 0.88, "bbox": [...]},
        ]
      }
    }
  ]
}
```

### 1.4 前端结果展示

**TaskCard 组件改造**：
- 检测到多模型结果时，分层显示
- 每层标题显示模型名称和类别
- 使用不同的边框颜色区分模型（色板：蓝、绿、紫、橙）

```jsx
{task.detection_mode === 'yolo' && task.results?.detections_by_model && (
  <div className="space-y-2">
    {Object.entries(task.results.detections_by_model).map(([modelId, detections]) => (
      <div key={modelId} className="border-l-4 border-blue-500 pl-3">
        <div className="text-xs font-medium text-ink-700">
          模型：{getModelName(modelId)}
        </div>
        <div className="text-xs text-ink-600">
          检测到 {detections.length} 个目标
        </div>
      </div>
    ))}
  </div>
)}
```

## 二、数据集合并功能

### 2.1 用户需求
- 在训练页面添加"合并数据集"功能
- 将源类别的数据追加到目标类别
- 源类别可选择保留或删除

### 2.2 前端实现

#### 位置
`TrainingPage.jsx` - 在类别列表上方添加操作按钮

#### UI 组件
```jsx
<DatasetMergeButton
  categories={categories}
  onMergeComplete={refetchCategories}
/>
```

**合并对话框**：
```
┌─────────────────────────────────────┐
│  合并数据集                          │
├─────────────────────────────────────┤
│  目标类别：[下拉选择]                │
│  源类别：  [下拉选择]                │
│                                      │
│  □ 合并后删除源类别                 │
│                                      │
│  预览：                              │
│  • 目标类别当前：120 张图片          │
│  • 源类别：      80 张图片           │
│  • 合并后：      200 张图片          │
│                                      │
│  [取消]  [确认合并]                 │
└─────────────────────────────────────┘
```

### 2.3 后端实现

#### API 端点
```python
@router.post("/api/categories/merge")
async def merge_datasets(
    target_category_id: str = Form(...),
    source_category_id: str = Form(...),
    delete_source: bool = Form(False),
) -> MergeResult:
    """
    将源类别的数据集合并到目标类别
    
    步骤：
    1. 验证两个类别存在且类型相同
    2. 复制源类别的原图到目标类别
    3. 复制源类别的标注到目标类别
    4. 更新目标类别的统计信息
    5. 如果 delete_source=True，删除源类别
    
    Returns:
        {
            "success": true,
            "merged_images": 80,
            "target_total_images": 200,
            "deleted_source": false
        }
    """
```

#### 实现逻辑
```python
async def merge_category_datasets(
    target_id: str,
    source_id: str,
    delete_source: bool
) -> MergeResult:
    # 1. 获取源类别的所有图片记录
    source_images = await get_category_images(source_id)
    
    # 2. 复制文件
    for img in source_images:
        # 复制原图
        source_raw = f"datasets/{source_id}/raw/{img.id}{img.ext}"
        target_raw = f"datasets/{target_id}/raw/{img.id}{img.ext}"
        shutil.copy2(source_raw, target_raw)
        
        # 复制标注文件（如果存在）
        source_label = f"annotations/{source_id}/{img.id}.txt"
        target_label = f"annotations/{target_id}/{img.id}.txt"
        if Path(source_label).exists():
            shutil.copy2(source_label, target_label)
    
    # 3. 更新数据库
    for img in source_images:
        img.category_id = target_id
        session.add(img)
    
    # 4. 重新计算目标类别统计
    await recompute_category_counts(target_id)
    
    # 5. 删除源类别（如果需要）
    if delete_source:
        await delete_category(source_id)
    
    await session.commit()
    
    return MergeResult(...)
```

### 2.4 验证逻辑
- 不允许目标和源相同
- 验证两个类别都存在
- 检查磁盘空间是否足够
- 合并前备份关键数据

## 三、实施步骤

### 阶段 1：前端基础框架（无后端依赖）
1. 创建 `DetectionModeTabs` 组件
2. 创建 `YoloModelMultiSelect` 组件
3. 改造 `DetectPage.jsx` 添加模式切换
4. 改造 `ImageDetectPage.jsx` 添加模式切换
5. 验证 UI 交互流程

### 阶段 2：后端 API 扩展
1. 修改检测 API 支持 `detection_mode` 和 `model_ids`
2. 实现多模型检测逻辑
3. 调整结果返回格式（分层）
4. 实现数据集合并 API

### 阶段 3：前后端集成
1. 前端调用新 API 格式
2. 前端展示分层检测结果
3. 添加数据集合并前端界面
4. 集成测试

### 阶段 4：测试与优化
1. 测试单模型/多模型检测
2. 测试模式切换
3. 测试数据集合并
4. 性能优化（多模型并发）

## 四、技术细节

### 4.1 多模型并发优化
```python
async def detect_with_multiple_models_async(
    frame: np.ndarray,
    model_ids: List[str],
    conf_threshold: float
) -> Dict[str, List[Detection]]:
    """异步并发检测，提升多模型性能"""
    tasks = []
    for model_id in model_ids:
        task = asyncio.create_task(
            detect_single_model(frame, model_id, conf_threshold)
        )
        tasks.append((model_id, task))
    
    results = {}
    for model_id, task in tasks:
        results[model_id] = await task
    
    return results
```

### 4.2 结果可视化色板
```javascript
const MODEL_COLORS = [
  'border-blue-500',    // 蓝色
  'border-green-500',   // 绿色
  'border-purple-500',  // 紫色
  'border-orange-500',  // 橙色
  'border-pink-500',    // 粉色
  'border-teal-500',    // 青色
]

function getModelColor(index) {
  return MODEL_COLORS[index % MODEL_COLORS.length]
}
```

### 4.3 数据集合并安全性
- 使用事务确保原子性
- 合并前创建备份点
- 记录操作日志
- 提供回滚机制（可选）

## 五、向后兼容性

### 5.1 API 兼容
- 保留旧的单模型 API 参数格式
- `model_id` 存在时自动转换为 `model_ids: [model_id]`
- `detection_mode` 默认为 `multimodal`

### 5.2 前端兼容
- 默认模式为多模态（保持现有用户习惯）
- 现有任务结果正常显示

## 六、文件清单

### 新增文件
- `frontend/src/components/DetectionModeTabs.jsx`
- `frontend/src/components/YoloModelMultiSelect.jsx`
- `frontend/src/components/DatasetMergeButton.jsx`
- `frontend/src/components/DatasetMergeDialog.jsx`

### 修改文件
- `frontend/src/pages/DetectPage.jsx`
- `frontend/src/pages/ImageDetectPage.jsx`
- `frontend/src/pages/TrainingPage.jsx`
- `frontend/src/services/api.js`
- `frontend/src/components/TaskCard.jsx`
- `backend/app/api/detection.py`
- `backend/app/api/dataset.py`
- `backend/app/services/detection_pipeline.py`
- `backend/app/models/schemas.py`

## 七、预估工作量

- 阶段 1：2-3 小时（前端 UI）
- 阶段 2：3-4 小时（后端逻辑）
- 阶段 3：2-3 小时（集成）
- 阶段 4：1-2 小时（测试）
- **总计：8-12 小时**

## 八、风险与注意事项

1. **多模型性能**：多个模型同时推理可能导致 GPU 内存不足
   - 缓解：提供串行/并行选项，或限制同时加载的模型数量

2. **数据集合并**：大数据集合并可能耗时较长
   - 缓解：添加进度条，后台任务执行

3. **结果展示**：多模型结果可能过于拥挤
   - 缓解：可折叠/展开每个模型的结果，默认折叠

4. **向后兼容**：确保现有功能不受影响
   - 缓解：保留旧 API，添加自动转换逻辑
