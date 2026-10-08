# 迭代标注与数据集合并功能实施总结

## 实施日期
2025年（根据用户批准的计划执行）

## 实施内容

根据用户批准的计划，本次实施完成了以下功能：

### 1. 迭代标注辅助模型功能

#### 后端改造
- **数据库模型** (`backend/app/db/models.py`)
  - 在 `AnnotationIterationRecord` 表中添加 `assist_model_id` 字段
  - 用于记录本轮标注使用的辅助模型ID

- **API 端点** (`backend/app/api/llm_annotation.py`)
  - 修改 `CreateAnnotationIterationRequest` schema，添加 `assist_model_id` 参数
  - 修改 `AnnotationIterationItem` 响应模型，包含 `assist_model_id`
  - 在 `create_annotation_iteration` 端点中添加辅助模型验证逻辑
  - 验证指定的辅助模型存在且可用

**核心逻辑：**
```python
# 验证辅助模型（如果指定）
assist_model_id = body.assist_model_id
if assist_model_id:
    assist_model = await session.get(TrainedModelRecord, assist_model_id)
    if assist_model is None:
        raise HTTPException(404, f"辅助模型 {assist_model_id} 不存在")
```

#### 前端实现
- **LLM 标注页面** (`frontend/src/pages/LLMAnnotationPage.jsx`)
  - 导入 `getModels` API 以获取已训练模型列表
  - 添加 `trainedModels` 状态存储可用的训练模型
  - 在迭代表单中添加辅助模型选择器（下拉菜单）
  - 初始化迭代表单时包含 `assistModelId: ''`
  - API 调用时传递 `assist_model_id` 参数

**UI 组件：**
```jsx
<select
  value={iterationForm.assistModelId || ''}
  onChange={(e) => setIterationForm((f) => ({ ...f, assistModelId: e.target.value }))}
  disabled={iterationSubmitting}
>
  <option value="">不使用辅助模型（纯LLM标注）</option>
  {trainedModels.map((m) => (
    <option key={m.id} value={m.id}>
      {m.name} {m.version > 1 ? `v${m.version}` : ''}
    </option>
  ))}
</select>
```

**注意事项：**
- 辅助模型参数已添加到 API 和数据库
- 完整的辅助标注逻辑（YOLO预标 + LLM修正）作为后续优化项
- 当前实现允许用户选择辅助模型，后端会记录但暂不执行预标注

---

### 2. 数据集合并功能

#### 后端实现
- **响应模型** (`backend/app/models/schemas.py`)
  - 新增 `DatasetMergeResult` schema
  - 包含合并结果统计信息

- **API 端点** (`backend/app/api/dataset.py`)
  - 新增 `POST /api/categories/merge` 端点
  - 支持将源类别数据合并到目标类别
  - 可选删除源类别

**合并流程：**
1. 验证目标和源类别存在
2. 复制源类别的原图到目标类别
3. 复制源类别的标注文件到目标类别
4. 更新数据库记录（修改 `category_id`）
5. 重新计算目标类别统计信息
6. 如果 `delete_source=True`，删除源类别和文件

**核心代码：**
```python
@router.post("/categories/merge", response_model=DatasetMergeResult)
async def merge_datasets(
    target_category_id: str = Form(...),
    source_category_id: str = Form(...),
    delete_source: bool = Form(False),
) -> DatasetMergeResult:
    # 复制文件
    for img in source_images:
        shutil.copy2(source_raw_path, target_raw_path)
        shutil.copy2(source_ann, target_ann)
        img.category_id = target_category_id
    
    # 更新统计
    await _recompute_counts(session, target_category_id)
    
    # 删除源类别（可选）
    if delete_source:
        await session.delete(source_cat)
        shutil.rmtree(source_raw_dir, ignore_errors=True)
```

#### 前端实现
- **API 服务** (`frontend/src/services/api.js`)
  - 新增 `mergeDatasets(targetId, sourceId, deleteSource)` 函数

- **合并对话框** (`frontend/src/components/DatasetMergeDialog.jsx`)
  - 新建独立的合并对话框组件
  - 目标类别选择器
  - 源类别选择器（自动排除目标类别）
  - "合并后删除源类别" 复选框
  - 实时预览合并后的图片数量
  - 删除源类别时显示警告提示

- **类别管理器** (`frontend/src/components/CategoryManager.jsx`)
  - 在标题栏添加"合并"按钮（Merge 图标）
  - 按钮在类别少于2个时禁用
  - 点击按钮打开合并对话框
  - 合并成功后刷新类别列表

**UI 示例：**
```
┌─────────────────────────────────────┐
│  合并数据集                          │
├─────────────────────────────────────┤
│  目标类别：[类别A (120张)]          │
│  源类别：  [类别B (80张)]           │
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

---

## 技术细节

### 数据库变更
- **新增字段**：`AnnotationIterationRecord.assist_model_id`
- **兼容性**：SQLAlchemy 的 `create_all()` 会自动添加新字段到现有表
- **迁移策略**：字段为 `Optional`，现有数据自动为 `NULL`

### API 兼容性
- 所有新增参数都是可选的（`Optional`）
- 不影响现有 API 调用
- 前端未传递 `assist_model_id` 时，后端按原逻辑执行

### 文件操作安全
- 使用 `shutil.copy2()` 保留文件元数据
- 合并前验证类别存在
- 删除操作使用 `ignore_errors=True` 避免崩溃
- 失败的图片合并会跳过并记录日志

---

## 测试验证

### 前端构建测试
```bash
npm run build
# ✓ built in 2.43s - 构建成功
```

### 后端语法检查
```bash
python3.10 -m py_compile app/api/llm_annotation.py  # ✓
python3.10 -m py_compile app/api/dataset.py         # ✓
python3.10 -m py_compile app/db/models.py           # ✓
python3.10 -m py_compile app/models/schemas.py      # ✓
```

---

## 使用方法

### 迭代标注使用辅助模型
1. 在 LLM 自动标注页面，查看已完成的标注任务
2. 点击"继续迭代标注"按钮
3. 在迭代表单中选择"辅助模型（可选）"下拉菜单
4. 选择已训练的模型或保持"不使用辅助模型"
5. 填写标注提示词和其他参数
6. 点击"开始下一轮"

### 数据集合并
1. 在训练页面，确保有至少2个类别
2. 点击类别管理器标题栏的"合并"按钮（双箭头图标）
3. 选择目标类别和源类别
4. 决定是否勾选"合并后删除源类别"
5. 查看预览信息
6. 点击"确认合并"
7. 等待合并完成，查看结果提示

---

## 已知限制与后续优化

### 1. 辅助模型标注逻辑
**当前状态：** API 参数和 UI 已完成，但完整的辅助标注流程（先 YOLO 预标，再 LLM 修正）尚未实现。

**后续优化：**
- 在 `llm_annotation.py` 中实现预标注逻辑
- 使用辅助模型进行初步检测
- 将检测框转换为 YOLO 格式
- 传递给 LLM 作为参考进行修正和补充

### 2. 数据集去重
**当前实现：** 简单复制所有源图片到目标类别

**后续优化：**
- 基于文件哈希检测重复图片
- 合并时提示用户重复数量
- 提供"跳过重复"或"覆盖"选项

### 3. 合并进度反馈
**当前实现：** 同步操作，大数据集可能耗时较长

**后续优化：**
- 改为后台异步任务
- 添加进度条显示
- 支持取消操作

---

## 文件清单

### 后端修改
- `backend/app/db/models.py` - 添加 `assist_model_id` 字段
- `backend/app/api/llm_annotation.py` - 迭代标注 API 扩展
- `backend/app/api/dataset.py` - 数据集合并 API
- `backend/app/models/schemas.py` - 新增 `DatasetMergeResult`

### 前端修改
- `frontend/src/pages/LLMAnnotationPage.jsx` - 添加辅助模型选择器
- `frontend/src/components/CategoryManager.jsx` - 添加合并按钮
- `frontend/src/services/api.js` - 新增 `mergeDatasets` API

### 前端新增
- `frontend/src/components/DatasetMergeDialog.jsx` - 合并对话框组件

---

## 总结

本次实施成功完成了用户批准计划中的核心功能：

✅ **迭代标注辅助模型**：用户可在迭代标注时选择已训练模型辅助（API 和 UI 完成，完整逻辑待实现）  
✅ **数据集合并**：支持将多个类别的数据集合并，便于扩充训练数据  
✅ **前后端集成**：所有功能前后端打通，构建测试通过  
✅ **向后兼容**：新增参数全部可选，不影响现有功能  

所有代码已验证语法正确，前端构建成功，可以启动服务进行功能测试。
