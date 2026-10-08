# YOLOE 训练使用指南

本文档说明如何为本项目的 YOLOE 检测后端训练 / 微调自定义模型，包括数据集结构、训练命令、以及如何把训练好的权重接回检测服务。

> 训练入口：`backend/YOLOWorld/train_yoloe.py`
> 数据集模板：`backend/YOLOWorld/dataset.yaml.example`

---

## 1. 环境准备

项目依赖已经包含 `ultralytics>=8.3.0`（YOLOE 通过 Ultralytics 提供）。如果是新环境：

```bash
cd backend
pip install -r requirements.txt
```

校验 YOLOE 是否可用：

```bash
python -c "from ultralytics import YOLOE; print('YOLOE OK')"
```

若提示 `ImportError`，升级 ultralytics：

```bash
pip install -U "ultralytics>=8.3.0"
```

> 训练脚本在 `YOLOE` 不可用时会自动回退到 `ultralytics.YOLO`，但推荐用 YOLOE 完整能力，请保证 ≥ 8.3。

---

## 2. 数据集结构

YOLOE 沿用 YOLO 标准目录布局，把图片和标签按 `train / val / test` 划分：

```
/your/dataset/
├── images/
│   ├── train/      # 训练集图片：image_0001.jpg, image_0002.jpg, ...
│   ├── val/        # 验证集图片
│   └── test/       # 测试集图片（可选）
└── labels/
    ├── train/      # 训练集标注：image_0001.txt, image_0002.txt, ...
    ├── val/        # 验证集标注
    └── test/       # 测试集标注（可选）
```

**关键约定**

- `labels/<split>/<filename>.txt` 必须与 `images/<split>/<filename>.{jpg,png,...}` **同名**（除后缀外）。
- 一张图对应一个 `.txt`，没有目标的图就放一个空 `.txt`。
- 每行一个目标，格式：

  ```
  <class_id> <cx> <cy> <w> <h>
  ```

  其中 `class_id` 从 0 开始；`cx, cy, w, h` 是相对于图片宽高的 **归一化值（0–1）** 的中心坐标和宽高。

  例：`0 0.512 0.480 0.310 0.622` 表示类别 0 的目标，中心点在图片中央偏左，框宽约 31%、高约 62%。

---

## 3. 编写数据集 YAML

复制模板并按你的数据填写：

```bash
cp backend/YOLOWorld/dataset.yaml.example backend/YOLOWorld/dataset.yaml
```

`dataset.yaml` 示例：

```yaml
path: /home/hmxh/datasets/iron_tower      # 数据集根目录（绝对路径）
train: images/train
val: images/val
test: images/test                          # 没有可省略

names:
  0: tower
  1: insulator
  2: vegetation
```

> Ultralytics 会按 `path + train/val/test` 的拼接去找图片目录，并自动到平级 `labels/` 下找同名标注。

---

## 4. 训练命令

进入项目根目录后运行：

### 4.1 单卡基础训练

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device cuda:0
```

### 4.2 从指定预训练权重开始

默认会读取 `backend/.env` 中的 `YOLOE_BASE_MODEL`。要显式指定：

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --model /home/hmxh/workspace/sodv3/SOD/backend/models/yolo/yoloe-11l-seg.pt \
    --epochs 100
```

### 4.3 多卡训练

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --device 0,1 \
    --batch 32
```

### 4.4 断点续训

中断后想从最近一次 checkpoint 继续：

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --name yoloe_exp \
    --resume
```

### 4.5 仅微调最后几层（快速实验）

冻结前 10 层，配合较小学习率：

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --epochs 30 \
    --freeze 10 \
    --lr0 0.001
```

### 4.6 CPU 调试（不推荐用于真正训练）

```bash
python backend/YOLOWorld/train_yoloe.py \
    --data backend/YOLOWorld/dataset.yaml \
    --device cpu \
    --epochs 1 --batch 2 --imgsz 320
```

---

## 5. CLI 参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | **必填** | 数据集 YAML 路径 |
| `--model` | `.env` 的 `YOLOE_BASE_MODEL` | 起始权重，YOLOE 默认 `yoloe-11l-seg.pt` |
| `--epochs` | 100 | 训练轮数 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--batch` | 16 | 批大小；`-1` 表示自动按显存匹配 |
| `--device` | `.env` 的 `DEVICE` | `cuda:0` / `cpu` / `0,1` |
| `--project` | `backend/runs/train` | 输出根目录 |
| `--name` | `yoloe_exp` | 实验子目录名 |
| `--resume` | False | 从最近 checkpoint 续训 |
| `--workers` | 8 | DataLoader 工作进程 |
| `--patience` | 50 | 早停轮数（验证集多少轮无提升就停） |
| `--freeze` | 无 | 冻结前 N 层 |
| `--lr0` | 无 | 初始学习率（不传走 Ultralytics 默认） |

完整帮助：

```bash
python backend/YOLOWorld/train_yoloe.py --help
```

---

## 6. 训练产物

训练完成后，产物位于：

```
backend/runs/train/yoloe_exp/
├── weights/
│   ├── best.pt          # 验证集最优的权重 ← 部署用这个
│   └── last.pt          # 最后一轮的权重（用于续训）
├── results.csv          # 每轮的 loss / mAP 等指标
├── results.png          # 训练曲线
├── confusion_matrix.png
└── val_batch*.jpg       # 验证集预测可视化
```

脚本在结束时会把 `best.pt` 的绝对路径打印到日志末尾，复制即可。

---

## 7. 将训练好的模型接回检测后端

只需改一行配置：

1. 打开 `backend/.env`，把 `YOLOE_BASE_MODEL` 改为你的 `best.pt` 绝对路径：

   ```env
   YOLOE_BASE_MODEL=/home/hmxh/workspace/sodv3/SOD/backend/runs/train/yoloe_exp/weights/best.pt
   ```

2. 重启后端：

   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   或直接重启 `start_all.sh`。

3. 日志里出现 `Using local YOLOE weights: .../best.pt` 即说明新权重已加载。

> 检测代码（`backend/app/services/yoloe_detector.py`）会自动识别本地权重文件，无需改动业务代码。

---

## 8. 常见问题

**Q1：报错 `CUDA out of memory`？**
A：减小 `--batch`（如 8 → 4）或 `--imgsz`（如 640 → 480），或换更小的预训练模型（`yoloe-11s-seg.pt`）。

**Q2：报错 `Dataset 'xxx' not found`？**
A：检查 `dataset.yaml` 里的 `path` 是不是绝对路径、`train/val` 子目录是否真的存在。Ultralytics 在第一次找不到时会尝试自动下载，离线模式下会失败。

**Q3：训练好的模型在推理时类别名不对？**
A：YOLOE 在 `set_classes(...)` 阶段会用前端 prompt 覆盖类别名。如果你只是想用训练时的类别表，前端 prompt 仍需把这些类别名写全（例如 `tower . insulator . vegetation`）。

**Q4：离线机器训练**
A：脚本默认设置了 `YOLO_OFFLINE=1`，但 **预训练权重需要提前下载到本地**，并通过 `--model /abs/path/to/xxx.pt` 显式传入。

**Q5：训练 mAP 一直很低**
A：先检查 (a) 标注是否归一化、(b) 类别 id 是否从 0 开始、(c) 图像和标签文件名是否严格对应。再考虑学习率、数据量、增广策略。

---

## 9. 端到端冒烟流程

最小验证（约 20 张图、1 轮训练即可跑通）：

```bash
# 1. 准备最小数据集（结构如第 2 节）
mkdir -p /tmp/mini/images/{train,val} /tmp/mini/labels/{train,val}
# 放几张图和对应 .txt，然后 ↓

cat > /tmp/mini/dataset.yaml <<'EOF'
path: /tmp/mini
train: images/train
val: images/val
names:
  0: object
EOF

# 2. 跑 1 轮训练
python backend/YOLOWorld/train_yoloe.py \
    --data /tmp/mini/dataset.yaml \
    --epochs 1 --batch 2 --imgsz 320 \
    --name smoke_test

# 3. 检查产物
ls backend/runs/train/smoke_test/weights/
# 应看到 best.pt 和 last.pt
```

跑通后即可换上真实数据集做正式训练。
