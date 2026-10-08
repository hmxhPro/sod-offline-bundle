"""
测试 LLM 标注服务（通义千问 Qwen）

提供商/模型/密钥均来自后端 .env 配置（create_annotator 不再接收参数）。
运行测试：
    python backend/tests/test_llm_annotation_manual.py
"""

import asyncio
import sys
from pathlib import Path

# 添加backend到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.llm_annotation import (
    QwenAnnotator,
    create_annotator,
    _parse_objects_response,
    BoundingBox,
)


def print_boxes(boxes):
    """打印边界框信息"""
    if not boxes:
        print("  未检测到目标")
        return

    print(f"  检测到 {len(boxes)} 个目标:")
    for i, box in enumerate(boxes, 1):
        print(f"    {i}. cx={box.cx:.3f}, cy={box.cy:.3f}, w={box.w:.3f}, h={box.h:.3f}")
        print(f"       YOLO格式: {box.to_yolo_line()}")


async def test_qwen():
    """用后端配置的 Qwen 标注器标注一张真实图片"""
    print("\n" + "=" * 60)
    print("测试 Qwen 标注器（使用后端 .env 配置）")
    print("=" * 60)

    if not settings.llm_configured:
        print("❌ 未配置 LLM_API_KEY，请先在 backend/.env 中设置后重试")
        return

    print(f"模型: {settings.LLM_MODEL}")
    print(f"接口: {settings.LLM_API_BASE}")

    annotator = create_annotator()

    test_image = input("请输入测试图片路径: ").strip()
    if not Path(test_image).exists():
        print(f"❌ 图片不存在: {test_image}")
        return

    prompt = input("请输入目标描述（例如：无人机）: ").strip() or "无人机"

    print(f"\n正在标注图片: {test_image}")
    print(f"提示词: {prompt}\n请稍候...")

    try:
        boxes = await annotator.annotate_image(
            image_path=test_image,
            prompt=prompt,
            image_width=1920,
            image_height=1080,
        )
        print("\n✅ 标注成功!")
        print_boxes(boxes)
    except Exception as e:
        print(f"\n❌ 标注失败: {e}")


def test_factory():
    """测试工厂函数（未配置时应拒绝，已配置时返回 QwenAnnotator）"""
    print("\n" + "=" * 60)
    print("测试标注器工厂函数")
    print("=" * 60)

    if settings.llm_configured:
        annotator = create_annotator()
        assert isinstance(annotator, QwenAnnotator)
        print(f"✅ 已配置：create_annotator() -> {type(annotator).__name__}")
    else:
        try:
            create_annotator()
            print("❌ 未配置时应抛出 RuntimeError")
        except RuntimeError as e:
            print(f"✅ 未配置时正确拒绝: {e}")


def test_parser():
    """测试响应解析（含 markdown 围栏与坐标越界裁剪）"""
    print("\n" + "=" * 60)
    print("测试响应解析")
    print("=" * 60)

    raw = '```json\n{"objects": [{"x1": 100, "y1": 200, "x2": 300, "y2": 600}]}\n```'
    boxes = _parse_objects_response(raw, 1000, 1000)
    assert len(boxes) == 1, boxes
    expected = "0 0.200000 0.400000 0.200000 0.400000"
    if boxes[0].to_yolo_line() == expected:
        print(f"✅ 解析+归一化正确: {boxes[0].to_yolo_line()}")
    else:
        print(f"❌ 期望 {expected}，实际 {boxes[0].to_yolo_line()}")

    empty = _parse_objects_response('{"objects": []}', 100, 100)
    print(f"✅ 空结果解析: {len(empty)} 个框")


def test_bbox():
    """测试边界框转换"""
    print("\n" + "=" * 60)
    print("测试边界框 YOLO 格式转换")
    print("=" * 60)

    box = BoundingBox(class_id=0, cx=0.5, cy=0.5, w=0.2, h=0.3, confidence=0.95)
    yolo_line = box.to_yolo_line()
    expected = "0 0.500000 0.500000 0.200000 0.300000"
    print(f"YOLO格式: {yolo_line}")
    print("✅ 转换正确" if yolo_line == expected else f"❌ 期望: {expected}")


async def main():
    print("\n" + "=" * 60)
    print("LLM 自动标注服务测试（Qwen）")
    print("=" * 60)
    print("\n可用测试:")
    print("1. 测试 Qwen 标注器（真实调用）")
    print("2. 测试工厂函数")
    print("3. 测试响应解析")
    print("4. 测试边界框转换")
    print("5. 运行离线测试（2+3+4）")

    choice = input("\n请选择测试 (1-5): ").strip()

    if choice == "1":
        await test_qwen()
    elif choice == "2":
        test_factory()
    elif choice == "3":
        test_parser()
    elif choice == "4":
        test_bbox()
    elif choice == "5":
        test_factory()
        test_parser()
        test_bbox()
    else:
        print("无效选择")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
