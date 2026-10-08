"""
app/utils/naming.py
-------------------
数据集/类别的底层唯一名带 14 位时间戳后缀（如 ``船只_20260620153012``，撞名再加 ``_1``）
以区分多次标注/训练。但**检测结果框上的类别标签**、训练 dataset.yaml 的 ``names``
应使用去掉时间戳的"干净名"（``船只``），避免把时间戳暴露给非技术用户。

``strip_timestamp_suffix`` 与前端 ``utils/displayName.js::splitTimestampName`` 行为一致：
仅当后缀确为合法的 ``YYYYMMDDHHMMSS`` 时间戳时才剥离，否则原样返回（不会误删名字里本就
存在的数字）。
"""

from __future__ import annotations

import re
from datetime import datetime

_TS_SUFFIX = re.compile(r"^(.*)_(\d{14})(?:_\d+)?$")


def strip_timestamp_suffix(name: str) -> str:
    """去掉类别名末尾的 ``_YYYYMMDDHHMMSS``（及可选撞名序号 ``_n``），返回干净展示名。

    后缀非合法日期时间则原样返回，避免误伤名字里恰好结尾的 14 位数字。
    """
    if not name:
        return name
    m = _TS_SUFFIX.match(name)
    if not m:
        return name
    base, digits = m.group(1), m.group(2)
    try:
        datetime.strptime(digits, "%Y%m%d%H%M%S")
    except ValueError:
        return name
    return base or name
