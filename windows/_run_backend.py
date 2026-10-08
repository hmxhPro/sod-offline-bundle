"""
windows/_run_backend.py
-----------------------
Windows 后端启动器。显式设置 asyncio 使用 ProactorEventLoop —— 训练功能依赖
create_subprocess_exec 派生子进程，而 SelectorEventLoop 不支持子进程会抛
NotImplementedError；Proactor 是 Windows 上的正确选择（新版 uvicorn 默认亦如此，
此处做兜底保证）。

以 backend 目录为工作目录运行（相对路径 ./uploads ./results 及 mobileclip_blt.ts
均依赖 CWD）。端口取环境变量 SOD_BACKEND_PORT，默认 8000。
"""
import os
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 以 uvicorn 字符串导入 app.main:app —— 需保证 backend(=当前工作目录) 在 sys.path 上。
sys.path.insert(0, os.getcwd())

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("SOD_BACKEND_PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_graceful_shutdown=30,
    )
