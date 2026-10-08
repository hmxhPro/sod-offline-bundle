#!/usr/bin/env python
"""
serve_frontend.py — 离线前端静态服务 + /api 反向代理（仅依赖打包环境，无需 nginx / node）
------------------------------------------------------------------------------------------
用打包好的 conda 环境运行即可：前端与 /api 同源（同一端口），因此无需写死服务器 IP、
无需重新构建前端。已支持 SSE 流式推送（实时检测帧）与大文件上传（请求/响应均流式转发）。

用法（在安装根目录下）：
    # 默认监听 8080，反代到本机 8000 的后端
    ./env/bin/python serve_frontend.py
    # 自定义端口 / 后端地址：
    FRONTEND_PORT=80 SOD_BACKEND=http://127.0.0.1:8000 ./env/bin/python serve_frontend.py
    # 后台运行：
    nohup ./env/bin/python serve_frontend.py > frontend.out 2>&1 &

然后浏览器访问： http://<服务器IP>:8080/
"""
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

SOD_HOME = Path(__file__).resolve().parent
DIST = (SOD_HOME / "frontend" / "dist").resolve()
BACKEND = os.environ.get("SOD_BACKEND", "http://127.0.0.1:8000")
PORT = int(os.environ.get("FRONTEND_PORT", "8080"))

# 逐跳(hop-by-hop)头部不应转发
_HOP = {
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade",
}

app = FastAPI(title="SOD Frontend + API proxy")
_client = httpx.AsyncClient(base_url=BACKEND, timeout=None)


@app.api_route("/api/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_api(path: str, request: Request):
    """把 /api/* 透传到后端，请求体与响应体均流式（兼容 SSE 与大文件上传）。"""
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    headers = [(k, v) for (k, v) in request.headers.raw
               if k.decode("latin1").lower() not in _HOP]
    req = _client.build_request(request.method, url,
                                headers=headers, content=request.stream())
    resp = await _client.send(req, stream=True)
    media = resp.headers.get("content-type")
    out_headers = {k.decode("latin1"): v.decode("latin1")
                   for (k, v) in resp.headers.raw
                   if k.decode("latin1").lower() not in _HOP | {"content-type"}}
    return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code,
                             headers=out_headers, media_type=media,
                             background=BackgroundTask(resp.aclose))


@app.get("/{full_path:path}")
async def spa(full_path: str):
    """静态文件；找不到则回退 index.html（支持前端路由刷新不 404）。"""
    candidate = (DIST / full_path).resolve()
    if full_path and str(candidate).startswith(str(DIST)) and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(DIST / "index.html")


if __name__ == "__main__":
    if not (DIST / "index.html").is_file():
        raise SystemExit(f"✗ 未找到前端构建产物：{DIST}/index.html")
    print(f"[serve_frontend] 前端目录: {DIST}")
    print(f"[serve_frontend] 反代后端: {BACKEND}")
    print(f"[serve_frontend] 监听:     http://0.0.0.0:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
