# ============================================================================
# windows/start.ps1  —  生产模式一键启动（数据库 + 后端 + 前端dist + 浏览器）
# ----------------------------------------------------------------------------
# 后端与前端以后台进程运行，日志写入 windows\logs\；PID 记录到 windows\run\，
# 供「停止.bat」结束。通过根目录「启动.bat」双击运行。
# ============================================================================

. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Stop'

Add-RuntimeToPath
Set-BackendEnv
Ensure-Dir $WinLogDir; Ensure-Dir $PidDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SOD 启动（生产模式）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path $PyExe)) {
  Write-Err "未找到 Python 环境（$PyExe）。请先双击「安装.bat」完成安装。"; exit 1
}

# ── 1. 数据库 ───────────────────────────────────────────────────────────────
Write-Section "1. 启动数据库"
Start-Pg

# ── 2. 后端（uvicorn，后台）────────────────────────────────────────────────
Write-Section "2. 启动后端 (:$BackendPort)"
if (Test-PortInUse $BackendPort) {
  Write-Warn "端口 $BackendPort 已被占用，跳过启动后端（可能已在运行）。"
} else {
  $out = Join-Path $WinLogDir 'backend.out.log'
  $err = Join-Path $WinLogDir 'backend.err.log'
  $env:SOD_BACKEND_PORT = "$BackendPort"
  $p = Start-Process -FilePath $PyExe -PassThru -WindowStyle Hidden `
        -WorkingDirectory $Backend `
        -ArgumentList @((Join-Path $WinDir '_run_backend.py')) `
        -RedirectStandardOutput $out -RedirectStandardError $err
  $p.Id | Out-File -Encoding ascii (Join-Path $PidDir 'backend.pid')
  Write-Info "后端进程 PID=$($p.Id)，日志：$out"
  Write-Info "首次启动会预加载检测模型，可能需要 30~90 秒…"
  if (Wait-Port $BackendPort 180) { Write-Ok "后端已就绪 (:$BackendPort)" }
  else { Write-Warn "等待后端超时；请查看日志 $err" }
}

# ── 3. 前端（serve_frontend.py 托管 dist + 反代 /api）───────────────────────
Write-Section "3. 启动前端 (:$FrontendPort)"
$dist = Join-Path $Frontend 'dist\index.html'
if (-not (Test-Path $dist)) {
  Write-Warn "未找到前端构建产物 frontend\dist。请重跑「安装.bat」或在 frontend 下 npm run build。"
} elseif (Test-PortInUse $FrontendPort) {
  Write-Warn "端口 $FrontendPort 已被占用，跳过启动前端。"
} else {
  $env:FRONTEND_PORT = "$FrontendPort"
  $env:SOD_BACKEND   = "http://127.0.0.1:$BackendPort"
  $out = Join-Path $WinLogDir 'frontend.out.log'
  $err = Join-Path $WinLogDir 'frontend.err.log'
  $p = Start-Process -FilePath $PyExe -PassThru -WindowStyle Hidden `
        -WorkingDirectory $SodRoot `
        -ArgumentList @('serve_frontend.py') `
        -RedirectStandardOutput $out -RedirectStandardError $err
  $p.Id | Out-File -Encoding ascii (Join-Path $PidDir 'frontend.pid')
  Write-Info "前端进程 PID=$($p.Id)，日志：$out"
  if (Wait-Port $FrontendPort 60) { Write-Ok "前端已就绪 (:$FrontendPort)" }
  else { Write-Warn "等待前端超时；请查看日志 $err" }
}

# ── 4. 打开浏览器 ───────────────────────────────────────────────────────────
$url = "http://localhost:$FrontendPort/"
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " 已启动。请访问： $url" -ForegroundColor Green
Write-Host " 停止服务：双击「停止.bat」" -ForegroundColor Green
Write-Host " 日志目录：$WinLogDir" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Start-Process $url
