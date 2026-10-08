# ============================================================================
# windows/start-dev.ps1  —  开发模式启动（数据库 + 后端 + Vite 热更新前端）
# ----------------------------------------------------------------------------
# 后端后台运行；前端用 Vite dev server（热更新），前台占用本窗口，关闭窗口即停前端。
# Vite 通过代理把 /api 转发到 http://localhost:8000，因此后端端口需保持 8000。
# 通过根目录「启动-开发模式.bat」双击运行。
# ============================================================================

. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Stop'

Add-RuntimeToPath
Set-BackendEnv
Ensure-Dir $WinLogDir; Ensure-Dir $PidDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SOD 启动（开发模式 / Vite 热更新）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Test-Path $PyExe)) {
  Write-Err "未找到 Python 环境（$PyExe）。请先双击「安装.bat」完成安装。"; exit 1
}
if ($BackendPort -ne 8000) {
  Write-Warn "开发模式下 Vite 代理写死转发到 localhost:8000（见 frontend\vite.config.js）。"
  Write-Warn "当前 SOD_BACKEND_PORT=$BackendPort 与之不符，/api 请求会失败。建议保持 8000。"
}

# ── 1. 数据库 ───────────────────────────────────────────────────────────────
Write-Section "1. 启动数据库"
Start-Pg

# ── 2. 后端（后台）─────────────────────────────────────────────────────────
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
  Write-Info "后端 PID=$($p.Id)，日志：$out（首次预加载模型约 30~90 秒）"
  if (Wait-Port $BackendPort 180) { Write-Ok "后端已就绪 (:$BackendPort)" }
  else { Write-Warn "等待后端超时；请查看日志 $err" }
}

# ── 3. 前端（Vite dev，前台运行）────────────────────────────────────────────
Write-Section "3. 启动前端热更新 (:$DevPort)"
$url = "http://localhost:$DevPort/"
Write-Host " Vite 就绪后请访问： $url" -ForegroundColor Green
Write-Host " 关闭本窗口 / Ctrl+C 即停止前端；后端仍在后台，停止请双击「停止.bat」。" -ForegroundColor Green
# 稍后自动打开浏览器（给 Vite 一点启动时间）。
Start-Job -ScriptBlock { param($u) Start-Sleep 5; Start-Process $u } -ArgumentList $url | Out-Null

Push-Location $Frontend
try {
  & $NpmCmd run dev -- --host --port $DevPort
} finally {
  Pop-Location
}
