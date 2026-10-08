# ============================================================================
# windows/stop.ps1  —  停止 SOD 所有服务（前端、后端、数据库）
# ----------------------------------------------------------------------------
# 读取 windows\run\ 下记录的 PID 结束后端/前端进程，并停止 PostgreSQL。
# 通过根目录「停止.bat」双击运行。
# ============================================================================

. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SOD 停止" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

function Stop-ByPidFile([string]$name, [string]$file) {
  if (-not (Test-Path $file)) { Write-Info "$name 无 PID 记录，跳过"; return }
  $procId = (Get-Content $file -Raw).Trim()
  if ($procId -match '^\d+$') {
    $p = Get-Process -Id ([int]$procId) -ErrorAction SilentlyContinue
    if ($p) {
      # /T 连带结束子进程（如 uvicorn reload / 训练子进程）。
      taskkill /F /T /PID $procId 2>&1 | Out-Null
      Write-Ok "已停止 $name (PID=$procId)"
    } else {
      Write-Info "$name (PID=$procId) 已不在运行"
    }
  }
  Remove-Item -Force $file -ErrorAction SilentlyContinue
}

Write-Section "停止前端 / 后端"
Stop-ByPidFile '前端' (Join-Path $PidDir 'frontend.pid')
Stop-ByPidFile '后端' (Join-Path $PidDir 'backend.pid')

Write-Section "停止数据库"
Stop-Pg

Write-Host "`n[完成] 所有服务已停止。" -ForegroundColor Green
