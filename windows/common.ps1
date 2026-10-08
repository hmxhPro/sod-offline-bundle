# ============================================================================
# windows/common.ps1  —  SOD Windows 脚本共享库（被其它 .ps1 dot-source 引用）
# ----------------------------------------------------------------------------
# 统一定义：安装根目录、各组件路径、下载地址/版本、日志与工具函数。
# 不要直接运行本文件；它只提供变量与函数。
# ============================================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# 老 Win10 的 PS 5.1 默认可能不启用 TLS 1.2，而 nodejs.org / EDB / Anaconda 均已
# 要求 TLS 1.2+；用 -bor 在现有协议上追加，不影响新系统的 SystemDefault。
try {
  [Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {}

# ── 路径 ────────────────────────────────────────────────────────────────────
# 本文件位于 <安装根>\windows\ 下，向上一级即安装根目录（SOD_HOME）。
$Script:SodRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Script:WinDir    = $PSScriptRoot
$Script:Backend   = Join-Path $SodRoot 'backend'
$Script:Frontend  = Join-Path $SodRoot 'frontend'

$Script:RuntimeDir = Join-Path $WinDir 'runtime'      # 便携组件解压/安装到这里
$Script:CacheDir   = Join-Path $WinDir 'downloads'    # 安装包缓存（可安装后删除）
$Script:WinLogDir  = Join-Path $WinDir 'logs'         # 服务/安装日志
$Script:PgData     = Join-Path $WinDir 'pgdata'       # PostgreSQL 数据目录
$Script:PidDir     = Join-Path $WinDir 'run'          # 记录后端/前端 PID

$Script:NodeDir    = Join-Path $RuntimeDir 'node'      # node.exe / npm.cmd 直接位于此
$Script:PgDir      = Join-Path $RuntimeDir 'pgsql'     # EDB zip 解压后 bin\ 在此下
$Script:CondaDir   = Join-Path $RuntimeDir 'miniconda' # Miniconda base

# conda 应用环境（Python + torch + 依赖）
$Script:CondaEnvName = 'sod'
$Script:CondaEnvDir  = Join-Path $CondaDir "envs\$CondaEnvName"
$Script:PyExe        = Join-Path $CondaEnvDir 'python.exe'
$Script:EnvScripts   = Join-Path $CondaEnvDir 'Scripts'

# 关键二进制
$Script:NodeExe  = Join-Path $NodeDir 'node.exe'
$Script:NpmCmd   = Join-Path $NodeDir 'npm.cmd'
$Script:PgBin    = Join-Path $PgDir 'bin'
$Script:CondaBat = Join-Path $CondaDir 'condabin\conda.bat'

# Ultralytics 配置/字体目录（对应 Linux 的 YOLO_CONFIG_DIR=<root>\.config）
$Script:YoloConfigDir = Join-Path $SodRoot '.config'
$Script:CjkFont       = Join-Path $SodRoot '.config\Ultralytics\Arial.Unicode.ttf'

# ── 端口（可用环境变量覆盖）────────────────────────────────────────────────
$Script:BackendPort  = if ($env:SOD_BACKEND_PORT)  { [int]$env:SOD_BACKEND_PORT }  else { 8000 }
$Script:FrontendPort = if ($env:SOD_FRONTEND_PORT) { [int]$env:SOD_FRONTEND_PORT } else { 8080 }
$Script:DevPort      = if ($env:SOD_DEV_PORT)      { [int]$env:SOD_DEV_PORT }      else { 5173 }
$Script:PgPort       = if ($env:SOD_PG_PORT)       { [int]$env:SOD_PG_PORT }       else { 5432 }

# ── 数据库（本地单机，localhost trust 认证）───────────────────────────────
$Script:PgSuperUser = 'postgres'
$Script:PgAppDb     = 'sod'
# asyncpg 连接串：trust 认证下密码被忽略，这里给占位密码即可。
$Script:DatabaseUrl = "postgresql+asyncpg://$PgSuperUser`:postgres@localhost:$PgPort/$PgAppDb"

# ── 下载地址与版本（如官方地址变动，改这里即可）────────────────────────────
# Node.js：优先从 index.json 动态选取最新 v20 LTS；失败则用此回退版本。
$Script:NodeFallbackVersion = 'v20.18.1'
$Script:NodeDistBase        = 'https://nodejs.org/dist'

# PostgreSQL：EnterpriseDB 官方“免安装二进制 zip”。按顺序尝试这些版本，
# 命中第一个可下载的即用（EDB 偶尔下线旧小版本）。
$Script:PgVersionCandidates = @('16.4-1', '16.6-1', '17.2-1', '16.3-1')
$Script:PgUrlTemplate       = 'https://get.enterprisedb.com/postgresql/postgresql-{0}-windows-x64-binaries.zip'

# Miniconda（Windows x64 最新版静默安装）。
$Script:MinicondaUrl = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'

# PyTorch CUDA 轮子索引。cu121 对 NVIDIA 驱动兼容性最广（>=527.41 即可）。
# 若目标机为很新的驱动且想用更高 CUDA，可改为 cu124 / cu126。
$Script:TorchIndexUrl = 'https://download.pytorch.org/whl/cu128'

# Python 版本（与原 Linux 环境一致）。
$Script:PythonVersion = '3.10'

# ── 日志/输出 ───────────────────────────────────────────────────────────────
function Write-Section([string]$msg) { Write-Host "`n==== $msg ====" -ForegroundColor Cyan }
function Write-Ok   ([string]$msg) { Write-Host "  [OK] $msg"   -ForegroundColor Green }
function Write-Warn ([string]$msg) { Write-Host "  [!]  $msg"   -ForegroundColor Yellow }
function Write-Err  ([string]$msg) { Write-Host "  [X]  $msg"   -ForegroundColor Red }
function Write-Info ([string]$msg) { Write-Host "  - $msg"      -ForegroundColor Gray }

# ── 通用工具 ────────────────────────────────────────────────────────────────

function Ensure-Dir([string]$path) {
  if (-not (Test-Path $path)) { New-Item -ItemType Directory -Force -Path $path | Out-Null }
}

# 校验安装根路径是否为纯 ASCII、无空格 —— Miniconda /D 与部分工具链对中文/空格
# 路径处理很脆弱，务必提前拦截给出可读错误。
function Test-SafeRootPath {
  $ascii = ($SodRoot -match '^[\x20-\x7E]+$')          # 全为可见 ASCII
  $hasSpace = ($SodRoot -match ' ')
  if (-not $ascii) {
    Write-Err "安装路径包含非 ASCII 字符（如中文）：$SodRoot"
    Write-Err "请把整个 sod-offline-bundle 文件夹移动到纯英文、无空格路径，例如 C:\sod ，再重试。"
    return $false
  }
  if ($hasSpace) {
    Write-Err "安装路径包含空格：$SodRoot"
    Write-Err "请移动到无空格路径，例如 C:\sod ，再重试。"
    return $false
  }
  # 软告警：路径过深易触发 Windows 260 字符上限（conda + torch 目录很深）。
  if ($SodRoot.Length -gt 32) {
    Write-Warn "安装路径较长（$($SodRoot.Length) 字符）：$SodRoot"
    Write-Warn "conda + PyTorch 目录很深，路径过长可能触发 Windows 260 字符上限导致装包失败。"
    Write-Warn "强烈建议把整个文件夹放到 C:\sod 这类短路径下再运行。"
  }
  return $true
}

# 带重试的下载（缓存到 CacheDir，已存在且非空则跳过）。
function Get-Download([string]$url, [string]$outFile, [int]$retries = 3) {
  if ((Test-Path $outFile) -and ((Get-Item $outFile).Length -gt 0)) {
    Write-Info "已缓存，跳过下载：$(Split-Path $outFile -Leaf)"
    return
  }
  Ensure-Dir (Split-Path $outFile -Parent)
  $tmp = "$outFile.part"
  for ($i = 1; $i -le $retries; $i++) {
    try {
      Write-Info "下载（$i/$retries）：$url"
      $old = $ProgressPreference
      $ProgressPreference = 'SilentlyContinue'   # 关进度条，Invoke-WebRequest 快很多
      Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing -TimeoutSec 1800
      $ProgressPreference = $old
      if ((Get-Item $tmp).Length -le 0) { throw "下载文件为空" }
      Move-Item -Force $tmp $outFile
      Write-Ok "下载完成：$(Split-Path $outFile -Leaf)"
      return
    } catch {
      Write-Warn "下载失败：$($_.Exception.Message)"
      if (Test-Path $tmp) { Remove-Item -Force $tmp -ErrorAction SilentlyContinue }
      if ($i -eq $retries) { throw "多次下载失败：$url" }
      Start-Sleep -Seconds 3
    }
  }
}

# 尝试多个候选 URL，返回第一个成功下载的本地文件路径。
function Get-DownloadFirst([string[]]$urls, [string]$outFile) {
  if ((Test-Path $outFile) -and ((Get-Item $outFile).Length -gt 0)) {
    Write-Info "已缓存，跳过下载：$(Split-Path $outFile -Leaf)"; return $outFile
  }
  foreach ($u in $urls) {
    try { Get-Download $u $outFile; return $outFile } catch { Write-Warn "换下一个地址…" }
  }
  throw "所有候选地址均下载失败。"
}

function Expand-Zip([string]$zip, [string]$dest) {
  Ensure-Dir $dest
  Write-Info "解压 $(Split-Path $zip -Leaf) → $dest"
  Expand-Archive -Path $zip -DestinationPath $dest -Force
}

# 端口占用检测。
function Test-PortInUse([int]$port) {
  try {
    $c = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    return ($null -ne $c)
  } catch {
    # 老系统没有 Get-NetTCPConnection，退回 netstat。
    $r = netstat -ano | Select-String ":$port\s" | Select-String 'LISTENING'
    return ($null -ne $r)
  }
}

# 等待某端口进入监听状态（用于等后端就绪）。
function Wait-Port([int]$port, [int]$timeoutSec = 120) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    if (Test-PortInUse $port) { return $true }
    Start-Sleep -Milliseconds 800
  }
  return $false
}

# 后端启动时注入的离线环境变量（对应 Linux run_backend.sh）。
function Set-BackendEnv {
  $env:HF_HUB_OFFLINE        = '1'
  $env:TRANSFORMERS_OFFLINE  = '1'
  $env:HF_DATASETS_OFFLINE   = '1'
  $env:YOLO_OFFLINE          = '1'
  $env:ULTRALYTICS_OFFLINE   = '1'
  $env:YOLO_CONFIG_DIR       = $YoloConfigDir
  $env:SOD_CJK_FONT          = $CjkFont
  $env:OPENCV_FFMPEG_LOGLEVEL = '-8'
  $env:AV_LOG_FORCE_NOCOLOR  = '1'
  $env:PYTHONUNBUFFERED       = '1'
}

# 把便携 Node 与 conda 环境 Scripts 放进当前进程 PATH 头部。
function Add-RuntimeToPath {
  $paths = @()
  if (Test-Path $NodeDir)    { $paths += $NodeDir }
  if (Test-Path $EnvScripts) { $paths += $EnvScripts }
  if (Test-Path $CondaEnvDir){ $paths += $CondaEnvDir }
  if (Test-Path $PgBin)      { $paths += $PgBin }
  if ($paths.Count -gt 0) { $env:PATH = ($paths -join ';') + ';' + $env:PATH }
}

# ── PostgreSQL 便携实例控制 ─────────────────────────────────────────────────

function Test-PgRunning {
  if (-not (Test-Path $PgData)) { return $false }
  & (Join-Path $PgBin 'pg_ctl.exe') status -D $PgData 2>&1 | Out-Null
  return ($LASTEXITCODE -eq 0)
}

function Start-Pg {
  if (Test-PgRunning) { Write-Ok "PostgreSQL 已在运行（端口 $PgPort）"; return }
  Ensure-Dir $WinLogDir
  $log = Join-Path $WinLogDir 'postgres.log'
  Write-Info "启动 PostgreSQL（数据目录 $PgData，端口 $PgPort）…"
  & (Join-Path $PgBin 'pg_ctl.exe') -D $PgData -l $log -o "-p $PgPort" -w start
  if ($LASTEXITCODE -ne 0) { throw "PostgreSQL 启动失败，详见 $log" }
  Write-Ok "PostgreSQL 已启动"
}

function Stop-Pg {
  if (-not (Test-Path (Join-Path $PgBin 'pg_ctl.exe'))) { return }
  if (Test-PgRunning) {
    Write-Info "停止 PostgreSQL…"
    & (Join-Path $PgBin 'pg_ctl.exe') -D $PgData -m fast stop 2>&1 | Out-Null
    Write-Ok "PostgreSQL 已停止"
  }
}
