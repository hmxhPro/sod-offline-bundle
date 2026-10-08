# ============================================================================
# windows/install.ps1  —  SOD Windows 一键安装（首次需联网）
# ----------------------------------------------------------------------------
# 联网下载并就地部署：便携 Node.js、PostgreSQL(zip 二进制)、Miniconda；
# 建 conda 环境并安装 CUDA 版 PyTorch + 全部后端依赖；初始化本地数据库；
# 生成 backend\.env；部署 Ultralytics 字体/配置；安装并构建前端。
#
# 装完即可断网离线使用。请通过根目录的「安装.bat」双击运行（自动放开脚本执行策略，无需管理员）。
# ============================================================================

. "$PSScriptRoot\common.ps1"

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SOD Windows 安装" -ForegroundColor Cyan
Write-Host "   安装根目录 : $SodRoot"
Write-Host "   运行时目录 : $RuntimeDir"
Write-Host "   目标算力   : NVIDIA GPU (CUDA, $TorchIndexUrl)"
Write-Host "============================================================" -ForegroundColor Cyan

# ── 0. 前置检查 ─────────────────────────────────────────────────────────────
Write-Section "0. 前置检查"
if (-not (Test-SafeRootPath)) { exit 1 }
Ensure-Dir $RuntimeDir; Ensure-Dir $CacheDir; Ensure-Dir $WinLogDir; Ensure-Dir $PidDir

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Ok "检测到 nvidia-smi（存在 NVIDIA 驱动）"
  try { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | ForEach-Object { Write-Info $_ } } catch {}
} else {
  Write-Warn "未检测到 nvidia-smi。将仍按 CUDA 版安装；若本机无 NVIDIA 显卡，"
  Write-Warn "请安装后把 backend\.env 里的 DEVICE=cuda:0 改为 DEVICE=cpu（速度较慢）。"
}

# ── 1. 便携 Node.js ─────────────────────────────────────────────────────────
Write-Section "1. 部署便携 Node.js"
if (Test-Path $NodeExe) {
  Write-Ok "Node 已存在，跳过：$NodeExe"
} else {
  # 动态选取最新 v20 LTS；失败则回退固定版本。
  $nodeVer = $NodeFallbackVersion
  try {
    $idx = Invoke-RestMethod -Uri "$NodeDistBase/index.json" -UseBasicParsing -TimeoutSec 60
    $v20 = $idx | Where-Object { $_.version -like 'v20.*' -and $_.lts } | Select-Object -First 1
    if ($v20) { $nodeVer = $v20.version }
  } catch { Write-Warn "无法读取 Node 版本索引，使用回退版本 $NodeFallbackVersion" }
  Write-Info "Node 版本：$nodeVer"

  $nodeZipName = "node-$nodeVer-win-x64.zip"
  $nodeZip = Join-Path $CacheDir $nodeZipName
  Get-Download "$NodeDistBase/$nodeVer/$nodeZipName" $nodeZip
  Expand-Zip $nodeZip $RuntimeDir
  $extracted = Join-Path $RuntimeDir "node-$nodeVer-win-x64"
  if (Test-Path $NodeDir) { Remove-Item -Recurse -Force $NodeDir }
  Move-Item $extracted $NodeDir
  if (-not (Test-Path $NodeExe)) { throw "Node 解压后未找到 node.exe" }
  Write-Ok "Node 就绪：$NodeExe"
}
& $NodeExe --version | ForEach-Object { Write-Info "node $_" }

# ── 2. 便携 PostgreSQL ──────────────────────────────────────────────────────
Write-Section "2. 部署便携 PostgreSQL"
if (Test-Path (Join-Path $PgBin 'postgres.exe')) {
  Write-Ok "PostgreSQL 已存在，跳过：$PgBin"
} else {
  $urls = $PgVersionCandidates | ForEach-Object { $PgUrlTemplate -f $_ }
  $pgZip = Join-Path $CacheDir 'postgresql-win-x64-binaries.zip'
  Get-DownloadFirst $urls $pgZip | Out-Null
  Expand-Zip $pgZip $RuntimeDir     # 解压后即 $RuntimeDir\pgsql\bin\...
  if (-not (Test-Path (Join-Path $PgBin 'postgres.exe'))) { throw "PostgreSQL 解压后未找到 postgres.exe" }
  Write-Ok "PostgreSQL 就绪：$PgBin"
}

# ── 3. Miniconda ────────────────────────────────────────────────────────────
Write-Section "3. 部署 Miniconda（Python 运行时）"
$condaExe = Join-Path $CondaDir 'Scripts\conda.exe'
if (Test-Path $condaExe) {
  Write-Ok "Miniconda 已存在，跳过：$CondaDir"
} else {
  $mc = Join-Path $CacheDir 'Miniconda3-Windows-x86_64.exe'
  Get-Download $MinicondaUrl $mc
  Write-Info "静默安装 Miniconda 到 $CondaDir …"
  # /D 参数必须放最后且不加引号（已校验路径无空格/中文）。
  $p = Start-Process -FilePath $mc -Wait -PassThru -ArgumentList `
    "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$CondaDir"
  if ($p.ExitCode -ne 0) { throw "Miniconda 安装失败（ExitCode=$($p.ExitCode)）" }
  if (-not (Test-Path $condaExe)) { throw "Miniconda 安装后未找到 conda.exe" }
  Write-Ok "Miniconda 就绪：$CondaDir"
}

# ── 4. 创建 conda 应用环境 ──────────────────────────────────────────────────
Write-Section "4. 创建 Python $PythonVersion 环境（conda: $CondaEnvName）"
if (Test-Path $PyExe) {
  Write-Ok "环境已存在，跳过创建：$CondaEnvDir"
} else {
  # 用 conda-forge 并 --override-channels：避开新版 conda 对 defaults 频道的
  # 服务条款(ToS)交互式确认，非交互安装才不会卡住。
  & $condaExe create -y -n $CondaEnvName -c conda-forge --override-channels "python=$PythonVersion"
  if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }
  if (-not (Test-Path $PyExe)) { throw "环境创建后未找到 python.exe" }
  Write-Ok "环境就绪：$PyExe"
}

# ── 5. 安装 Python 依赖（CUDA torch + requirements）─────────────────────────
Write-Section "5. 安装 Python 依赖"
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
& $PyExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "升级 pip 失败" }

Write-Info "安装 CUDA 版 PyTorch（$TorchIndexUrl）…（体积较大，请耐心）"
& $PyExe -m pip install torch torchvision --index-url $TorchIndexUrl
if ($LASTEXITCODE -ne 0) { throw "安装 PyTorch 失败" }

$req = Join-Path $Backend 'requirements.txt'
Write-Info "安装后端依赖 requirements.txt（torch 已排除，不会被覆盖为 CPU 版）…"
& $PyExe -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw "安装 requirements.txt 失败" }

# serve_frontend.py（前端生产托管+反代）依赖 httpx，但未列入 requirements。
Write-Info "安装前端托管所需的 httpx …"
& $PyExe -m pip install httpx
if ($LASTEXITCODE -ne 0) { throw "安装 httpx 失败" }

# YOLOE 开放词表：ultralytics.nn.text_model 在 import 时就需要 clip 模块。
Write-Info "安装 YOLOE 文本编码器 CLIP …"
& $PyExe -m pip install "git+https://github.com/ultralytics/CLIP.git"
if ($LASTEXITCODE -ne 0) { Write-Warn "CLIP 安装失败，开放词表检测将不可用。" }

# ByteTrack 跨帧 ID：缺 cython_bbox 会静默降级为 passthrough。
Write-Info "安装 ByteTrack 依赖 cython_bbox …"
& $PyExe -m pip install cython_bbox
if ($LASTEXITCODE -ne 0) { Write-Warn "cython_bbox 安装失败（通常缺 C++ Build Tools），跟踪将降级为逐帧新 ID。" }

# 验证 torch 能用 CUDA（仅告警，不阻断）。
$cuda = & $PyExe -c "import torch; print(torch.cuda.is_available())" 2>$null
if ("$cuda".Trim() -eq 'True') {
  $gpu = & $PyExe -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
  Write-Ok "PyTorch CUDA 可用：$gpu"
} else {
  Write-Warn "PyTorch 无法使用 CUDA（torch.cuda.is_available()=False）。"
  Write-Warn "常见原因：无 NVIDIA 显卡 / 驱动过旧（cu121 需驱动 >= 527.41）。"
  Write-Warn "可继续，但需把 backend\.env 的 DEVICE 改为 cpu，或更新显卡驱动后重试。"
}

# ── 6. 初始化本地数据库 ─────────────────────────────────────────────────────
Write-Section "6. 初始化 PostgreSQL 任务/数据库"
if (-not (Test-Path (Join-Path $PgData 'PG_VERSION'))) {
  Ensure-Dir $PgData
  Write-Info "initdb 初始化数据目录（localhost trust 认证，UTF8）…"
  # 本地单机使用：host/local 均 trust。数据仅监听 localhost，见下方 start 脚本。
  & (Join-Path $PgBin 'initdb.exe') -D $PgData -U $PgSuperUser -A trust -E UTF8 --locale=C
  if ($LASTEXITCODE -ne 0) { throw "initdb 失败" }
  Write-Ok "数据目录已初始化：$PgData"
} else {
  Write-Ok "数据目录已存在，跳过 initdb"
}

Start-Pg
# 创建应用数据库（幂等）。trust 认证下无需密码。
$exists = & (Join-Path $PgBin 'psql.exe') -h localhost -p $PgPort -U $PgSuperUser -d postgres -tAc `
  "SELECT 1 FROM pg_database WHERE datname='$PgAppDb'" 2>$null
if ("$exists".Trim() -ne '1') {
  & (Join-Path $PgBin 'createdb.exe') -h localhost -p $PgPort -U $PgSuperUser $PgAppDb
  if ($LASTEXITCODE -ne 0) { throw "创建数据库 $PgAppDb 失败" }
  Write-Ok "已创建数据库：$PgAppDb"
} else {
  Write-Ok "数据库已存在：$PgAppDb"
}
# 应用建表在后端启动时由 init_db() 自动完成。

# ── 7. 生成 backend\.env ────────────────────────────────────────────────────
Write-Section "7. 生成后端配置 backend\.env"
$envOut = Join-Path $Backend '.env'
if (Test-Path $envOut) {
  Copy-Item -Force $envOut "$envOut.bak"
  Write-Warn ".env 已存在，备份为 .env.bak 后覆盖"
}
$rootFwd = $SodRoot -replace '\\','/'
# 模板是 UTF-8：必须显式指定编码。PS 5.1 的 Get-Content 默认按系统 ANSI(GBK) 解码，
# 中文行尾字节会吞掉 CRLF 的 CR，导致下一行并入注释、DATABASE_URL 替换失配（已踩坑）。
$tpl = Get-Content (Join-Path $SodRoot 'env.template') -Raw -Encoding UTF8
$tpl = $tpl -replace '__SOD_HOME__', $rootFwd
$tpl = [regex]::Replace($tpl, '(?m)^DATABASE_URL=.*$', ('DATABASE_URL=' + $DatabaseUrl))
# 无 BOM 写出，避免 python-dotenv 解析首行异常。
[System.IO.File]::WriteAllText($envOut, $tpl, (New-Object System.Text.UTF8Encoding($false)))
Write-Ok "已写入 $envOut（DEVICE=cuda:0；权重指向 backend/models/yolo；DATABASE_URL 指向本地库）"

# 权重存在性检查（仅告警）。
foreach ($w in @('models\yolo\yoloe-11l-seg.pt','models\yolo\yolo11l.pt')) {
  if (Test-Path (Join-Path $Backend $w)) { Write-Ok "权重就绪：$w" }
  else { Write-Warn "缺失权重：backend\$w（检测/训练相应功能将不可用）" }
}

# ── 8. Ultralytics 字体 / 配置 ──────────────────────────────────────────────
Write-Section "8. 部署 Ultralytics 字体与配置"
$cfgDir = Join-Path $YoloConfigDir 'Ultralytics'
$settings = Join-Path $cfgDir 'settings.json'
if (Test-Path $settings) {
  & $PyExe (Join-Path $WinDir '_fix_ultra_settings.py') $settings $Backend
} else {
  Write-Warn "未找到 $settings（应随包提供）。中文标注字体可能不生效。"
}
if (Test-Path $CjkFont) { Write-Ok "中文字体就绪：$CjkFont" }
else { Write-Warn "缺失中文字体 Arial.Unicode.ttf，检测框中文可能显示为方块。" }

# YOLOE 文本编码器：Linux 用软链，Windows 直接拷贝一份到 backend\weights\ 以防万一。
$mobileSrc = Join-Path $Backend 'mobileclip_blt.ts'
$weightsDir = Join-Path $Backend 'weights'
if (Test-Path $mobileSrc) {
  Ensure-Dir $weightsDir
  Copy-Item -Force $mobileSrc (Join-Path $weightsDir 'mobileclip_blt.ts')
  Write-Ok "已拷贝 mobileclip_blt.ts → backend\weights\"
} else {
  Write-Warn "缺失 backend\mobileclip_blt.ts（YOLOE 开放词表检测离线必需）。"
}

# ── 9. 前端：安装依赖 + 构建 dist ───────────────────────────────────────────
Write-Section "9. 安装并构建前端（支持 npm run dev 与后端托管 dist 两种模式）"
Add-RuntimeToPath
$env:npm_config_loglevel = 'warn'
Push-Location $Frontend
try {
  Write-Info "npm install …（首次较慢）"
  & $NpmCmd install
  if ($LASTEXITCODE -ne 0) { throw "npm install 失败" }
  Write-Info "npm run build（生成 frontend\dist，供后端托管的生产模式使用）…"
  & $NpmCmd run build
  if ($LASTEXITCODE -ne 0) { throw "npm run build 失败" }
  Write-Ok "前端依赖与 dist 就绪"
} finally {
  Pop-Location
}

# ── 完成 ────────────────────────────────────────────────────────────────────
$elapsed = [int]((Get-Date) - $startTime).TotalSeconds
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " 安装完成（用时 ${elapsed}s）" -ForegroundColor Green
Write-Host "------------------------------------------------------------"
Write-Host " 生产模式（推荐，双击）: 启动.bat"
Write-Host "     → 起 数据库 + 后端(:$BackendPort) + 前端(:$FrontendPort)，自动打开浏览器"
Write-Host " 开发模式（含热更新）  : 启动-开发模式.bat"
Write-Host "     → 起 数据库 + 后端(:$BackendPort) + Vite dev(:$DevPort)"
Write-Host " 停止所有服务          : 停止.bat"
Write-Host "------------------------------------------------------------"
Write-Host " 如需 LLM 自动标注：把 API Key 填入 backend\.env 的 LLM_API_KEY。"
Write-Host " 详细说明见：Windows部署文档.md"
Write-Host "============================================================" -ForegroundColor Green
