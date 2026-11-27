# 裸机部署推送脚本（本地 Windows 侧运行，配合服务器上的 deploy/bootstrap.sh）
#
# 编码约束：本文件的字符串一律只用 ASCII，中文只写在注释里。
# 原因是 Windows PowerShell 5.1 读取无 BOM 文件时使用系统 ANSI 码页（GBK），
# 中文字节的最后一个往往是 GBK 前导字节，会把紧随其后的闭合引号当配对尾字节吞掉，
# 于是字符串不再闭合，报出一堆假的 "&& 不是有效语句分隔符"，且报错行号远离真正的元凶。
# 换行符不是合法的 GBK 尾字节，所以注释里的中文怎么乱码都不会破坏语法结构。
# 验证方法（改完必跑）：
#   $e=$null; [Management.Automation.Language.Parser]::ParseFile("$PWD\deploy.ps1",[ref]$null,[ref]$e); $e
param([switch]$Backend,[switch]$Frontend,[switch]$Env,[switch]$Migrate,[switch]$Sync,[switch]$Clean,[switch]$Up,[switch]$DryRun,[string]$Server="root@sh03-ssh.gpuhome.cc",[int]$Port=30245,[string]$Remote="/root/rivermind-data/ai-picture-editor")

# 不能设 Stop：ssh/scp 的正常输出与进度全走 stderr，5.1 会把它当异常抛出，脚本会半途死掉。
# 因此每条命令后显式判退出码。
$ErrorActionPreference = "Continue"

function Step([string]$m) { Write-Host ("==> " + $m) -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host ("FAIL: " + $m) -ForegroundColor Red; exit 1 }

function Invoke-Remote([string]$cmd, [string]$why) {
    if ($DryRun) { Write-Host ("  [dry-run] ssh " + $Server + "  " + $cmd) -ForegroundColor DarkGray; return }
    ssh -p $Port $Server $cmd
    if ($LASTEXITCODE -ne 0) { Fail ($why + " (ssh exit " + $LASTEXITCODE + ")") }
}

# DryRun 必须同时挡住 scp，否则"只想看看"会把 .env 真的推上去
function Invoke-Scp([string]$src, [string]$dst, [string]$why) {
    if ($DryRun) { Write-Host ("  [dry-run] scp " + $src + " -> " + $dst) -ForegroundColor DarkGray; return }
    scp -P $Port $src $dst
    if ($LASTEXITCODE -ne 0) { Fail $why }
}

if (-not ($Backend -or $Frontend -or $Env -or $Migrate -or $Sync -or $Up)) { $Backend = $true }

if ($Up) { Invoke-Remote "retouch up; sleep 6; retouch status; retouch health" "cannot start services"; exit }

if ($dirty = git status --porcelain) { Write-Host "WARN: local changes are uncommitted, commit first" -ForegroundColor Yellow }

$pkg = Join-Path $env:TEMP "ape-code.tgz"

if ($Backend) {
    if (-not (Test-Path backend\pyproject.toml)) { Fail "not a repo root, cd into the project first" }
    $ex = @("--exclude=.venv","--exclude=node_modules","--exclude=__pycache__","--exclude=.pytest_cache","--exclude=.ruff_cache","--exclude=dist","--exclude=.env","--exclude=.idea","--exclude=.git")
    Step "pack sources (backend, migrations, frontend/src, deploy)"
    tar -czf $pkg $ex .
    if ($LASTEXITCODE -ne 0) { Fail "local tar failed" }
    $list = tar -tzf $pkg
    if ($LASTEXITCODE -ne 0) { Fail "cannot read the tarball, is tar available?" }
    Write-Host ("  entries: " + @($list).Count)
    $bad = $list | Where-Object { $_ -match "/\.venv/" -or $_ -match "frontend/dist/" -or $_ -match "node_modules/" }
    if ($bad) { $bad | Select-Object -First 5; Fail "tarball contains files that must not be shipped" }
    if (-not (Test-Path $pkg)) { Fail ("cannot find " + $pkg) }

    $stamp = Get-Date -Format "MMdd-HHmm"
    if ($Clean) {
        # 破坏性操作：先确认远端那份 tarball 完整可读，再删旧代码
        $apply = "tar -tzf /tmp/ape-code.tgz > /dev/null && rm -rf backend/app backend/migrations && tar xzf /tmp/ape-code.tgz"
    } else {
        $apply = "tar xzf /tmp/ape-code.tgz"
    }
    Step "upload (scp prints progress on stderr, that is normal)"
    Invoke-Scp $pkg ($Server + ":/tmp/ape-code.tgz") "upload failed, remote extract will not run"
    Invoke-Remote ("cd " + $Remote + " && cp -a backend/app /tmp/app.bak-" + $stamp + " && " + $apply + " && find backend -name __pycache__ -type d -exec rm -rf {} + && echo CODE_UPDATED") ("extract failed, previous code is in /tmp/app.bak-" + $stamp)
}

if ($Sync) { Invoke-Remote ("bash -lc 'cd " + $Remote + "/backend && uv sync --all-extras'") "uv sync failed" }

if ($Migrate) { Invoke-Remote ("cd " + $Remote + "/backend && .venv/bin/alembic heads && .venv/bin/alembic current && .venv/bin/alembic upgrade head && .venv/bin/alembic current") "migration failed" }

if ($Env) {
    if (-not (Test-Path .env)) { Fail "no local .env" }
    if (Select-String -Path ".env" -Pattern "dev-only-secret" -Quiet) { Fail "local JWT_SECRET is still the template default, generate a random one first" }
    if ($DryRun) {
        # 上传与比对是一件事，只预览就预览整件事，不去 diff 一个没上传的旧文件
        Write-Host ("  [dry-run] scp .env -> " + $Server + ":/tmp/env.new, diff against the server copy, then ask y/N before install -m 600") -ForegroundColor DarkGray
    } else {
        Invoke-Scp ".env" ($Server + ":/tmp/env.new") "env upload failed"
        ssh -p $Port $Server ("cp -a " + $Remote + "/.env /tmp/env.server.bak; diff " + $Remote + "/.env /tmp/env.new")
        $yes = Read-Host "overwrite the server .env with the local one? y/N"
        if ($yes -eq "y") {
            Invoke-Remote ("install -m 600 /tmp/env.new " + $Remote + "/.env && grep -E '^(APP_ENV|JWT_SECRET|MATTING_PROVIDER|OCR_PROVIDER|DATABASE_URL|REDIS_URL)=' " + $Remote + "/.env") "env overwrite failed"
            Write-Host "NOTE: if DATABASE_URL/REDIS_URL/S3_ENDPOINT changed, re-run bash deploy/bootstrap.sh so supervisor config follows them" -ForegroundColor Yellow
        }
    }
}

if ($Frontend) {
    Step "build frontend"
    if ($DryRun) {
        Write-Host "  [dry-run] npm run build" -ForegroundColor DarkGray
    } else {
        Push-Location frontend
        npm.cmd run build
        $buildCode = $LASTEXITCODE
        Pop-Location
        if ($buildCode -ne 0) { Fail "frontend build failed, nothing to ship" }
        tar -czf (Join-Path $env:TEMP "ape-dist.tgz") -C frontend dist
        if ($LASTEXITCODE -ne 0) { Fail "dist pack failed" }
    }
    Invoke-Scp (Join-Path $env:TEMP "ape-dist.tgz") ($Server + ":/tmp/ape-dist.tgz") "dist upload failed"
    # 先解到 dist.new 再原子换名；首次部署时没有 dist，if/fi 保证不误报
    Invoke-Remote ("cd " + $Remote + "/frontend && rm -rf dist.new && mkdir dist.new && tar -C dist.new -xzf /tmp/ape-dist.tgz && if [ -d dist ]; then mv dist dist.old; fi && mv dist.new/dist dist && rm -rf dist.old && echo DIST_UPDATED") "dist swap failed, previous copy may be in dist.old"
}

Invoke-Remote "retouch restart; sleep 8; retouch status; retouch health; ps -o args= -C postgres | head -1" "restart failed, run manually: retouch status; retouch logs app"
