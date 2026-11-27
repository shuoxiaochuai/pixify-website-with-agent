# deploy.ps1 - push local changes to the bare-metal server.
# Server side counterpart: deploy/bootstrap.sh
#
# ENCODING RULE: this file must stay 100% ASCII, comments included.
# Windows PowerShell 5.1 reads a BOM-less file with the system ANSI codepage (CP936 here).
# A UTF-8 Chinese char ends in a byte that CP936 treats as a lead byte, so the decoder
# swallows the NEXT byte as an invalid trail byte - a closing quote, or even the newline
# at the end of a comment line. Lines then merge, braces get orphaned, and the reported
# line numbers point far away from the real cause.
# CRLF masks this (the \r gets swallowed instead of the \n) and a BOM masks it too, but
# both depend on editors behaving, so neither is a fix. ASCII-only is the only fix.
# Self-check after every edit:
#   $e=$null; [Management.Automation.Language.Parser]::ParseFile("$PWD\deploy.ps1",[ref]$null,[ref]$e); $e
param([switch]$Backend,[switch]$Frontend,[switch]$Env,[switch]$Migrate,[switch]$Sync,[switch]$Clean,[switch]$Up,[switch]$DryRun,[string]$Server="root@sh03-ssh.gpuhome.cc",[int]$Port=30245,[string]$Remote="/root/rivermind-data/ai-picture-editor")

# Never set Stop here: ssh and scp write normal progress to stderr, and 5.1 would turn
# that into a terminating error. Every command is checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

function Step([string]$m) { Write-Host ("==> " + $m) -ForegroundColor Cyan }
function Fail([string]$m) { Write-Host ("FAIL: " + $m) -ForegroundColor Red; exit 1 }

function Invoke-Remote([string]$cmd, [string]$why) {
    if ($DryRun) { Write-Host ("  [dry-run] ssh " + $Server + "  " + $cmd) -ForegroundColor DarkGray; return }
    ssh -p $Port $Server $cmd
    if ($LASTEXITCODE -ne 0) { Fail ($why + " (ssh exit " + $LASTEXITCODE + ")") }
}

# DryRun must block scp as well, otherwise a preview would really upload .env
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
        # Destructive: verify the uploaded tarball is readable BEFORE deleting the old code.
        # The whole remote chain ends with && echo, so any failure gives a non-zero exit.
        $apply = "tar -tzf /tmp/ape-code.tgz > /dev/null && rm -rf backend/app backend/migrations && tar xzf /tmp/ape-code.tgz"
    } else {
        $apply = "tar xzf /tmp/ape-code.tgz"
    }
    Step "upload (scp prints progress on stderr, that is normal)"
    Invoke-Scp $pkg ($Server + ":/tmp/ape-code.tgz") "upload failed, remote extract will not run"
    Invoke-Remote ("cd " + $Remote + " && cp -a backend/app /tmp/app.bak-" + $stamp + " && " + $apply + " && find backend -name __pycache__ -type d -exec rm -rf {} + && echo CODE_UPDATED") ("extract failed, previous code is in /tmp/app.bak-" + $stamp)
}

# The remote command below must not contain a single double quote. PowerShell passes
# arguments to native programs (ssh.exe) through the MSVCRT command line, which eats every
# embedded " as a quoting delimiter, so the remote bash receives an UNQUOTED command.
# Two consequences, both observed in production:
#   1. '"$UVBIN" sync' degraded to running coreutils /bin/sync ("unrecognized option").
#   2. '[ -x "$UVBIN" ]' became '[ -x $UVBIN ]', which is TRUE when the value is empty,
#      so the whole fallback chain was skipped instead of firing.
# Verified locally by handing the same string to bash.exe: the unquoted glob */.venv/* was
# expanded by the shell and find died with "paths must precede expression".
# Therefore: quote-free, and ${UV:-/nonexistent} keeps the word non-empty when the server's
# /etc/retouch/env has no UV= key (it had none until bootstrap.sh regenerated it).
$syncCmd = 'set -a; [ -f /etc/retouch/env ] && . /etc/retouch/env; set +a; cd ' + $Remote + '/backend || exit 1; UVBIN=${UV:-/nonexistent}; [ -x $UVBIN ] || UVBIN=$(command -v uv || echo /nonexistent); [ -x $UVBIN ] || UVBIN=$HOME/.local/bin/uv; [ -x $UVBIN ] || { echo uv not found: run bash deploy/bootstrap.sh; exit 1; }; echo uv=$UVBIN; $UVBIN --version; $UVBIN sync --all-extras'
if ($Sync) { Invoke-Remote $syncCmd "uv sync failed" }

if ($Migrate) { Invoke-Remote ("cd " + $Remote + "/backend && .venv/bin/alembic heads && .venv/bin/alembic current && .venv/bin/alembic upgrade head && .venv/bin/alembic current") "migration failed" }

if ($Env) {
    if (-not (Test-Path .env)) { Fail "no local .env" }
    if (Select-String -Path ".env" -Pattern "dev-only-secret" -Quiet) { Fail "local JWT_SECRET is still the template default, generate a random one first" }
    if ($DryRun) {
        # Upload and diff are one action; a preview previews the whole thing
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
    # Extract into dist.new then swap names; on a first deploy there is no dist yet,
    # so the if/fi guard keeps the && chain from breaking.
    Invoke-Remote ("cd " + $Remote + "/frontend && rm -rf dist.new && mkdir dist.new && tar -C dist.new -xzf /tmp/ape-dist.tgz && if [ -d dist ]; then mv dist dist.old; fi && mv dist.new/dist dist && rm -rf dist.old && echo DIST_UPDATED") "dist swap failed, previous copy may be in dist.old"
}

Invoke-Remote "retouch restart; sleep 8; retouch status; retouch health; ps -o args= -C postgres | head -1" "restart failed, run manually: retouch status; tail -n 60 /var/log/retouch/app.log"
