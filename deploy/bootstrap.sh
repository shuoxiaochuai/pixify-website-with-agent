#!/usr/bin/env bash
# 裸机（无 docker daemon）部署重建脚本，在服务器上以 root 执行：
#
#   bash deploy/bootstrap.sh                        # 全量重建/补齐，可反复执行
#   bash deploy/bootstrap.sh --check                # 只报告缺什么，不改任何东西
#   bash deploy/bootstrap.sh --skip-apt             # 跳过系统包安装（离线或已装好时）
#   bash deploy/bootstrap.sh --skip-sync --skip-migrate   # 只重建系统侧文件，不动 venv 与库
#
# 选项：--repo=DIR  仓库位置，默认取脚本自身所在目录的上一层
#       --svc=DIR   数据目录，默认 /root/rivermind-data/retouch-svc，无该盘则 /srv/retouch
#
# 端口与凭据不写死在脚本里，而是从 $REPO/.env 解析，因此改 .env 后重跑即全链路一致。
#
# 覆盖的对象全部位于数据盘之外，因此实例被重置后需要重跑本脚本：
#   /etc/retouch/env            路径与端口的唯一来源
#   /etc/retouch/supervisord.conf  进程定义（绝对路径在此烧死）
#   /usr/local/bin/retouch      统一操作入口
#   /usr/local/bin/retouch-backup  数据库 + 对象存储打包备份
#   apt 包 / /usr/local/bin/minio / uv
#
# 数据（$SVC/pg、$SVC/minio-data）与代码（$REPO）都在仓库/数据盘一侧，本脚本不会删除它们。

set -euo pipefail

if [[ -n "${1:-}" && "$1" != *"--"* ]]; then
  echo "首个参数必须是选项，收到：$1" >&2
  exit 2
fi

SKIP_APT=0
SKIP_SYNC=0
SKIP_MIGRATE=0
CHECK_ONLY=0
OPT_SVC=""
OPT_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    --skip-sync) SKIP_SYNC=1 ;;
    --skip-migrate) SKIP_MIGRATE=1 ;;
    --repo=*) OPT_REPO="${1#*=}" ;;
    --svc=*) OPT_SVC="${1#*=}" ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed '$d'
      exit 0
      ;;
    *) ;;
  esac
  shift
done

log() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m警告\033[0m %s\n' "$*"; }
die() { printf '\033[31m错误\033[0m %s\n' "$*" >&2; exit 1; }

MISSING=()

# ---------------------------------------------------------------- 路径与配置
# 脚本自身位置决定了仓库根，因此仓库放在哪里都不影响重建。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "$OPT_REPO" ]] && REPO="$OPT_REPO"
[[ -f "$REPO/backend/pyproject.toml" ]] || die "仓库路径不对：$REPO 下没有 backend/pyproject.toml"

# 已有的 env 优先，保证重复执行不会把路径改回默认值。
if [[ -z "$OPT_SVC" && -f /etc/retouch/env ]]; then
  OPT_SVC="$(sed -n 's|^SVC=||p' /etc/retouch/env | head -1)"
fi
if [[ -n "$OPT_SVC" ]]; then
  SVC="$OPT_SVC"
elif [[ -d /root/rivermind-data ]]; then
  SVC=/root/rivermind-data/retouch-svc
else
  SVC=/srv/retouch
fi

ENV_FILE="$REPO/.env"
[[ -f "$ENV_FILE" ]] || die "缺少 $ENV_FILE，先把配置文件放好再跑本脚本"

# 端口与凭据一律从 .env 解析，脚本产物与配置文件不会各说各话。
DATABASE_URL="$(sed -n 's|^DATABASE_URL=||p' "$ENV_FILE" | head -1)"
REDIS_URL="$(sed -n 's|^REDIS_URL=||p' "$ENV_FILE" | head -1)"
S3_ENDPOINT="$(sed -n 's|^S3_ENDPOINT=||p' "$ENV_FILE" | head -1)"
API_PORT="$(sed -n 's|^API_PORT=||p' "$ENV_FILE" | head -1)"
APP_ENV="$(sed -n 's|^APP_ENV=||p' "$ENV_FILE" | head -1)"

PG_USER=retouch PG_PASS=retouch_dev PG_HOST=localhost PG_PORT=7311 PG_DB=retouch
if [[ "$DATABASE_URL" =~ postgresql(\+asyncpg)?://([^:]+):([^@]*)@([^/:]+):([0-9]+)/([^?]+) ]]; then
  PG_USER="${BASH_REMATCH[2]}" PG_PASS="${BASH_REMATCH[3]}"
  PG_HOST="${BASH_REMATCH[4]}" PG_PORT="${BASH_REMATCH[5]}" PG_DB="${BASH_REMATCH[6]}"
fi
REDIS_PORT=7312
[[ "$REDIS_URL" =~ :([0-9]+) ]] && REDIS_PORT="${BASH_REMATCH[1]}"
S3_PORT=7313
[[ "$S3_ENDPOINT" =~ :([0-9]+) ]] && S3_PORT="${BASH_REMATCH[1]}"
S3_CONSOLE_PORT=$((S3_PORT + 1))
API_PORT="${API_PORT:-7302}"

if [[ "$PG_HOST" != "localhost" && "$PG_HOST" != "127.0.0.1" ]]; then
  warn "DATABASE_URL 指向 $PG_HOST，那是 docker-compose 的服务名。裸机部署需要改成 localhost:7311"
fi
if [[ "${APP_ENV:-development}" == "production" ]]; then
  warn "APP_ENV=production 会给会话 cookie 加 Secure；若用 http:// 访问会永远无法保持登录"
fi
# 只警告不中断：默认密钥会让任何拿到源码的人伪造登录态，公网暴露前必须换
if grep -q 'JWT_SECRET=.*dev-only-secret' "$ENV_FILE"; then
  warn "JWT_SECRET 仍是模板默认值，请 openssl rand -hex 32 换一个"
fi

PGBIN=""
for candidate in $(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tac); do
  PGBIN="$candidate"
  break
done

if [[ $CHECK_ONLY -eq 1 ]]; then
  log "只读检查"
  printf '  %-26s %s\n' "REPO" "$REPO" "  SVC（推断）" "$SVC"
  printf '  %-26s %s\n' "PG 端口/库" "$PG_PORT/$PG_DB" "  API 端口" "$API_PORT"
  for item in "$PGBIN/postgres:postgres 二进制" "$(command -v redis-server || echo redis-server):redis 二进制" \
    "/usr/local/bin/minio:MinIO 二进制" "$(command -v supervisord || echo supervisord):supervisor" \
    "$(command -v uv || echo uv):uv" "$SVC/pg/PG_VERSION:数据库数据目录" \
    "$SVC/minio-data:对象存储数据" "/etc/retouch/supervisord.conf:进程定义" \
    "/usr/local/bin/retouch:retouch 命令" "$REPO/frontend/dist/index.html:前端产物" \
    "$REPO/backend/.venv/bin/uvicorn:虚拟环境"; do
    path="${item%%:*}" label="${item#*:}"
    if [[ -n "$path" && -e "$path" ]]; then printf '  \033[32mOK\033[0m   %s\n' "$label"
    else printf '  \033[31m缺失\033[0m %s\n' "$label"; MISSING+=("$label"); fi
  done
  [[ ${#MISSING[@]} -eq 0 ]] || { echo; echo "缺失 ${#MISSING[@]} 项，跑 bash deploy/bootstrap.sh 补齐"; }
  exit 0
fi

[[ $EUID -eq 0 ]] || die "需要 root（postgres 以独立用户运行，且要写 /etc 与 /usr/local/bin）"

log "仓库 $REPO"
log "数据 $SVC"
log "端口 api=$API_PORT pg=$PG_PORT redis=$REDIS_PORT s3=$S3_PORT console=$S3_CONSOLE_PORT"

# ---------------------------------------------------------------- 系统依赖
if [[ $SKIP_APT -eq 0 ]]; then
  NEED=()
  dpkg -s postgresql >/dev/null 2>&1 || NEED+=(postgresql)
  dpkg -s redis-server >/dev/null 2>&1 || NEED+=(redis-server)
  dpkg -s supervisor >/dev/null 2>&1 || NEED+=(supervisor)
  dpkg -s cron >/dev/null 2>&1 || NEED+=(cron)
  dpkg -s fonts-wqy-microhei >/dev/null 2>&1 || NEED+=(fonts-wqy-microhei)
  dpkg -s curl >/dev/null 2>&1 || NEED+=(curl)
  # 中文字体缺失不会让服务起不来，但会让导出的文字层变成方框
  if [[ ${#NEED[@]} -gt 0 ]]; then
    log "安装系统包：${NEED[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends "${NEED[@]}"
  else
    log "系统包已齐全"
  fi
else
  log "跳过系统包安装"
fi

command -v redis-server >/dev/null || die "没有 redis-server，去掉 --skip-apt 或手工安装"
command -v supervisord >/dev/null || die "没有 supervisord，去掉 --skip-apt 或手工安装"
PGBIN="$(for candidate in $(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tac); do echo "$candidate"; break; done)"
[[ -n "$PGBIN" ]] || die "找不到 postgres 二进制，apt 是否装成功？"
log "PG 二进制 $PGBIN"
id postgres >/dev/null 2>&1 || die "系统里没有 postgres 用户，postgresql 包未正确安装"

if [[ ! -x /usr/local/bin/minio ]]; then
  log "下载 MinIO 二进制"
  curl -fL -o /usr/local/bin/minio https://dl.min.io/server/minio/release/linux-amd64/minio \
    || die "MinIO 下载失败，可从本地 scp 一个到 /usr/local/bin/minio"
  chmod +x /usr/local/bin/minio
fi

if ! command -v uv >/dev/null 2>&1 && [[ ! -x /root/.local/bin/uv ]]; then
  log "安装 uv"
  # Ubuntu 24.04 起系统 Python 是 externally-managed，直接 pip 会被拒
  python3 -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null \
    || python3 -m pip install -U --break-system-packages uv -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh \
    || die "uv 装不上：pip 与官方脚本都不通，从本地 scp 一个 uv 到 /usr/local/bin/"
fi
UV="$(command -v uv || echo /root/.local/bin/uv)"
[[ -x "$UV" ]] || die "找不到可执行的 uv"
log "uv $UV"

# ---------------------------------------------------------------- 目录与数据库集群
mkdir -p "$SVC"/{pg,redis,socket,minio-data} /var/log/retouch /etc/retouch

if [[ ! -f "$SVC/pg/PG_VERSION" ]]; then
  log "初始化数据库集群（首次部署才会执行）"
  chown -R postgres:postgres "$SVC/pg" "$SVC/socket"
  chmod 700 "$SVC/pg"
  su -s /bin/bash postgres -c \
    "$PGBIN/initdb -D $SVC/pg -U $PG_USER --auth-local=trust --auth-host=scram-sha-256 -E UTF8"
else
  log "检测到已有数据目录，跳过 initdb"
fi
# 数据盘挂载点权限变化会让 postgres 读不到目录，每次重建都校正一遍。
# 700 是 postgres 的硬性要求：数据目录 group/world 可读会直接拒绝启动。
chown -R postgres:postgres "$SVC/pg" "$SVC/socket"
chmod 700 "$SVC/pg"
chmod 755 "$SVC/socket"
# postgres 以独立用户运行，需要整条路径每一层都有穿越位（x），少一层就是
# "Permission denied" 起不来。上一轮部署正是卡在 /root 与 $SVC 这两层，
# 所以这里每次都校正一遍，排查用 namei -l "$SVC/pg" 一次摊开整条路径。
chmod o+x "$SVC" "$(dirname "$SVC")" 2>/dev/null || true
[[ "$SVC" == /root/* ]] && chmod o+x /root 2>/dev/null || true
chmod 750 "$SVC/minio-data" "$SVC/redis" 2>/dev/null || true

# ---------------------------------------------------------------- /etc/retouch/env
log "写 /etc/retouch/env"
DATA_ROOT="$(dirname "$SVC")"
cat > /etc/retouch/env <<EOF
# 由 deploy/bootstrap.sh 生成，勿手工编辑；改配置请改 $ENV_FILE 后重跑脚本
REPO=$REPO
SVC=$SVC
PGBIN=$PGBIN
DATA_ROOT=$DATA_ROOT
UV=$UV
UV_LINK_MODE=copy
UV_CACHE_DIR=$DATA_ROOT/.uv-cache
UV_PYTHON_INSTALL_DIR=$DATA_ROOT/.uv-python
API_PORT=$API_PORT
EOF
chmod 644 /etc/retouch/env
# 立即生效：后续 uv / alembic 用的就是这份定义，含数据盘上的缓存与解释器目录
set -a; . /etc/retouch/env; set +a

# ---------------------------------------------------------------- 进程定义
if [[ -f /etc/retouch/supervisord.conf ]]; then
  cp -a /etc/retouch/supervisord.conf "/etc/retouch/supervisord.conf.bak-$(date +%m%d-%H%M)"
fi
log "写 /etc/retouch/supervisord.conf"
cat > /etc/retouch/supervisord.conf <<EOF
; 由 deploy/bootstrap.sh 生成，绝对路径已烧死；改配置请改 .env 后重跑脚本
[supervisord]
logfile=/var/log/retouch/supervisord.log
pidfile=/run/retouch-supervisord.pid
childlogdir=/var/log/retouch
nodaemon=false
user=root

[unix_http_server]
file=/run/retouch.sock
chmod=0700

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///run/retouch.sock

[program:pg]
command=$PGBIN/postgres -D $SVC/pg -p $PG_PORT -c listen_addresses=127.0.0.1 -k $SVC/socket
directory=/tmp
user=postgres
priority=10
startsecs=5
autorestart=true
stdout_logfile=/var/log/retouch/pg.log
stderr_logfile=/var/log/retouch/pg.log

[program:redis]
command=$(command -v redis-server) --port $REDIS_PORT --bind 127.0.0.1 --dir $SVC/redis
priority=20
startsecs=3
autorestart=true
stdout_logfile=/var/log/retouch/redis.log
stderr_logfile=/var/log/retouch/redis.log

[program:minio]
command=/usr/local/bin/minio server $SVC/minio-data --address 127.0.0.1:$S3_PORT --console-address 127.0.0.1:$S3_CONSOLE_PORT
environment=MINIO_ROOT_USER="$PG_USER",MINIO_ROOT_PASSWORD="$PG_PASS"
priority=30
startsecs=5
autorestart=true
stdout_logfile=/var/log/retouch/minio.log
stderr_logfile=/var/log/retouch/minio.log

[program:app]
command=$REPO/backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $API_PORT
directory=$REPO/backend
user=root
environment=HOME="/root",PYTHONPATH="$REPO/backend",PATH="$REPO/backend/.venv/bin:/usr/local/bin:/usr/bin:/bin"
priority=50
startsecs=10
autorestart=true
stdout_logfile=/var/log/retouch/app.log
stderr_logfile=/var/log/retouch/app.log

[program:worker]
command=$REPO/backend/.venv/bin/arq app.worker.WorkerSettings
directory=$REPO/backend
user=root
environment=HOME="/root",PYTHONPATH="$REPO/backend",PATH="$REPO/backend/.venv/bin:/usr/local/bin:/usr/bin:/bin"
priority=60
startsecs=10
autorestart=true
stdout_logfile=/var/log/retouch/worker.log
stderr_logfile=/var/log/retouch/worker.log

[program:cron]
command=/usr/sbin/cron -f
priority=70
autorestart=true
stdout_logfile=/var/log/retouch/cron.log
stderr_logfile=/var/log/retouch/cron.log

[group:retouch]
programs=pg,redis,minio,app,worker
EOF

# ---------------------------------------------------------------- 命令行入口
log "写 /usr/local/bin/retouch 与 retouch-backup"
cat > /usr/local/bin/retouch <<EOF
#!/usr/bin/env bash
CONF=/etc/retouch/supervisord.conf
ctl() { supervisorctl -c "\$CONF" "\$@"; }
# supervisorctl 不会自己把守护进程拉起来。bootstrap 重建配置前会先 shutdown，
# 如果这里不兜底，紧接着的 infra 就会对着一个没连接的 socket 静默失败。
ensure() { ctl status >/dev/null 2>&1 || { supervisord -c "\$CONF" >/dev/null 2>&1; sleep 3; }; }
case "\${1:-status}" in
  up)      ensure; ctl start all >/dev/null 2>&1; sleep 3; ctl status ;;
  infra)   ensure; ctl start retouch:pg retouch:redis retouch:minio >/dev/null 2>&1; sleep 4; ctl status ;;
  stop)    ctl stop all ;;
  restart) ensure; ctl restart retouch:* ;;
  status)  ctl status ;;
  logs)   tail -n 80 -f /var/log/retouch/\${2:-app}.log ;;
  check)  bash $REPO/deploy/bootstrap.sh --check ;;
  health) curl -s http://127.0.0.1:${API_PORT}/api/health; echo ;;
  *) echo "用法: retouch up|infra|stop|restart|status|logs [app|worker|pg|redis|minio]|check|health" ;;
esac
EOF
chmod 755 /usr/local/bin/retouch

cat > /usr/local/bin/retouch-backup <<EOF
#!/usr/bin/env bash
set -a; . /etc/retouch/env; set +a
STAMP=\$(date +%Y%m%d-%H%M)
DEST=\${RETBACK:-\$SVC/backup}
mkdir -p "\$DEST"
supervisorctl -c /etc/retouch/supervisord.conf stop retouch:app retouch:worker >/dev/null 2>&1 || true
su -s /bin/bash postgres -c "$PGBIN/pg_dump -h \$SVC/socket -p $PG_PORT -U $PG_USER $PG_DB" > "\$DEST/postgres-\$STAMP.sql" || echo "pg_dump 失败，先确认 postgres 在跑"
tar czf "\$DEST/config-\$STAMP.tar.gz" -C / etc/retouch usr/local/bin/retouch usr/local/bin/retouch-backup -C "\$REPO" .env || echo "config 备份失败"
tar czf "\$DEST/minio-\$STAMP.tar.gz" -C "\$SVC" minio-data || echo "minio 备份失败"
retouch up >/dev/null 2>&1 || true
ls -lh "\$DEST"
EOF
chmod 755 /usr/local/bin/retouch-backup

# 实例重启后没有 systemd 可依赖，这一钩子只能在“有人登录”时把服务带回来。
# 无人值守时它不会触发，补一次用：ssh -p <端口> root@<主机> "retouch up"
touch /root/.bashrc
grep -q 'retouch/env' /root/.bashrc || \
  printf '\nset -a; [ -f /etc/retouch/env ] && . /etc/retouch/env; set +a\n' >> /root/.bashrc
grep -q 'retouch up' /root/.bashrc || \
  printf '\n[ -x /usr/local/bin/retouch ] && retouch up >/dev/null 2>&1\n' >> /root/.bashrc

# ---------------------------------------------------------------- 起服务并建库
if supervisorctl -c /etc/retouch/supervisord.conf status >/dev/null 2>&1; then
  log "重载 supervisor 配置"
  supervisorctl -c /etc/retouch/supervisord.conf shutdown || true
  sleep 3
  pkill -f "supervisord -c /etc/retouch/supervisord.conf" 2>/dev/null || true
  sleep 1
fi
retouch infra

log "等待 postgres 就绪"
for _ in $(seq 1 30); do
  "$PGBIN/pg_isready" -h "$SVC/socket" -p "$PG_PORT" -q && break
  sleep 1
done
"$PGBIN/pg_isready" -h "$SVC/socket" -p "$PG_PORT" || die "postgres 没起来，看 /var/log/retouch/pg.log"

# 集群是新建的话角色还没有密码；已有集群重复执行同样安全
log "校正数据库账号与库"
cat > /tmp/retouch-pg-init.sql <<EOF
ALTER USER $PG_USER WITH PASSWORD '$PG_PASS';
EOF
chgrp postgres /tmp/retouch-pg-init.sql
chmod 640 /tmp/retouch-pg-init.sql
su -s /bin/bash postgres -c "$PGBIN/psql -h $SVC/socket -p $PG_PORT -U $PG_USER -d postgres -f /tmp/retouch-pg-init.sql" \
  || warn "无法给角色 $PG_USER 设置密码：集群里的角色与 .env 不一致，核对 DATABASE_URL 后重跑"
rm -f /tmp/retouch-pg-init.sql
su -s /bin/bash postgres -c "$PGBIN/createdb -h $SVC/socket -p $PG_PORT -U $PG_USER $PG_DB" 2>/dev/null \
  || log "数据库 $PG_DB 已存在"

# ---------------------------------------------------------------- 应用侧
if [[ $SKIP_SYNC -eq 0 && ! -x "$REPO/backend/.venv/bin/uvicorn" ]]; then
  log "重建虚拟环境"
  mkdir -p "$(dirname "$UV_CACHE_DIR")"
  cd "$REPO/backend" && "$UV" python install 3.13 && "$UV" sync --all-extras
elif [[ ! -x "$REPO/backend/.venv/bin/uvicorn" ]]; then
  die "虚拟环境不存在且已跳过同步，手工执行：cd $REPO/backend && $UV sync --all-extras"
fi

if [[ $SKIP_MIGRATE -eq 0 ]]; then
  log "数据库迁移"
  cd "$REPO/backend" && .venv/bin/alembic upgrade head
fi

retouch restart
sleep 8
log "状态与健康检查"
retouch status
retouch health
printf '  postgres 实际数据目录：%s\n' "$(ps -o args= -C postgres | head -1 | sed -n 's/.*-D \([^ ]*\).*/\1/p')"
[[ -d "$REPO/frontend/dist" ]] || warn "没有 frontend/dist，首页会返回 {\"detail\":\"Not Found\"}，本地 build 后 scp 上来"
[[ -d /root/.u2net ]] || warn "没有 /root/.u2net，MATTING_PROVIDER=auto 时首次抠图会卡在下载权重"

cat <<EOF

重建完成。日常操作：
  retouch status / health / logs worker / restart
  retouch-backup                       数据库与图片打包到 $SVC/backup
  bash deploy/bootstrap.sh --check     只读体检
EOF
