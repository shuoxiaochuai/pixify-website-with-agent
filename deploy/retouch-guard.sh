#!/usr/bin/env bash
# retouch 服务栈幂等守卫。
#
# 容器没有 systemd，/opt/boot.sh 只负责 sshd 和 jupyter；supervisord 一旦缺席
# （容器重启、进程被杀），网页就是 502 bad gateway。本脚本由三处调用：
#   /opt/start.sh   容器每次启动（平台钩子，need_service=1 时由 boot.sh 调用）
#   /root/.bashrc   ssh 登录（非交互命令也生效）
#   /root/.ssh/rc   ssh 认证后兜底
# 服务齐全时开销约 0.1s，60 秒内的重复调用直接返回。--force 跳过节流。
CONF=/etc/retouch/supervisord.conf
LOG=/var/log/retouch/autostart.log
STAMP=/run/retouch-guard.stamp

command -v supervisorctl >/dev/null 2>&1 || exit 0

mkdir -p /var/log/retouch /run 2>/dev/null || true

if [ "${1:-}" != "--force" ]; then
  now=$(date +%s)
  last=$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt 60 ]; then
    exit 0
  fi
fi
touch "$STAMP" 2>/dev/null || true

# supervisord 被强杀（OOM、平台清扫、kill -9）后，它的子进程会被 PID1 收养并继续占着
# 7302/7311/7312/7313，直接重启 supervisord 会因端口冲突全部变成 FATAL/BACKOFF，
# 严重时 arq 会跑出两个实例重复消费任务。所以拉起前先把孤儿优雅停掉。
cleanup_orphans() {
  local pat pids="" alive p
  local pats=(
    '/usr/lib/postgresql/14/bin/postgres -D'
    'redis-server 127.0.0.1:7312'
    'minio server /root/rivermind-data/retouch-svc/minio-data'
    'uvicorn app.main:app --host 0.0.0.0 --port 7302'
    'arq app.worker.WorkerSettings'
    '/usr/sbin/cron -f'
  )
  self=$$
  parent=$(ps -o ppid= -p "$self" 2>/dev/null | tr -d ' ')
  for pat in "${pats[@]}"; do
    for p in $(pgrep -f -- "$pat" 2>/dev/null); do
      [ "$p" = "$self" ] || [ "$p" = "$parent" ] || pids="$pids $p"
    done
  done
  pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  [ -z "${pids// /}" ] && return 0
  echo "清理孤儿进程:$pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do
    sleep 1
    alive=""
    for p in $pids; do
      kill -0 "$p" 2>/dev/null && alive="$alive $p"
    done
    [ -z "${alive// /}" ] && break
  done
  # pg 的 SIGTERM 要花几秒做 checkpoint，等 6 秒还活着就强杀，避免卡住整个恢复流程。
  for p in $alive; do kill -KILL "$p" 2>/dev/null || true; done
  sleep 1
}

{
  echo "== $(date '+%F %T') retouch-guard 触发 =="
  if ! supervisorctl -c "$CONF" status >/dev/null 2>&1; then
    # 容器重启时数据盘（PVC）可能晚于 start.sh 挂载，pg 起不来会连锁拖垮整个栈，
    # 这里最多等 120 秒；路径取自 /etc/retouch/env 的 REPO。
    [ -f /etc/retouch/env ] && . /etc/retouch/env
    wait_path="${REPO:-/root/rivermind-data/ai-picture-editor}"
    for _ in $(seq 1 24); do
      [ -d "$wait_path" ] && break
      sleep 5
    done
    echo "supervisord 不在，拉起"
    cleanup_orphans
    supervisord -c "$CONF"
    sleep 5
  fi
  supervisorctl -c "$CONF" start all
  supervisorctl -c "$CONF" status
} >>"$LOG" 2>&1