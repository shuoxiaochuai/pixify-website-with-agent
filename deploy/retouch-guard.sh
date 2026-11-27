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
    supervisord -c "$CONF"
    sleep 5
  fi
  supervisorctl -c "$CONF" start all
  supervisorctl -c "$CONF" status
} >>"$LOG" 2>&1