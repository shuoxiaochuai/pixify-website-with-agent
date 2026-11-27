#!/usr/bin/env bash
# retouch 常驻看门狗。
#
# 与 retouch-guard 的分工：
#   guard 是「一次性检查」，只在容器启动（/opt/start.sh）或 ssh 登录（.bashrc/.ssh/rc）时被调用；
#   本脚本是常驻循环，由 /etc/profile.d 在容器启动时拉起，并 setsid 脱离父进程组，
#   所以 supervisord 被杀之后仍能在 20 秒内把整个服务栈拉回来。
#
# 为什么需要它：平台在无卡模式下给 PID1 注入 need_service=0，/opt/boot.sh 不会执行
# /opt/start.sh，guard 的容器启动钩子直接失效；此时若没人 ssh 登录，业务会一直是 502。
CONF=/etc/retouch/supervisord.conf
LOG=/var/log/retouch/watchdog.log
INTERVAL=20

mkdir -p /var/log/retouch 2>/dev/null || true
echo "== $(date '+%F %T') watchdog 启动 pid=$$ ==" >>"$LOG"

while true; do
  if ! supervisorctl -c "$CONF" status >/dev/null 2>&1; then
    # 顺带记下 PID1 的启动时间，用来区分「容器被平台重建」和「supervisord 单独被杀」。
    pid1=$(ps -o lstart= -p 1 2>/dev/null | tr -s ' ')
    echo "== $(date '+%F %T') supervisord 不在，拉起（pid1 started:${pid1:-unknown}）==" >>"$LOG"
    /usr/local/bin/retouch-guard --force >>"$LOG" 2>&1 || true
  fi
  sleep "$INTERVAL"
done