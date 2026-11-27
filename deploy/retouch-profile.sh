#!/bin/sh
# retouch-autostart：容器启动与 ssh 登录时恢复业务栈，勿删。
#
# 为什么挂在这里：/opt/boot.sh 每次启动都会 `source /etc/profile`，与 need_service 无关；
# 而无卡模式下平台注入 need_service=0，boot.sh 不会执行 /opt/start.sh，
# /etc/profile.d 是容器里唯一稳定执行的钩子。
#
# 本文件被 source，必须静默、幂等、快速返回。

[ "$(id -u 2>/dev/null)" = "0" ] || return 0

# boot.sh 会用 ${SSH_PUB_KEY} 覆盖 authorized_keys，平台未配置该变量时直接清空，这里补回部署公钥。
mkdir -p /root/.ssh 2>/dev/null || true
chmod 700 /root/.ssh 2>/dev/null || true
if ! grep -q 'qoder-deploy' /root/.ssh/authorized_keys 2>/dev/null; then
  echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHmBfGb0DZnGKjXRKWxvvpaX/mqQhoAHA0C5v2FnQRpG qoder-deploy' >>/root/.ssh/authorized_keys 2>/dev/null || true
fi
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true

# 常驻看门狗：不受 supervisord 管理，supervisord 被杀后 20 秒内自愈。
if [ -x /usr/local/bin/retouch-watchdog ]; then
  if ! pgrep -f '/usr/local/bin/retouch-watchdog' >/dev/null 2>&1; then
    setsid nohup /usr/local/bin/retouch-watchdog >>/var/log/retouch/watchdog.log 2>&1 &
  fi
fi