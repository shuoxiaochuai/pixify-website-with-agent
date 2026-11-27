#!/usr/bin/env bash
# Keep this script runnable even when a supervisor calls `sh /opt/start.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  if [ -x /bin/bash ]; then
    exec /bin/bash "$0" "$@"
  fi
  exec bash "$0" "$@"
fi

set -euo pipefail

# --- retouch-autostart: 容器启动即恢复业务服务栈，勿删 ---
# 放在 code-server 之前：PASSWORD 缺失时 code-server 段会因 set -e 退出，
# 若本块在其后会一起被跳过，业务就还是 502。
# boot.sh 会用 ${SSH_PUB_KEY} 覆盖 authorized_keys，平台未配置时是直接清空，这里补回部署公钥。
mkdir -p /root/.ssh /var/log/retouch || true
chmod 700 /root/.ssh 2>/dev/null || true
grep -q 'qoder-deploy' /root/.ssh/authorized_keys 2>/dev/null ||
  echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHmBfGb0DZnGKjXRKWxvvpaX/mqQhoAHA0C5v2FnQRpG qoder-deploy' >>/root/.ssh/authorized_keys || true
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
if [ -x /usr/local/bin/retouch-guard ]; then
  nohup /usr/local/bin/retouch-guard >>/var/log/retouch/autostart.log 2>&1 &
fi

if [ -x /usr/bin/code-server ]; then
    : "${PASSWORD:?PASSWORD is required to start code-server}"
    /usr/bin/code-server \
        --locale zh-cn \
        --app-name "智川云 Code Server" \
        --i18n /etc/code-server/gpuhome-zh-cn.json \
        --auth password \
        --bind-addr 0.0.0.0:8080 \
        /root >/tmp/code-server.log 2>&1 &
fi