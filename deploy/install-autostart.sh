#!/usr/bin/env bash
# 一次性补丁：给容器加上 retouch 服务栈自恢复 + SSH 公钥持久化。
#
# 背景：
#   1. /opt/boot.sh 每次容器启动都会用 ${SSH_PUB_KEY} 覆盖 /root/.ssh/authorized_keys，
#      平台未配置该变量时会直接清空，导致免密登录失效。
#   2. /opt/boot.sh 只在 need_service=1 时执行 /opt/start.sh；无卡模式下平台注入
#      need_service=0，那段自启完全不执行，容器一重启网页就是 502 bad gateway。
#   3. 唯一与 need_service 无关的启动钩子是 boot.sh 里的 `source /etc/profile`，
#      所以自启必须挂到 /etc/profile.d（本脚本的 PROFILE 段）。
#   4. supervisord 单独被杀时也要能在 20 秒内自愈，由常驻 watchdog 负责。
#   5. /root/.bashrc 里的自启写在交互判断之后，非交互 ssh 执行不到。
#
# 用法：scp deploy/retouch-guard.sh deploy/retouch-watchdog.sh deploy/retouch-profile.sh \
#             deploy/install-autostart.sh root@host:/tmp/，
#       然后 bash /tmp/install-autostart.sh
set -euo pipefail

GUARD=/usr/local/bin/retouch-guard
WATCHDOG=/usr/local/bin/retouch-watchdog
PROFILE=/etc/profile.d/zz-retouch-autostart.sh
STARTSH=/opt/start.sh
BASHRC=/root/.bashrc
SSHRC=/root/.ssh/rc
PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHmBfGb0DZnGKjXRKWxvvpaX/mqQhoAHA0C5v2FnQRpG qoder-deploy'

echo '--- 1. 安装 retouch-guard'
install -m 755 /tmp/retouch-guard.sh "$GUARD"
echo "supervisord: $(command -v supervisord || echo '未找到')"

echo '--- 1b. 安装常驻 watchdog 与 /etc/profile.d 启动钩子（need_service=0 时唯一的自启路径）'
install -d -m 755 /var/log/retouch
install -m 755 /tmp/retouch-watchdog.sh "$WATCHDOG"
install -m 644 /tmp/retouch-profile.sh "$PROFILE"
if pgrep -f '/usr/local/bin/retouch-watchdog' >/dev/null 2>&1; then
  echo 'watchdog 已在运行'
else
  setsid nohup "$WATCHDOG" >>/var/log/retouch/watchdog.log 2>&1 &
  echo 'watchdog 已启动'
fi
echo '--- 2. 补回 SSH 公钥（boot.sh 启动时会清空 authorized_keys）'
mkdir -p /root/.ssh
chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys
grep -q 'qoder-deploy' /root/.ssh/authorized_keys || echo "$PUBKEY" >>/root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

echo '--- 3. 挂到 /opt/start.sh（容器启动钩子，boot.sh 不会覆盖它）'
cp -n "$STARTSH" "${STARTSH}.bak-retouch" 2>/dev/null || true
if grep -q 'retouch-autostart' "$STARTSH"; then
  echo 'start.sh 已打过补丁，跳过'
else
  cat >>"$STARTSH" <<EOF

# --- retouch-autostart: 容器启动即恢复业务服务栈，勿删 ---
mkdir -p /root/.ssh /var/log/retouch
chmod 700 /root/.ssh 2>/dev/null || true
grep -q 'qoder-deploy' /root/.ssh/authorized_keys 2>/dev/null ||
  echo '$PUBKEY' >>/root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
if [ -x /usr/local/bin/retouch-guard ]; then
  nohup /usr/local/bin/retouch-guard >>/var/log/retouch/autostart.log 2>&1 &
fi
EOF
  echo 'start.sh 已追加'
fi

echo '--- 4. .bashrc 兜底（插在交互判断之前）'
python3 - "$BASHRC" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
if 'retouch-guard' in text:
    print('bashrc 已存在，跳过')
else:
    line = '[ -x /usr/local/bin/retouch-guard ] && /usr/local/bin/retouch-guard >/dev/null 2>&1\n'
    anchor = '[ -z "$PS1" ] && return'
    if anchor in text:
        text = text.replace(anchor, line + anchor, 1)
    else:
        text = line + text
    path.write_text(text)
    print('bashrc 已插入')
PY

echo '--- 5. .ssh/rc 兜底（ssh 认证后执行，不依赖 shell 类型）'
if [ -f "$SSHRC" ] && grep -q 'retouch-guard' "$SSHRC"; then
    echo '.ssh/rc 已存在，跳过'
else
    {
        echo '#!/bin/bash'
        echo '# retouch 兜底：任何 ssh 登录都执行（含非交互），输出全部丢弃'
        echo '[ -f /etc/ssh/sshrc ] && . /etc/ssh/sshrc >/dev/null 2>&1'
        echo '[ -x /usr/local/bin/retouch-guard ] && /usr/local/bin/retouch-guard >/dev/null 2>&1'
        echo 'exit 0'
    } >"$SSHRC"
    chmod 755 "$SSHRC"
    echo '.ssh/rc 已写入'
fi

echo '--- 完成，当前状态'
supervisorctl -c /etc/retouch/supervisord.conf status || true