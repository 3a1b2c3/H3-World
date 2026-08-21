#!/usr/bin/env bash
# 根盘写入巡检。缓存重定向靠环境变量和 sitecustomize，理论上都覆盖了，
# 但总有工具用自己的硬编码路径 —— 这个脚本是用来发现"理论之外"的。
# 用法: bash code/check_rootfs.sh [起始时间，默认 1 小时前]
SINCE="${1:-$(date -d '1 hour ago' '+%Y-%m-%d %H:%M')}"
echo "=== 根盘 ==="; df -h / | tail -1
USED=$(df / | awk 'NR==2{print $5}' | tr -d '%')
[ "$USED" -ge 85 ] && echo "⚠️  已超 85%，需要处理"
echo
echo "=== $SINCE 之后写入根盘的地方（>1MB 才列）==="
for d in /root/.cache /root/.triton /root/.local /root/miniconda3/pkgs \
         /tmp/torchinductor_root /var/tmp /root/.conda; do
    [ -e "$d" ] || continue
    n=$(find "$d" -newermt "$SINCE" -type f 2>/dev/null | wc -l)
    b=$(find "$d" -newermt "$SINCE" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{print s+0}')
    [ "$n" -gt 0 ] && printf "  %-28s %5d 个文件  %8.1f MB\n" "$d" "$n" "$(echo "scale=1;$b/1048576"|bc)"
done
echo "（空 = 干净）"
