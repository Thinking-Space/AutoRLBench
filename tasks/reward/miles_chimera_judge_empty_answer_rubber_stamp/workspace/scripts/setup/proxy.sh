#!/bin/bash
set -x 
# 根据MASTER_ADDR和WORLD_SIZE，把所有的MASTER和WORKER的IP地址都加入到NO_PROXY中
# 如果MASTER_ADDR以pytorchjob开头，则把所有master和worker的域名和ip都加到no_proxy中
# MASTER_ADDR=pytorchjob-miles-135389-master-0
# WORKER_ADDR=pytorchjob-miles-135389-worker-0
# 否则，只把hostname和ip加到no_proxy中
NO_PROXY_LIST="localhost,127.0.0.1,$(hostname -i | tr ' ' ','),$(hostname)"
# 定义解析 IP 的函数
resolve_ip() {
    local host=$1
    if [[ -n "$host" ]]; then
        local ip=""
        local slept_seconds=0
        echo "[proxy] resolving host: ${host}" >&2
        while [[ -z "$ip" ]]; do
            ip=$(getent ahosts "$host" | awk '{print $1; exit}')
            if [[ -z "$ip" ]]; then
                slept_seconds=$((slept_seconds + 10))
                echo "[proxy] resolve failed, retrying: ${host} (slept ${slept_seconds}s)" >&2
                sleep 10
            fi
        done
        echo "[proxy] resolved ${host} -> ${ip}" >&2
        echo "$ip"
    fi
}
# Wait for the workers to start
# sleep $((120 * ${WORLD_SIZE}))

if [ -n "$MASTER_ADDR" ]; then
    NO_PROXY_LIST="${NO_PROXY_LIST},${MASTER_ADDR}"
    # 解析 Master IP
    MASTER_IP=$(resolve_ip "$MASTER_ADDR")
    if [[ -n "$MASTER_IP" ]]; then
        NO_PROXY_LIST="${NO_PROXY_LIST},${MASTER_IP}"
    fi

    if [[ "$MASTER_ADDR" == pytorchjob* ]]; then
        # 提取前缀和后缀，支持带有域名后缀的情况
        if [[ "$MASTER_ADDR" =~ (.*)-master-0(.*) ]]; then
            PREFIX="${BASH_REMATCH[1]}"
            SUFFIX="${BASH_REMATCH[2]}"
            if [ -n "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
                # 假设 WORLD_SIZE 是总节点数
                # Master 是 master-0，Worker 是 worker-0 到 worker-(WORLD_SIZE-2)
                for ((i=0; i<WORLD_SIZE-1; i++)); do
                    WORKER_HOST="${PREFIX}-worker-${i}${SUFFIX}"
                    NO_PROXY_LIST="${NO_PROXY_LIST},${WORKER_HOST}"
                    # 解析 Worker IP
                    WORKER_IP=$(resolve_ip "$WORKER_HOST")
                    if [[ -n "$WORKER_IP" ]]; then
                        NO_PROXY_LIST="${NO_PROXY_LIST},${WORKER_IP}"
                    fi
                done
            fi
        fi
    fi
fi
export NO_PROXY=$NO_PROXY_LIST
export no_proxy=$NO_PROXY_LIST
export http_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export HTTP_PROXY=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export https_proxy=http://whitelist-proxy.cybertron.svc.cluster.local:7891
export HTTPS_PROXY=http://whitelist-proxy.cybertron.svc.cluster.local:7891