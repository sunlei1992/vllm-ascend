#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
STORE="$ROOT/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
IMAGE="mooncake-tcp-p2p"
NET="mooncake-p2p"
MASTER="mooncake-master"

test -f "$STORE/layerwise_pd.py"
test -f "$STORE/mooncake_tcp_transport.py"
test -f "$ROOT/tests/ut/distributed/ascend_store/mooncake_tcp/client.py"

docker build -q -t "$IMAGE" -f "$ROOT/tests/ut/distributed/ascend_store/mooncake_tcp/Dockerfile" \
  "$ROOT/tests/ut/distributed/ascend_store/mooncake_tcp" >/dev/null
docker network create "$NET" >/dev/null 2>&1 || true
docker rm -f "$MASTER" >/dev/null 2>&1 || true
cleanup() {
  docker rm -f "$MASTER" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$MASTER" --network "$NET" "$IMAGE" mooncake_master >/dev/null
sleep 3
docker run --rm --network "$NET" \
  -v "$STORE/layerwise_pd.py:/app/layerwise_pd.py:ro" \
  -v "$STORE/mooncake_tcp_transport.py:/app/mooncake_tcp_transport.py:ro" \
  -v "$ROOT/tests/ut/distributed/ascend_store/mooncake_tcp/client.py:/app/client.py:ro" \
  "$IMAGE" python /app/client.py 2>&1 | tail -5
