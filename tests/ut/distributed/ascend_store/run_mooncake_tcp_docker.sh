#!/usr/bin/env bash
# Reproduce the real Mooncake TCP layerwise round-trip on a Linux arm64 Docker host.
#
# This is a transport smoke test, not an NPU test.  It installs the non-CUDA
# Mooncake wheel, starts mooncake_master in the container, and exercises the
# session/range adapter with CPU buffers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LAYERWISE_PD="$ROOT/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_pd.py"
TRANSPORT="$ROOT/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/mooncake_tcp_transport.py"

test -f "$LAYERWISE_PD"
test -f "$TRANSPORT"

docker run --rm --platform linux/arm64 \
  -v "$LAYERWISE_PD:/app/layerwise_pd.py:ro" \
  -v "$TRANSPORT:/app/mooncake_tcp_transport.py:ro" \
  python:3.12-slim-trixie bash -lc '
    set -e
    apt-get update -qq >/dev/null
    apt-get install -y -qq libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2 >/dev/null 2>&1
    pip install --no-cache-dir -q mooncake-transfer-engine-non-cuda

    mooncake_master > /tmp/mooncake-master.log 2>&1 &
    master_pid=$!
    trap "kill $master_pid 2>/dev/null || true" EXIT
    sleep 3

    python - <<"PY"
import sys
import time

sys.path.insert(0, "/app")
from layerwise_pd import LayerwiseConsumer, LayerwisePublisher
from mooncake_tcp_transport import connect_mooncake_tcp_session

NUM_LAYERS = 78
NUM_SHARDS = 2
slots = [[0]]
shared = list(range(1, NUM_LAYERS))
for shard in range(3):
    slots.append(shared[shard::3])

transport = connect_mooncake_tcp_session(
    master_server_addr="127.0.0.1:50051",
    num_shards=NUM_SHARDS,
    key_prefix=f"docker-{time.time_ns()}",
    global_segment_size=4 * 1024**3,
    local_buffer_size=512 * 1024**2,
)
publisher = LayerwisePublisher(slots=slots, transport=transport)
consumer = LayerwiseConsumer(transport=transport, num_shards=NUM_SHARDS)

for layer_id in range(NUM_LAYERS):
    key = f"layer{layer_id}"
    publisher.publish(
        layer_id,
        key,
        {0: f"main-{layer_id}".encode(), 1: f"indexer-{layer_id}".encode()},
    )

for layer_id in range(NUM_LAYERS):
    key = f"layer{layer_id}"
    got = consumer.receive(layer_id, key)
    expected = f"main-{layer_id}indexer-{layer_id}".encode()
    assert got == expected, (layer_id, got, expected)
    transport.release(key)

print("REAL_MOONCAKE_TCP_OK", NUM_LAYERS)
PY
  ' 2>&1 | tail -30
