"""P1: closed-loop slot-fence stress for the layerwise P/D prototype."""

from __future__ import annotations

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    build_layerwise_cache_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    InMemoryTCPTransport,
    LayerwiseConsumer,
    LayerwisePublisher,
)

NUM_LAYERS = 78
NUM_PREFIXES = 24
REQUESTS_PER_PREFIX = 40
NUM_SHARDS = 2


def _payload(prefix: int, request: int, layer: int, shard: int) -> bytes:
    return f"{prefix}:{request}:{layer}:{shard}".encode()


def test_24_prefixes_40_requests_slot_fence() -> None:
    layout = build_layerwise_cache_layout(
        NUM_LAYERS,
        {
            "layerwise_num_shared_buffers": 3,
            "layerwise_independent_layers": [0],
        },
    )
    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=layout.storage_indices, transport=transport)
    consumer = LayerwiseConsumer(transport=transport, num_shards=NUM_SHARDS)

    expected = {}
    for prefix in range(NUM_PREFIXES):
        for request in range(REQUESTS_PER_PREFIX):
            for layer in range(NUM_LAYERS):
                key = f"p{prefix}-r{request}-l{layer}"
                shards = {shard: _payload(prefix, request, layer, shard) for shard in range(NUM_SHARDS)}
                expected[(prefix, request, layer)] = shards[0] + shards[1]
                publisher.publish(layer, key, shards)

    assert len(transport.wait_put_order) == NUM_PREFIXES * REQUESTS_PER_PREFIX * NUM_LAYERS
    for prefix in range(NUM_PREFIXES):
        for request in range(REQUESTS_PER_PREFIX):
            for layer in range(NUM_LAYERS):
                key = f"p{prefix}-r{request}-l{layer}"
                assert consumer.receive(layer, key) == expected[(prefix, request, layer)]
