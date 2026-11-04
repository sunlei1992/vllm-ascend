"""GLM-shaped synthetic end-to-end test for the layerwise shared-buffer path.

This test does not run GLM. It builds CPU tensors with the same *shape class*
as GLM-5.3's main MLA + indexer caches (78 physical layers, two cache entries
per layer), reuses three shared slots, and exercises the P publisher,
transport, and D consumer end to end.
"""

from __future__ import annotations

import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    build_layerwise_cache_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    InMemoryTCPTransport,
    LayerwiseConsumer,
    LayerwisePublisher,
)

NUM_LAYERS = 78
BLOCKS = 2
MAIN_BYTES = 64
INDEXER_BYTES = 16
NUM_SHARDS = 2


class GLMShapeCodec:
    """Composite main+indexer codec for one physical layer."""

    def __init__(self) -> None:
        self._targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._main_nbytes: dict[int, int] = {}

    def bind(self, layer_id: int, main: torch.Tensor, indexer: torch.Tensor) -> None:
        self._targets[layer_id] = (main, indexer)
        self._main_nbytes[layer_id] = main.numel() * main.element_size()

    def encode(self, layer_id: int, main: torch.Tensor, indexer: torch.Tensor) -> dict[int, bytes]:
        del layer_id
        main_raw = main.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        idx_raw = indexer.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        raw = main_raw + idx_raw
        chunk = (len(raw) + NUM_SHARDS - 1) // NUM_SHARDS
        return {shard: raw[shard * chunk : (shard + 1) * chunk] for shard in range(NUM_SHARDS)}

    def decode(self, layer_id: int, payload: bytes) -> None:
        main, indexer = self._targets[layer_id]
        split = self._main_nbytes[layer_id]
        main.copy_(torch.frombuffer(bytearray(payload[:split]), dtype=torch.uint8).view(main.dtype).reshape(main.shape))
        indexer.copy_(
            torch.frombuffer(bytearray(payload[split:]), dtype=torch.uint8).view(indexer.dtype).reshape(indexer.shape)
        )


def make_layer(layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    main = torch.arange(BLOCKS * MAIN_BYTES, dtype=torch.uint8).reshape(BLOCKS, MAIN_BYTES) + layer_id
    indexer = torch.arange(BLOCKS * INDEXER_BYTES, dtype=torch.uint8).reshape(BLOCKS, INDEXER_BYTES) + layer_id
    return main, indexer


def test_glm_shape_78_layers_with_three_shared_slots() -> None:
    layout = build_layerwise_cache_layout(
        NUM_LAYERS,
        {
            "layerwise_num_shared_buffers": 3,
            "layerwise_independent_layers": [0],
        },
    )
    assert layout.has_layer_reuse is True
    assert len(layout.storage_indices) == 4
    assert sum(len(slot) for slot in layout.storage_indices) == NUM_LAYERS

    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=layout.storage_indices, transport=transport)
    consumer = LayerwiseConsumer(transport=transport, num_shards=NUM_SHARDS)
    producer_codec = GLMShapeCodec()
    consumer_codec = GLMShapeCodec()
    targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for layer_id in range(NUM_LAYERS):
        main, indexer = make_layer(layer_id)
        key = f"req0-layer{layer_id}"
        publisher.publish(layer_id, key, producer_codec.encode(layer_id, main, indexer))
        target_main = torch.zeros_like(main)
        target_indexer = torch.zeros_like(indexer)
        targets[layer_id] = (target_main, target_indexer)
        consumer_codec.bind(layer_id, target_main, target_indexer)

    for layer_id in range(NUM_LAYERS):
        payload = consumer.receive(layer_id, f"req0-layer{layer_id}")
        consumer_codec.decode(layer_id, payload)
        expected_main, expected_indexer = make_layer(layer_id)
        actual_main, actual_indexer = targets[layer_id]
        assert torch.equal(actual_main, expected_main)
        assert torch.equal(actual_indexer, expected_indexer)

    # Three shared slots + one independent slot must have been reused.
    assert len(transport.wait_put_order) == NUM_LAYERS


def test_glm_shape_multiple_requests_reuse_slots() -> None:
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
    producer_codec = GLMShapeCodec()
    consumer_codec = GLMShapeCodec()
    targets: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    for request_id in range(3):
        for layer_id in range(NUM_LAYERS):
            key = f"req{request_id}-layer{layer_id}"
            main, indexer = make_layer(layer_id)
            publisher.publish(layer_id, key, producer_codec.encode(layer_id, main, indexer))
            target_main = torch.zeros_like(main)
            target_indexer = torch.zeros_like(indexer)
            targets[(request_id, layer_id)] = (target_main, target_indexer)

    for request_id in range(3):
        for layer_id in range(NUM_LAYERS):
            target_main, target_indexer = targets[(request_id, layer_id)]
            consumer_codec.bind(layer_id, target_main, target_indexer)
            payload = consumer.receive(layer_id, f"req{request_id}-layer{layer_id}")
            consumer_codec.decode(layer_id, payload)
            expected_main, expected_indexer = make_layer(layer_id)
            assert torch.equal(target_main, expected_main)
            assert torch.equal(target_indexer, expected_indexer)

    assert len(transport.wait_put_order) == 3 * NUM_LAYERS
