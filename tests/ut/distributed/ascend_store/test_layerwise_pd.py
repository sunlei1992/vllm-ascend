"""Tests for the layerwise P/D prototype.

These tests exercise the cross-layer slot fence, TP shard aggregation and
failure retry without a Mooncake server or an NPU.  The transport is the
in-memory fake; a real Mooncake TCP/URMA adapter will implement the same
``LayerwisePDTransport`` contract.
"""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    build_layerwise_cache_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    InMemoryTCPTransport,
    LayerwiseConsumer,
    LayerwisePublisher,
)


def make_publisher(slots, transport) -> LayerwisePublisher:
    return LayerwisePublisher(slots=slots, transport=transport)


def test_slot_reuse_waits_for_previous_publish() -> None:
    transport = InMemoryTCPTransport()
    publisher = make_publisher(((0, 2), (1, 3)), transport)

    publisher.publish(0, "layer0", {0: b"a0"})
    publisher.publish(1, "layer1", {0: b"b0"})
    publisher.publish(2, "layer2", {0: b"c0"})

    assert transport.wait_put_order == ["layer0", "layer1", "layer2"]
    assert transport.put_order == [("layer0", 0), ("layer1", 0), ("layer2", 0)]
    assert publisher.published_key(2) == "layer2"


def test_consumer_aggregates_tp_shards_in_rank_order() -> None:
    transport = InMemoryTCPTransport()
    publisher = make_publisher(((0,),), transport)
    publisher.publish(0, "layer0", {0: b"rank0-", 1: b"rank1-", 2: b"rank2"})

    consumer = LayerwiseConsumer(transport=transport, num_shards=3)
    assert consumer.receive(0, "layer0") == b"rank0-rank1-rank2"


def test_failed_put_can_be_retried_without_overwriting_slot() -> None:
    transport = InMemoryTCPTransport()
    publisher = make_publisher(((0, 1),), transport)

    publisher.publish(0, "layer0", {0: b"old"})
    transport.fail_put_keys.add("layer1")

    with pytest.raises(RuntimeError, match="injected put failure"):
        publisher.publish(1, "layer1", {0: b"new"})
    assert publisher.published_key(1) is None
    assert transport.get_layer("layer0", 0) == b"old"

    transport.fail_put_keys.remove("layer1")
    publisher.publish(1, "layer1", {0: b"new"})
    assert publisher.published_key(1) == "layer1"
    assert transport.get_layer("layer1", 0) == b"new"


def test_consumer_rejects_missing_tp_shard() -> None:
    transport = InMemoryTCPTransport()
    publisher = make_publisher(((0,),), transport)
    publisher.publish(0, "layer0", {0: b"only-rank0"})

    consumer = LayerwiseConsumer(transport=transport, num_shards=2)
    with pytest.raises(KeyError, match="shard=1"):
        consumer.receive(0, "layer0")


def test_roundtrip_all_layers_through_shared_slots() -> None:
    transport = InMemoryTCPTransport()
    slots = ((0, 3), (1, 4), (2, 5))
    publisher = make_publisher(slots, transport)
    expected: dict[int, bytes] = {}
    for layer_id in range(6):
        payload = bytes([layer_id]) * 8
        expected[layer_id] = payload + payload
        publisher.publish(layer_id, f"layer{layer_id}", {0: payload, 1: payload})

    consumer = LayerwiseConsumer(transport=transport, num_shards=2)
    received = consumer.receive_all({layer_id: f"layer{layer_id}" for layer_id in range(6)})
    assert received == {layer_id: expected[layer_id] for layer_id in range(6)}


def test_slots_match_existing_reuse_layout_shape() -> None:
    layout = build_layerwise_cache_layout(
        6,
        {
            "layerwise_num_shared_buffers": 2,
            "layerwise_independent_layers": [0],
        },
    )
    assert layout.has_layer_reuse is True
    assert len(layout.storage_indices) == 3
    assert layout.storage_indices[0] == [0]
    assert sorted(layer for slot in layout.storage_indices for layer in slot) == list(range(6))


def test_duplicate_layer_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than one slot"):
        make_publisher(((0, 1), (1, 2)), InMemoryTCPTransport())
