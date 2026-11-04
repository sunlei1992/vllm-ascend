"""P5: failure injection and checksum consistency for shared slots."""

from __future__ import annotations

import hashlib

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    InMemoryTCPTransport,
    LayerwiseConsumer,
    LayerwisePublisher,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_failed_put_does_not_advance_slot_and_retry_is_clean() -> None:
    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=((0, 1),), transport=transport)
    consumer = LayerwiseConsumer(transport=transport, num_shards=1)

    old = b"old-layer0"
    publisher.publish(0, "layer0", {0: old})
    transport.fail_put_keys.add("layer1")

    with pytest.raises(RuntimeError, match="injected put failure"):
        publisher.publish(1, "layer1", {0: b"new-layer1"})
    assert publisher.published_key(1) is None
    assert consumer.receive(0, "layer0") == old

    transport.fail_put_keys.remove("layer1")
    publisher.publish(1, "layer1", {0: b"new-layer1"})
    assert consumer.receive(1, "layer1") == b"new-layer1"
    assert consumer.receive(0, "layer0") == old


def test_reused_slot_preserves_previous_object_checksum() -> None:
    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=((0, 1),), transport=transport)
    consumer = LayerwiseConsumer(transport=transport, num_shards=2)

    first = b"first" * 128
    second = b"second" * 128
    first_checksum = _sha(first + first)

    publisher.publish(0, "layer0", {0: first, 1: first})
    publisher.publish(1, "layer1", {0: second, 1: second})

    # The shared slot now belongs to layer1, but layer0's object is durable.
    assert consumer.receive(0, "layer0") == first + first
    assert _sha(consumer.receive(0, "layer0")) == first_checksum
    assert consumer.receive(1, "layer1") == second + second


def test_missing_shard_is_detected_by_checksum_roundtrip() -> None:
    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=((0,),), transport=transport)
    publisher.publish(0, "layer0", {0: b"rank0", 1: b"rank1"})

    consumer = LayerwiseConsumer(transport=transport, num_shards=2)
    payload = consumer.receive(0, "layer0")
    assert _sha(payload) == _sha(b"rank0rank1")

    # Remove one shard: the next read must fail rather than return a partial
    # checksum.
    del transport._data[("layer0", 1)]
    with pytest.raises(KeyError, match="shard=1"):
        consumer.receive(0, "layer0")
