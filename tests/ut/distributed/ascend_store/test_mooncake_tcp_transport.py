"""Tests for the Mooncake TCP adapter, using a fake DistributedStore."""

from __future__ import annotations

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    LayerwiseConsumer,
    LayerwisePublisher,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_tcp_transport import (
    MooncakeTCPTransport,
)


class FakeMooncakeStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.put_calls: list[tuple[str, bytes]] = []
        self.removed: list[str] = []
        self.fail_put_keys: set[str] = set()

    def put(self, key: str, value: bytes) -> int:
        if key in self.fail_put_keys:
            return -1
        self.data[key] = value
        self.put_calls.append((key, value))
        return 0

    def get(self, key: str) -> bytes:
        return self.data.get(key, b"")

    def remove(self, key: str) -> int:
        self.data.pop(key, None)
        self.removed.append(key)
        return 0


def test_transport_roundtrip_through_publisher_and_consumer() -> None:
    store = FakeMooncakeStore()
    transport = MooncakeTCPTransport(store, key_prefix="test")
    publisher = LayerwisePublisher(slots=((0,),), transport=transport)
    publisher.publish(0, "layer0", {0: b"rank0-", 1: b"rank1"})

    consumer = LayerwiseConsumer(transport=transport, num_shards=2)
    assert consumer.receive(0, "layer0") == b"rank0-rank1"
    assert store.data["test:layer0:0"] == b"rank0-"
    assert store.data["test:layer0:1"] == b"rank1"


def test_transport_missing_shard_raises() -> None:
    transport = MooncakeTCPTransport(FakeMooncakeStore(), key_prefix="missing")
    with pytest.raises(KeyError, match="missing Mooncake object"):
        transport.get_layer("layer0", 3)


def test_transport_put_failure_does_not_mark_pending() -> None:
    store = FakeMooncakeStore()
    store.fail_put_keys.add("failed:layer0:0")
    transport = MooncakeTCPTransport(store, key_prefix="failed")

    with pytest.raises(RuntimeError, match="Mooncake put failed"):
        transport.put_layer("layer0", 0, 0, b"payload")
    with pytest.raises(KeyError, match="unknown pending put"):
        transport.wait_put("layer0")


def test_release_removes_all_shards_for_layer() -> None:
    store = FakeMooncakeStore()
    transport = MooncakeTCPTransport(store, key_prefix="release")
    transport.put_layer("layer0", 0, 0, b"a")
    transport.put_layer("layer0", 0, 1, b"b")
    transport.wait_put("layer0")

    transport.release("layer0")

    assert store.removed == ["release:layer0:0", "release:layer0:1"]
    assert store.data == {}


def test_key_prefix_is_required() -> None:
    with pytest.raises(ValueError, match="key_prefix"):
        MooncakeTCPTransport(FakeMooncakeStore(), key_prefix="")
