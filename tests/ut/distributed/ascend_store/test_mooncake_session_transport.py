"""Tests for the one-object-per-layer Mooncake session transport."""

from __future__ import annotations

import ctypes

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    LayerwiseConsumer,
    LayerwisePublisher,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_tcp_transport import (
    MooncakeSessionTransport,
)


class FakeSessionStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.sessions: dict[str, bytearray] = {}
        self.calls: list[str] = []
        self.fail_range = False

    def register_buffer(self, _address: int, _size: int) -> int:
        return 0

    def unregister_buffer(self, _address: int) -> int:
        return 0

    def batch_put_session_start(self, keys: list[str], sizes: list[int]) -> list[int]:
        self.calls.append("put_start")
        for key, size in zip(keys, sizes, strict=True):
            self.sessions[key] = bytearray(size)
        return [0] * len(keys)

    def batch_put_from_multi_buffer_ranges(self, keys, all_ptrs, all_sizes, all_offsets):
        self.calls.append("put_range")
        if self.fail_range:
            return [-1] * len(keys)
        result = []
        for key, ptrs, sizes, offsets in zip(keys, all_ptrs, all_sizes, all_offsets, strict=True):
            session = self.sessions[key]
            total = 0
            for ptr, size, offset in zip(ptrs, sizes, offsets, strict=True):
                session[offset : offset + size] = ctypes.string_at(ptr, size)
                total += size
            result.append(total)
        return result

    def batch_put_session_end(self, keys: list[str]) -> list[int]:
        self.calls.append("put_end")
        for key in keys:
            self.data[key] = bytes(self.sessions.pop(key))
        return [0] * len(keys)

    def batch_put_session_revoke(self, keys: list[str]) -> list[int]:
        self.calls.append("put_revoke")
        for key in keys:
            self.sessions.pop(key, None)
        return [0] * len(keys)

    def get_size(self, key: str) -> int:
        return len(self.data.get(key, b""))

    def batch_get_session_start(self, keys: list[str]) -> list[int]:
        self.calls.append("get_start")
        return [0 if key in self.data else -1 for key in keys]

    def batch_get_into_multi_buffer_ranges(self, keys, all_ptrs, all_sizes, all_offsets):
        self.calls.append("get_range")
        result = []
        for key, ptrs, sizes, offsets in zip(keys, all_ptrs, all_sizes, all_offsets, strict=True):
            payload = self.data[key]
            total = 0
            for ptr, size, offset in zip(ptrs, sizes, offsets, strict=True):
                ctypes.memmove(ptr, payload[offset : offset + size], size)
                total += size
            result.append(total)
        return result

    def batch_get_session_end(self, _keys: list[str]) -> int:
        self.calls.append("get_end")
        return 0

    def remove(self, key: str) -> int:
        self.data.pop(key, None)
        return 0


def test_session_transport_roundtrip_uses_one_object_per_layer() -> None:
    store = FakeSessionStore()
    transport = MooncakeSessionTransport(store, num_shards=2, key_prefix="s")
    publisher = LayerwisePublisher(slots=((0,),), transport=transport)
    publisher.publish(0, "layer0", {0: b"rank0-", 1: b"rank1"})

    assert list(store.data) == ["s:layer0"]
    assert "put_start" in store.calls
    assert "put_range" in store.calls
    assert "put_end" in store.calls

    consumer = LayerwiseConsumer(transport=transport, num_shards=2)
    assert consumer.receive(0, "layer0") == b"rank0-rank1"
    assert "get_start" in store.calls
    assert "get_range" in store.calls
    assert "get_end" in store.calls


def test_session_transport_caches_decoded_layer() -> None:
    store = FakeSessionStore()
    transport = MooncakeSessionTransport(store, num_shards=2, key_prefix="cache")
    publisher = LayerwisePublisher(slots=((0,),), transport=transport)
    publisher.publish(0, "layer0", {0: b"a", 1: b"b"})

    assert transport.get_layer("layer0", 0) == b"a"
    calls = list(store.calls)
    assert transport.get_layer("layer0", 1) == b"b"
    assert store.calls == calls


def test_session_transport_requires_all_shards_before_wait() -> None:
    store = FakeSessionStore()
    transport = MooncakeSessionTransport(store, num_shards=2, key_prefix="missing")
    transport.put_layer("layer0", 0, 0, b"only")

    with pytest.raises(ValueError, match="has 1 shards, expected 2"):
        transport.wait_put("layer0")


def test_session_transport_revokes_on_range_failure() -> None:
    store = FakeSessionStore()
    store.fail_range = True
    transport = MooncakeSessionTransport(store, num_shards=1, key_prefix="fail")
    transport.put_layer("layer0", 0, 0, b"payload")

    with pytest.raises(RuntimeError, match="put range failed"):
        transport.wait_put("layer0")
    assert "put_revoke" in store.calls
    assert store.data == {}


def test_session_transport_rejects_duplicate_or_out_of_range_shard() -> None:
    store = FakeSessionStore()
    transport = MooncakeSessionTransport(store, num_shards=2, key_prefix="dup")
    transport.put_layer("layer0", 0, 0, b"a")
    with pytest.raises(ValueError, match="duplicate shard"):
        transport.put_layer("layer0", 0, 0, b"b")
    with pytest.raises(ValueError, match="outside"):
        transport.put_layer("layer0", 0, 2, b"c")
