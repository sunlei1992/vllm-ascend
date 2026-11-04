"""Tests for the CPU tensor codec used by the P/D prototype."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd_codec import (
    CPUByteCodec,
)


def test_codec_roundtrip_preserves_dtype_and_shape() -> None:
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    target = torch.zeros_like(source)
    producer = CPUByteCodec(num_shards=2)
    consumer = CPUByteCodec(num_shards=2)
    consumer.bind_layer("layer0", target)

    shards = producer.encode("layer0", source, "key")
    consumer.decode("layer0", b"".join(shards[shard] for shard in sorted(shards)))

    assert torch.equal(target, source)


def test_codec_handles_uneven_byte_chunks() -> None:
    source = torch.arange(5, dtype=torch.uint8)
    target = torch.zeros_like(source)
    producer = CPUByteCodec(num_shards=2)
    consumer = CPUByteCodec(num_shards=2)
    consumer.bind_layer("layer0", target)

    shards = producer.encode("layer0", source, "key")
    assert len(shards[0]) == 3
    assert len(shards[1]) == 2
    consumer.decode("layer0", b"".join(shards[shard] for shard in sorted(shards)))

    assert torch.equal(target, source)


def test_codec_rejects_unbound_destination() -> None:
    codec = CPUByteCodec(num_shards=1)
    with pytest.raises(KeyError, match="no bound destination"):
        codec.decode("missing", b"")


def test_codec_rejects_wrong_payload_size() -> None:
    codec = CPUByteCodec(num_shards=1)
    codec.bind_layer("layer0", torch.zeros(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="payload"):
        codec.decode("layer0", b"short")
