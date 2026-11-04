"""Lifecycle tests for the layerwise P/D connector skeleton."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    InMemoryTCPTransport,
    LayerwisePublisher,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd_codec import (
    CPUByteCodec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd_connector import (
    LayerwisePDConnector,
    LayerwisePDMetadata,
    LayerwisePDRequestMeta,
)


def make_connector(role: KVConnectorRole, kv_role: str) -> LayerwisePDConnector:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config={},
        )
    )
    return LayerwisePDConnector(config, role, SimpleNamespace(prefix_cacheable_group_ids=()))


def test_producer_scheduler_tracks_requests() -> None:
    connector = make_connector(KVConnectorRole.SCHEDULER, "kv_producer")
    request = SimpleNamespace(request_id="req-1")

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks=None, num_external_tokens=0)
    meta = connector.build_connector_meta(None)

    assert connector.is_producer is True
    assert connector.is_consumer is False
    assert meta.role == "kv_producer"
    assert [item.request_id for item in meta.requests] == ["req-1"]
    assert connector.request_finished(request, []) == (False, None)
    assert connector.build_connector_meta(None).requests == ()


def test_consumer_scheduler_tracks_requests() -> None:
    connector = make_connector(KVConnectorRole.SCHEDULER, "kv_consumer")
    request = SimpleNamespace(request_id="req-2")

    connector.update_state_after_alloc(request, blocks=None, num_external_tokens=0)
    meta = connector.build_connector_meta(None)

    assert connector.is_consumer is True
    assert meta.role == "kv_consumer"
    assert [item.key for item in meta.requests] == ["req-2"]


def test_worker_hooks_record_layers_without_transport() -> None:
    connector = make_connector(KVConnectorRole.WORKER, "kv_producer")
    connector.register_kv_caches({"layer0": object(), "layer1": object()})

    connector.save_kv_layer("layer0", object(), attn_metadata=None)
    connector.save_kv_layer("layer1", object(), attn_metadata=None)
    connector.wait_for_save()
    connector.start_load_kv(forward_context=None)
    connector.wait_for_layer_load("layer0")

    assert connector._worker_layer_names == ["layer0", "layer1"]
    assert connector._finished_layers == {"layer0", "layer1"}


def test_shutdown_clears_state() -> None:
    connector = make_connector(KVConnectorRole.SCHEDULER, "kv_producer")
    request = SimpleNamespace(request_id="req-3")
    connector.update_state_after_alloc(request, blocks=None, num_external_tokens=0)
    connector.shutdown()

    assert connector.build_connector_meta(None).requests == ()


class FakeCodec:
    def __init__(self) -> None:
        self.decoded: dict[str, bytes] = {}

    def encode(self, layer_name: str, kv_layer: object, key: str) -> dict[int, bytes]:
        del key
        return {0: bytes(kv_layer)}

    def decode(self, layer_name: str, payload: bytes) -> None:
        self.decoded[layer_name] = payload


def test_worker_attaches_runtime_and_publishes_layer() -> None:
    transport = InMemoryTCPTransport()
    connector = make_connector(KVConnectorRole.WORKER, "kv_producer")
    connector.attach_layerwise_runtime(
        transport=transport,
        slots=(("layer0",),),
        num_shards=1,
        codec=FakeCodec(),
    )
    connector.set_active_request("req-1")
    connector.save_kv_layer("layer0", b"layer-payload", attn_metadata=None)

    assert transport.get_layer("req-1", 0) == b"layer-payload"
    assert connector._finished_layers == {"layer0"}


def test_worker_consumer_receives_and_decodes_layer() -> None:
    transport = InMemoryTCPTransport()
    publisher = LayerwisePublisher(slots=(("layer0",),), transport=transport)
    publisher.publish("layer0", "req-2", {0: b"from-producer"})

    codec = FakeCodec()
    connector = make_connector(KVConnectorRole.WORKER, "kv_consumer")
    connector.attach_layerwise_runtime(
        transport=transport,
        slots=(("layer0",),),
        num_shards=1,
        codec=codec,
    )
    connector.set_active_request("req-2")
    connector.wait_for_layer_load("layer0")

    assert codec.decoded == {"layer0": b"from-producer"}


def test_start_load_kv_reads_active_key_from_metadata() -> None:
    connector = make_connector(KVConnectorRole.WORKER, "kv_consumer")
    connector.bind_connector_metadata(
        LayerwisePDMetadata(
            role="kv_consumer",
            requests=(LayerwisePDRequestMeta(request_id="req-3", key="key-3"),),
        )
    )
    connector.start_load_kv(forward_context=None)

    assert connector._active_key == "key-3"


def test_attach_runtime_rejects_scheduler_role() -> None:
    connector = make_connector(KVConnectorRole.SCHEDULER, "kv_producer")
    with pytest.raises(RuntimeError, match="only be attached on the worker"):
        connector.attach_layerwise_runtime(
            transport=InMemoryTCPTransport(),
            slots=(("layer0",),),
            num_shards=1,
            codec=FakeCodec(),
        )


def test_connector_tensor_roundtrip_through_publisher_and_consumer() -> None:
    transport = InMemoryTCPTransport()
    source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    target = torch.zeros_like(source)

    producer = make_connector(KVConnectorRole.WORKER, "kv_producer")
    producer.attach_layerwise_runtime(
        transport=transport,
        slots=(("layer0",),),
        num_shards=2,
        codec=CPUByteCodec(num_shards=2),
    )
    producer.register_kv_caches({"layer0": source})
    producer.set_active_request("req-tensor")
    producer.save_kv_layer("layer0", source, attn_metadata=None)

    consumer = make_connector(KVConnectorRole.WORKER, "kv_consumer")
    consumer.attach_layerwise_runtime(
        transport=transport,
        slots=(("layer0",),),
        num_shards=2,
        codec=CPUByteCodec(num_shards=2),
    )
    consumer.register_kv_caches({"layer0": target})
    consumer.set_active_request("req-tensor")
    consumer.wait_for_layer_load("layer0")

    assert torch.equal(target, source)
