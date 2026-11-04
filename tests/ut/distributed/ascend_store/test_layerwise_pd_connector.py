"""Lifecycle tests for the layerwise P/D connector skeleton."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd_connector import (
    LayerwisePDConnector,
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
