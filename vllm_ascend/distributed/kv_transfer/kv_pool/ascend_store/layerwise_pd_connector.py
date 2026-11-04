"""Skeleton P/D connector for the layerwise shared-buffer prototype.

This is intentionally a thin lifecycle skeleton: it wires the scheduler and
worker hooks, carries per-request metadata, and leaves the actual layer
publish/consume calls to a transport implementation.  It is not registered as
a production connector yet; the goal is to make the vLLM integration points
testable before the data plane is connected.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (
    LayerwiseConsumer,
    LayerwisePDTransport,
    LayerwisePublisher,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class LayerwisePDRequestMeta:
    request_id: str
    key: str
    num_shards: int = 1


class LayerwisePDCodec(Protocol):
    """Convert a vLLM KV layer to/from per-TP-shard bytes."""

    def encode(self, layer_name: str, kv_layer: Any, key: str) -> Mapping[int, bytes]: ...

    def decode(self, layer_name: str, payload: bytes) -> None: ...


@dataclass
class LayerwisePDMetadata(KVConnectorMetadata):
    role: str = ""
    requests: tuple[LayerwisePDRequestMeta, ...] = ()


class LayerwisePDConnector(KVConnectorBase_V1):
    """Lifecycle-only P/D connector skeleton.

    ``kv_role`` is ``kv_producer`` on P and ``kv_consumer`` on D.  The
    scheduler tracks admitted requests; the worker records layer hooks.  A
    real transport can be attached to ``transport`` and used by
    ``save_kv_layer`` / ``wait_for_layer_load`` without changing the vLLM
    lifecycle wiring.
    """

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._pending: OrderedDict[str, LayerwisePDRequestMeta] = OrderedDict()
        self._worker_layer_names: list[str] = []
        self._finished_layers: set[str] = set()
        self.is_producer = self.kv_role in ("kv_producer", "kv_both")
        self.is_consumer = self.kv_role in ("kv_consumer", "kv_both")
        # The real transport/runtime is attached by the worker after KV caches
        # and the Mooncake store are initialized.  The scheduler connector
        # only needs the request metadata.
        self.transport: LayerwisePDTransport | None = None
        self._publisher: LayerwisePublisher | None = None
        self._consumer: LayerwiseConsumer | None = None
        self._codec: LayerwisePDCodec | None = None
        self._active_key: str | None = None
        self._kv_caches: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Scheduler side
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        # Skeleton: no external lookup yet.  Returning (0, False) keeps the
        # request on the normal local-prefill path.
        del request, num_computed_tokens
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        del blocks
        if num_external_tokens:
            logger.warning(
                "LayerwisePDConnector skeleton does not load external tokens yet; request=%s external=%d",
                request.request_id,
                num_external_tokens,
            )
        self._pending[request.request_id] = LayerwisePDRequestMeta(
            request_id=request.request_id,
            key=request.request_id,
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> LayerwisePDMetadata:
        del scheduler_output
        return LayerwisePDMetadata(
            role=self.kv_role,
            requests=tuple(self._pending.values()),
        )

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        self._pending.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def attach_layerwise_runtime(
        self,
        *,
        transport: LayerwisePDTransport,
        slots: tuple[tuple[str, ...], ...],
        num_shards: int,
        codec: LayerwisePDCodec,
    ) -> None:
        """Attach the real transport/codec to a worker connector.

        The scheduler-side connector does not need this; the worker calls it
        after the Mooncake store and KV caches are initialized.
        """
        if self.role != KVConnectorRole.WORKER:
            raise RuntimeError("layerwise runtime can only be attached on the worker")
        self.transport = transport
        self._publisher = LayerwisePublisher(slots=slots, transport=transport)
        self._consumer = LayerwiseConsumer(transport=transport, num_shards=num_shards)
        self._codec = codec
        bind = getattr(codec, "bind_layer", None)
        if callable(bind):
            for layer_name, tensor in self._kv_caches.items():
                bind(layer_name, tensor)

    def set_active_request(self, key: str) -> None:
        self._active_key = key

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._worker_layer_names = sorted(kv_caches)
        self._kv_caches = dict(kv_caches)
        bind = getattr(self._codec, "bind_layer", None)
        if callable(bind):
            for layer_name, tensor in self._kv_caches.items():
                bind(layer_name, tensor)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        if not self.has_connector_metadata():
            return
        metadata = self._get_connector_metadata()
        if isinstance(metadata, LayerwisePDMetadata) and metadata.requests:
            self._active_key = metadata.requests[0].key

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.transport is None or self._consumer is None or self._codec is None:
            return
        if not self.is_consumer or self._active_key is None:
            return
        payload = self._consumer.receive(layer_name, self._active_key)
        self._codec.decode(layer_name, payload)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del attn_metadata, kwargs
        if self.transport is not None and self._publisher is not None and self._codec is not None:
            if not self.is_producer or self._active_key is None:
                return
            shards = self._codec.encode(layer_name, kv_layer, self._active_key)
            self._publisher.publish(layer_name, self._active_key, shards)
        elif self.is_producer:
            logger.debug("LayerwisePDConnector producer has no transport attached; layer=%s", layer_name)
        self._finished_layers.add(layer_name)

    def wait_for_save(self) -> None:
        if self.transport is None and self.is_producer:
            return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        return set(), set()

    def shutdown(self) -> None:
        self._pending.clear()
        self._finished_layers.clear()
        self._active_key = None
