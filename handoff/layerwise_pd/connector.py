"""Transport/codec/slot-fence composition; no vLLM dependency."""

from __future__ import annotations

from collections.abc import Mapping

from .interfaces import LayerCodec, LayerTransport, ReuseLayout
from .slot_manager import FencedSlotManager


class LayerwiseConnector:
    def __init__(
        self,
        layout: ReuseLayout,
        transport: LayerTransport,
        codec: LayerCodec,
        *,
        num_shards: int,
    ) -> None:
        self._transport = transport
        self._codec = codec
        self._num_shards = num_shards
        self._slots = FencedSlotManager(layout, transport)

    def publish(self, layer_id: int, key: str, tensor) -> None:
        lease = self._slots.begin(layer_id, key)
        shards = self._codec.encode(layer_id, tensor, key)
        for shard_id in sorted(shards):
            self._transport.put_layer(key, layer_id, shard_id, shards[shard_id])
        self._slots.finish(lease)

    def consume(self, layer_id: int, key: str) -> None:
        parts = [self._transport.get_layer(key, shard_id) for shard_id in range(self._num_shards)]
        self._transport.wait_get(key)
        self._codec.decode(layer_id, b"".join(parts))

    def publish_all(self, tensors: Mapping[int, object], keys: Mapping[int, str]) -> None:
        for layer_id in sorted(tensors):
            self.publish(layer_id, keys[layer_id], tensors[layer_id])

    def consume_all(self, keys: Mapping[int, str]) -> None:
        for layer_id in sorted(keys):
            self.consume(layer_id, keys[layer_id])
