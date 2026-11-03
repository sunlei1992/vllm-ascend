"""Layerwise P/D transport prototype.

This module keeps the P-side shared-slot lifetime separate from the actual
transport.  A real Mooncake TCP/URMA adapter implements ``LayerwisePDTransport``;
the publisher/consumer below are transport-agnostic and can be tested with an
in-memory fake before the data plane exists.

The model:
  - P owns a small number of physical slots; several layers share each slot.
  - Publishing a layer must wait until the previous owner of the slot has been
    published (put completed), otherwise the previous layer's bytes could be
    overwritten while D is still reading them.
  - D receives every TP shard of a layer and concatenates the shards in rank
    order into its own (for example TP1) layout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


class LayerwisePDTransport(Protocol):
    """Small transport contract used by the P/D coordinator."""

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None: ...

    def wait_put(self, key: str) -> None: ...

    def get_layer(self, key: str, shard_id: int) -> bytes: ...

    def wait_get(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...


@dataclass
class LayerwisePublisher:
    """P-side slot/publish coordinator.

    ``slots`` is the same shape as ``LayerwiseReuseLayout.buffer_slots``:
    one tuple per physical slot, containing the layer ids that share it.
    """

    slots: Sequence[Sequence[int]]
    transport: LayerwisePDTransport
    _layer_to_slot: dict[int, int] = field(init=False, repr=False)
    _slot_last_key: list[str | None] = field(init=False, repr=False)
    _published_keys: dict[int, str] = field(init=False, repr=False)
    _committed_keys: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._layer_to_slot = {}
        for slot_id, layers in enumerate(self.slots):
            for layer_id in layers:
                if layer_id in self._layer_to_slot:
                    raise ValueError(f"layer {layer_id} appears in more than one slot")
                self._layer_to_slot[layer_id] = slot_id
        if len(self._layer_to_slot) != sum(len(layer) for layer in self.slots):
            raise ValueError("duplicate layer ids in slots")
        self._slot_last_key = [None] * len(self.slots)
        self._published_keys = {}
        self._committed_keys = set()

    def publish(self, layer_id: int, key: str, shards: Mapping[int, bytes]) -> None:
        """Publish one layer after all of its TP shards are in the slot.

        The caller is responsible for filling the slot before calling this
        method.  This method only handles the cross-layer fence: a slot cannot
        be reused until its previous key has completed its put.
        """
        try:
            slot_id = self._layer_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} is not assigned to any slot") from exc
        if not shards:
            raise ValueError(f"layer {layer_id} has no TP shards")

        previous = self._slot_last_key[slot_id]
        if previous is not None and previous not in self._committed_keys:
            # The previous owner must be durable before this layer can reuse
            # the same physical slot.
            self.transport.wait_put(previous)
            self._committed_keys.add(previous)

        for shard_id in sorted(shards):
            self.transport.put_layer(key, layer_id, shard_id, shards[shard_id])
        self.transport.wait_put(key)
        self._committed_keys.add(key)

        self._slot_last_key[slot_id] = key
        self._published_keys[layer_id] = key

    def published_key(self, layer_id: int) -> str | None:
        return self._published_keys.get(layer_id)

    def slot_for(self, layer_id: int) -> int:
        return self._layer_to_slot[layer_id]


@dataclass
class LayerwiseConsumer:
    """D-side layer receiver.

    ``num_shards`` is the P-side TP size.  The consumer concatenates shards in
    ascending shard order to form the D-side (for example TP1) layer payload.
    """

    transport: LayerwisePDTransport
    num_shards: int

    def receive(self, layer_id: int, key: str) -> bytes:
        parts: list[bytes] = []
        for shard_id in range(self.num_shards):
            parts.append(self.transport.get_layer(key, shard_id))
        self.transport.wait_get(key)
        if len(parts) != self.num_shards:
            raise ValueError(f"layer {layer_id} missing TP shards")
        return b"".join(parts)

    def receive_all(self, layers: Mapping[int, str]) -> dict[int, bytes]:
        return {layer_id: self.receive(layer_id, key) for layer_id, key in sorted(layers.items())}


class InMemoryTCPTransport:
    """In-memory stand-in for a future Mooncake TCP transport.

    The real adapter will use Mooncake Store with ``protocol=tcp``; this fake
    keeps the same key/shard shape so the coordinator can be tested without a
    Mooncake server or an NPU.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, int], bytes] = {}
        self._pending_puts: set[str] = set()
        self._pending_gets: set[str] = set()
        self.put_order: list[tuple[str, int]] = []
        self.wait_put_order: list[str] = []
        self.release_order: list[str] = []
        self.fail_put_keys: set[str] = set()

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None:
        if key in self.fail_put_keys:
            raise RuntimeError(f"injected put failure for {key}")
        self._data[(key, shard_id)] = bytes(payload)
        self._pending_puts.add(key)
        self.put_order.append((key, shard_id))

    def wait_put(self, key: str) -> None:
        if key not in self._pending_puts:
            # A real transport should reject waiting on an unknown key.
            raise KeyError(f"unknown pending put: {key}")
        self._pending_puts.remove(key)
        self.wait_put_order.append(key)

    def get_layer(self, key: str, shard_id: int) -> bytes:
        try:
            return self._data[(key, shard_id)]
        except KeyError as exc:
            raise KeyError(f"missing layer key={key!r} shard={shard_id}") from exc

    def wait_get(self, key: str) -> None:
        self._pending_gets.discard(key)

    def release(self, key: str) -> None:
        self.release_order.append(key)
        for k in [k for k in self._data if k[0] == key]:
            del self._data[k]

    def has(self, key: str, shard_id: int) -> bool:
        return (key, shard_id) in self._data
