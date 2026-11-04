"""Stable contracts for the layerwise shared-buffer handoff.

Only these interfaces may be shared between the CPU reference implementation
and the Ascend implementation.  The intranet implementation fills the
``ascend_pseudocode.py`` bodies; it must not change these signatures without
stopping and reporting an interface mismatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    name: str
    page_size_bytes: int


@dataclass(frozen=True)
class ReuseLayout:
    slots: tuple[tuple[int, ...], ...]
    independent_layers: tuple[int, ...]
    prefetch_map: Mapping[int, int]

    @property
    def layer_to_slot(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for slot_id, layer_ids in enumerate(self.slots):
            for layer_id in layer_ids:
                if layer_id in result:
                    raise ValueError(f"layer {layer_id} appears in more than one slot")
                result[layer_id] = slot_id
        return result


class SharedBufferPlanner(Protocol):
    def plan(
        self,
        layers: Sequence[LayerSpec],
        *,
        num_shared_buffers: int,
        independent_layers: Sequence[int] = (),
    ) -> ReuseLayout: ...


class LayerTransport(Protocol):
    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None: ...

    def wait_put(self, key: str) -> None: ...

    def get_layer(self, key: str, shard_id: int) -> bytes: ...

    def wait_get(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...


class LayerCodec(Protocol):
    def bind_layer(self, layer_id: int, tensor) -> None: ...

    def encode(self, layer_id: int, tensor, key: str) -> Mapping[int, bytes]: ...

    def decode(self, layer_id: int, payload: bytes) -> None: ...


class DeviceEvent(Protocol):
    def record(self) -> None: ...

    def wait(self, timeout_s: float | None = None) -> bool: ...
