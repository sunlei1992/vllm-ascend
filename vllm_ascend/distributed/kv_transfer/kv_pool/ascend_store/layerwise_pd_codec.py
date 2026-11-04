"""CPU tensor codec for the layerwise P/D prototype.

The codec is the boundary between a vLLM KV tensor and the byte-oriented
LayerwisePDTransport.  It does not attempt to reproduce the production MLA /
indexer layout; it is a small, testable implementation that preserves dtype,
shape and byte order while splitting a tensor across TP shards.

A production codec will replace this with the real main/indexer page views.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class CPUByteCodec:
    """Split a CPU tensor into TP shards and copy received bytes back.

    ``bind_layer`` records the destination tensor used by ``decode``.  A
    producer only needs ``encode``; a consumer binds the D-side KV tensors
    before ``wait_for_layer_load``.
    """

    def __init__(self, num_shards: int) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        self._num_shards = num_shards
        self._layers: dict[str, torch.Tensor] = {}
        self._shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    def bind_layer(self, layer_name: str, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"layer {layer_name!r} is not a tensor")
        self._layers[layer_name] = tensor
        self._shapes[layer_name] = (tuple(tensor.shape), tensor.dtype)

    def _raw(self, tensor: torch.Tensor) -> bytes:
        if not torch.is_tensor(tensor):
            raise TypeError(f"expected a tensor, got {type(tensor).__name__}")
        return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    def encode(self, layer_name: str, kv_layer: Any, key: str) -> dict[int, bytes]:
        del key
        raw = self._raw(kv_layer)
        chunk = math.ceil(len(raw) / self._num_shards) if raw else 0
        if chunk == 0:
            return {shard: b"" for shard in range(self._num_shards)}
        return {shard: raw[shard * chunk : (shard + 1) * chunk] for shard in range(self._num_shards)}

    def decode(self, layer_name: str, payload: bytes) -> None:
        if layer_name not in self._layers:
            raise KeyError(f"layer {layer_name!r} has no bound destination tensor")
        shape, dtype = self._shapes[layer_name]
        expected = self._layers[layer_name].numel() * self._layers[layer_name].element_size()
        if len(payload) != expected:
            raise ValueError(f"layer {layer_name!r} payload {len(payload)} bytes, expected {expected}")
        # torch.frombuffer needs a writable buffer; the transport payload is
        # immutable, so copy once into a bytearray.
        buffer = bytearray(payload)
        decoded = torch.frombuffer(buffer, dtype=torch.uint8).view(dtype).reshape(shape)
        self._layers[layer_name].copy_(decoded)
