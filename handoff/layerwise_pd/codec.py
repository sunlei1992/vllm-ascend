"""CPU tensor codec; Ascend implementation is in ascend_pseudocode.py."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .interfaces import LayerCodec


class CPUByteCodec(LayerCodec):
    def __init__(self, num_shards: int) -> None:
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        self._num_shards = num_shards
        self._dest: dict[int, torch.Tensor] = {}

    def bind_layer(self, layer_id: int, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"layer {layer_id} is not a tensor")
        self._dest[layer_id] = tensor

    def encode(self, layer_id: int, tensor: torch.Tensor, key: str) -> Mapping[int, bytes]:
        del layer_id, key
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        chunk = math.ceil(len(raw) / self._num_shards) if raw else 0
        return {shard: raw[shard * chunk : (shard + 1) * chunk] for shard in range(self._num_shards)}

    def decode(self, layer_id: int, payload: bytes) -> None:
        try:
            target = self._dest[layer_id]
        except KeyError as exc:
            raise KeyError(f"layer {layer_id} has no bound destination") from exc
        expected = target.numel() * target.element_size()
        if len(payload) != expected:
            raise ValueError(f"layer {layer_id} payload {len(payload)} != {expected}")
        decoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(target.dtype).reshape(target.shape)
        target.copy_(decoded)
