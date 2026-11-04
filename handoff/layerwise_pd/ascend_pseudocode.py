"""NPU-only bodies for the intranet agent.

Fill ONLY the method bodies marked ``PSEUDOCODE``.  Do not change signatures,
class names, constructor fields or the interfaces in ``interfaces.py``.  If a
signature cannot express the NPU reality, stop and record the mismatch in
``RESULTS.md`` instead of changing the contract.
"""

from __future__ import annotations

from typing import Any


class NpuEvent:
    """DeviceEvent backed by torch.npu.Event."""

    def __init__(self) -> None:
        # PSEUDOCODE:
        # self._event = torch.npu.Event()
        raise NotImplementedError("T02: create torch.npu.Event")

    def record(self) -> None:
        # PSEUDOCODE:
        # self._event.record()
        raise NotImplementedError("T02: record NPU event")

    def wait(self, timeout_s: float | None = None) -> bool:
        # PSEUDOCODE:
        # return bool(self._event.wait(timeout_s))
        raise NotImplementedError("T02: wait NPU event")


class AscendLayerCodec:
    """LayerCodec for NPU MLA/SFA layer tensors.

    The CPU reference is ``codec.CPUByteCodec``.  This class must produce the
    same logical payload: all main/indexer bytes for one physical layer,
    split into TP shards in a deterministic order.
    """

    def __init__(self, num_shards: int) -> None:
        self._num_shards = num_shards
        self._dest: dict[int, Any] = {}

    def bind_layer(self, layer_id: int, tensor: Any) -> None:
        # PSEUDOCODE:
        # 1. Keep the destination tensor reference.
        # 2. Record dtype/shape/nbytes for validation.
        raise NotImplementedError("T03: bind NPU layer tensor")

    def encode(self, layer_id: int, tensor: Any, key: str) -> dict[int, bytes]:
        # PSEUDOCODE:
        # 1. tensor is the NPU per-layer KV view.
        # 2. main = tensor.main_mla_view(); indexer = tensor.indexer_view()
        # 3. raw = main_bytes + indexer_bytes, preserving dtype/layout.
        # 4. split raw into self._num_shards contiguous chunks.
        # 5. return {shard_id: bytes}
        raise NotImplementedError("T03: encode NPU KV layer")

    def decode(self, layer_id: int, payload: bytes) -> None:
        # PSEUDOCODE:
        # 1. split payload at the recorded main/indexer nbytes.
        # 2. copy main bytes into the bound main view.
        # 3. copy indexer bytes into the bound indexer view.
        # 4. assert the written byte count matches.
        raise NotImplementedError("T03: decode NPU KV layer")


class AscendStoreTransport:
    """LayerTransport over the intranet Ascend store (Mooncake/Memcache).

    Use the existing vllm-ascend store wrapper; do not reimplement the
    transport in this package unless the wrapper cannot expose put/get.
    """

    def __init__(self, store: Any, *, key_prefix: str) -> None:
        self._store = store
        self._prefix = key_prefix
        self._pending: set[str] = set()

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None:
        # PSEUDOCODE:
        # object_key = f"{self._prefix}:{key}:{shard_id}"
        # rc = self._store.put(object_key, payload)
        # if rc != 0: raise RuntimeError(...)
        raise NotImplementedError("T04: put into Ascend store")

    def wait_put(self, key: str) -> None:
        # PSEUDOCODE: wait for the store's completion/lease fence.
        raise NotImplementedError("T04: wait Ascend store put")

    def get_layer(self, key: str, shard_id: int) -> bytes:
        # PSEUDOCODE: payload = store.get(...); raise KeyError if empty.
        raise NotImplementedError("T04: get from Ascend store")

    def wait_get(self, key: str) -> None:
        # PSEUDOCODE: wait for the store's read completion.
        raise NotImplementedError("T04: wait Ascend store get")

    def release(self, key: str) -> None:
        # PSEUDOCODE: best-effort remove/release; never mask the primary error.
        raise NotImplementedError("T04: release Ascend store object")


class AscendDeviceCopier:
    """H2D/D2H copy contract used by the codec/store bridge."""

    def h2d(self, host_buffer: Any, device_buffer: Any, size: int) -> None:
        raise NotImplementedError("T05: host-to-device copy")

    def d2h(self, device_buffer: Any, host_buffer: Any, size: int) -> None:
        raise NotImplementedError("T05: device-to-host copy")
