# SPDX-License-Identifier: Apache-2.0
"""Pull transport backend selection for the SFA P/D connector.

``memfabric`` is the existing Ascend path.  ``mooncake`` reuses Mooncake
TransferEngine (``initialize`` + ``batch_register_memory`` +
``batch_transfer_sync_read``) so the same read thread can pull from P's
registered buffers over TCP locally or over the ``ascend`` transport on NPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

BACKEND_MEMFABRIC = "memfabric"
BACKEND_MOONCAKE = "mooncake"
PULL_BACKENDS = (BACKEND_MEMFABRIC, BACKEND_MOONCAKE)


def _extra(vllm_config: Any) -> dict[str, Any]:
    config = getattr(vllm_config, "kv_transfer_config", None)
    return (getattr(config, "kv_connector_extra_config", None) or {}) if config else {}


def resolve_transfer_backend(vllm_config: Any) -> str:
    backend = _extra(vllm_config).get("transfer_backend", BACKEND_MEMFABRIC)
    if backend not in PULL_BACKENDS:
        raise ValueError(
            f"Unsupported transfer_backend={backend!r}; choose one of {PULL_BACKENDS}."
        )
    return backend


def resolve_mooncake_options(vllm_config: Any) -> dict[str, str]:
    extra = _extra(vllm_config)
    return {
        "metadata": str(extra.get("mooncake_metadata", "P2PHANDSHAKE")),
        "protocol": str(extra.get("mooncake_transfer_protocol", "ascend")),
        "device": str(extra.get("mooncake_device_name", "")),
    }


class MemfabricPullEngine:
    """Thin adapter over the existing global memfabric engine."""

    def __init__(self, hostname: str, role: str, protocol: str | None, device_id: int):
        from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (
            global_memfabric_te,
        )

        global_memfabric_te.configure(
            role=role,
            device_id=device_id,
            transfer_protocol=protocol,
        )
        self._engine = global_memfabric_te.get_transfer_engine(hostname)
        self.unique_id = global_memfabric_te.unique_id

    def get_rpc_port(self) -> int:
        return self._engine.get_rpc_port()

    def register_memory(self, ptrs: Sequence[int], sizes: Sequence[int]) -> int:
        for ptr, size in zip(ptrs, sizes, strict=True):
            ret = self._engine.register_memory(int(ptr), int(size))
            if ret != 0:
                return int(ret)
        return 0

    def batch_transfer_sync_read(
        self,
        peer: str,
        local_ptrs: Sequence[int],
        peer_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        return int(
            self._engine.batch_transfer_sync_read(
                peer, list(local_ptrs), list(peer_ptrs), list(lengths)
            )
        )


class MooncakePullEngine:
    """Mooncake TransferEngine adapter with the same contract."""

    def __init__(self, hostname: str, options: dict[str, str]):
        from mooncake.engine import TransferEngine  # type: ignore

        self._engine = TransferEngine()
        rc = self._engine.initialize(
            hostname,
            options["metadata"],
            options["protocol"],
            options["device"],
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed rc={rc}")
        # For the TCP build ``unique_id`` is the peer hostname used by
        # batch_transfer_sync_read; the NPU build uses a session id instead.
        self.unique_id = hostname
        self._registered = False

    def get_rpc_port(self) -> int:
        return 0

    def register_memory(self, ptrs: Sequence[int], sizes: Sequence[int]) -> int:
        rc = self._engine.batch_register_memory(list(ptrs), list(sizes))
        self._registered = rc == 0
        return int(rc)

    def batch_transfer_sync_read(
        self,
        peer: str,
        local_ptrs: Sequence[int],
        peer_ptrs: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        return int(
            self._engine.batch_transfer_sync_read(
                peer, list(local_ptrs), list(peer_ptrs), list(lengths)
            )
        )


def make_pull_engine(
    vllm_config: Any,
    *,
    role: str,
    hostname: str,
    device_id: int,
    memfabric_protocol: str | None = None,
) -> MemfabricPullEngine | MooncakePullEngine:
    backend = resolve_transfer_backend(vllm_config)
    if backend == BACKEND_MEMFABRIC:
        return MemfabricPullEngine(hostname, role, memfabric_protocol, device_id)
    return MooncakePullEngine(hostname, resolve_mooncake_options(vllm_config))
