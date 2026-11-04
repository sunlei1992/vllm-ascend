"""Mooncake Store TCP adapter for the layerwise P/D prototype.

This adapter keeps the transport contract small: each layer/TP-shard pair is
one Mooncake object.  A later optimized version can switch to
``batch_put_session_start`` + ``batch_put_from_multi_buffer_ranges`` so a
whole layer is written with one session; the coordinator above does not need
to change.

The real ``mooncake`` package is imported lazily so the CPU tests can run with
a fake store on machines that do not have the wheel installed (for example
macOS Docker is required for the real TCP test).
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Protocol


class MooncakeStoreLike(Protocol):
    def put(self, key: str, value: bytes) -> int: ...

    def get(self, key: str) -> bytes: ...

    def remove(self, key: str) -> int: ...


class MooncakeTCPTransport:
    """Layerwise transport backed by a Mooncake DistributedStore.

    ``key_prefix`` isolates deployments and test runs.  The object key shape
    is ``<prefix>:<layer key>:<shard id>``; D resolves the same shape.
    """

    def __init__(self, store: MooncakeStoreLike, *, key_prefix: str = "layerwise") -> None:
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        self._store = store
        self._key_prefix = key_prefix
        self._pending_puts: set[str] = set()
        self._keys_by_layer: dict[str, list[str]] = {}

    def _object_key(self, key: str, shard_id: int) -> str:
        return f"{self._key_prefix}:{key}:{shard_id}"

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None:
        object_key = self._object_key(key, shard_id)
        rc = self._store.put(object_key, bytes(payload))
        if rc != 0:
            raise RuntimeError(f"Mooncake put failed key={object_key!r} rc={rc}")
        self._pending_puts.add(key)
        self._keys_by_layer.setdefault(key, []).append(object_key)

    def wait_put(self, key: str) -> None:
        # Mooncake ``put`` is synchronous; the pending set exists so the
        # coordinator still exercises the same fence as an async transport.
        if key not in self._pending_puts:
            raise KeyError(f"unknown pending put: {key}")
        self._pending_puts.remove(key)

    def get_layer(self, key: str, shard_id: int) -> bytes:
        object_key = self._object_key(key, shard_id)
        payload = self._store.get(object_key)
        if not payload:
            raise KeyError(f"missing Mooncake object: {object_key}")
        return payload

    def wait_get(self, key: str) -> None:
        # Mooncake ``get`` is synchronous.
        return None

    def release(self, key: str) -> None:
        """Best-effort cleanup; leases/eviction remain Store-owned."""
        for object_key in self._keys_by_layer.pop(key, []):
            # A leased object can be removed later by the Store; cleanup
            # must not mask the actual P/D error path.
            with suppress(Exception):
                self._store.remove(object_key)

    def close(self) -> None:
        # The caller owns the Store lifecycle.
        return None


def connect_mooncake_tcp(
    *,
    master_server_addr: str,
    local_hostname: str = "localhost",
    metadata_server: str = "P2PHANDSHAKE",
    global_segment_size: int = 512 * 1024 * 1024,
    local_buffer_size: int = 128 * 1024 * 1024,
    protocol: str = "tcp",
    key_prefix: str = "layerwise",
) -> MooncakeTCPTransport:
    """Create a TCP-backed transport; import Mooncake only when called."""
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:  # pragma: no cover - depends on wheel/image
        raise RuntimeError(
            "mooncake is not installed; run the real TCP test in a Linux image with the Mooncake wheel"
        ) from exc

    store: Any = MooncakeDistributedStore()
    rc = store.setup(
        local_hostname,
        metadata_server,
        global_segment_size,
        local_buffer_size,
        protocol,
        "",
        master_server_addr,
    )
    if rc != 0:
        raise RuntimeError(f"Mooncake setup failed rc={rc}")
    return MooncakeTCPTransport(store, key_prefix=key_prefix)
