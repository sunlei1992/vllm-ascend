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

import ctypes
import struct
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


# ---------------------------------------------------------------------------
# Session/range transport: one object per layer, one range write per shard.
# ---------------------------------------------------------------------------

_LAYER_MAGIC = b"LPD1"
_LAYER_HEADER = struct.Struct("<4sI")


def _encode_layer(shards: list[bytes]) -> bytes:
    sizes = [len(shard) for shard in shards]
    header = _LAYER_HEADER.pack(_LAYER_MAGIC, len(sizes))
    header += struct.pack(f"<{len(sizes)}Q", *sizes)
    return header + b"".join(shards)


def _decode_layer(payload: bytes) -> list[bytes]:
    if len(payload) < _LAYER_HEADER.size:
        raise ValueError("layer payload is shorter than its header")
    magic, count = _LAYER_HEADER.unpack_from(payload, 0)
    if magic != _LAYER_MAGIC:
        raise ValueError(f"bad layer magic: {magic!r}")
    header_size = _LAYER_HEADER.size + 8 * count
    if len(payload) < header_size:
        raise ValueError("layer payload is shorter than its shard table")
    sizes = struct.unpack_from(f"<{count}Q", payload, _LAYER_HEADER.size)
    if sum(sizes) != len(payload) - header_size:
        raise ValueError("layer payload size does not match its shard table")
    result: list[bytes] = []
    offset = header_size
    for size in sizes:
        result.append(payload[offset : offset + size])
        offset += size
    return result


class MooncakeSessionTransport:
    """Layerwise transport with one Mooncake object per layer.

    ``put_layer`` buffers shards; ``wait_put`` writes the header and all shards
    through one put session and one ``batch_put_from_multi_buffer_ranges`` call.
    ``get_layer`` opens one get session per layer, reads the whole object into a
    registered host buffer, and caches the decoded shards for subsequent calls.
    """

    def __init__(self, store: Any, *, num_shards: int, key_prefix: str = "layerwise-session") -> None:
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        self._store = store
        self._num_shards = num_shards
        self._key_prefix = key_prefix
        self._pending: dict[str, dict[int, bytes]] = {}
        self._cache: dict[str, list[bytes]] = {}

    def _object_key(self, key: str) -> str:
        return f"{self._key_prefix}:{key}"

    def _register(self, data: bytes) -> tuple[Any, int]:
        buffer = ctypes.create_string_buffer(len(data))
        ctypes.memmove(buffer, data, len(data))
        address = ctypes.addressof(buffer)
        rc = self._store.register_buffer(address, len(data))
        if rc != 0:
            raise RuntimeError(f"Mooncake register_buffer failed rc={rc}")
        return buffer, address

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None:
        if shard_id < 0 or shard_id >= self._num_shards:
            raise ValueError(f"shard_id {shard_id} outside [0, {self._num_shards})")
        shards = self._pending.setdefault(key, {})
        if shard_id in shards:
            raise ValueError(f"duplicate shard {shard_id} for layer key {key!r}")
        shards[shard_id] = bytes(payload)

    def wait_put(self, key: str) -> None:
        shards = self._pending.get(key)
        if shards is None:
            raise KeyError(f"unknown pending put: {key}")
        if len(shards) != self._num_shards:
            raise ValueError(f"layer key {key!r} has {len(shards)} shards, expected {self._num_shards}")
        object_key = self._object_key(key)
        payload = _encode_layer([shards[i] for i in range(self._num_shards)])
        buffer, address = self._register(payload)
        session_open = False
        try:
            starts = self._store.batch_put_session_start([object_key], [len(payload)])
            if starts != [0]:
                raise RuntimeError(f"Mooncake put session start failed: {starts}")
            session_open = True
            rcs = self._store.batch_put_from_multi_buffer_ranges(
                [object_key],
                [[address]],
                [[len(payload)]],
                [[0]],
            )
            if rcs != [len(payload)]:
                raise RuntimeError(f"Mooncake put range failed: {rcs}")
            ends = self._store.batch_put_session_end([object_key])
            if ends != [0]:
                raise RuntimeError(f"Mooncake put session end failed: {ends}")
            session_open = False
        except Exception:
            if session_open:
                with suppress(Exception):
                    self._store.batch_put_session_revoke([object_key])
            raise
        finally:
            self._store.unregister_buffer(address)
        self._pending.pop(key, None)
        self._cache.pop(key, None)

    def _fetch(self, key: str) -> list[bytes]:
        if key in self._cache:
            return self._cache[key]
        object_key = self._object_key(key)
        size = self._store.get_size(object_key)
        if size <= 0:
            raise KeyError(f"missing Mooncake object: {object_key}")
        buffer = ctypes.create_string_buffer(size)
        address = ctypes.addressof(buffer)
        rc = self._store.register_buffer(address, size)
        if rc != 0:
            raise RuntimeError(f"Mooncake register_buffer failed rc={rc}")
        session_open = False
        try:
            starts = self._store.batch_get_session_start([object_key])
            if starts != [0]:
                raise RuntimeError(f"Mooncake get session start failed: {starts}")
            session_open = True
            rcs = self._store.batch_get_into_multi_buffer_ranges(
                [object_key],
                [[address]],
                [[size]],
                [[0]],
            )
            if rcs != [size]:
                raise RuntimeError(f"Mooncake get range failed: {rcs}")
            ends = self._store.batch_get_session_end([object_key])
            if ends != 0:
                raise RuntimeError(f"Mooncake get session end failed: {ends}")
            session_open = False
            shards = _decode_layer(ctypes.string_at(address, size))
        except Exception:
            if session_open:
                with suppress(Exception):
                    self._store.batch_get_session_end([object_key])
            raise
        finally:
            self._store.unregister_buffer(address)
        if len(shards) != self._num_shards:
            raise ValueError(f"object {object_key} has {len(shards)} shards, expected {self._num_shards}")
        self._cache[key] = shards
        return shards

    def get_layer(self, key: str, shard_id: int) -> bytes:
        if shard_id < 0 or shard_id >= self._num_shards:
            raise ValueError(f"shard_id {shard_id} outside [0, {self._num_shards})")
        return self._fetch(key)[shard_id]

    def wait_get(self, key: str) -> None:
        return None

    def release(self, key: str) -> None:
        self._cache.pop(key, None)
        with suppress(Exception):
            self._store.remove(self._object_key(key))


def connect_mooncake_tcp_session(
    *,
    master_server_addr: str,
    num_shards: int,
    local_hostname: str = "localhost",
    metadata_server: str = "P2PHANDSHAKE",
    global_segment_size: int = 4 * 1024**3,
    local_buffer_size: int = 512 * 1024**2,
    protocol: str = "tcp",
    key_prefix: str = "layerwise-session",
) -> MooncakeSessionTransport:
    """Create a session/range transport with the same TCP setup as the fallback."""
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
    return MooncakeSessionTransport(store, num_shards=num_shards, key_prefix=key_prefix)
