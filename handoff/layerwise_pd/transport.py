"""In-memory transport used by CPU UT and as the contract reference."""

from __future__ import annotations

from .interfaces import LayerTransport


class MemoryTransport(LayerTransport):
    def __init__(self) -> None:
        self._data: dict[tuple[str, int], bytes] = {}
        self._pending_puts: set[str] = set()
        self._pending_gets: set[str] = set()
        self.put_order: list[tuple[str, int]] = []
        self.wait_put_order: list[str] = []
        self.fail_put_keys: set[str] = set()
        self.fail_get_keys: set[str] = set()

    def put_layer(self, key: str, layer_id: int, shard_id: int, payload: bytes) -> None:
        del layer_id
        if key in self.fail_put_keys:
            raise RuntimeError(f"injected put failure: {key}")
        self._data[(key, shard_id)] = bytes(payload)
        self._pending_puts.add(key)
        self.put_order.append((key, shard_id))

    def wait_put(self, key: str) -> None:
        if key not in self._pending_puts:
            raise KeyError(f"unknown pending put: {key}")
        self._pending_puts.remove(key)
        self.wait_put_order.append(key)

    def get_layer(self, key: str, shard_id: int) -> bytes:
        if key in self.fail_get_keys:
            raise RuntimeError(f"injected get failure: {key}")
        try:
            return self._data[(key, shard_id)]
        except KeyError as exc:
            raise KeyError(f"missing layer key={key!r} shard={shard_id}") from exc

    def wait_get(self, key: str) -> None:
        self._pending_gets.discard(key)

    def release(self, key: str) -> None:
        for item in [item for item in self._data if item[0] == key]:
            del self._data[item]

    def has(self, key: str, shard_id: int) -> bool:
        return (key, shard_id) in self._data
