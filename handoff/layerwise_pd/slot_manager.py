"""Slot ownership and save-completion fence."""

from __future__ import annotations

from dataclasses import dataclass

from .interfaces import LayerTransport, ReuseLayout


@dataclass(frozen=True)
class SlotLease:
    slot_id: int
    layer_id: int
    key: str


class FencedSlotManager:
    """Ensure a shared slot is not reused before its previous put committed."""

    def __init__(self, layout: ReuseLayout, transport: LayerTransport) -> None:
        self._layout = layout
        self._transport = transport
        self._layer_to_slot = layout.layer_to_slot
        self._slot_key: list[str | None] = [None] * len(layout.slots)
        self._slot_owner: list[int | None] = [None] * len(layout.slots)
        self._committed: set[str] = set()

    def slot_for(self, layer_id: int) -> int:
        try:
            return self._layer_to_slot[layer_id]
        except KeyError as exc:
            raise ValueError(f"layer {layer_id} is not in the reuse layout") from exc

    def begin(self, layer_id: int, key: str) -> SlotLease:
        slot_id = self.slot_for(layer_id)
        previous = self._slot_key[slot_id]
        if previous is not None and previous not in self._committed:
            self._transport.wait_put(previous)
            self._committed.add(previous)
        self._slot_owner[slot_id] = layer_id
        return SlotLease(slot_id=slot_id, layer_id=layer_id, key=key)

    def finish(self, lease: SlotLease) -> None:
        if self._slot_owner[lease.slot_id] not in (None, lease.layer_id):
            raise RuntimeError(f"slot {lease.slot_id} is owned by another layer")
        self._transport.wait_put(lease.key)
        self._committed.add(lease.key)
        self._slot_owner[lease.slot_id] = lease.layer_id
        self._slot_key[lease.slot_id] = lease.key

    def owner(self, slot_id: int) -> int | None:
        return self._slot_owner[slot_id]

    def last_key(self, slot_id: int) -> str | None:
        return self._slot_key[slot_id]
