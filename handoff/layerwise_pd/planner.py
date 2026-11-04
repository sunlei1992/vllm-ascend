"""Pure layer-to-slot planner; no NPU, no vLLM, no I/O."""

from __future__ import annotations

from collections.abc import Sequence

from .interfaces import LayerSpec, ReuseLayout


class RoundRobinPlanner:
    """Assign independent layers alone, then round-robin the rest.

    This is the public CPU reference for ``build_layerwise_reuse_layout``'s
    shared-buffer grouping.  The real vLLM Ascend planner also validates cache
    specs and handles MTP/indexer entries; those checks remain on the Ascend
    side.
    """

    def plan(
        self,
        layers: Sequence[LayerSpec],
        *,
        num_shared_buffers: int,
        independent_layers: Sequence[int] = (),
    ) -> ReuseLayout:
        if num_shared_buffers < 1:
            raise ValueError("num_shared_buffers must be positive")
        if not layers:
            raise ValueError("layers must not be empty")

        by_id = {layer.layer_id: layer for layer in layers}
        if len(by_id) != len(layers):
            raise ValueError("duplicate layer ids")
        independent = tuple(dict.fromkeys(independent_layers))
        unknown = set(independent) - set(by_id)
        if unknown:
            raise ValueError(f"independent layers not found: {sorted(unknown)}")

        shared = [layer for layer in layers if layer.layer_id not in set(independent)]
        if shared:
            page_sizes = {layer.page_size_bytes for layer in shared}
            if len(page_sizes) != 1:
                raise ValueError(f"shared layers must have identical page sizes: {sorted(page_sizes)}")

        slots: list[tuple[int, ...]] = [(layer_id,) for layer_id in independent]
        for buffer_index in range(num_shared_buffers):
            slot = tuple(layer.layer_id for layer in shared[buffer_index::num_shared_buffers])
            if slot:
                slots.append(slot)

        prefetch: dict[int, int] = {}
        for slot in slots:
            for previous, current in zip(slot, slot[1:]):
                prefetch[current] = previous

        return ReuseLayout(
            slots=tuple(slots),
            independent_layers=independent,
            prefetch_map=prefetch,
        )
