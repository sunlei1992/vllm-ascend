"""CPU integration harness for the layerwise KV path with a fake store.

The real layer transfer code computes raw addresses and byte counts; this
module drives it with CPU tensors and an address-based in-memory store, so the
packing/unpacking of a layer (main MLA page + indexer page) is checked without
an NPU or a Mooncake deployment.
"""

import ctypes

import numpy as np
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

BLOCK = 2
BLOCKS = 4
MAIN_BYTES = 64
IDX_BYTES = 16


class FakeAddressStore:
    """Address-based store: (key, inner offset) -> bytes, no transfer engine."""

    def __init__(self):
        self.data: dict[tuple[str, int], bytes] = {}

    def put(self, key: str, gva: int, addr: int, size: int) -> None:
        self.data[(key, gva)] = ctypes.string_at(addr, size)

    def get(self, key: str, gva: int, addr: int, size: int) -> None:
        payload = self.data[(key, gva)]
        assert len(payload) == size
        ctypes.memmove(addr, payload, size)

    def entries(self) -> int:
        return len(self.data)


def make_tensor(rows: int, width: int) -> torch.Tensor:
    return torch.zeros(rows, width, dtype=torch.uint8)


def make_builder(addrs, lens, offsets, num_layers) -> LayerBatchBuilder:
    db = ChunkedTokenDatabase([object()], [BLOCK], None)
    db.set_group_buffers(
        {0: addrs},
        {0: lens},
        {0: lens},
        group_num_layers={0: num_layers},
        group_layer_cache_entry_offsets={0: offsets},
    )
    return LayerBatchBuilder(db, page_size_bytes=sum(lens), num_layers=num_layers)


def fill(tensor: torch.Tensor, base: int) -> None:
    for block in range(BLOCKS):
        tensor[block].fill_(base + block)


def expected(tensor: torch.Tensor, base: int) -> bool:
    # Only the saved block (index 2) is moved in these tests.
    return int(tensor[2][0]) == base + 2


def save_layer(store, builder, layer_id, key):
    addr, size, gva = builder._build_transfer_arrays(np.asarray([2]), np.asarray([0]), layer_id)
    for a, s, g in zip(addr.tolist(), size.tolist(), gva.tolist()):
        store.put(key, g, a, s)


def load_layer(store, builder, layer_id, key):
    addr, size, gva = builder._build_transfer_arrays(np.asarray([2]), np.asarray([0]), layer_id)
    for a, s, g in zip(addr.tolist(), size.tolist(), gva.tolist()):
        store.get(key, g, a, s)


def test_roundtrip_covers_main_and_indexer() -> None:
    """Each layer carries a main page and an indexer page; both must round-trip."""
    tensors, addrs, lens = [], [], []
    for _ in range(3):
        main, indexer = make_tensor(BLOCKS, MAIN_BYTES), make_tensor(BLOCKS, IDX_BYTES)
        tensors += [main, indexer]
        addrs += [main.data_ptr(), indexer.data_ptr()]
        lens += [MAIN_BYTES, IDX_BYTES]
    builder = make_builder(addrs, lens, [0, 2, 4, 6], 3)
    store = FakeAddressStore()

    for layer in range(3):
        fill(tensors[2 * layer], 0x10 + layer * 0x10)
        fill(tensors[2 * layer + 1], 0x80 + layer * 0x10)
        save_layer(store, builder, layer, f"layer{layer}")
    assert store.entries() == 6

    for tensor in tensors:
        tensor.zero_()
    for layer in range(3):
        load_layer(store, builder, layer, f"layer{layer}")
    for layer in range(3):
        assert expected(tensors[2 * layer], 0x10 + layer * 0x10)
        assert expected(tensors[2 * layer + 1], 0x80 + layer * 0x10)


def test_reused_slot_reloads_from_store_after_overwrite() -> None:
    """Two layers share one physical slot: loading the first must undo the overwrite."""
    main, indexer = make_tensor(BLOCKS, MAIN_BYTES), make_tensor(BLOCKS, IDX_BYTES)
    # Both physical layers map to the same slot buffers.
    builder = make_builder(
        [main.data_ptr(), indexer.data_ptr(), main.data_ptr(), indexer.data_ptr()],
        [MAIN_BYTES, IDX_BYTES, MAIN_BYTES, IDX_BYTES],
        [0, 2, 4],
        2,
    )
    store = FakeAddressStore()

    fill(main, 0x11)
    fill(indexer, 0x91)
    save_layer(store, builder, 0, "layer0")

    # Layer 1 reuses the same buffers and overwrites them.
    fill(main, 0x22)
    fill(indexer, 0xA2)
    save_layer(store, builder, 1, "layer1")

    # Loading layer 0 must restore its own bytes from the store.
    main.zero_()
    indexer.zero_()
    load_layer(store, builder, 0, "layer0")
    assert expected(main, 0x11)
    assert expected(indexer, 0x91)


def test_layer_without_indexer_transfers_main_only() -> None:
    """Non-uniform layouts still address each layer through its own offsets."""
    main0, indexer0, main1 = (
        make_tensor(BLOCKS, MAIN_BYTES),
        make_tensor(BLOCKS, IDX_BYTES),
        make_tensor(BLOCKS, MAIN_BYTES),
    )
    builder = make_builder(
        [main0.data_ptr(), indexer0.data_ptr(), main1.data_ptr()],
        [MAIN_BYTES, IDX_BYTES, MAIN_BYTES],
        [0, 2, 3],
        2,
    )
    addr, size, _ = builder._build_transfer_arrays(np.asarray([2]), np.asarray([0]), 1)
    assert addr.tolist() == [main1.data_ptr() + 2 * MAIN_BYTES]
    assert size.tolist() == [MAIN_BYTES]
