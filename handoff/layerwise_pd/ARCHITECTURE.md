# Layerwise shared-buffer handoff architecture

## 1. Goal

Make the layerwise shared-buffer feature executable by a weak intranet agent
without asking it to make design decisions.

Boundary:

- CPU reference: complete and testable here.
- NPU implementation: only the bodies in `ascend_pseudocode.py` are filled
  by the intranet agent.
- The agent must not change interfaces, class relationships, data structures
  or lifecycle semantics.

## 2. Module map

```text
interfaces.py       stable Protocol/dataclass contracts
planner.py          pure layer -> slot grouping
events.py           CPU ThreadingEvent / NoopEvent
transport.py        in-memory transport for CPU UT
slot_manager.py     slot ownership + previous-put fence
codec.py            CPU tensor <-> TP-shard bytes
connector.py        planner + slot manager + transport + codec composition
ascend_pseudocode.py NPU-only bodies to implement
tests/               CPU contract tests
```

`connector.py` has no vLLM import. The production vllm-ascend connector is
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_pd_connector.py`;
the classes here are the stable contract it should conform to.

## 3. Class relationship

```text
RoundRobinPlanner ──plan──> ReuseLayout
                               │
                               ├──> FencedSlotManager ──> LayerTransport
                               │
LayerCodec ────────────────────┴──> LayerwiseConnector
DeviceEvent (NPU) ────────────────> FencedSlotManager / connector timing
```

## 4. Data structures

### LayerSpec

```python
LayerSpec(layer_id: int, name: str, page_size_bytes: int)
```

Used only by the planner. Shared layers must have the same page size.

### ReuseLayout

```python
ReuseLayout(
    slots: tuple[tuple[int, ...], ...],
    independent_layers: tuple[int, ...],
    prefetch_map: Mapping[int, int],
)
```

Example for 28 layers, 3 shared buffers, layer 0 independent:

```text
slots = [(0,), (1, 10, 19, 28?...), ...]
```

`prefetch_map[current] = previous` means "current may reuse the slot only
after previous has been published".

### SlotLease

```python
SlotLease(slot_id, layer_id, key)
```

The lease is valid only between `begin()` and `finish()`.

## 5. Invariants

1. One layer appears in exactly one slot.
2. A slot is not reused until the previous owner's `wait_put(previous_key)`
   has completed.
3. `finish()` runs only after all shard puts for the current key.
4. A failed put must not advance `_slot_key` or `_committed`.
5. `decode()` runs only after all shards for the key are available.
6. Protocol signatures in `interfaces.py` are frozen.

## 6. P/D mapping

| Role | Operation | Contract |
| :-- | :-- | :-- |
| P | layer compute done | `save_kv_layer` → codec.encode → transport.put → slot.finish |
| D | before layer compute | transport.get → codec.decode → wait_for_layer_load returns |
| both | slot reuse | `FencedSlotManager.begin` waits previous put |

## 7. Intranet implementation boundary

The agent may edit only:

```text
ascend_pseudocode.py
```

If it must edit any other file, stop and write the reason to `RESULTS.md`.

## 8. Non-goals

- No GLM/Qwen3 full-model performance here.
- No Mooncake replacement in the CPU package.
- No change to vLLM core allocator in this handoff package.
