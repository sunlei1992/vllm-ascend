# NPU pseudocode notes

The authoritative stubs are in `ascend_pseudocode.py`.

## NpuEvent

```python
class NpuEvent:
    def __init__(self):
        self._event = torch.npu.Event()

    def record(self):
        self._event.record()

    def wait(self, timeout_s=None):
        return bool(self._event.wait(timeout_s))
```

## AscendLayerCodec.encode

1. Validate `tensor` is an NPU tensor.
2. Extract the main MLA bytes and indexer bytes in a deterministic order.
3. Concatenate: `main_bytes + indexer_bytes`.
4. Split into `num_shards` contiguous chunks.
5. Return `{shard: bytes}`.

The CPU reference `codec.CPUByteCodec` is the byte-level contract; the Ascend
version only changes how main/indexer views are obtained.

## AscendLayerCodec.decode

1. Split payload at the recorded main nbytes.
2. Copy main bytes into the bound main view.
3. Copy indexer bytes into the bound indexer view.
4. Validate written byte counts.

## AscendStoreTransport

Use the existing wrapper; do not implement Mooncake from scratch.

```python
object_key = f"{prefix}:{key}:{shard_id}"
rc = store.put(object_key, payload)
if rc != 0:
    raise RuntimeError(...)
```

`wait_put`/`wait_get` must respect the store's completion/lease semantics.

## Stop condition

If the real NPU layout cannot be expressed by these signatures, stop and write
the mismatch in `RESULTS.md`; do not change the contract.
