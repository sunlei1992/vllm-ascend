# Intranet tasks

Rules:

- Run tasks in order.
- Every command must have exit code 0; otherwise stop and record `FAIL`.
- Do not modify `interfaces.py`, `planner.py`, `slot_manager.py`,
  `connector.py`, `codec.py`, `transport.py`, or the tests.
- Only fill the bodies in `ascend_pseudocode.py`.
- If an interface cannot express the NPU reality, stop and record the exact
  signature mismatch in `RESULTS.md`; do not redesign.

Set variables once:

```bash
cd <vllm-ascend-repo>
export HANDOFF="$PWD/handoff/layerwise_pd"
export PYTHONPATH="$PWD"
```

## T00 — CPU contract

Command:

```bash
python -m pytest -q handoff/layerwise_pd/tests
```

Expected:

```text
6 passed
```

Stop if any test fails.

## T01 — NPU environment probe

Command:

```bash
python - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("npu_count", torch.npu.device_count())
PY
npu-smi info | head -20
```

Expected: `npu_count >= 1`.

Stop and record if import or device count fails.

## T02 — NpuEvent

File: `handoff/layerwise_pd/ascend_pseudocode.py`, class `NpuEvent`.

Implement `record()` and `wait()` with `torch.npu.Event`.

Command:

```bash
python - <<'PY'
import torch, torch_npu
from handoff.layerwise_pd.ascend_pseudocode import NpuEvent
e = NpuEvent()
e.record()
assert e.wait(5)
print("T02_PASS")
PY
```

Expected: `T02_PASS`.

## T03 — AscendLayerCodec

File: `ascend_pseudocode.py`, class `AscendLayerCodec`.

Implement `bind_layer`, `encode`, `decode` for one NPU tensor; keep the same
logical payload as `codec.CPUByteCodec`.

Command:

```bash
python - <<'PY'
import torch, torch_npu
from handoff.layerwise_pd.ascend_pseudocode import AscendLayerCodec
codec = AscendLayerCodec(num_shards=2)
src = torch.arange(32, dtype=torch.float32).npu()
dst = torch.zeros_like(src)
codec.bind_layer(0, dst)
shards = codec.encode(0, src, "k")
assert sorted(shards) == [0, 1]
codec.decode(0, b"".join(shards[i] for i in sorted(shards)))
assert torch.equal(dst.cpu(), src.cpu())
print("T03_PASS")
PY
```

Expected: `T03_PASS`.

## T04 — AscendStoreTransport

File: `ascend_pseudocode.py`, class `AscendStoreTransport`.

Implement using the existing vllm-ascend store wrapper. Do not add a new
transport backend.

Command: follow the store's own smoke test; expected `T04_PASS`.

If no store is available, record `SKIP_NO_STORE` and continue.

## T05 — 4 layers / 2 slots

After T02/T03 pass, run the exact NPU smoke:

```bash
python handoff/layerwise_pd/tests_npu_smoke.py
```

Expected:

```text
NPU_SMOKE_PASS
```

This tests the NPU codec + slot fence without vLLM. If it fails, do not edit
the test; record the traceback and stop.

## T06 — 28 layers / 4 slots

Run the layout test with 28 layers and 3 shared buffers. Expected:

```text
T06_PASS tensors=4 layers=28
```

## T07 — decode compare

Run 4 layers / 2 slots with `max_tokens=4`; compare against the baseline
without shared buffer. Expected identical token list.

Expected:

```text
T07_PASS tokens=<same-as-baseline>
```

## T08 — 24 prefixes

Run 24 prefixes × 2 requests first, then × 40 only if T08a passes.

Expected:

```text
T08a_PASS prefixes=24 requests=2
T08b_PASS prefixes=24 requests=40
```

## T09 — metrics

Record only:

```text
KV blocks / HBM / TTFT / throughput
```

Do not claim performance from a single run.

Expected:

```text
T09_PASS
```
