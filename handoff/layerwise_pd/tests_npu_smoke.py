"""T05: NPU codec + slot-fence smoke, without vLLM/model.

Run only after T02/T03 pass:

    python handoff/layerwise_pd/tests_npu_smoke.py

Expected: NPU_SMOKE_PASS
"""

from __future__ import annotations

import torch

from handoff.layerwise_pd.connector import LayerwiseConnector
from handoff.layerwise_pd.interfaces import LayerSpec
from handoff.layerwise_pd.planner import RoundRobinPlanner
from handoff.layerwise_pd.transport import MemoryTransport


def main() -> None:
    import torch_npu  # noqa: F401

    torch.manual_seed(0)
    layout = RoundRobinPlanner().plan(
        [LayerSpec(i, f"layer{i}", 4096) for i in range(4)],
        num_shared_buffers=1,
        independent_layers=[0],
    )
    from handoff.layerwise_pd.ascend_pseudocode import AscendLayerCodec

    transport = MemoryTransport()
    producer_codec = AscendLayerCodec(num_shards=2)
    consumer_codec = AscendLayerCodec(num_shards=2)
    producer = LayerwiseConnector(layout, transport, producer_codec, num_shards=2)
    consumer = LayerwiseConnector(layout, transport, consumer_codec, num_shards=2)

    sources = {i: (torch.arange(32, dtype=torch.float32) + i).npu() for i in range(4)}
    targets = {i: torch.zeros(32, dtype=torch.float32).npu() for i in range(4)}
    for layer_id, tensor in targets.items():
        consumer_codec.bind_layer(layer_id, tensor)

    producer.publish_all(sources, {i: f"key{i}" for i in range(4)})
    consumer.consume_all({i: f"key{i}" for i in range(4)})

    for layer_id in range(4):
        assert torch.equal(targets[layer_id].cpu(), sources[layer_id].cpu())
    print("NPU_SMOKE_PASS")


if __name__ == "__main__":
    main()
