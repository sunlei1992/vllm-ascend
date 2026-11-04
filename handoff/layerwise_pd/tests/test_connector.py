import torch

from handoff.layerwise_pd.codec import CPUByteCodec
from handoff.layerwise_pd.connector import LayerwiseConnector
from handoff.layerwise_pd.interfaces import LayerSpec
from handoff.layerwise_pd.planner import RoundRobinPlanner
from handoff.layerwise_pd.transport import MemoryTransport


def make_layout():
    return RoundRobinPlanner().plan(
        [LayerSpec(i, f"layer{i}", 4096) for i in range(4)],
        num_shared_buffers=1,
        independent_layers=[0],
    )


def test_shared_buffer_tensor_roundtrip():
    layout = make_layout()
    transport = MemoryTransport()
    producer_codec = CPUByteCodec(num_shards=2)
    consumer_codec = CPUByteCodec(num_shards=2)
    producer = LayerwiseConnector(layout, transport, producer_codec, num_shards=2)
    consumer = LayerwiseConnector(layout, transport, consumer_codec, num_shards=2)

    sources = {i: torch.arange(8, dtype=torch.float32) + i for i in range(4)}
    targets = {i: torch.zeros_like(tensor) for i, tensor in sources.items()}
    for layer_id, tensor in targets.items():
        consumer_codec.bind_layer(layer_id, tensor)

    producer.publish_all(sources, {i: f"key{i}" for i in range(4)})
    consumer.consume_all({i: f"key{i}" for i in range(4)})

    for layer_id in range(4):
        assert torch.equal(targets[layer_id], sources[layer_id])


def test_24_prefixes_40_requests_with_two_slots():
    layout = make_layout()
    transport = MemoryTransport()
    producer = LayerwiseConnector(layout, transport, CPUByteCodec(2), num_shards=2)
    consumer_codec = CPUByteCodec(2)
    target = torch.zeros(8, dtype=torch.float32)
    for layer_id in range(4):
        consumer_codec.bind_layer(layer_id, target)

    for prefix in range(24):
        for request in range(40):
            for layer_id in range(4):
                key = f"p{prefix}-r{request}-l{layer_id}"
                producer.publish(layer_id, key, torch.full((8,), prefix + layer_id, dtype=torch.float32))

    assert len(transport.wait_put_order) == 24 * 40 * 4
