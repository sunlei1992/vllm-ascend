import sys
import time

sys.path.insert(0, "/app")
from layerwise_pd import LayerwiseConsumer, LayerwisePublisher
from mooncake_tcp_transport import connect_mooncake_tcp_session

NUM_PREFIXES = 24
NUM_LAYERS = 6
NUM_SHARDS = 2

transport = connect_mooncake_tcp_session(
    master_server_addr="mooncake-master:50051",
    num_shards=NUM_SHARDS,
    key_prefix=f"cross-{time.time_ns()}",
    global_segment_size=4 * 1024**3,
    local_buffer_size=512 * 1024**2,
)
slots = [[layer] for layer in range(NUM_LAYERS)]
publisher = LayerwisePublisher(slots=slots, transport=transport)
consumer = LayerwiseConsumer(transport=transport, num_shards=NUM_SHARDS)

for prefix in range(NUM_PREFIXES):
    for layer in range(NUM_LAYERS):
        key = f"p{prefix}-l{layer}"
        publisher.publish(
            layer,
            key,
            {0: f"p{prefix}".encode(), 1: f"l{layer}".encode()},
        )

for prefix in range(NUM_PREFIXES):
    for layer in range(NUM_LAYERS):
        key = f"p{prefix}-l{layer}"
        got = consumer.receive(layer, key)
        assert got == f"p{prefix}l{layer}".encode(), (key, got)
        transport.release(key)

print("CROSS_CONTAINER_MOONCAKE_OK", NUM_PREFIXES * NUM_LAYERS)
