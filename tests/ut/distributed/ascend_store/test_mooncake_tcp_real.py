"""Opt-in real Mooncake TCP round-trip.

Run in a Linux image with the Mooncake wheel and a running ``mooncake_master``:

    MOONCAKE_MASTER=127.0.0.1:50051 \
      pytest -q tests/ut/distributed/ascend_store/test_mooncake_tcp_real.py

The test is skipped by default; it validates the transport against a real
Store without requiring an NPU.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("mooncake", reason="Mooncake wheel is not installed")

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_pd import (  # noqa: E402
    LayerwiseConsumer,
    LayerwisePublisher,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mooncake_tcp_transport import (  # noqa: E402
    connect_mooncake_tcp,
)


@pytest.mark.skipif(
    not os.getenv("MOONCAKE_MASTER"),
    reason="set MOONCAKE_MASTER to run the real Mooncake TCP test",
)
def test_real_tcp_layer_roundtrip() -> None:
    key = f"layerwise-pd-{os.getpid()}-{time.time_ns()}"
    transport = connect_mooncake_tcp(
        master_server_addr=os.environ["MOONCAKE_MASTER"],
        key_prefix="ut",
        global_segment_size=4 * 1024**3,
        local_buffer_size=512 * 1024**2,
    )
    publisher = LayerwisePublisher(slots=((0,),), transport=transport)
    payload0 = b"rank0" * 1024
    payload1 = b"rank1" * 1024
    try:
        publisher.publish(0, key, {0: payload0, 1: payload1})
        consumer = LayerwiseConsumer(transport=transport, num_shards=2)
        assert consumer.receive(0, key) == payload0 + payload1
    finally:
        transport.release(key)
