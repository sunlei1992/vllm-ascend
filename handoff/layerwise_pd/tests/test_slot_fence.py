import pytest

from handoff.layerwise_pd.interfaces import LayerSpec
from handoff.layerwise_pd.planner import RoundRobinPlanner
from handoff.layerwise_pd.slot_manager import FencedSlotManager
from handoff.layerwise_pd.transport import MemoryTransport


def test_previous_put_is_waited_before_reuse():
    layout = RoundRobinPlanner().plan(
        [LayerSpec(i, f"l{i}", 1) for i in range(4)],
        num_shared_buffers=1,
        independent_layers=[0],
    )
    transport = MemoryTransport()
    slots = FencedSlotManager(layout, transport)

    lease0 = slots.begin(1, "key0")
    transport.put_layer("key0", 1, 0, b"a")
    slots.finish(lease0)

    slots.begin(2, "key1")  # must wait key0
    assert transport.wait_put_order == ["key0"]


def test_failed_put_does_not_advance_and_retry_works():
    layout = RoundRobinPlanner().plan(
        [LayerSpec(i, f"l{i}", 1) for i in range(3)],
        num_shared_buffers=1,
        independent_layers=[0],
    )
    transport = MemoryTransport()
    slots = FencedSlotManager(layout, transport)
    lease0 = slots.begin(1, "k0")
    transport.put_layer("k0", 1, 0, b"a")
    slots.finish(lease0)

    transport.fail_put_keys.add("k1")
    lease1 = slots.begin(2, "k1")
    with pytest.raises(RuntimeError):
        transport.put_layer("k1", 2, 0, b"b")
    assert slots.last_key(lease1.slot_id) == "k0"
