from handoff.layerwise_pd.interfaces import LayerSpec
from handoff.layerwise_pd.planner import RoundRobinPlanner


def layers(n=28):
    return [LayerSpec(i, f"model.layers.{i}.attn", 4096) for i in range(n)]


def test_28_layers_to_four_slots():
    layout = RoundRobinPlanner().plan(layers(28), num_shared_buffers=3, independent_layers=[0])
    assert len(layout.slots) == 4
    assert [len(slot) for slot in layout.slots] == [1, 9, 9, 9]
    assert layout.independent_layers == (0,)
    assert set(layout.prefetch_map) == set(range(1, 28)) - {1, 2, 3}


def test_rejects_duplicate_layers():
    planner = RoundRobinPlanner()
    try:
        planner.plan([LayerSpec(0, "a", 1), LayerSpec(0, "b", 1)], num_shared_buffers=1)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate error")
