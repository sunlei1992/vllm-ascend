"""Unit tests for the pull transport backend selection."""

import sys
import types
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.utils import pull_engine


def make_config(extra):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra)
    )


def test_default_backend_is_memfabric():
    assert pull_engine.resolve_transfer_backend(make_config({})) == pull_engine.BACKEND_MEMFABRIC


def test_mooncake_backend_selected():
    config = make_config({"transfer_backend": "mooncake"})
    assert pull_engine.resolve_transfer_backend(config) == pull_engine.BACKEND_MOONCAKE


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported transfer_backend"):
        pull_engine.resolve_transfer_backend(make_config({"transfer_backend": "nixl"}))


def test_mooncake_options_defaults():
    opts = pull_engine.resolve_mooncake_options(make_config({}))
    assert opts == {"metadata": "P2PHANDSHAKE", "protocol": "ascend", "device": ""}


def test_mooncake_engine_contract(monkeypatch):
    calls = {}

    class FakeTE:
        def initialize(self, host, metadata, protocol, device):
            calls["init"] = (host, metadata, protocol, device)
            return 0

        def batch_register_memory(self, ptrs, sizes):
            calls["register"] = (list(ptrs), list(sizes))
            return 0

        def batch_transfer_sync_read(self, peer, local, peers, lengths):
            calls["read"] = (peer, list(local), list(peers), list(lengths))
            return 0

    fake_engine = types.ModuleType("mooncake.engine")
    fake_engine.TransferEngine = FakeTE
    fake_pkg = types.ModuleType("mooncake")
    fake_pkg.engine = fake_engine
    monkeypatch.setitem(sys.modules, "mooncake", fake_pkg)
    monkeypatch.setitem(sys.modules, "mooncake.engine", fake_engine)

    engine = pull_engine.MooncakePullEngine(
        "pull-p",
        {"metadata": "http://meta:8080/metadata", "protocol": "tcp", "device": ""},
    )
    assert engine.unique_id == "pull-p"
    assert engine.register_memory([11, 12], [21, 22]) == 0
    assert engine.batch_transfer_sync_read("pull-p", [1], [2], [3]) == 0
    assert calls["init"] == ("pull-p", "http://meta:8080/metadata", "tcp", "")
    assert calls["register"] == ([11, 12], [21, 22])
    assert calls["read"] == ("pull-p", [1], [2], [3])


def test_make_pull_engine_dispatch(monkeypatch):
    seen = []

    def fake_memfabric(*args, **kwargs):
        seen.append("memfabric")
        return "memfabric-engine"

    def fake_mooncake(*args, **kwargs):
        seen.append("mooncake")
        return "mooncake-engine"

    monkeypatch.setattr(pull_engine, "MemfabricPullEngine", fake_memfabric)
    monkeypatch.setattr(pull_engine, "MooncakePullEngine", fake_mooncake)

    mem_cfg = make_config({})
    assert pull_engine.make_pull_engine(
        mem_cfg, role="Decode", hostname="h", device_id=0
    ) == "memfabric-engine"

    mc_cfg = make_config({"transfer_backend": "mooncake"})
    assert pull_engine.make_pull_engine(
        mc_cfg, role="Decode", hostname="h", device_id=0
    ) == "mooncake-engine"
    assert seen == ["memfabric", "mooncake"]
