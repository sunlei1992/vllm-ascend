"""Regression tests for the layerwise step-end save fence (Plan B)."""

import threading

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)


class FakeQueue:
    def __init__(self):
        self.unfinished_tasks = 0
        self.all_tasks_done = threading.Condition()


class FakeSendThread:
    def __init__(self):
        self.request_queue = FakeQueue()
        self.failed = False

    def raise_if_failed(self):
        if self.failed:
            raise RuntimeError("send failed")


def make_worker(events, prefetch):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.layer_save_finished_events = events
    worker.prefetch_layer_map = prefetch
    worker.block_key_hybrid = False
    worker.use_block_key_layerwise = True
    return worker


def run_with_timeout(fn, timeout=2.0):
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "step-end save fence blocked"


def test_fence_returns_after_reuse_consumer_cleared_source_event():
    events = [threading.Event() for _ in range(3)]
    worker = make_worker(events, {1: 0})
    # The reuse load consumed layer 0's one-shot save event.
    events[0].set()
    events[0].clear()

    run_with_timeout(lambda: worker._wait_for_final_layer_save(3, FakeSendThread()))


def test_fence_clears_only_non_source_events():
    events = [threading.Event() for _ in range(3)]
    worker = make_worker(events, {1: 0})
    events[2].set()

    worker._wait_for_final_layer_save(3, FakeSendThread())

    assert not events[2].is_set()
    # Source-layer events are left to the reuse consumer.
    assert not events[0].is_set()
