# Layerwise shared-buffer handoff

This directory is the stable CPU reference and the intranet task package for
the layerwise shared-buffer work.

Read in this order:

1. `ARCHITECTURE.md` — class relationships, data structures, invariants.
2. `TASKS.md` — exact commands and expected output.
3. `PSEUDOCODE.md` — NPU implementation notes.
4. `ascend_pseudocode.py` — only file the intranet agent may edit.
5. `RESULTS_TEMPLATE.md` — PASS/FAIL record.

CPU check:

```bash
PYTHONPATH=. python -m pytest -q handoff/layerwise_pd/tests
```

Expected: `6 passed`.
