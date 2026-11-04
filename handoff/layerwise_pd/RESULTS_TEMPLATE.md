# Results

| Task | Status | Exit code | Evidence |
| :-- | :-- | :-- | :-- |
| T00 | PASS/FAIL | 0/1 | `6 passed` |
| T01 | PASS/FAIL | 0/1 | npu_count=... |
| T02 | PASS/FAIL | 0/1 | `T02_PASS` |
| T03 | PASS/FAIL | 0/1 | `T03_PASS` |
| T04 | PASS/FAIL/SKIP | 0/1 | no store |
| T05 | PASS/FAIL | 0/1 | `T05_PASS tensors=2` |
| T06 | PASS/FAIL | 0/1 | `T06_PASS tensors=4 layers=28` |
| T07 | PASS/FAIL | 0/1 | token equality |
| T08a | PASS/FAIL | 0/1 | 24x2 |
| T08b | PASS/FAIL | 0/1 | 24x40 |
| T09 | PASS/FAIL | 0/1 | metrics |

Do not paste large logs. Keep failing logs under `/tmp/<task>.log`.
