# Test diagnostics

The `tools/` package contains helpers for reporting elapsed time and available machine/GPU details in live integration tests.

| File | Responsibility |
| --- | --- |
| `stats.py` | Wraps selected live tests with timing and benchmark summaries. |

These helpers report diagnostics; they do not determine whether application or model behavior is correct. Offline tests should not require a GPU or hardware-monitoring libraries. Keep timing output out of correctness assertions unless a test explicitly defines a stable performance budget.
