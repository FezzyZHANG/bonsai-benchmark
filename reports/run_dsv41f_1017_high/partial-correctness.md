# Partial correctness — `run_dsv41f_1017_high`

The verifier harness crashed with **SIGSEGV (exit 139) during the `greedy`
suite** at 21:53 PDT on 2026-10-27. These are the suites completed before the
crash. The missing `greedy` suite has weight 0.30 and is treated as 0 here, so
this subtotal is a **lower bound, not an acceptance score**.

| suite | score | weight | contribution |
|---|---:|---:|---:|
| tokenizer | 1.0000 | 0.10 | 0.10000 |
| dequant | 1.0000 | 0.15 | 0.15000 |
| logits | 0.9965 | 0.30 | 0.29895 |
| positions | 0.9788 | 0.15 | 0.14682 |
| greedy | — (crashed) | 0.30 | 0.00000 |
| **subtotal (4/5)** | | | **0.69577** |

The acceptance gate is 0.90. The missing greedy suite would need ≥0.681 to
reach it.

Root cause: `src/main.zig:773` declares `var scores: [8192]f32` in the
full-attention path, while the loop writes `scores[t]` for `t < p + 1`. The
long greedy case evaluates an 8192-token prompt and then 64 teacher-forced
steps; the first continuation is at `pos = 8192`, so `scores[8192]` overflows
the array. See `records/run_dsv41f_1017_high/AUDIT.md` and the verifier failure
report in this directory (`report.md`, `report.json`).

The verifier harness has since been revised to print a weighted subtotal after
each suite and write `agent_result.json` after each suite, so a future crash
will preserve the finished suites' subtotal in the report automatically.
