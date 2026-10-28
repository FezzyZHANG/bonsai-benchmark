#!/bin/sh
# Print the wall-clock time remaining in the benchmark run.
# The runner exports BENCH_DEADLINE_EPOCH (Unix seconds).
if [ -z "${BENCH_DEADLINE_EPOCH:-}" ]; then
  echo "no benchmark deadline set (BENCH_DEADLINE_EPOCH is empty)"
  exit 1
fi
now=$(date +%s)
left=$((BENCH_DEADLINE_EPOCH - now))
if [ "$left" -le 0 ]; then
  echo "deadline reached: the run is being terminated"
  exit 0
fi
printf '%dh %02dm %02ds remaining (deadline in %d s)\n' \
  $((left / 3600)) $((left % 3600 / 60)) $((left % 60)) "$left"
