#!/bin/bash
echo "=== timespec ==="
grep -rn "pub const timespec" /opt/zig/lib/std/os/linux.zig | head
echo "=== clockid_t ==="
grep -rn "pub const clockid_t" /opt/zig/lib/std/os/linux.zig | head
echo "=== MONOTONIC ==="
grep -rn "MONOTONIC" /opt/zig/lib/std/os/linux.zig | head
echo "=== clock_gettime ctx ==="
sed -n '2425,2450p' /opt/zig/lib/std/os/linux.zig
echo "=== posix clock ==="
grep -rn "clock_gettime" /opt/zig/lib/std/posix.zig | head
