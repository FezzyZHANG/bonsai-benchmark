#!/bin/bash
echo "=== std/time.zig ==="
grep -n "pub const\|pub fn" /opt/zig/lib/std/time.zig
echo "=== time dir ==="
ls /opt/zig/lib/std/time/
echo "=== Instant ==="
grep -rn "pub const Instant\|pub fn now\|pub fn read" /opt/zig/lib/std/time/ | head -40
