#!/bin/bash
echo "=== clock_gettime in std ==="
grep -rn "pub fn clock_gettime" /opt/zig/lib/std/ | head
echo "=== nanosleep ==="
grep -rn "pub fn nanosleep" /opt/zig/lib/std/ | head
echo "=== CLOCK const ==="
grep -rn "pub const CLOCK" /opt/zig/lib/std/ | head
echo "=== std.debug.print ==="
grep -rn "pub fn print" /opt/zig/lib/std/debug.zig | head
echo "=== page_allocator ==="
grep -rn "pub const page_allocator" /opt/zig/lib/std/heap.zig | head
