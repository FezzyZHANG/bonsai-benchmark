#!/bin/bash
ls -la /opt/zig/lib/std/ | head -60
echo "=== process.zig size ==="
wc -l /opt/zig/lib/std/process.zig
grep -n "pub " /opt/zig/lib/std/process.zig | head -40
echo "=== Build.zig exists? ==="
ls -la /opt/zig/lib/std/Build.zig
echo "=== getenv anywhere ==="
grep -rln "getenv" /opt/zig/lib/std/ | head -20
