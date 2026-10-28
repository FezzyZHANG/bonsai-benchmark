#!/bin/bash
echo "=== Build.zig env refs ==="
grep -n "env" /opt/zig/lib/std/Build.zig | head -40
echo "=== Build dir ==="
ls /opt/zig/lib/std/Build/
echo "=== Graph env ==="
grep -rn "env_map\|environ" /opt/zig/lib/std/Build/ | head -30
echo "=== Environ ==="
grep -n "pub fn" /opt/zig/lib/std/process/Environ.zig | head -20
