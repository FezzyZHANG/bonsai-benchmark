#!/bin/bash
echo "=== process getenv ==="
grep -n "pub fn .*[Ee]nv\|pub const EnvMap" /opt/zig/lib/std/process.zig | head -30
echo "=== Build env_map ==="
grep -n "env_map" /opt/zig/lib/std/Build.zig | head -10
echo "=== posix getenv ==="
grep -n "pub fn getenv" /opt/zig/lib/std/posix.zig
