#!/bin/bash
grep -n "pub fn add\|rpath" /opt/zig/lib/std/Build/Module.zig | head -40
echo "=== Compile rpath ==="
grep -n "rpath" /opt/zig/lib/std/Build/Step/Compile.zig | head -20
