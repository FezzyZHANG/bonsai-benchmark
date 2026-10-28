#!/bin/bash
sed -n '9790,9805p' /opt/zig/lib/std/os/linux.zig
grep -n "use_kernel_timespec" /opt/zig/lib/std/os/linux.zig | head
grep -n "pub const kernel_timespec" /opt/zig/lib/std/os/linux.zig
