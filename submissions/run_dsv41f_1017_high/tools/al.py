import re
lines=open("/opt/zig/lib/std/array_list.zig").read().splitlines()
for i,l in enumerate(lines):
    if re.search(r"pub fn (append|ensureTotalCapacity|toOwnedSlice|appendAssumeCapacity|deinit|clearRetainingCapacity)\b", l):
        print(i,l.strip()[:120])
