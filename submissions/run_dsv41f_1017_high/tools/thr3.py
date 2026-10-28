import re
lines=open("/opt/zig/lib/std/Io.zig").read().splitlines()
# find Mutex and Condition definitions
for i,l in enumerate(lines):
    if re.match(r"pub const (Mutex|Condition) = ", l):
        print("=== line",i,l)
        for j in range(i, min(i+60,len(lines))):
            print(j, lines[j][:140])
