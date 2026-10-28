import re
txt=open("/opt/zig/lib/std/Thread.zig").read()
for m in re.finditer(r"pub (const|fn) (\w+)", txt):
    print(m.group(1), m.group(2))
print("---- std.zig top")
t2=open("/opt/zig/lib/std/std.zig").read().splitlines()
for l in t2:
    if "Mutex" in l or "Condition" in l or "Thread" in l: print(l.strip()[:120])
