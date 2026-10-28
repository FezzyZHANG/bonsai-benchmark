import re
p="/opt/zig/lib/std/fs.zig"
txt=open(p).read().splitlines()
for i,l in enumerate(txt[:200]):
    print(i,l)
