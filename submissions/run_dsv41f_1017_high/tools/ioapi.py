import re
p="/opt/zig/lib/std/Io.zig"
txt=open(p).read().splitlines()
for i,l in enumerate(txt):
    if re.search(r"pub fn (cwd|openFile|openFileAbsolute|readFileAlloc|readToEndAlloc|stat)\b", l) or re.match(r"pub const (Dir|File|Io)\b", l):
        print(i,l)
print("---- Dir/File files:")
for f in ["/opt/zig/lib/std/Io/Dir.zig","/opt/zig/lib/std/Io/File.zig"]:
    try: lines=open(f).read().splitlines()
    except Exception as e: print(f,e); continue
    print("==",f)
    for i,l in enumerate(lines):
        if re.search(r"pub fn (openFile|readFileAlloc|readToEndAlloc|stat|close|createFile)\b", l):
            print(i,l)
