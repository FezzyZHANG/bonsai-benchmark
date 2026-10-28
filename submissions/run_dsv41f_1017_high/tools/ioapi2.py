import re
for f in ["/opt/zig/lib/std/Io.zig","/opt/zig/lib/std/Io/Dir.zig","/opt/zig/lib/std/Io/File.zig","/opt/zig/lib/std/Io/Threaded.zig"]:
    try: lines=open(f).read().splitlines()
    except Exception as e:
        print("ERR",f,e); continue
    print("==",f)
    for i,l in enumerate(lines):
        if re.search(r"(pub fn (cwd|init|deinit|io)|pub const (Dir|File|Io|Threaded)\b|openFileAbsolute|readFileAlloc)", l):
            print(i,l[:160])
