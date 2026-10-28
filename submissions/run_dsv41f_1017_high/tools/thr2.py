import subprocess, glob, re
hits=[]
for f in glob.glob("/opt/zig/lib/std/**/*.zig", recursive=True):
    try: txt=open(f).read()
    except: continue
    for m in re.finditer(r"pub const (Mutex|Condition|Futex|Semaphore)\b", txt):
        hits.append((f,m.group(1)))
for h in sorted(set(hits)): print(h)
