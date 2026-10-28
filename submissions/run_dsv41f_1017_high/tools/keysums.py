import re,sys
names=sys.argv[2:]
entries={}
cur=None
for ln in open(sys.argv[1]).read().splitlines():
    m=re.match(r"common_debug_cb_eval:\s+(.*?) = \((f32|f16)\)\s+(.*)$", ln)
    if m:
        cur=m.group(1).strip(); entries[cur]={"op":m.group(3),"vals":[],"sum":None}
        continue
    if cur is None: continue
    s=ln.strip()
    if s.startswith("sum ="): entries[cur]["sum"]=float(s.split("=")[1])
    else:
        for num in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", s):
            entries[cur]["vals"].append(float(num))
for n in names:
    for k in entries:
        if k==n or k.startswith(n+" "):
            e=entries[k]
            print(k,"|",e["op"][:80],"| sum=",e["sum"],"| first6=",e["vals"][:6]," last6=",e["vals"][-6:])
