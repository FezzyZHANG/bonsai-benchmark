import re, sys, json
path = sys.argv[1] if len(sys.argv)>1 else "/workspace/tools/ldfull.txt"
text=open(path).read().splitlines()
entries=[]
cur=None
for ln in text:
    m=re.match(r"common_debug_cb_eval:\s+(.*?) = \((f32|f16)\)\s+(.*)$", ln)
    if m:
        if cur: entries.append(cur)
        cur={"name":m.group(1).strip(),"op":m.group(3),"vals":[],"sum":None}
        continue
    if cur is None: continue
    s=ln.strip()
    if s.startswith("sum ="):
        cur["sum"]=float(s.split("=")[1])
    else:
        for num in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", s):
            cur["vals"].append(float(num))
if cur: entries.append(cur)
for e in entries:
    if "--all" in sys.argv or re.search(sys.argv[2] if len(sys.argv)>2 else ".*", e["name"]):
        print(e["name"], "|", e["op"][:90], "| sum=", e["sum"], "| nums=", e["vals"][:24])
