import gguf, os
d = os.path.dirname(gguf.__file__)
print(d)
for root, dirs, files in os.walk(d):
    for f in files:
        p = os.path.join(root, f)
        try:
            txt = open(p, "r", errors="ignore").read()
        except Exception:
            continue
        if "PTQ1_0" in txt:
            print("HIT", p)
            for i, line in enumerate(txt.splitlines()):
                if "PTQ1_0" in line or "PTQ" in line:
                    print(i, line)
