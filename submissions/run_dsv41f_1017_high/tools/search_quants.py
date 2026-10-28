import re
p="/opt/bonsai/gguf-py/gguf/quants.py"
txt=open(p).read()
print(len(txt))
for i,line in enumerate(txt.splitlines()):
    if "PTQ" in line or "TQ1" in line or "dequant" in line.lower() and ("def " in line):
        print(i, line)
