import subprocess, sys, struct, numpy as np

cmds = sys.argv[1]
inp = open(cmds).read()
p = subprocess.run(["/workspace/ref/bin/bonsai-ref"], input=inp, text=True, capture_output=True, timeout=600)
print("STDOUT:")
print(p.stdout[:5000])
print("STDERR:")
print(p.stderr[-3000:])
