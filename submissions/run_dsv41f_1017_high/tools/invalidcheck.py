import subprocess
import sys

REF = "/workspace/ref/bin/bonsai-ref"
MODEL = "/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"

cases = [
    b"\xff",
    b"\xc3",
    b"\xc3\x28",
    b"\x80\x80",
    b"a\xffb",
    b"\xed\xa0\x80",
    b"\xf0\x9f\x92",
    b"\x00\x01\x02",
    b"\xf5\x80\x80\x80",
    b"hello\xffworld",
    b"\xff \xff",
    b"\xffa",
    b"a\xff",
    b"\xc3\xa9\xff\xc3\xa9",
]

cmds = ["load " + MODEL]
for c in cases:
    cmds.append("tokenize " + c.hex())
cmds.append("quit")
p = subprocess.run([REF], input=("\n".join(cmds) + "\n").encode(), stdout=subprocess.PIPE)
lines = p.stdout.decode().splitlines()
assert lines[0] == "ok", lines[0]
exp = []
for line in lines[1:]:
    parts = line.split()
    if parts[0] == "ok":
        n = int(parts[1])
        exp.append([int(x) for x in parts[2:2 + n]])
    else:
        exp.append(None)

with open("/workspace/tools/tokdata/testhex.txt", "w") as fh:
    for c in cases:
        fh.write(c.hex() + "\n")

p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
got = []
for line in p.stderr.decode().splitlines():
    if line.startswith("BATCH "):
        parts = line.split()
        n = int(parts[2])
        got.append([int(x) for x in parts[3:3 + n]])

bad = 0
for c, e, g in zip(cases, exp, got):
    same = e == g
    if not same:
        bad += 1
    print(("OK " if same else "BAD"), c.hex(), "exp", e, "got", g, file=sys.stderr)
print("invalid utf8 bad=%d" % bad, file=sys.stderr)
