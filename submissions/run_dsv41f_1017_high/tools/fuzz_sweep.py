import subprocess
import sys

from oracle import run

CHUNK = 1000
texts = []
labels = []

cp = 0x20
while cp <= 0x10FFFF:
    chars = []
    start = cp
    while cp <= 0x10FFFF and len(chars) < CHUNK:
        if not (0xD800 <= cp <= 0xDFFF):
            chars.append(chr(cp))
        cp += 1
    if not chars:
        continue
    texts.append("x".join(chars))
    labels.append((start, cp))

print("chunks", len(texts), file=sys.stderr)

with open("/workspace/tools/tokdata/testhex.txt", "w", encoding="utf-8") as fh:
    for s in texts:
        fh.write(s.encode("utf-8").hex())
        fh.write("\n")

exp = run(texts)
p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
lines = p.stderr.decode("utf-8", "replace").splitlines()
got = []
for line in lines:
    if line.startswith("BATCH "):
        parts = line.split()
        n = int(parts[2])
        got.append([int(x) for x in parts[3:3 + n]])

print("batch", len(got), "oracle", len(exp), file=sys.stderr)
assert len(got) == len(exp)

bad = 0
for (start, end), e, g, s in zip(labels, exp, got, texts):
    if e != g:
        bad += 1
        if bad <= 10:
            print("MISMATCH chunk 0x%X-0x%X" % (start, end), file=sys.stderr)
            # first differing id
            for i, (a, b) in enumerate(zip(e, g)):
                if a != b:
                    print("  first diff at id %d: exp %d got %d (ids %r vs %r)"
                          % (i, a, b, e[max(0, i - 4):i + 8], g[max(0, i - 4):i + 8]), file=sys.stderr)
                    break
            print("  len exp %d got %d" % (len(e), len(g)), file=sys.stderr)
            print("  text head %r" % s[:80], file=sys.stderr)
print("SWEEP chunks=%d mismatch=%d" % (len(texts), bad), file=sys.stderr)
