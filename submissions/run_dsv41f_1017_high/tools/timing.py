import subprocess
import sys
import time

corpus = open("/workspace/data/samples/corpus.txt", "r", encoding="utf-8").read()

# baseline: no batch work (just vocab load + init + vectors + corpus once)
open("/workspace/tools/tokdata/testhex.txt", "w").close()
t0 = time.perf_counter()
p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
base = time.perf_counter() - t0
lines = p.stderr.decode().splitlines()
print("baseline_s %.3f" % base, file=sys.stderr)
for ln in lines[:4]:
    print("  " + ln, file=sys.stderr)

# workload: corpus x10 via batch, plus the built-in single corpus pass
n = 10
with open("/workspace/tools/tokdata/testhex.txt", "w", encoding="utf-8") as fh:
    for _ in range(n):
        fh.write(corpus.encode("utf-8").hex())
        fh.write("\n")
t0 = time.perf_counter()
p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
work = time.perf_counter() - t0
print("workload_s %.3f for %d x %d bytes" % (work, n, len(corpus.encode())), file=sys.stderr)
elapsed = work - base
print("batch_s %.3f  bytes_per_s %.0f" % (elapsed, n * len(corpus.encode()) / max(elapsed, 1e-9)), file=sys.stderr)
for ln in p.stderr.decode().splitlines():
    if ln.startswith("BATCH") or ln.startswith("CORPUS") or ln.startswith("EMPTY") or ln.startswith("BOS"):
        print("  " + ln[:200], file=sys.stderr)
