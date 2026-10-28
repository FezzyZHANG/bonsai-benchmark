import itertools
import random
import subprocess
import sys

from oracle import run

alpha = ["a", "1", " ", "\t", "\n", "\r", "!", "'", "s", "é", "́", " ", "🥑", "<", "|", ">"]
texts = []
for n in (1, 2, 3):
    for combo in itertools.product(alpha, repeat=n):
        texts.append("".join(combo))

random.seed(99)
words = ["", "a", "hi", " ", "x y", "\n", "!"]
specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</think>",
            "<|fim_prefix|>", "<tts_text_bos_single>", "<|image_pad|>"]
for _ in range(400):
    s = random.choice(words)
    for _ in range(random.randint(1, 3)):
        s += random.choice(specials) + random.choice(words)
    texts.append(s)

print("n texts", len(texts), file=sys.stderr)
with open("/workspace/tools/tokdata/testhex.txt", "w", encoding="utf-8") as fh:
    for s in texts:
        fh.write(s.encode().hex() + "\n")

exp = run(texts)
p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
got = []
for line in p.stderr.decode().splitlines():
    if line.startswith("BATCH "):
        parts = line.split()
        n = int(parts[2])
        got.append([int(x) for x in parts[3:3 + n]])
print("batch", len(got), "oracle", len(exp), file=sys.stderr)
assert len(got) == len(exp)
bad = 0
for s, e, g in zip(texts, exp, got):
    if e != g:
        bad += 1
        if bad <= 15:
            print("MISMATCH %r exp=%r got=%r" % (s, e, g), file=sys.stderr)
print("EXHAUSTIVE total=%d mismatch=%d" % (len(texts), bad), file=sys.stderr)
