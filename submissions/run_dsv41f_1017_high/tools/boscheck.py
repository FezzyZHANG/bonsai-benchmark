import sys

from oracle import run

texts = ["Hi!", "abc", "<|im_start|>", " hello", "42"]
a = run(texts, add_bos=0)
b = run(texts, add_bos=1)
for t, x, y in zip(texts, a, b):
    print(("OK " if x == y else "DIFF"), repr(t), x, y, file=sys.stderr)
