import collections
import json
import subprocess
import sys

tokens = [bytes.fromhex(l).decode("utf-8") for l in open("/workspace/tools/tokdata/tokens.hex")]
merges = [bytes.fromhex(l).decode("utf-8") for l in open("/workspace/tools/tokdata/merges.hex")]
types = [int(l) for l in open("/workspace/tools/tokdata/types.txt")]

c = collections.Counter(tokens)
dups = [(k, v) for k, v in c.items() if v > 1]
print("duplicate token strings:", len(dups), file=sys.stderr)
for k, v in dups[:10]:
    print("  ", repr(k), v, [i for i, t in enumerate(tokens) if t == k][:6], file=sys.stderr)

multispace = [m for m in merges if m.count(" ") != 1]
print("merges with != 1 space:", len(multispace), multispace[:5], file=sys.stderr)

specials = [tokens[i] for i, t in enumerate(types) if t in (2, 3, 4)]
print("n specials", len(specials), file=sys.stderr)
prefix_pairs = [(a, b) for a in specials for b in specials if a != b and b.startswith(a)]
print("prefix pairs (a prefix of b):", prefix_pairs, file=sys.stderr)

# oracle check for the overlapping specials and some weird cases
from oracle import run

texts = [
    "<tts_text_bos_single>",
    "<tts_text_bos>",
    "<tts_text_bos_singlex",
    "x<tts_text_bos_single>y",
    "x<tts_text_bos>y",
    "<tts_text_bos_single><tts_text_bos>",
    "<|im_start|><|im_start|>",
    "<|im_start|>'s",
    "'s<|im_start|>",
    "<|im_start|> <|im_start|>",
    "a<|im_end|>b",
    "<tool_call><think>",
    "<tts_text_bos_single_single>",
]
exp = run(texts)
with open("/workspace/tools/tokdata/testhex.txt", "w", encoding="utf-8") as fh:
    for s in texts:
        fh.write(s.encode().hex() + "\n")
p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
got = []
for line in p.stderr.decode().splitlines():
    if line.startswith("BATCH "):
        parts = line.split()
        n = int(parts[2])
        got.append([int(x) for x in parts[3:3 + n]])
bad = 0
for s, e, g in zip(texts, exp, got):
    tag = "OK " if e == g else "BAD"
    if e != g:
        bad += 1
    print(tag, repr(s), "exp", e, "got", g, file=sys.stderr)
print("special overlap check bad=%d" % bad, file=sys.stderr)
