import json
import random
import subprocess
import sys
import unicodedata

from oracle import run

texts = []


def add(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8", "surrogateescape")
    texts.append(s)


vec = json.load(open("/workspace/TOKENIZER-VECTORS.json"))
for v in vec:
    add(v["text"])

samples = "/workspace/data/samples/"
short = open(samples + "short.txt", encoding="utf-8").read()
corpus = open(samples + "corpus.txt", encoding="utf-8").read()
add(short)
add(short.rstrip("\n"))
add(corpus)
add(corpus[:2000])
add(corpus[2000:5000])
add(corpus[-1500:])

# curated
curated = [
    " ", "  ", "   ", "\t", "\n", "\r", "\r\n", "\n\n", "\r\n\r\n",
    "\r\n \r\n", " \r\n", "\r\n ", "  \n  ", "\n  x", "  \n x", "x  \n",
    "\t \t", "\t\t\t", " \t\n \t\n", "   trailing", "trailing   ",
    "a", "ab", "a b", " a", "  a", "a  ", "a  b", "a\tb", "a\nb", "a\r\nb",
    "'s", "'S", "'t", "'T", "'re", "'RE", "'rE", "'ve", "'VE", "'m", "'M",
    "'ll", "'LL", "'lL", "'d", "'D", "'x", "'r", "'l", "'", "''", "'''s",
    "a's", "a's's", "we're", "I'VE", "he'll", "it'd", "don't", "can't",
    "1", "12", "123", "1.5", "1,000", "x1", "1x", "9a9", "3.14e-10",
    "Hello, world!", "Hello, world! ", "a-b_c", "--", "---", "_", "__",
    "()", "()[]{}", "()[]{} ", "{}", "#!/bin/bash\n", "int main() { return 0; }",
    "a" * 100, "ab" * 100, " " * 100, "\t" * 50, "\n" * 50,
    "a" * 50 + " " + "b" * 50,
    "…", "—", "“quote”", "naïve", "café", "mañana", "Zurich",
    "é", "ñ", "à́̂", "́", "́a",
    "áb̂c", " ́a", "́ a", "1́",
    "한국어", "中文", "日本語", "русский", "עברית", "العربية", "हिन्दी",
    "🥑", "🧪", "👨‍👩‍👧‍👦", "🇺🇸", "👍🏽", "☺", "☺️",
    " ", " a", "a b", " ", " a", "a b",
    "　", "a　b", "　a", " x", "\u0085x", "\x0bx", "\x0cx",
    "​", "a​b", "﻿", "a﻿b",
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|im_start|>x",
    "x<|im_start|>", "x<|im_start|>y", "<|im_end|><|im_start|>",
    "a<|im_end|>b<|im_start|>c", "<|im_start|>user<|im_end|>",
    "<|im_end|>user\n<|im_start|>assistant",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "x<tool_call>y", "<|file_sep|>", "<|fim_prefix|>",
    "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>",
    "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    "<|im_end", "<|im_end|", "<|im_star|>", "<|im_start|", "|im_start|>",
    "[PAD248077]", "[PAD248077]a", "a[PAD248077]b", "[PAD248044]",
    "<|repo_name|>", "<|object_ref_start|>", "<|box_start|>", "<|quad_end|>",
    "\x00", "a\x00b", "\x00\"'s", "\x7f", "a\x7fb",
    " \t\r\n x \t\r\n", "x \t\r\n y", "\r\n\r\n\r\n", " \r \n \r ",
    "e.g. i.e. U.S.A.", "3.5%", "$100", "C++", "a->b", "a::b",
    "  \t  x  \t  ", "\n\t\n\t", " \n\n ", "\r\n\n\r", "\t\r\t\n\t",
]
for s in curated:
    add(s)

random.seed(20251028)

# character pools by broad script/category
pools = {
    "ascii_letters": [chr(c) for c in range(0x41, 0x5B)] + [chr(c) for c in range(0x61, 0x7B)],
    "digits": list("0123456789"),
    "punct": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
    "combining": [chr(c) for c in range(0x300, 0x36F)] + ["⃐", "⃑", "҃", "҄"],
    "latin1": [chr(c) for c in range(0xA0, 0x100)],
    "greek": [chr(c) for c in range(0x386, 0x3D0)],
    "cyrillic": [chr(c) for c in range(0x400, 0x460)],
    "hebrew": [chr(c) for c in range(0x5D0, 0x5EB)],
    "arabic": [chr(c) for c in range(0x620, 0x650)],
    "devanagari": [chr(c) for c in range(0x900, 0x960)],
    "cjk": [chr(c) for c in range(0x4E00, 0x4E80)],
    "hangul": [chr(c) for c in range(0xAC00, 0xAC40)],
    "hiragana": [chr(c) for c in range(0x3041, 0x3097)],
    "symbols": ["€", "£", "©", "®", "°", "±", "×", "÷", "∞", "≠", "≤", "→", "★", "♥"],
    "emoji": ["🥑", "🧪", "😀", "😃", "🔥", "👍", "🎉", "🚀", "🧠", "💡"],
    "whitespace": [" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x85", " ", " ",
                   " ", " ", " ", " ", " ", " ", " ",
                   " ", " ", " ", " ", " ", " ", " ",
                   " ", "　"],
    "numbers": ["0", "1", "9", "½", "Ⅳ", "①", "²", "٣", "१", "一", "〇"],
    "letters_uni": ["é", "ñ", "ü", "ß", "æ", "ø", "Ω", "π", "Я", "中", "あ", "한", "א", "ع", "अ"],
    "marks_uni": ["́", "̃", "̈", "ً", "़", "ั", "゙"],
    "specials_misc": ["<", ">", "|", "<|", "|>", "<|im_start|>", "<|im_end|>", "<tool_call>",
                      "[PAD248077]", "<|endoftext|>"],
}

pool_names = list(pools.keys())

for _ in range(1200):
    kind = random.randrange(6)
    n = random.randint(1, 25)
    if kind == 0:
        names = random.sample(pool_names, random.randint(1, 4))
        chars = [c for nm in names for c in pools[nm]]
        s = "".join(random.choice(chars) for _ in range(n))
    elif kind == 1:
        names = random.sample(pool_names, random.randint(1, 3))
        s = ""
        for _ in range(n):
            nm = random.choice(names)
            s += random.choice(pools[nm])
    elif kind == 2:
        # whitespace heavy
        s = "".join(random.choice(pools["whitespace"]) for _ in range(random.randint(1, 10)))
        if random.random() < 0.5:
            s += random.choice(pools["ascii_letters"]) + "".join(random.choice(pools["whitespace"]) for _ in range(random.randint(0, 4)))
    elif kind == 3:
        # word-ish with apostrophes and digits
        base = random.choice(["don", "can", "I", "we", "he", "she", "it", "they", "a", "x"])
        tail = random.choice(["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'RE", "'LL", "'", "s", "ll"])
        s = base + tail
        if random.random() < 0.5:
            s = random.choice([" ", "  ", "\t", "\n", ""]) + s + random.choice(["", " ", ".", ",", "!", "123"])
    elif kind == 4:
        # punctuation/number runs
        chars = pools["punct"] + pools["digits"] + [" ", "\t", "\n", "\r"]
        s = "".join(random.choice(chars) for _ in range(random.randint(1, 20)))
    else:
        # random unicode scalar values from interesting ranges
        ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF),
                  (0x400, 0x4FF), (0x590, 0x5FF), (0x600, 0x6FF), (0x900, 0x97F),
                  (0xE00, 0xE7F), (0x1100, 0x11FF), (0x1E00, 0x1EFF), (0x2000, 0x206F),
                  (0x20A0, 0x20BF), (0x2100, 0x214F), (0x3000, 0x303F), (0x4E00, 0x4EFF),
                  (0xAC00, 0xAD00), (0x1F300, 0x1F5FF), (0x1F600, 0x1F64F), (0x1F900, 0x1F9FF)]
        s = ""
        for _ in range(n):
            lo, hi = random.choice(ranges)
            cp = random.randint(lo, hi)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x41
            s += chr(cp)
    add(s)

# single-codepoint probes over sampled code points
cps = set()
for _ in range(600):
    cps.add(random.randint(0x20, 0x10FFFF))
for lo, hi in [(0x300, 0x370), (0x2000, 0x2070), (0xA0, 0x100), (0x1F300, 0x1FAFF), (0xE00, 0xE80), (0x900, 0x980)]:
    for c in range(lo, hi, 7):
        cps.add(c)
for cp in sorted(cps):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    add(chr(cp))
    add("a" + chr(cp) + "b")
    add(" " + chr(cp) + " ")

print("n texts", len(texts), file=sys.stderr)

# write hex list for the Zig harness
with open("/workspace/tools/tokdata/testhex.txt", "w", encoding="utf-8") as fh:
    for s in texts:
        fh.write(s.encode("utf-8", "surrogatepass").hex())
        fh.write("\n")

exp = run(texts)
with open("/workspace/tools/tokdata/oracle.json", "w") as fh:
    json.dump(exp, fh)

p = subprocess.run(["/workspace/tools/toktest"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
lines = p.stderr.decode("utf-8", "replace").splitlines()
got = []
for line in lines:
    if line.startswith("BATCH "):
        parts = line.split()
        n = int(parts[2])
        got.append([int(x) for x in parts[3:3 + n]])

print("batch results", len(got), "oracle", len(exp), "texts", len(texts), file=sys.stderr)
assert len(got) == len(exp), (len(got), len(exp))

bad = 0
shown = 0
for s, e, g in zip(texts, exp, got):
    if e is None:
        print("oracle error for", repr(s)[:100], file=sys.stderr)
        continue
    if e != g:
        bad += 1
        if shown < 25:
            shown += 1
            print("MISMATCH text=%r" % s[:200], file=sys.stderr)
            print("  exp=%r" % (e[:60],), file=sys.stderr)
            print("  got=%r" % (g[:60],), file=sys.stderr)
print("TOTAL %d mismatch %d" % (len(texts), bad), file=sys.stderr)
for line in lines:
    if line.startswith("VECTORS") or line.startswith("CORPUS"):
        print(line, file=sys.stderr)
