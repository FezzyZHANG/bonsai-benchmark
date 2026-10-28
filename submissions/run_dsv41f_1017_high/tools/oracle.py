import json
import subprocess
import sys

REF = "/workspace/ref/bin/bonsai-ref"
MODEL = "/workspace/data/Ternary-Bonsai-2-4B-PTQ1_0.gguf"

def run(texts, add_bos=0):
    cmds = ["load " + MODEL]
    for t in texts:
        cmds.append("tokenize " + t.encode("utf-8").hex() + (" 1" if add_bos else ""))
    cmds.append("quit")
    p = subprocess.run([REF], input=("\n".join(cmds) + "\n").encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = p.stdout.decode("utf-8", "replace").splitlines()
    out = []
    it = iter(lines)
    first = next(it)
    assert first == "ok", first
    for line in it:
        parts = line.split()
        if parts[0] == "ok":
            n = int(parts[1])
            out.append([int(x) for x in parts[2:2 + n]])
        else:
            out.append(None)
    return out

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as fh:
        texts = json.load(fh)
    res = run(texts)
    json.dump(res, sys.stdout)
