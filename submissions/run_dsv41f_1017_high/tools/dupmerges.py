import collections

merges = [l.strip() for l in open("/workspace/tools/tokdata/merges.hex")]
print("n merges", len(merges))
c = collections.Counter(merges)
dups = [(k, v) for k, v in c.items() if v > 1]
print("duplicate merge lines", len(dups), dups[:5])
