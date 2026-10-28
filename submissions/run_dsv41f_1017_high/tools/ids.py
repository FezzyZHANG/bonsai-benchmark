import sys

tokens = [bytes.fromhex(l) for l in open("/workspace/tools/tokdata/tokens.hex")]
for a in sys.argv[1:]:
    for i in a.split(","):
        i = int(i)
        print(i, repr(tokens[i]), tokens[i].hex(), file=sys.stderr)
