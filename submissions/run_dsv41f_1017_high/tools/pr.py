import sys
p=sys.argv[1]; a=int(sys.argv[2]); b=int(sys.argv[3])
lines=open(p).read().splitlines()
for i in range(a,b):
    print(i, lines[i])
