import os
for root, dirs, files in os.walk("/opt/bonsai"):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
    depth = root.count("/")
    print("  "*depth, root, "files:", len(files))
    for f in files[:40]:
        print("  "*(depth+1), f)
