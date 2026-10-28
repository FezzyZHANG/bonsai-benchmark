import shutil, os
for p in ["/workspace/zig-out", "/workspace/.zig-cache"]:
    if os.path.exists(p): shutil.rmtree(p)
print("cleared")
