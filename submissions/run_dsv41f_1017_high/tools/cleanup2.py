import shutil, os
for p in ["/workspace/tools/cleanbuild"]:
    if os.path.exists(p):
        shutil.rmtree(p)
for f in ["/workspace/src/vk_bindings_raw.zig", "/workspace/src/toktest.zig"]:
    if os.path.exists(f):
        os.remove(f)
print("done")
