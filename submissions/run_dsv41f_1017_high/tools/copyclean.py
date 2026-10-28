import shutil, os
dst="/workspace/tools/cleanbuild"
for f in ["build.zig","NOTES.md"]:
    shutil.copyfile("/workspace/"+f, dst+"/"+f)
os.makedirs(dst+"/src", exist_ok=True)
for f in os.listdir("/workspace/src"):
    if f.endswith(".zig"):
        shutil.copyfile("/workspace/src/"+f, dst+"/src/"+f)
print(sorted(os.listdir(dst)), sorted(os.listdir(dst+"/src")))
