import shutil, os
os.makedirs("/workspace/tools/refsrc", exist_ok=True)
for f in ["constants.py","quants.py","gguf_reader.py","vocab.py","gguf_writer.py","tensor_mapping.py"]:
    shutil.copyfile("/opt/bonsai/gguf-py/gguf/"+f, "/workspace/tools/refsrc/"+f)
print("done")
