import struct
data=open("/workspace/tools/matvec.spv","rb").read()
words=struct.unpack("<%dI"%(len(data)//4), data)
out=["// Auto-generated from tools/matvec.comp (glslc --target-env=vulkan1.4 -O).",
     "pub const words = [_]u32{"]
line="    "
for w in words:
    line += "0x%08x,"%w
    if len(line)>96:
        out.append(line); line="    "
if line.strip(): out.append(line)
out.append("};")
open("/workspace/src/matvec_spv.zig","w").write("\n".join(out)+"\n")
print("words",len(words))
