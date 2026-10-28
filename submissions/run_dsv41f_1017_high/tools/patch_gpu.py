p="/workspace/src/main.zig"
s=open(p).read()
s=s.replace("matvecPtq(ctx.model.pool, g, ","matvecPtq(ctx.model.pool, ctx.model.gpu, g, ")
s=s.replace("    uploadWeightsToVk(m);","    setupGpu(m);")
s=s.replace("""    if (m.vk) |p| {
        for (m.vk_bufs) |b| p.destroyBuffer(b);
        if (m.vk_bufs.len > 0) alloc.free(m.vk_bufs);
        p.deinit();
        alloc.destroy(p);
    }""","""    if (m.gpu) |gp| {
        gp.vk.destroyBuffer(gp.xbuf);
        gp.vk.destroyBuffer(gp.obuf);
        for (gp.bufs) |b| {
            if (b) |bb| gp.vk.destroyBuffer(bb);
        }
        alloc.free(gp.bufs);
        gp.vk.destroyPipeline(gp.pipe);
        gp.vk.deinit();
        alloc.destroy(gp.vk);
        alloc.destroy(gp);
        m.gpu = null;
    }""")
open(p,"w").write(s)
print("ok")
