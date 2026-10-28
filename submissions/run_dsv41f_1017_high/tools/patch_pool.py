p="/workspace/src/main.zig"
s=open(p).read()
s=s.replace("""fn matvecPtq(g: *const Gguf, t: *const Tensor, x: []const f32, out: []f32) void {
    const rows: usize = @intCast(t.ne[1]);
    const job = MVJob{ .g = g, .t = t, .x = x, .out = out, .r0 = 0, .r1 = rows, .rows = rows };
    if (g_pool) |p| {""","""fn matvecPtq(pool: ?*Pool, g: *const Gguf, t: *const Tensor, x: []const f32, out: []f32) void {
    const rows: usize = @intCast(t.ne[1]);
    const job = MVJob{ .g = g, .t = t, .x = x, .out = out, .r0 = 0, .r1 = rows, .rows = rows };
    if (pool) |p| {""")
s=s.replace("matvecPtq(g, L.wq.?,","matvecPtq(ctx.model.pool, g, L.wq.?,")
s=s.replace("matvecPtq(g, L.wk.?,","matvecPtq(ctx.model.pool, g, L.wk.?,")
s=s.replace("matvecPtq(g, L.wv.?,","matvecPtq(ctx.model.pool, g, L.wv.?,")
s=s.replace("matvecPtq(g, L.wo.?,","matvecPtq(ctx.model.pool, g, L.wo.?,")
s=s.replace("matvecPtq(g, L.wqkv.?,","matvecPtq(ctx.model.pool, g, L.wqkv.?,")
s=s.replace("matvecPtq(g, L.wgate.?,","matvecPtq(ctx.model.pool, g, L.wgate.?,")
s=s.replace("matvecPtq(g, L.ssout.?,","matvecPtq(ctx.model.pool, g, L.ssout.?,")
s=s.replace("matvecPtq(g, L.ffn_gate,","matvecPtq(ctx.model.pool, g, L.ffn_gate,")
s=s.replace("matvecPtq(g, L.ffn_up,","matvecPtq(ctx.model.pool, g, L.ffn_up,")
s=s.replace("matvecPtq(g, L.ffn_down,","matvecPtq(ctx.model.pool, g, L.ffn_down,")
s=s.replace("matvecPtq(g, m.token_embd,","matvecPtq(ctx.model.pool, g, m.token_embd,")
s=s.replace("    signs9216: []const f32,\n    n_vocab: i32,\n    vk: ?*vkmod.Vk = null,","    signs9216: []const f32,\n    n_vocab: i32,\n    pool: ?*Pool = null,\n    vk: ?*vkmod.Vk = null,")
s=s.replace("    g_pool = Pool.create(alloc) catch null;\n    return m;","    m.pool = Pool.create(alloc) catch null;\n    return m;")
s=s.replace("""    if (g_pool) |p| {
        p.destroy(alloc);
        g_pool = null;
    }""","""    if (m.pool) |p| {
        p.destroy(alloc);
        m.pool = null;
    }""")
s=s.replace("var g_pool: ?*Pool = null;\n\n","")
open(p,"w").write(s)
print("patched")
