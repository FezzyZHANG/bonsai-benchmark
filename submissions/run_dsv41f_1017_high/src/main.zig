const std = @import("std");
const ggufmod = @import("gguf.zig");
const tokmod = @import("tokenizer.zig");
const vkmod = @import("vk");

const Tensor = ggufmod.Tensor;
const Gguf = ggufmod.Gguf;

const E_INVALID: i32 = -1;
const E_CAPACITY: i32 = -2;
const E_IO: i32 = -3;
const E_UNSUPPORTED: i32 = -4;
const E_STATE: i32 = -5;
const E_INTERNAL: i32 = -6;

threadlocal var err_buf: [512]u8 = undefined;
threadlocal var err_len: usize = 0;

fn clearErr() void {
    err_len = 0;
}

fn setErr(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(err_buf[0 .. err_buf.len - 1], fmt, args) catch {
        err_len = 0;
        return;
    };
    err_len = s.len;
    err_buf[s.len] = 0;
}

// ---------------- model ----------------

const n_embd = 2560;
const n_ff = 9216;
const n_head = 16;
const n_head_kv = 4;
const head_dim = 256;
const n_rot = 64;
const n_layer = 32;
const n_vocab_fallback = 248320;
const qk_dim = 128; // GDN head dim (k and v)
const n_k_heads = 16;
const n_v_heads = 32;
const n_qkv = 8192;
const ssm_state_dim = qk_dim * qk_dim; // 16384 per head
const hd512: f32 = 1.0 / 22.627416997969522; // 1/sqrt(512)

const Layer = struct {
    is_attn: bool,
    attn_norm: *const Tensor,
    post_attn_norm: *const Tensor,
    ffn_gate: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    // attention
    wq: ?*const Tensor = null,
    wk: ?*const Tensor = null,
    wv: ?*const Tensor = null,
    wo: ?*const Tensor = null,
    q_norm: ?*const Tensor = null,
    k_norm: ?*const Tensor = null,
    // ssm
    wqkv: ?*const Tensor = null,
    wgate: ?*const Tensor = null,
    ssout: ?*const Tensor = null,
    walpha: ?*const Tensor = null,
    wbeta: ?*const Tensor = null,
    ssm_a: ?*const Tensor = null,
    dt_bias: ?*const Tensor = null,
    conv1d: ?*const Tensor = null,
    ssm_norm: ?*const Tensor = null,
};

const Model = struct {
    alloc: std.mem.Allocator,
    g: *Gguf,
    tok: tokmod.Tokenizer,
    layers: []Layer,
    token_embd: *const Tensor,
    output_norm: *const Tensor,
    signs2560: []const f32,
    signs4096: []const f32,
    signs9216: []const f32,
    n_vocab: i32,
    pool: ?*Pool = null,
    gpu: ?*Gpu = null,
};

const Gpu = struct {
    vk: *vkmod.Vk,
    pipe: vkmod.Pipeline,
    bufs: []?vkmod.Buffer,
    xbuf: vkmod.Buffer,
    obuf: vkmod.Buffer,
    xmap: [*]u8,
    omap: [*]u8,
};

fn tensorIndex(g: *const Gguf, t: *const Tensor) usize {
    return (@intFromPtr(t) - @intFromPtr(&g.tensors[0])) / @sizeOf(Tensor);
}

fn findT(g: *Gguf, comptime fmt: []const u8, args: anytype) !*const Tensor {
    var buf: [96]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return g.lookup(name) orelse error.MissingTensor;
}

fn findName(g: *Gguf, name: []const u8) !*const Tensor {
    return g.lookup(name) orelse error.MissingTensor;
}

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;

fn readWholeFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    const pathz = try alloc.alloc(u8, path.len + 1);
    defer alloc.free(pathz);
    @memcpy(pathz[0..path.len], path);
    pathz[path.len] = 0;
    const fd = open(@ptrCast(pathz.ptr), 0);
    if (fd < 0) return error.OpenFailed;
    defer _ = close(fd);
    const end = lseek(fd, 0, 2);
    if (end <= 0) return error.OpenFailed;
    _ = lseek(fd, 0, 0);
    const size: usize = @intCast(end);
    const buf = try alloc.alloc(u8, size);
    var off: usize = 0;
    while (off < size) {
        const n = read(fd, buf.ptr + off, size - off);
        if (n <= 0) return error.ReadFailed;
        off += @intCast(n);
    }
    return buf;
}

fn openModel(path: []const u8) !*Model {
    const alloc = std.heap.c_allocator;
    const data = try readWholeFile(alloc, path);
    if (data.len < 1024) return error.BadGguf;
    errdefer alloc.free(data);
    const g = try ggufmod.parse(alloc, data);

    const tokens = g.tokens;
    if (tokens.len == 0) return error.BadGguf;
    const t = try tokmod.Tokenizer.init(alloc, tokens, g.token_types, g.merges, g.bos_id, g.eos_id, g.add_bos);

    const signs = g.signs;
    if (signs.len < 15872) return error.BadGguf;
    const s2560 = signs[0..2560];
    const s4096 = signs[2560..6656];
    const s9216 = signs[6656..15872];

    const layers = try alloc.alloc(Layer, n_layer);
    for (layers, 0..) |*L, il| {
        const is_attn = ((il + 1) % 4) == 0;
        L.* = .{
            .is_attn = is_attn,
            .attn_norm = try findT(g, "blk.{d}.attn_norm.weight", .{il}),
            .post_attn_norm = try findT(g, "blk.{d}.post_attention_norm.weight", .{il}),
            .ffn_gate = try findT(g, "blk.{d}.ffn_gate.weight", .{il}),
            .ffn_up = try findT(g, "blk.{d}.ffn_up.weight", .{il}),
            .ffn_down = try findT(g, "blk.{d}.ffn_down.weight", .{il}),
        };
        if (is_attn) {
            L.wq = try findT(g, "blk.{d}.attn_q.weight", .{il});
            L.wk = try findT(g, "blk.{d}.attn_k.weight", .{il});
            L.wv = try findT(g, "blk.{d}.attn_v.weight", .{il});
            L.wo = try findT(g, "blk.{d}.attn_output.weight", .{il});
            L.q_norm = try findT(g, "blk.{d}.attn_q_norm.weight", .{il});
            L.k_norm = try findT(g, "blk.{d}.attn_k_norm.weight", .{il});
        } else {
            L.wqkv = try findT(g, "blk.{d}.attn_qkv.weight", .{il});
            L.wgate = try findT(g, "blk.{d}.attn_gate.weight", .{il});
            L.ssout = try findT(g, "blk.{d}.ssm_out.weight", .{il});
            L.walpha = try findT(g, "blk.{d}.ssm_alpha.weight", .{il});
            L.wbeta = try findT(g, "blk.{d}.ssm_beta.weight", .{il});
            L.ssm_a = try findT(g, "blk.{d}.ssm_a", .{il});
            L.dt_bias = try findT(g, "blk.{d}.ssm_dt.bias", .{il});
            L.conv1d = try findT(g, "blk.{d}.ssm_conv1d.weight", .{il});
            L.ssm_norm = try findT(g, "blk.{d}.ssm_norm.weight", .{il});
        }
    }
    const m = try alloc.create(Model);
    m.* = .{
        .alloc = alloc,
        .g = g,
        .tok = t,
        .layers = layers,
        .token_embd = try findName(g, "token_embd.weight"),
        .output_norm = try findName(g, "output_norm.weight"),
        .signs2560 = s2560,
        .signs4096 = s4096,
        .signs9216 = s9216,
        .n_vocab = if (g.n_vocab > 0) g.n_vocab else n_vocab_fallback,
    };
    // Bring up the Vulkan device and make the packed model resident in
    // device-local memory. If Vulkan is unavailable the engine keeps working.
    setupGpu(m);
    // The worker pool is only used when matmuls run on the CPU; with the GPU
    // path active it would just spin and steal cores from the orchestration.
    if (m.gpu == null) m.pool = Pool.create(alloc) catch null;
    return m;
}

const matvec_spv = @import("matvec_spv.zig");

fn setupGpu(m: *Model) void {
    const alloc = m.alloc;
    const p = alloc.create(vkmod.Vk) catch return;
    p.* = vkmod.Vk.init(alloc) catch {
        alloc.destroy(p);
        return;
    };
    const pipe = p.createComputePipelineEx(&matvec_spv.words, 3, 4) catch {
        p.deinit();
        alloc.destroy(p);
        return;
    };
    const bufs = alloc.alloc(?vkmod.Buffer, m.g.tensors.len) catch {
        p.destroyPipeline(pipe);
        p.deinit();
        alloc.destroy(p);
        return;
    };
    @memset(bufs, null);
    // Upload every PTQ1_0 tensor into its own device-local buffer; the packed
    // weights stay ternary on the GPU.
    for (m.g.tensors, 0..) |*t, i| {
        if (t.ggml_type != 143) continue;
        const d = m.g.tensorData(t);
        if (d.len == 0) continue;
        bufs[i] = p.uploadDeviceLocal(d) catch null;
    }
    const usage: u32 = vkmod.buffer_usage_storage;
    const xbuf = p.createHostBuffer(9216 * 4, usage) catch {
        for (bufs) |b| {
            if (b) |bb| p.destroyBuffer(bb);
        }
        alloc.free(bufs);
        p.destroyPipeline(pipe);
        p.deinit();
        alloc.destroy(p);
        return;
    };
    const obuf = p.createHostBuffer(248320 * 4, usage) catch {
        p.destroyBuffer(xbuf);
        for (bufs) |b| {
            if (b) |bb| p.destroyBuffer(bb);
        }
        alloc.free(bufs);
        p.destroyPipeline(pipe);
        p.deinit();
        alloc.destroy(p);
        return;
    };
    const xmap = p.mapHostBuffer(xbuf) catch return;
    const omap = p.mapHostBuffer(obuf) catch return;
    const gp = alloc.create(Gpu) catch return;
    gp.* = .{ .vk = p, .pipe = pipe, .bufs = bufs, .xbuf = xbuf, .obuf = obuf, .xmap = xmap, .omap = omap };
    m.gpu = gp;
}

fn tryGpuMatvec(gp: *Gpu, tidx: usize, cols: usize, rows: usize, x: []const f32, out: []f32) bool {
    if (cols == 0 or rows == 0 or cols > 9216 or rows > 248320) return false;
    if ((cols % 128) != 0) return false;
    const wbuf = gp.bufs[tidx] orelse return false;
    const vk = gp.vk;
    @memcpy(gp.xmap[0 .. cols * 4], std.mem.sliceAsBytes(x[0..cols]));
    var pc: [4]u8 = undefined;
    std.mem.writeInt(u32, &pc, @intCast(cols), .little);
    const groups: u32 = @intCast((rows + 63) / 64);
    vk.dispatchPush(gp.pipe, &.{ wbuf, gp.xbuf, gp.obuf }, &pc, groups, 1, 1) catch return false;
    @memcpy(std.mem.sliceAsBytes(out[0..rows]), gp.omap[0 .. rows * 4]);
    return true;
}

fn closeModel(m: *Model) void {
    const alloc = m.alloc;
    if (m.pool) |p| {
        p.destroy(alloc);
        m.pool = null;
    }
    if (m.gpu) |gp| {
        gp.vk.unmapHostBuffer(gp.xbuf);
        gp.vk.unmapHostBuffer(gp.obuf);
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
    }
    m.tok.deinit();
    for (m.g.tensors) |t| alloc.free(@constCast(t.name));
    alloc.free(m.g.tensors);
    alloc.free(m.g.tokens);
    alloc.free(m.g.merges);
    alloc.free(@constCast(m.g.token_types));
    alloc.free(@constCast(m.g.signs));
    alloc.destroy(m.g);
    alloc.free(@constCast(m.g.data));
    alloc.free(m.layers);
    alloc.destroy(m);
}

// ---------------- math ----------------

inline fn f16Of(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn fwht(v: []f32) void {
    const n = v.len;
    var len: usize = 1;
    while (len < n) : (len *= 2) {
        var i: usize = 0;
        while (i < n) : (i += 2 * len) {
            var j: usize = 0;
            while (j < len) : (j += 1) {
                const a = v[i + j];
                const b = v[i + j + len];
                v[i + j] = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

fn hadamardBlock(v: []f32) void {
    fwht(v);
    const n: f32 = @floatFromInt(v.len);
    const s = 1.0 / @sqrt(n);
    for (v) |*x| x.* *= s;
}

/// sign then hadamard, per block of 512; len must be a multiple of 512.
fn transformSignH(v: []f32, signs: []const f32) void {
    for (v, 0..) |*x, i| x.* *= signs[i];
    var off: usize = 0;
    while (off < v.len) : (off += 512) hadamardBlock(v[off .. off + 512]);
}

/// hadamard then sign (embedding lookup path).
fn transformHSign(v: []f32, signs: []const f32) void {
    var off: usize = 0;
    while (off < v.len) : (off += 512) hadamardBlock(v[off .. off + 512]);
    for (v, 0..) |*x, i| x.* *= signs[i];
}

fn rmsnorm(x: []const f32, w: []const f32, eps: f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, 0..) |v, i| out[i] = v * inv * w[i];
}

inline fn siluF(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

inline fn sigmoidF(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Dequantize a full row of a PTQ1_0 tensor into out (ne[0] floats).
fn dequantRow(g: *const Gguf, t: *const Tensor, row: usize, out: []f32) void {
    const cols: usize = @intCast(t.ne[0]);
    const nblk = cols / 128;
    const raw = g.tensorData(t);
    const base = row * nblk * 28;
    var blk: [128]f32 = undefined;
    for (0..nblk) |b| {
        ggufmod.dequantBlock(raw[base + b * 28 ..][0..28], &blk);
        @memcpy(out[b * 128 ..][0..128], &blk);
    }
}

inline fn planeDot(raw: []const u8, x: []const f32, comptime n: usize, comptime ndig: usize, comptime xoff: usize) @Vector(n, f32) {
    const rv: @Vector(n, u8) = raw[0..n].*;
    var acc: @Vector(n, f32) = @splat(0.0);
    var mult: u16 = 1;
    inline for (0..ndig) |dg| {
        const wv: @Vector(n, u16) = @intCast(rv);
        const prod = (wv * @as(@Vector(n, u16), @splat(mult))) & @as(@Vector(n, u16), @splat(255));
        const v = (prod * @as(@Vector(n, u16), @splat(3))) >> @as(@Vector(n, u4), @splat(8));
        const q: @Vector(n, i16) = @as(@Vector(n, i16), @intCast(v)) - @as(@Vector(n, i16), @splat(1));
        const qf: @Vector(n, f32) = @floatFromInt(q);
        const xv: @Vector(n, f32) = x[xoff + dg * n ..][0..n].*;
        acc += qf * xv;
        mult = (mult * 3) & 0xFF;
    }
    return acc;
}

inline fn blockDot(raw: []const u8, x: []const f32) f32 {
    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[26..28], .little))));
    var acc: @Vector(16, f32) = planeDot(raw[0..16], x, 16, 5, 0);
    const mid = planeDot(raw[16..24], x, 8, 5, 80);
    inline for (0..8) |k| acc[k] += mid[k];
    const tail = planeDot(raw[24..26], x, 2, 4, 120);
    inline for (0..2) |k| acc[k] += tail[k];
    return @reduce(.Add, acc) * d;
}

fn matvecRows(g: *const Gguf, t: *const Tensor, x: []const f32, out: []f32, r0: usize, r1: usize) void {
    const cols: usize = @intCast(t.ne[0]);
    const nblk = cols / 128;
    const raw = g.tensorData(t);
    for (r0..r1) |r| {
        const base = r * nblk * 28;
        var acc: @Vector(16, f32) = @splat(0.0);
        for (0..nblk) |b| {
            const xs = x[b * 128 ..][0..128];
            acc += blockDotAcc(raw[base + b * 28 ..][0..28], xs);
        }
        out[r] = @reduce(.Add, acc);
    }
}

inline fn blockDotAcc(raw: []const u8, x: []const f32) @Vector(16, f32) {
    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[26..28], .little))));
    var acc: @Vector(16, f32) = planeDot(raw[0..16], x, 16, 5, 0);
    const mid = planeDot(raw[16..24], x, 8, 5, 80);
    inline for (0..8) |k| acc[k] += mid[k];
    const tail = planeDot(raw[24..26], x, 2, 4, 120);
    inline for (0..2) |k| acc[k] += tail[k];
    return acc * @as(@Vector(16, f32), @splat(d));
}

const MVJob = struct {
    g: *const Gguf,
    t: *const Tensor,
    x: []const f32,
    out: []f32,
    r0: usize,
    r1: usize,
    rows: usize = 0,
    fn run(job: MVJob) void {
        matvecRows(job.g, job.t, job.x, job.out, job.r0, job.r1);
    }
};

const n_workers: usize = 19;

const Pool = struct {
    threads: []std.Thread,
    job: ?MVJob = null,
    gen: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    active: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    nworkers: usize,

    fn worker(self: *Pool, idx: usize) void {
        var seen: u64 = 0;
        while (true) {
            while (true) {
                if (self.stop.load(.acquire)) return;
                const g = self.gen.load(.acquire);
                if (g != seen) {
                    seen = g;
                    break;
                }
                std.atomic.spinLoopHint();
            }
            const job = self.job.?;
            const total = self.nworkers + 1;
            const chunk = (job.rows + total - 1) / total;
            const r0 = @min(job.rows, (idx + 1) * chunk);
            const r1 = @min(job.rows, (idx + 2) * chunk);
            if (r0 < r1) matvecRows(job.g, job.t, job.x, job.out, r0, r1);
            _ = self.active.fetchSub(1, .release);
        }
    }

    fn dispatch(self: *Pool, job: MVJob) void {
        if (job.rows < 128) {
            matvecRows(job.g, job.t, job.x, job.out, job.r0, job.r1);
            return;
        }
        self.job = job;
        self.active.store(@intCast(self.nworkers), .release);
        _ = self.gen.fetchAdd(1, .release);
        const total = self.nworkers + 1;
        const chunk = (job.rows + total - 1) / total;
        matvecRows(job.g, job.t, job.x, job.out, 0, @min(job.rows, chunk));
        while (self.active.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    fn create(alloc: std.mem.Allocator) !*Pool {
        const p = try alloc.create(Pool);
        p.* = .{ .threads = &.{}, .nworkers = 0 };
        const n = @min(@as(usize, n_workers), (std.Thread.getCpuCount() catch 4) -| 1);
        if (n == 0) return p;
        p.threads = try alloc.alloc(std.Thread, n);
        p.nworkers = n;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            p.threads[i] = std.Thread.spawn(.{}, Pool.worker, .{ p, i }) catch {
                p.nworkers = i;
                break;
            };
        }
        return p;
    }

    fn destroy(self: *Pool, alloc: std.mem.Allocator) void {
        self.stop.store(true, .release);
        for (self.threads[0..self.nworkers]) |th| th.join();
        if (self.threads.len > 0) alloc.free(self.threads);
        alloc.destroy(self);
    }
};

/// out[o] = sum_i W[o][i] * x[i] for a PTQ1_0 weight, using the stored ternary weights.
fn matvecPtq(pool: ?*Pool, gpu: ?*Gpu, g: *const Gguf, t: *const Tensor, x: []const f32, out: []f32) void {
    const rows: usize = @intCast(t.ne[1]);
    const cols: usize = @intCast(t.ne[0]);
    if (gpu) |gp| {
        if (tryGpuMatvec(gp, tensorIndex(g, t), cols, rows, x, out)) return;
    }
    const job = MVJob{ .g = g, .t = t, .x = x, .out = out, .r0 = 0, .r1 = rows, .rows = rows };
    if (pool) |p| {
        p.dispatch(job);
    } else {
        matvecRows(g, t, x, out, 0, rows);
    }
}

/// F32 weight matrix: out[o] = sum_i W[o][i] x[i], W row-major [rows][cols].
fn matvecF32(g: *const Gguf, t: *const Tensor, x: []const f32, out: []f32) void {
    const cols: usize = @intCast(t.ne[0]);
    const rows: usize = @intCast(t.ne[1]);
    const raw = g.tensorData(t);
    const w: [*]const f32 = @ptrCast(@alignCast(raw.ptr));
    for (0..rows) |r| {
        var vacc: @Vector(8, f32) = @splat(0.0);
        const row = w + r * cols;
        var i: usize = 0;
        while (i + 8 <= cols) : (i += 8) {
            const bv: @Vector(8, f32) = row[i..][0..8].*;
            const xv: @Vector(8, f32) = x[i..][0..8].*;
            vacc += bv * xv;
        }
        var acc = @reduce(.Add, vacc);
        while (i < cols) : (i += 1) acc += row[i] * x[i];
        out[r] = acc;
    }
}

fn sumF(v: []const f32) f32 {
    var s: f32 = 0;
    for (v) |x| s += x;
    return s;
}

fn f32Data(g: *const Gguf, t: *const Tensor) []const f32 {
    const raw = g.tensorData(t);
    const n = ggufmod.elemCount(t);
    const p: [*]const f32 = @ptrCast(@alignCast(raw.ptr));
    return p[0..n];
}

// ---------------- context ----------------

const Context = struct {
    alloc: std.mem.Allocator,
    model: *Model,
    n_ctx: i32,
    pos: i32,
    kv_k: [][]f32,
    kv_v: [][]f32,
    ssm_state: [][]f32,
    conv_state: [][]f32,
    // scratch
    x: []f32,
    xn: []f32,
    xh: []f32, // up to 9216
    qkv: []f32, // 8192
    cs: []f32, // 8192
    z: []f32, // 4096
    out1: []f32, // 4096
    out2: []f32, // 4096
    qbuf: []f32, // 8192 (attn q+gate)
    kcur: []f32, // 1024
    vcur: []f32, // 1024
    alpha: []f32, // 32
    beta: []f32, // 32
    gbuf: []f32, // 32
    ffg: []f32, // 9216
    ffu: []f32, // 9216
    sw: []f32, // 9216
    logits: []f32,
};

fn allocF32(alloc: std.mem.Allocator, n: usize) ![]f32 {
    const p = try alloc.alloc(f32, n);
    return p;
}

fn createContext(m: *Model, n_ctx: i32, n_batch: i32) !*Context {
    _ = n_batch;
    if (n_ctx < 1) return error.BadArg;
    const alloc = m.alloc;
    const ctx = try alloc.create(Context);
    ctx.* = undefined;
    ctx.alloc = alloc;
    ctx.model = m;
    ctx.n_ctx = n_ctx;
    ctx.pos = 0;
    const nctx: usize = @intCast(n_ctx);
    var n_attn: usize = 0;
    var n_ssm: usize = 0;
    for (m.layers) |L| {
        if (L.is_attn) n_attn += 1 else n_ssm += 1;
    }
    ctx.kv_k = try alloc.alloc([]f32, n_attn);
    ctx.kv_v = try alloc.alloc([]f32, n_attn);
    for (0..n_attn) |i| {
        ctx.kv_k[i] = try allocF32(alloc, nctx * head_dim * n_head_kv);
        ctx.kv_v[i] = try allocF32(alloc, nctx * head_dim * n_head_kv);
    }
    ctx.ssm_state = try alloc.alloc([]f32, n_ssm);
    ctx.conv_state = try alloc.alloc([]f32, n_ssm);
    for (0..n_ssm) |i| {
        ctx.ssm_state[i] = try allocF32(alloc, n_v_heads * ssm_state_dim);
        @memset(ctx.ssm_state[i], 0);
        ctx.conv_state[i] = try allocF32(alloc, 3 * n_qkv);
        @memset(ctx.conv_state[i], 0);
    }
    ctx.x = try allocF32(alloc, n_embd);
    ctx.xn = try allocF32(alloc, n_embd);
    ctx.xh = try allocF32(alloc, n_ff);
    ctx.qkv = try allocF32(alloc, n_qkv);
    ctx.cs = try allocF32(alloc, n_qkv);
    ctx.z = try allocF32(alloc, 4096);
    ctx.out1 = try allocF32(alloc, 4096);
    ctx.out2 = try allocF32(alloc, n_embd);
    ctx.qbuf = try allocF32(alloc, 8192);
    ctx.kcur = try allocF32(alloc, 1024);
    ctx.vcur = try allocF32(alloc, 1024);
    ctx.alpha = try allocF32(alloc, 32);
    ctx.beta = try allocF32(alloc, 32);
    ctx.gbuf = try allocF32(alloc, 32);
    ctx.ffg = try allocF32(alloc, n_ff);
    ctx.ffu = try allocF32(alloc, n_ff);
    ctx.sw = try allocF32(alloc, n_ff);
    ctx.logits = try allocF32(alloc, @intCast(m.n_vocab));
    return ctx;
}

fn destroyContext(ctx: *Context) void {
    const alloc = ctx.alloc;
    for (ctx.kv_k) |s| alloc.free(s);
    for (ctx.kv_v) |s| alloc.free(s);
    for (ctx.ssm_state) |s| alloc.free(s);
    for (ctx.conv_state) |s| alloc.free(s);
    alloc.free(ctx.kv_k);
    alloc.free(ctx.kv_v);
    alloc.free(ctx.ssm_state);
    alloc.free(ctx.conv_state);
    alloc.free(ctx.x);
    alloc.free(ctx.xn);
    alloc.free(ctx.xh);
    alloc.free(ctx.qkv);
    alloc.free(ctx.cs);
    alloc.free(ctx.z);
    alloc.free(ctx.out1);
    alloc.free(ctx.out2);
    alloc.free(ctx.qbuf);
    alloc.free(ctx.kcur);
    alloc.free(ctx.vcur);
    alloc.free(ctx.alpha);
    alloc.free(ctx.beta);
    alloc.free(ctx.gbuf);
    alloc.free(ctx.ffg);
    alloc.free(ctx.ffu);
    alloc.free(ctx.sw);
    alloc.free(ctx.logits);
    alloc.destroy(ctx);
}

fn resetContext(ctx: *Context) void {
    ctx.pos = 0;
    for (ctx.ssm_state) |s| @memset(s, 0);
    for (ctx.conv_state) |s| @memset(s, 0);
}

fn ropeInPlace(v: []f32, nheads: usize, stride: usize, pos: i32, freq_base: f32) void {
    const half = n_rot / 2;
    var h: usize = 0;
    while (h < nheads) : (h += 1) {
        const base = h * stride;
        var i: usize = 0;
        while (i < half) : (i += 1) {
            const exponent: f32 = @floatFromInt(2 * i);
            const nd: f32 = @floatFromInt(n_rot);
            const inv_freq = 1.0 / std.math.pow(f32, freq_base, exponent / nd);
            const ang = @as(f32, @floatFromInt(pos)) * inv_freq;
            const c = @cos(ang);
            const s = @sin(ang);
            const x0 = v[base + i];
            const x1 = v[base + i + half];
            v[base + i] = x0 * c - x1 * s;
            v[base + i + half] = x0 * s + x1 * c;
        }
    }
}

const gdn_scale: f32 = 1.0 / 11.313708498984761; // 1/sqrt(128)

fn evalToken(ctx: *Context, token: i32, want_logits: bool) i32 {
    const m = ctx.model;
    const g = m.g;
    if (token < 0 or token >= m.n_vocab) {
        setErr("token id {d} out of range", .{token});
        return E_INVALID;
    }
    // embedding lookup: e = signs * H(row)
    dequantRow(g, m.token_embd, @intCast(token), ctx.x);
    transformHSign(ctx.x, m.signs2560);

    var attn_idx: usize = 0;
    var ssm_idx: usize = 0;
    const eps: f32 = if (g.eps > 0) g.eps else 1e-6;
    const freq_base: f32 = if (g.rope_freq_base > 0) g.rope_freq_base else 10000.0;

    for (m.layers, 0..) |L, il| {
        rmsnorm(ctx.x, f32Data(g, L.attn_norm), eps, ctx.xn);
        if (L.is_attn) {
            transformSignH(ctx.xn, m.signs2560);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wq.?, ctx.xn, ctx.qbuf);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wk.?, ctx.xn, ctx.kcur);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wv.?, ctx.xn, ctx.vcur);
            const qw = f32Data(g, L.q_norm.?);
            const kw = f32Data(g, L.k_norm.?);
            // per-head rms norm + rope
            {
                var h: usize = 0;
                var tmp: [head_dim]f32 = undefined;
                while (h < n_head) : (h += 1) {
                    const qs = ctx.qbuf[h * 2 * head_dim ..][0..head_dim];
                    rmsnorm(qs, qw, eps, &tmp);
                    @memcpy(qs, &tmp);
                }
                h = 0;
                while (h < n_head_kv) : (h += 1) {
                    const ks = ctx.kcur[h * head_dim ..][0..head_dim];
                    rmsnorm(ks, kw, eps, &tmp);
                    @memcpy(ks, &tmp);
                }
            }
            ropeInPlace(ctx.qbuf[0 .. n_head * 2 * head_dim], n_head, 2 * head_dim, ctx.pos, freq_base);
            ropeInPlace(ctx.kcur, n_head_kv, head_dim, ctx.pos, freq_base);
            // store kv
            const kcache = ctx.kv_k[attn_idx];
            const vcache = ctx.kv_v[attn_idx];
            const p: usize = @intCast(ctx.pos);
            const kvrow = p * head_dim * n_head_kv;
            @memcpy(kcache[kvrow..][0 .. head_dim * n_head_kv], ctx.kcur);
            @memcpy(vcache[kvrow..][0 .. head_dim * n_head_kv], ctx.vcur);
            // attention
            const scale: f32 = 1.0 / 16.0;
            var scores: [8192]f32 = undefined;
            var h: usize = 0;
            while (h < n_head) : (h += 1) {
                const kvh = h / (n_head / n_head_kv);
                const q = ctx.qbuf[h * 2 * head_dim ..][0..head_dim];
                const npos = p + 1;
                var mx: f32 = -1e30;
                var t: usize = 0;
                while (t < npos) : (t += 1) {
                    const krow = kcache[t * head_dim * n_head_kv + kvh * head_dim ..][0..head_dim];
                    var acc: @Vector(8, f32) = @splat(0.0);
                    var d: usize = 0;
                    while (d < head_dim) : (d += 8) {
                        const a: @Vector(8, f32) = q[d..][0..8].*;
                        const b: @Vector(8, f32) = krow[d..][0..8].*;
                        acc += a * b;
                    }
                    const sc = @reduce(.Add, acc) * scale;
                    scores[t] = sc;
                    if (sc > mx) mx = sc;
                }
                var sum: f32 = 0;
                t = 0;
                while (t < npos) : (t += 1) {
                    const e = @exp(scores[t] - mx);
                    scores[t] = e;
                    sum += e;
                }
                const inv = 1.0 / sum;
                const o = ctx.out1[h * head_dim ..][0..head_dim];
                @memset(o, 0);
                t = 0;
                while (t < npos) : (t += 1) {
                    const pw = scores[t] * inv;
                    const vrow = vcache[t * head_dim * n_head_kv + kvh * head_dim ..][0..head_dim];
                    var d: usize = 0;
                    while (d < head_dim) : (d += 8) {
                        const pv: @Vector(8, f32) = vrow[d..][0..8].*;
                        const ov: @Vector(8, f32) = o[d..][0..8].*;
                        o[d..][0..8].* = ov + pv * @as(@Vector(8, f32), @splat(pw));
                    }
                }
            }
            // gate sigmoid * attn: per head, gate follows q in qbuf
            for (0..4096) |i| {
                const head = i / head_dim;
                const off = head * 2 * head_dim + head_dim + (i % head_dim);
                ctx.out1[i] *= sigmoidF(ctx.qbuf[off]);
            }
            transformSignH(ctx.out1, m.signs4096);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wo.?, ctx.out1, ctx.out2);
            attn_idx += 1;
        } else {
            matvecF32(g, L.walpha.?, ctx.xn, ctx.alpha);
            matvecF32(g, L.wbeta.?, ctx.xn, ctx.beta);
            transformSignH(ctx.xn, m.signs2560);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wqkv.?, ctx.xn, ctx.qkv);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.wgate.?, ctx.xn, ctx.z);
            // conv1d over qkv (pre-activation), state = 3 previous
            const w = f32Data(g, L.conv1d.?);
            const st = ctx.conv_state[ssm_idx];
            for (0..n_qkv) |ch| {
                const acc = w[ch * 4 + 0] * st[ch] + w[ch * 4 + 1] * st[n_qkv + ch] + w[ch * 4 + 2] * st[2 * n_qkv + ch] + w[ch * 4 + 3] * ctx.qkv[ch];
                ctx.cs[ch] = siluF(acc);
            }
            // shift state and store current qkv
            @memcpy(st[0..n_qkv], st[n_qkv .. 2 * n_qkv]);
            @memcpy(st[n_qkv .. 2 * n_qkv], st[2 * n_qkv .. 3 * n_qkv]);
            @memcpy(st[2 * n_qkv .. 3 * n_qkv], ctx.qkv[0..n_qkv]);
            // l2norm over q,k (first 4096 as 32 heads)
            {
                var h: usize = 0;
                while (h < 32) : (h += 1) {
                    const seg = ctx.cs[h * qk_dim ..][0..qk_dim];
                    var ss: f32 = 0;
                    for (seg) |v| ss += v * v;
                    const inv = 1.0 / @sqrt(ss + 1e-6);
                    for (seg) |*v| v.* *= inv;
                }
            }
            const a_data = f32Data(g, L.ssm_a.?);
            const dt_data = f32Data(g, L.dt_bias.?);
            for (0..32) |h| {
                const sp = @log(1.0 + @exp(ctx.alpha[h] + dt_data[h]));
                ctx.gbuf[h] = sp * a_data[h];
            }
            for (0..32) |h| ctx.beta[h] = sigmoidF(ctx.beta[h]);
            // gated delta rule
            const S = ctx.ssm_state[ssm_idx];
            var h: usize = 0;
            while (h < n_v_heads) : (h += 1) {
                const kh = h % n_k_heads;
                const qh = ctx.cs[kh * qk_dim ..][0..qk_dim];
                const kk = ctx.cs[(n_k_heads + kh) * qk_dim ..][0..qk_dim];
                const vv = ctx.cs[4096 + h * qk_dim ..][0..qk_dim];
                const Sh = S[h * ssm_state_dim ..][0..ssm_state_dim];
                const dec = @exp(ctx.gbuf[h]);
                var dv: usize = 0;
                while (dv < qk_dim) : (dv += 1) {
                    const row = Sh[dv * qk_dim ..][0..qk_dim];
                    for (row) |*s| s.* *= dec;
                }
                var delta: [qk_dim]f32 = undefined;
                dv = 0;
                while (dv < qk_dim) : (dv += 1) {
                    const row = Sh[dv * qk_dim ..][0..qk_dim];
                    var acc: @Vector(8, f32) = @splat(0.0);
                    var dk: usize = 0;
                    while (dk < qk_dim) : (dk += 8) {
                        const a: @Vector(8, f32) = row[dk..][0..8].*;
                        const b: @Vector(8, f32) = kk[dk..][0..8].*;
                        acc += a * b;
                    }
                    const pred = @reduce(.Add, acc);
                    delta[dv] = ctx.beta[h] * (vv[dv] - pred);
                }
                dv = 0;
                while (dv < qk_dim) : (dv += 1) {
                    const row = Sh[dv * qk_dim ..][0..qk_dim];
                    const dd = delta[dv];
                    var dk: usize = 0;
                    while (dk < qk_dim) : (dk += 8) {
                        const kv: @Vector(8, f32) = kk[dk..][0..8].*;
                        const rv: @Vector(8, f32) = row[dk..][0..8].*;
                        row[dk..][0..8].* = rv + kv * @as(@Vector(8, f32), @splat(dd));
                    }
                }
                var obuf: [qk_dim]f32 = undefined;
                dv = 0;
                while (dv < qk_dim) : (dv += 1) {
                    const row = Sh[dv * qk_dim ..][0..qk_dim];
                    var acc: @Vector(8, f32) = @splat(0.0);
                    var dk: usize = 0;
                    while (dk < qk_dim) : (dk += 8) {
                        const a: @Vector(8, f32) = row[dk..][0..8].*;
                        const b: @Vector(8, f32) = qh[dk..][0..8].*;
                        acc += a * b;
                    }
                    obuf[dv] = @reduce(.Add, acc) * gdn_scale;
                }
                // rmsnorm over v head, times silu(z)
                const ssw = f32Data(g, L.ssm_norm.?);
                var ss: f32 = 0;
                for (obuf) |v| ss += v * v;
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(qk_dim)) + eps);
                const headpos = 2 * (h % 16) + h / 16;
                var d: usize = 0;
                while (d < qk_dim) : (d += 1) {
                    ctx.out1[headpos * qk_dim + d] = obuf[d] * inv * ssw[d] * siluF(ctx.z[h * qk_dim + d]);
                }
            }
            transformSignH(ctx.out1, m.signs4096);
            matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.ssout.?, ctx.out1, ctx.out2);
            ssm_idx += 1;
        }
        // residual
        for (0..n_embd) |i| ctx.x[i] += ctx.out2[i];
        // post attention norm + ffn
        rmsnorm(ctx.x, f32Data(g, L.post_attn_norm), eps, ctx.xn);
        transformSignH(ctx.xn, m.signs2560);
        matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.ffn_gate, ctx.xn, ctx.ffg);
        matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.ffn_up, ctx.xn, ctx.ffu);
        for (0..n_ff) |i| ctx.sw[i] = siluF(ctx.ffg[i]) * ctx.ffu[i];
        transformSignH(ctx.sw, m.signs9216);
        matvecPtq(ctx.model.pool, ctx.model.gpu, g, L.ffn_down, ctx.sw, ctx.out2);
        for (0..n_embd) |i| ctx.x[i] += ctx.out2[i];
        _ = il;
    }
    ctx.pos += 1;
    if (want_logits) {
        rmsnorm(ctx.x, f32Data(g, m.output_norm), eps, ctx.xn);
        transformSignH(ctx.xn, m.signs2560);
        matvecPtq(ctx.model.pool, ctx.model.gpu, g, m.token_embd, ctx.xn, ctx.logits);
    }
    return 0;
}

// ---------------- ABI ----------------

fn asModel(p: ?*anyopaque) ?*Model {
    return @ptrCast(@alignCast(p));
}
fn asContext(p: ?*anyopaque) ?*Context {
    return @ptrCast(@alignCast(p));
}

export fn bonsai_abi_version() callconv(.c) u32 {
    return 1;
}

export fn bonsai_last_error() callconv(.c) ?[*:0]const u8 {
    if (err_len == 0) return null;
    return @ptrCast(&err_buf);
}

export fn bonsai_model_open(gguf_path: ?[*:0]const u8, out_model: ?*?*anyopaque) callconv(.c) i32 {
    clearErr();
    if (out_model) |p| p.* = null;
    const pathp = gguf_path orelse {
        setErr("null gguf path", .{});
        return E_INVALID;
    };
    if (out_model == null) {
        setErr("null out_model", .{});
        return E_INVALID;
    }
    const path = std.mem.span(pathp);
    const m = openModel(path) catch |e| {
        setErr("failed to open model: {s}", .{@errorName(e)});
        return E_IO;
    };
    out_model.?.* = @ptrCast(m);
    return 0;
}

export fn bonsai_model_close(model: ?*anyopaque) callconv(.c) void {
    clearErr();
    if (asModel(model)) |m| closeModel(m);
}

export fn bonsai_n_vocab(model: ?*const anyopaque) callconv(.c) i32 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse {
        setErr("null model", .{});
        return E_INVALID;
    };
    return m.n_vocab;
}

export fn bonsai_n_ctx_train(model: ?*const anyopaque) callconv(.c) i32 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse {
        setErr("null model", .{});
        return E_INVALID;
    };
    return m.g.n_ctx_train;
}

export fn bonsai_tokenize(model: ?*const anyopaque, utf8: ?[*:0]const u8, add_bos: i32, ids: ?[*]i32, capacity: i32) callconv(.c) i32 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse {
        setErr("null model", .{});
        return E_INVALID;
    };
    const up = utf8 orelse {
        setErr("null text", .{});
        return E_INVALID;
    };
    const text = std.mem.span(up);
    const mm: *Model = @constCast(m);
    const out = mm.tok.tokenize(m.alloc, text, add_bos != 0) catch |e| {
        setErr("tokenize failed: {s}", .{@errorName(e)});
        return E_INTERNAL;
    };
    defer m.alloc.free(out);
    const n: i32 = @intCast(out.len);
    if (ids == null) return n;
    if (capacity < n) {
        setErr("capacity {d} < required {d}", .{ capacity, n });
        return E_CAPACITY;
    }
    @memcpy(ids.?[0..out.len], out);
    return n;
}

export fn bonsai_tensor_count(model: ?*const anyopaque) callconv(.c) i32 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse return E_INVALID;
    return @intCast(m.g.tensors.len);
}

export fn bonsai_tensor_info(model: ?*const anyopaque, index: i32, name: ?*?[*:0]const u8, ggml_type: ?*i32, rank: ?*i32, ne: ?[*]i64) callconv(.c) i32 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse return E_INVALID;
    if (index < 0 or index >= @as(i32, @intCast(m.g.tensors.len))) {
        setErr("tensor index out of range", .{});
        return E_INVALID;
    }
    const t = &m.g.tensors[@intCast(index)];
    if (name) |np| np.* = @ptrCast(t.name.ptr);
    if (ggml_type) |gp| gp.* = @intCast(t.ggml_type);
    if (rank) |rp| rp.* = @intCast(t.n_dims);
    if (ne) |nep| {
        for (0..4) |i| nep[i] = @intCast(t.ne[i]);
    }
    return 0;
}

export fn bonsai_tensor_dequant(model: ?*const anyopaque, name: ?[*:0]const u8, out: ?[*]f32, capacity: i64) callconv(.c) i64 {
    clearErr();
    const p: ?*const Model = @ptrCast(@alignCast(model));
    const m = p orelse return E_INVALID;
    const np = name orelse return E_INVALID;
    const t = m.g.lookup(std.mem.span(np)) orelse {
        setErr("unknown tensor", .{});
        return E_INVALID;
    };
    const count = ggufmod.elemCount(t);
    if (capacity < @as(i64, @intCast(count))) {
        setErr("capacity {d} < required {d}", .{ capacity, count });
        return E_CAPACITY;
    }
    const op = out orelse return E_INVALID;
    if (t.ggml_type == 0) {
        const raw = m.g.tensorData(t);
        const src: [*]const f32 = @ptrCast(@alignCast(raw.ptr));
        @memcpy(op[0..count], src[0..count]);
    } else if (t.ggml_type == 1) {
        const raw = m.g.tensorData(t);
        for (0..count) |i| {
            op[i] = f16Of(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little));
        }
    } else if (t.ggml_type == 143) {
        const cols: usize = @intCast(t.ne[0]);
        const rows = count / cols;
        const nblk = cols / 128;
        const raw = m.g.tensorData(t);
        var blk: [128]f32 = undefined;
        for (0..rows) |r| {
            for (0..nblk) |b| {
                ggufmod.dequantBlock(raw[(r * nblk + b) * 28 ..][0..28], &blk);
                @memcpy(op[r * cols + b * 128 ..][0..128], &blk);
            }
        }
    } else {
        setErr("unsupported tensor type {d}", .{t.ggml_type});
        return E_UNSUPPORTED;
    }
    return @intCast(count);
}

export fn bonsai_context_create(model: ?*anyopaque, n_ctx: i32, n_batch: i32, out_ctx: ?*?*anyopaque) callconv(.c) i32 {
    clearErr();
    if (out_ctx) |p| p.* = null;
    const m = asModel(model) orelse {
        setErr("null model", .{});
        return E_INVALID;
    };
    if (out_ctx == null) return E_INVALID;
    const ctx = createContext(m, n_ctx, n_batch) catch |e| {
        setErr("context create failed: {s}", .{@errorName(e)});
        return if (e == error.BadArg) E_INVALID else E_INTERNAL;
    };
    out_ctx.?.* = @ptrCast(ctx);
    return 0;
}

export fn bonsai_context_close(ctx: ?*anyopaque) callconv(.c) void {
    clearErr();
    if (asContext(ctx)) |c| destroyContext(c);
}

export fn bonsai_context_reset(ctx: ?*anyopaque) callconv(.c) i32 {
    clearErr();
    const c = asContext(ctx) orelse return E_INVALID;
    resetContext(c);
    return 0;
}

export fn bonsai_eval(ctx: ?*anyopaque, tokens: ?[*]const i32, n_tokens: i32, logits: ?[*]f32, logits_capacity: i32) callconv(.c) i32 {
    clearErr();
    const c = asContext(ctx) orelse {
        setErr("null context", .{});
        return E_INVALID;
    };
    if (n_tokens < 1) {
        setErr("n_tokens must be >= 1", .{});
        return E_INVALID;
    }
    if (tokens == null) {
        setErr("null tokens", .{});
        return E_INVALID;
    }
    if (logits != null and logits_capacity < c.model.n_vocab) {
        setErr("logits capacity too small", .{});
        return E_CAPACITY;
    }
    if (c.pos + n_tokens > c.n_ctx) {
        setErr("context overflow: pos {d} + {d} > {d}", .{ c.pos, n_tokens, c.n_ctx });
        return E_STATE;
    }
    const n: usize = @intCast(n_tokens);
    for (0..n) |i| {
        const last = (i + 1) == n;
        const rc = evalToken(c, tokens.?[i], last and logits != null);
        if (rc != 0) return rc;
    }
    if (logits) |lp| {
        @memcpy(lp[0..@intCast(c.model.n_vocab)], c.logits);
    }
    return 0;
}

fn argmaxFloats(v: []const f32) i32 {
    var best: i32 = 0;
    var bestv: f32 = v[0];
    for (v[1..], 1..) |x, i| {
        if (x > bestv) {
            bestv = x;
            best = @intCast(i);
        }
    }
    return best;
}

export fn bonsai_generate_greedy(ctx: ?*anyopaque, prompt: ?[*]const i32, n_prompt: i32, n_predict: i32, flags: u32, out_ids: ?[*]i32) callconv(.c) i32 {
    clearErr();
    const c = asContext(ctx) orelse {
        setErr("null context", .{});
        return E_INVALID;
    };
    if (n_predict < 0 or n_prompt < 0) return E_INVALID;
    if (n_predict > 0 and out_ids == null) {
        setErr("null out_ids", .{});
        return E_INVALID;
    }
    if (n_prompt > 0) {
        if (prompt == null) return E_INVALID;
        if (c.pos + n_prompt > c.n_ctx) return E_STATE;
        const n: usize = @intCast(n_prompt);
        for (0..n) |i| {
            const last = (i + 1) == n;
            const rc = evalToken(c, prompt.?[i], last);
            if (rc != 0) return rc;
        }
    } else if (c.pos == 0) {
        setErr("empty prompt with empty context", .{});
        return E_INVALID;
    }
    const ignore_eos = (flags & 1) != 0;
    const eos = c.model.g.eos_id;
    var generated: i32 = 0;
    var i: i32 = 0;
    while (i < n_predict) : (i += 1) {
        if (c.pos >= c.n_ctx) break;
        const id = argmaxFloats(c.logits[0..@intCast(c.model.n_vocab)]);
        if (!ignore_eos and eos >= 0 and id == eos) break;
        out_ids.?[@intCast(generated)] = id;
        generated += 1;
        const rc = evalToken(c, id, true);
        if (rc != 0) return rc;
    }
    return generated;
}
