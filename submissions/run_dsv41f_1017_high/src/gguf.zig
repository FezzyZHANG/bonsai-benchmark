const std = @import("std");

pub const Tensor = struct {
    name: []const u8,
    ggml_type: u32,
    n_dims: u32,
    ne: [4]u64,
    offset: u64,
};

pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

pub const Reader = struct {
    buf: []const u8,
    pos: usize = 0,
    ok: bool = true,

    fn need(self: *Reader, n: usize) bool {
        if (self.pos + n > self.buf.len) {
            self.ok = false;
            return false;
        }
        return true;
    }
    pub fn u8v(self: *Reader) u8 {
        if (!self.need(1)) return 0;
        const v = self.buf[self.pos];
        self.pos += 1;
        return v;
    }
    pub fn u16v(self: *Reader) u16 {
        if (!self.need(2)) return 0;
        const v = std.mem.readInt(u16, self.buf[self.pos..][0..2], .little);
        self.pos += 2;
        return v;
    }
    pub fn u32v(self: *Reader) u32 {
        if (!self.need(4)) return 0;
        const v = std.mem.readInt(u32, self.buf[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }
    pub fn i32v(self: *Reader) i32 {
        return @bitCast(self.u32v());
    }
    pub fn u64v(self: *Reader) u64 {
        if (!self.need(8)) return 0;
        const v = std.mem.readInt(u64, self.buf[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }
    pub fn i64v(self: *Reader) i64 {
        return @bitCast(self.u64v());
    }
    pub fn f32v(self: *Reader) f32 {
        return @bitCast(self.u32v());
    }
    pub fn f64v(self: *Reader) f64 {
        return @bitCast(self.u64v());
    }
    pub fn str(self: *Reader) []const u8 {
        const n = self.u64v();
        if (n > self.buf.len or !self.need(@intCast(n))) return "";
        const s = self.buf[self.pos .. self.pos + @as(usize, @intCast(n))];
        self.pos += @intCast(n);
        return s;
    }
    pub fn skip(self: *Reader, t: ValueType) void {
        switch (t) {
            .u8, .i8, .bool => _ = self.u8v(),
            .u16, .i16 => _ = self.u16v(),
            .u32, .i32, .f32 => _ = self.u32v(),
            .u64, .i64, .f64 => _ = self.u64v(),
            .string => _ = self.str(),
            .array => {
                const et: ValueType = @enumFromInt(self.u32v());
                const n = self.u64v();
                var i: u64 = 0;
                while (i < n and self.ok) : (i += 1) self.skip(et);
            },
        }
    }
};

pub const Gguf = struct {
    alloc: std.mem.Allocator,
    data: []const u8,
    tensors: []Tensor,
    data_offset: u64,
    alignment: u64 = 32,

    // metadata we care about
    n_vocab: i32 = 0,
    n_ctx_train: i32 = 0,
    n_layer: i32 = 0,
    n_embd: i32 = 0,
    n_ff: i32 = 0,
    n_head: i32 = 0,
    n_head_kv: i32 = 0,
    n_rot: i32 = 0,
    key_length: i32 = 0,
    value_length: i32 = 0,
    eps: f32 = 1e-6,
    rope_freq_base: f32 = 10000.0,
    conv_kernel: i32 = 4,
    ssm_state: i32 = 0,
    ssm_groups: i32 = 0,
    ssm_dt_rank: i32 = 0,
    ssm_inner: i32 = 0,
    full_attn_interval: i32 = 4,
    bos_id: i32 = -1,
    eos_id: i32 = -1,
    add_bos: bool = false,
    tokens: [][]const u8 = &.{},
    token_types: []const u32 = &.{},
    merges: [][]const u8 = &.{},
    signs: []const f32 = &.{},
    sign_widths: [4]i32 = .{ 0, 0, 0, 0 },
    n_sign_widths: usize = 0,
    block_size: i32 = 512,

    pub fn lookup(self: *const Gguf, name: []const u8) ?*const Tensor {
        for (self.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn tensorData(self: *const Gguf, t: *const Tensor) []const u8 {
        const off = self.data_offset + t.offset;
        const end = @min(self.data.len, off + nbytes(t));
        return self.data[@intCast(off)..@intCast(end)];
    }
};

pub fn elemCount(t: *const Tensor) usize {
    var n: usize = 1;
    for (t.ne) |d| n *= @intCast(d);
    return n;
}

pub fn nbytes(t: *const Tensor) usize {
    return switch (t.ggml_type) {
        0 => elemCount(t) * 4,
        1 => elemCount(t) * 2,
        143 => (@as(usize, @intCast(t.ne[0])) / 128 * 28) * (elemCount(t) / @as(usize, @intCast(t.ne[0]))),
        else => elemCount(t) * 4,
    };
}

pub const pow3 = [5]u8{ 1, 3, 9, 27, 81 };

/// Decode one PTQ1_0 block (28 bytes) into 128 f32 values (scale applied).
pub inline fn dequantBlock(blk: []const u8, out: *[128]f32) void {
    const d: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[26..28], .little))));
    // chunk 0: bytes 0..16, 5 digits, digit-major over 16
    var idx: usize = 0;
    inline for (0..5) |dg| {
        inline for (0..16) |j| {
            const v: u8 = blk[j] *% @as(u8, pow3[dg]);
            const q: i32 = @as(i32, (@as(u16, v) * 3) >> 8) - 1;
            out[idx] = d * @as(f32, @floatFromInt(q));
            idx += 1;
        }
    }
    inline for (0..5) |dg| {
        inline for (0..8) |j| {
            const v: u8 = blk[16 + j] *% @as(u8, pow3[dg]);
            const q: i32 = @as(i32, (@as(u16, v) * 3) >> 8) - 1;
            out[idx] = d * @as(f32, @floatFromInt(q));
            idx += 1;
        }
    }
    inline for (0..4) |dg| {
        inline for (0..2) |j| {
            const v: u8 = blk[24 + j] *% @as(u8, pow3[dg]);
            const q: i32 = @as(i32, (@as(u16, v) * 3) >> 8) - 1;
            out[idx] = d * @as(f32, @floatFromInt(q));
            idx += 1;
        }
    }
}

pub const Error = error{ BadGguf, OutOfMemory };

pub fn parse(alloc: std.mem.Allocator, data: []const u8) Error!*Gguf {
    const g = try alloc.create(Gguf);
    g.* = .{ .alloc = alloc, .data = data, .tensors = &.{}, .data_offset = 0 };
    var r = Reader{ .buf = data };
    if (!std.mem.eql(u8, data[0..4], "GGUF")) return error.BadGguf;
    r.pos = 4;
    const version = r.u32v();
    if (version < 2 or version > 3) return error.BadGguf;
    const n_tensors = r.u64v();
    const n_kv = r.u64v();

    var tok_list = std.ArrayList([]const u8).empty;
    var merge_list = std.ArrayList([]const u8).empty;
    var type_list = std.ArrayList(u32).empty;
    var sign_list = std.ArrayList(f32).empty;

    var ki: u64 = 0;
    while (ki < n_kv) : (ki += 1) {
        const key = r.str();
        if (!r.ok) return error.BadGguf;
        const vt: ValueType = @enumFromInt(r.u32v());
        if (std.mem.eql(u8, key, "tokenizer.ggml.tokens")) {
            const et: ValueType = @enumFromInt(r.u32v());
            _ = et;
            const n = r.u64v();
            try tok_list.ensureTotalCapacity(alloc, @intCast(n));
            var i: u64 = 0;
            while (i < n) : (i += 1) tok_list.appendAssumeCapacity(r.str());
            g.n_vocab = @intCast(n);
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.merges")) {
            const et: ValueType = @enumFromInt(r.u32v());
            _ = et;
            const n = r.u64v();
            try merge_list.ensureTotalCapacity(alloc, @intCast(n));
            var i: u64 = 0;
            while (i < n) : (i += 1) merge_list.appendAssumeCapacity(r.str());
        } else if (std.mem.eql(u8, key, "tokenizer.ggml.token_type")) {
            const et: ValueType = @enumFromInt(r.u32v());
            _ = et;
            const n = r.u64v();
            try type_list.ensureTotalCapacity(alloc, @intCast(n));
            var i: u64 = 0;
            while (i < n) : (i += 1) type_list.appendAssumeCapacity(r.u32v());
        } else if (std.mem.eql(u8, key, "prism.hadamard.sign_values")) {
            const et: ValueType = @enumFromInt(r.u32v());
            _ = et;
            const n = r.u64v();
            try sign_list.ensureTotalCapacity(alloc, @intCast(n));
            var i: u64 = 0;
            while (i < n) : (i += 1) sign_list.appendAssumeCapacity(@floatFromInt(r.i32v()));
        } else if (std.mem.eql(u8, key, "prism.hadamard.sign_widths")) {
            const et: ValueType = @enumFromInt(r.u32v());
            _ = et;
            const n = r.u64v();
            g.n_sign_widths = @intCast(n);
            var i: u64 = 0;
            while (i < n) : (i += 1) {
                if (i < 4) g.sign_widths[@intCast(i)] = r.i32v();
            }
        } else if (std.mem.eql(u8, key, "prism.hadamard.block_size")) {
            g.block_size = r.i32v();
        } else if (vt == .string) {
            const s = r.str();
            try setString(g, key, s);
        } else if (vt == .u32 or vt == .i32 or vt == .f32) {
            const v = r.u32v();
            try setScalar(g, key, v, vt);
        } else if (vt == .bool) {
            _ = r.u8v();
        } else {
            r.skip(vt);
        }
        if (!r.ok) return error.BadGguf;
    }
    g.tokens = try tok_list.toOwnedSlice(alloc);
    g.merges = try merge_list.toOwnedSlice(alloc);
    g.token_types = try type_list.toOwnedSlice(alloc);
    g.signs = try sign_list.toOwnedSlice(alloc);

    g.tensors = try alloc.alloc(Tensor, @intCast(n_tensors));
    for (g.tensors) |*t| {
        const raw_name = r.str();
        const zname = try alloc.alloc(u8, raw_name.len + 1);
        @memcpy(zname[0..raw_name.len], raw_name);
        zname[raw_name.len] = 0;
        t.name = zname[0..raw_name.len];
        t.n_dims = r.u32v();
        t.ne = .{ 1, 1, 1, 1 };
        var d: u32 = 0;
        while (d < t.n_dims and d < 4) : (d += 1) t.ne[d] = r.u64v();
        t.ggml_type = r.u32v();
        t.offset = r.u64v();
    }
    const align_to: u64 = if (g.alignment == 0) 32 else g.alignment;
    g.data_offset = std.mem.alignForward(u64, r.pos, align_to);
    if (g.data_offset > data.len) return error.BadGguf;
    return g;
}

fn setString(g: *Gguf, key: []const u8, s: []const u8) Error!void {
    _ = g;
    _ = key;
    _ = s;
}

fn setScalar(g: *Gguf, key: []const u8, v: u32, vt: ValueType) Error!void {
    const as_i: i32 = @bitCast(v);
    const as_f: f32 = @bitCast(v);
    if (std.mem.eql(u8, key, "qwen35.context_length")) g.n_ctx_train = as_i else if (std.mem.eql(u8, key, "qwen35.block_count")) g.n_layer = as_i else if (std.mem.eql(u8, key, "qwen35.embedding_length")) g.n_embd = as_i else if (std.mem.eql(u8, key, "qwen35.feed_forward_length")) g.n_ff = as_i else if (std.mem.eql(u8, key, "qwen35.attention.head_count")) g.n_head = as_i else if (std.mem.eql(u8, key, "qwen35.attention.head_count_kv")) g.n_head_kv = as_i else if (std.mem.eql(u8, key, "qwen35.attention.key_length")) g.key_length = as_i else if (std.mem.eql(u8, key, "qwen35.attention.value_length")) g.value_length = as_i else if (std.mem.eql(u8, key, "qwen35.rope.dimension_count")) g.n_rot = as_i else if (std.mem.eql(u8, key, "qwen35.attention.layer_norm_rms_epsilon")) g.eps = as_f else if (std.mem.eql(u8, key, "qwen35.rope.freq_base")) g.rope_freq_base = as_f else if (std.mem.eql(u8, key, "qwen35.ssm.conv_kernel")) g.conv_kernel = as_i else if (std.mem.eql(u8, key, "qwen35.ssm.state_size")) g.ssm_state = as_i else if (std.mem.eql(u8, key, "qwen35.ssm.group_count")) g.ssm_groups = as_i else if (std.mem.eql(u8, key, "qwen35.ssm.time_step_rank")) g.ssm_dt_rank = as_i else if (std.mem.eql(u8, key, "qwen35.ssm.inner_size")) g.ssm_inner = as_i else if (std.mem.eql(u8, key, "qwen35.full_attention_interval")) g.full_attn_interval = as_i else if (std.mem.eql(u8, key, "tokenizer.ggml.bos_token_id")) g.bos_id = as_i else if (std.mem.eql(u8, key, "tokenizer.ggml.eos_token_id")) g.eos_id = as_i else if (std.mem.eql(u8, key, "tokenizer.ggml.add_bos_token")) g.add_bos = (v != 0) else if (std.mem.eql(u8, key, "general.alignment")) g.alignment = @intCast(v);
    _ = vt;
}
