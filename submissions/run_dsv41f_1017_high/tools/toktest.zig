const std = @import("std");
const tokmod = @import("../src/tokenizer.zig");

const Vector = struct { text: []const u8, ids: []i32 };

fn readAll(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    var f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    return try f.readToEndAlloc(alloc, 1 << 30);
}

fn hexDecode(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
    const out = try alloc.alloc(u8, s.len / 2);
    _ = try std.fmt.hexToBytes(out, s);
    return out;
}

fn loadHexLines(alloc: std.mem.Allocator, path: []const u8) ![]const []const u8 {
    const data = try readAll(alloc, path);
    var list = std.ArrayList([]const u8).empty;
    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try list.append(alloc, try hexDecode(alloc, line));
    }
    return try list.toOwnedSlice(alloc);
}

fn loadTypes(alloc: std.mem.Allocator, path: []const u8) ![]const u32 {
    const data = try readAll(alloc, path);
    var list = std.ArrayList(u32).empty;
    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try list.append(alloc, try std.fmt.parseInt(u32, line, 10));
    }
    return try list.toOwnedSlice(alloc);
}

fn writeOut(data: []const u8) void {
    std.fs.File.stdout().writeAll(data) catch {};
}

pub fn main() !void {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const alloc = arena_state.allocator();

    const tokens = try loadHexLines(alloc, "/workspace/tools/tokdata/tokens.hex");
    const merges = try loadHexLines(alloc, "/workspace/tools/tokdata/merges.hex");
    const types = try loadTypes(alloc, "/workspace/tools/tokdata/types.txt");

    var timer = try std.time.Timer.start();
    var tk = try tokmod.Tokenizer.init(alloc, tokens, types, merges, 248044, 248046, false);
    const init_ns = timer.read();
    defer tk.deinit();

    var out_buf = std.ArrayList(u8).empty;

    // ---- visible JSON vectors ----
    const vdata = try readAll(alloc, "/workspace/TOKENIZER-VECTORS.json");
    const parsed = try std.json.parseFromSlice([]Vector, alloc, vdata, .{});
    var pass: usize = 0;
    var fail: usize = 0;
    for (parsed.value) |v| {
        const got = try tk.tokenize(alloc, v.text, false);
        var ok = got.len == v.ids.len;
        if (ok) {
            for (got, v.ids) |g, e| {
                if (g != e) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            pass += 1;
        } else {
            fail += 1;
            try out_buf.writer(alloc).print("FAIL vector text={s}\n  got=", .{v.text});
            for (got) |g| try out_buf.writer(alloc).print("{d} ", .{g});
            try out_buf.writer(alloc).print("\n  exp=");
            for (v.ids) |e| try out_buf.writer(alloc).print("{d} ", .{e});
            try out_buf.writer(alloc).print("\n", .{});
        }
    }
    try out_buf.writer(alloc).print("VECTORS pass={d} fail={d}\n", .{ pass, fail });

    // timing: corpus
    const corpus = try readAll(alloc, "/workspace/data/samples/corpus.txt");
    timer.reset();
    const cids = try tk.tokenize(alloc, corpus, false);
    const corpus_ns = timer.read();
    try out_buf.writer(alloc).print("corpus bytes={d} ids={d} time_ms={d}\n", .{ corpus.len, cids.len, corpus_ns / 1_000_000 });
    try out_buf.writer(alloc).print("init_ms={d}\n", .{init_ns / 1_000_000});

    // ---- optional batch: one hex text per line ----
    if (std.fs.cwd().openFile("/workspace/tools/tokdata/testhex.txt", .{})) |f| {
        defer f.close();
        const bdata = try f.readToEndAlloc(alloc, 1 << 30);
        var it = std.mem.splitScalar(u8, bdata, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            const text = try hexDecode(alloc, line);
            const ids = try tk.tokenize(alloc, text, false);
            try out_buf.writer(alloc).print("ok {d}", .{ids.len});
            for (ids) |id| try out_buf.writer(alloc).print(" {d}", .{id});
            try out_buf.writer(alloc).append(alloc, '\n');
        }
    } else |_| {}

    writeOut(out_buf.items);
}
