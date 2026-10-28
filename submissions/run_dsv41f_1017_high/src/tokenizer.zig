//! GPT-2 byte-level BPE tokenizer with the "qwen35" pre-tokenizer regex.
//!
//! Public interface:
//!   Tokenizer.init(alloc, tokens, token_types, merges, bos_id, eos_id, add_bos) !Tokenizer
//!   Tokenizer.deinit(*Tokenizer) void
//!   Tokenizer.tokenize(alloc, text, add_bos) ![]i32
//!
//! Token strings in the GGUF are already in the GPT-2 byte-to-unicode
//! representation.  Merges are "left right" strings.

const std = @import("std");
const uni = @import("unicode_tables.zig");

// ---------------------------------------------------------------------------
// merge pair key
// ---------------------------------------------------------------------------

const Pair = struct { left: []const u8, right: []const u8 };

const PairCtx = struct {
    pub fn hash(_: PairCtx, p: Pair) u64 {
        const h1 = std.hash.Wyhash.hash(0x9e3779b97f4a7c15, p.left);
        const h2 = std.hash.Wyhash.hash(0xc2b2ae3d27d4eb4f, p.right);
        return (h1 *% 0x100000001b3) ^ h2;
    }
    pub fn eql(_: PairCtx, a: Pair, b: Pair) bool {
        return std.mem.eql(u8, a.left, b.left) and std.mem.eql(u8, a.right, b.right);
    }
};

const Special = struct {
    text: []const u8,
    id: i32,
};

const Node = struct {
    text: []const u8,
    prev: i32,
    next: i32,
    pos: u32,
    alive: bool,
};

const HeapEntry = struct {
    rank: u32,
    pos: u32,
    a: u32,
    b: u32,
};

fn heapLess(x: HeapEntry, y: HeapEntry) bool {
    if (x.rank != y.rank) return x.rank < y.rank;
    return x.pos < y.pos;
}

fn heapPush(heap: *std.ArrayList(HeapEntry), alloc: std.mem.Allocator, e: HeapEntry) !void {
    try heap.append(alloc, e);
    var i = heap.items.len - 1;
    while (i > 0) {
        const parent = (i - 1) / 2;
        if (!heapLess(heap.items[i], heap.items[parent])) break;
        std.mem.swap(HeapEntry, &heap.items[i], &heap.items[parent]);
        i = parent;
    }
}

fn heapPop(heap: *std.ArrayList(HeapEntry)) HeapEntry {
    const top = heap.items[0];
    const n_after = heap.items.len - 1;
    if (n_after == 0) {
        heap.items.len = 0;
        return top;
    }
    heap.items[0] = heap.items[n_after];
    heap.items.len = n_after;
    var i: usize = 0;
    const n = heap.items.len;
    while (true) {
        var c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n and heapLess(heap.items[c + 1], heap.items[c])) c += 1;
        if (!heapLess(heap.items[c], heap.items[i])) break;
        std.mem.swap(HeapEntry, &heap.items[i], &heap.items[c]);
        i = c;
    }
    return top;
}

// ---------------------------------------------------------------------------
// GPT-2 byte -> unicode code point
// ---------------------------------------------------------------------------

fn byteToCp(b: u8) u21 {
    // Standard GPT-2 mapping:
    //   33..126, 161..172, 174..255  -> identical code point
    //   everything else              -> 256 + n  (n in increasing byte order)
    if (b >= 33 and b <= 126) return b;
    if (b >= 161 and b <= 172) return b;
    if (b >= 174) return b;
    if (b <= 32) return 256 + @as(u21, b);
    if (b <= 160) return 162 + @as(u21, b); // 289..322
    return 323; // b == 173
}

// ---------------------------------------------------------------------------
// unicode category helpers
// ---------------------------------------------------------------------------

fn inRanges(ranges: []const [2]u21, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r[0]) {
            hi = mid;
        } else if (cp > r[1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

fn isLetter(cp: u21) bool {
    return inRanges(&uni.letters, cp);
}

fn isMark(cp: u21) bool {
    return inRanges(&uni.marks, cp);
}

fn isNumber(cp: u21) bool {
    return inRanges(&uni.numbers, cp);
}

fn isWhitespace(cp: u21) bool {
    return inRanges(&uni.whitespace_ranges, cp);
}

fn isLm(cp: u21) bool {
    return isLetter(cp) or isMark(cp);
}

const Decoded = struct {
    cp: u21,
    len: usize,
};

fn decodeAt(text: []const u8, pos: usize) Decoded {
    const b = text[pos];
    if (b < 0x80) return .{ .cp = b, .len = 1 };
    const l = std.unicode.utf8ByteSequenceLength(b) catch return .{ .cp = 0xFFFD, .len = 1 };
    if (pos + l > text.len) return .{ .cp = 0xFFFD, .len = 1 };
    const cp = std.unicode.utf8Decode(text[pos .. pos + l]) catch return .{ .cp = 0xFFFD, .len = 1 };
    return .{ .cp = cp, .len = l };
}

fn cpLenAt(text: []const u8, pos: usize) usize {
    return decodeAt(text, pos).len;
}

// ---------------------------------------------------------------------------
// qwen35 pre-tokenizer
// ---------------------------------------------------------------------------
//
// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
// |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
// |\p{N}
// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
// |\s*[\r\n]+
// |\s+(?!\S)
// |\s+
//
// Every code point is matched by at least one alternative, so a simple
// left-to-right scan at each position (leftmost-first alternation, greedy with
// the standard backtracking) covers the whole input.

fn alt4Class(cp: u21) bool {
    return !(isWhitespace(cp) or isLetter(cp) or isMark(cp) or isNumber(cp));
}

fn lmRunEnd(text: []const u8, start: usize) usize {
    var p = start;
    while (p < text.len) {
        const d = decodeAt(text, p);
        if (!isLm(d.cp)) break;
        p += d.len;
    }
    return p;
}

fn alt4RunEnd(text: []const u8, start: usize) usize {
    var p = start;
    while (p < text.len) {
        const d = decodeAt(text, p);
        if (!alt4Class(d.cp)) break;
        p += d.len;
    }
    return p;
}

/// Returns the end offset (> pos) of the pre-token match starting at pos.
fn pretokEnd(text: []const u8, pos: usize) usize {
    const d0 = decodeAt(text, pos);
    const c0 = d0.cp;
    const p1 = pos + d0.len;

    // 1) contractions
    if (c0 == '\'') {
        if (p1 < text.len) {
            const d1 = decodeAt(text, p1);
            const c1 = d1.cp;
            const p2 = p1 + d1.len;
            if (c1 == 's' or c1 == 'S' or c1 == 't' or c1 == 'T' or
                c1 == 'm' or c1 == 'M' or c1 == 'd' or c1 == 'D')
            {
                return p2;
            }
            if (c1 == 'r' or c1 == 'R' or c1 == 'v' or c1 == 'V') {
                if (p2 < text.len) {
                    const d2 = decodeAt(text, p2);
                    if (d2.cp == 'e' or d2.cp == 'E') return p2 + d2.len;
                }
            } else if (c1 == 'l' or c1 == 'L') {
                if (p2 < text.len) {
                    const d2 = decodeAt(text, p2);
                    if (d2.cp == 'l' or d2.cp == 'L') return p2 + d2.len;
                }
            }
        }
    }

    // 2) optional non-CR/LF/non-letter/non-number code point + letters/marks
    if (!(c0 == '\r' or c0 == '\n' or isLetter(c0) or isNumber(c0))) {
        if (p1 < text.len) {
            const d1 = decodeAt(text, p1);
            if (isLm(d1.cp)) return lmRunEnd(text, p1);
        }
    }
    if (isLm(c0)) return lmRunEnd(text, pos);

    // 3) a single number code point
    if (isNumber(c0)) return p1;

    // 4) optional ' ' + run of "other" code points + trailing CR/LF
    {
        var p = pos;
        var ok = false;
        if (c0 == ' ') {
            if (p1 < text.len) {
                const d1 = decodeAt(text, p1);
                if (alt4Class(d1.cp)) {
                    p = p1;
                    ok = true;
                }
            }
        } else if (alt4Class(c0)) {
            ok = true;
        }
        if (ok) {
            p = alt4RunEnd(text, p);
            while (p < text.len and (text[p] == '\r' or text[p] == '\n')) p += 1;
            return p;
        }
    }

    // whitespace run analysis for alternatives 5..7
    var wend = pos;
    var count: usize = 0;
    var last_start = pos;
    var last_crlf: ?usize = null;
    {
        var q = pos;
        while (q < text.len) {
            const d = decodeAt(text, q);
            if (!isWhitespace(d.cp)) break;
            if (d.cp == '\r' or d.cp == '\n') last_crlf = q;
            last_start = q;
            count += 1;
            q += d.len;
        }
        wend = q;
    }

    // 5) \s*[\r\n]+
    if (last_crlf) |j| return j + 1;

    // 6) \s+(?!\S): whole run at EOS, else give back the last whitespace char
    if (count >= 1) {
        if (wend == text.len) return wend;
        if (count >= 2) return last_start;
    }

    // 7) \s+
    if (count >= 1) return wend;

    // Should be unreachable: be safe and consume one code point.
    return p1;
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    tokens: []const []const u8,
    special_first: [256]bool,
    token_map: std.StringHashMapUnmanaged(i32),
    merge_map: std.HashMapUnmanaged(Pair, u32, PairCtx, 80),
    specials: []const Special,
    bos_id: i32,
    eos_id: i32,
    add_bos_default: bool,

    pub fn init(
        alloc: std.mem.Allocator,
        tokens: []const []const u8,
        token_types: []const u32,
        merges: []const []const u8,
        bos_id: i32,
        eos_id: i32,
        add_bos: bool,
    ) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(alloc);
        errdefer arena.deinit();
        const a = arena.allocator();

        // Copy token strings into one owned buffer (ids are array indexes).
        var total: usize = 0;
        for (tokens) |t| total += t.len;
        const buf = try a.alloc(u8, total);
        const owned = try a.alloc([]const u8, tokens.len);
        var off: usize = 0;
        for (tokens, 0..) |t, i| {
            @memcpy(buf[off .. off + t.len], t);
            owned[i] = buf[off .. off + t.len];
            off += t.len;
        }

        var token_map = std.StringHashMapUnmanaged(i32){};
        try token_map.ensureTotalCapacity(a, @intCast(tokens.len));
        for (owned, 0..) |t, i| {
            const gop = try token_map.getOrPut(a, t);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }

        var merge_map = std.HashMapUnmanaged(Pair, u32, PairCtx, 80){};
        try merge_map.ensureTotalCapacity(a, @intCast(merges.len));
        for (merges, 0..) |line, i| {
            const sp = std.mem.indexOfScalar(u8, line, ' ') orelse continue;
            const key = try a.dupe(u8, line);
            const p = Pair{ .left = key[0..sp], .right = key[sp + 1 ..] };
            const gop = try merge_map.getOrPut(a, p);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }

        // Special (atomic) tokens: control (3), user-defined (4), unknown (2).
        // Type 5 ("unused") tokens are not matched atomically.
        var specials = std.ArrayList(Special).empty;
        var special_first = std.mem.zeroes([256]bool);
        const n_types = @min(tokens.len, token_types.len);
        for (owned, 0..) |t, i| {
            if (i >= n_types) break;
            const ty = token_types[i];
            if (ty == 2 or ty == 3 or ty == 4) {
                if (t.len == 0) continue;
                try specials.append(a, .{ .text = t, .id = @intCast(i) });
                special_first[t[0]] = true;
            }
        }

        return .{
            .arena = arena,
            .tokens = owned,
            .special_first = special_first,
            .token_map = token_map,
            .merge_map = merge_map,
            .specials = try specials.toOwnedSlice(a),
            .bos_id = bos_id,
            .eos_id = eos_id,
            .add_bos_default = add_bos,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Returns newly allocated token ids. The caller owns the returned slice.
    /// `add_bos` is accepted for interface compatibility; this model has
    /// tokenizer.ggml.add_bos_token = false, so it never prepends BOS.
    pub fn tokenize(self: *Tokenizer, alloc: std.mem.Allocator, text: []const u8, add_bos: bool) ![]i32 {
        _ = add_bos;
        var out = std.ArrayList(i32).empty;
        errdefer out.deinit(alloc);

        var tmp = std.heap.ArenaAllocator.init(alloc);
        defer tmp.deinit();
        const ta = tmp.allocator();

        if (text.len == 0) {
            return try out.toOwnedSlice(alloc);
        }

        var seg_start: usize = 0;
        var pos: usize = 0;
        while (pos < text.len) {
            if (self.matchSpecialAt(text, pos)) |sp| {
                if (pos > seg_start) {
                    try self.tokenizeSegment(ta, text[seg_start..pos], &out, alloc);
                }
                try out.append(alloc, sp.id);
                pos += sp.text.len;
                seg_start = pos;
            } else {
                pos += cpLenAt(text, pos);
            }
        }
        if (seg_start < text.len) {
            try self.tokenizeSegment(ta, text[seg_start..], &out, alloc);
        }

        return try out.toOwnedSlice(alloc);
    }

    fn matchSpecialAt(self: *const Tokenizer, text: []const u8, pos: usize) ?Special {
        if (!self.special_first[text[pos]]) return null;
        var best: ?Special = null;
        for (self.specials) |sp| {
            if (sp.text.len > text.len - pos) continue;
            if (!std.mem.eql(u8, text[pos .. pos + sp.text.len], sp.text)) continue;
            if (best == null or sp.text.len > best.?.text.len) best = sp;
        }
        return best;
    }

    fn pairRank(self: *const Tokenizer, left: []const u8, right: []const u8) ?u32 {
        return self.merge_map.get(.{ .left = left, .right = right });
    }

    fn tokenizeSegment(
        self: *const Tokenizer,
        ta: std.mem.Allocator,
        seg: []const u8,
        out: *std.ArrayList(i32),
        out_alloc: std.mem.Allocator,
    ) !void {
        var pos: usize = 0;
        while (pos < seg.len) {
            var end = pretokEnd(seg, pos);
            if (end <= pos) end = pos + cpLenAt(seg, pos);
            try self.bpePiece(ta, seg[pos..end], out, out_alloc);
            pos = end;
        }
    }

    fn bpePiece(
        self: *const Tokenizer,
        ta: std.mem.Allocator,
        piece: []const u8,
        out: *std.ArrayList(i32),
        out_alloc: std.mem.Allocator,
    ) !void {
        const n = piece.len;
        if (n == 0) return;

        // Fast path: a single byte is always one symbol (no merges possible).
        if (n == 1) {
            const cp = byteToCp(piece[0]);
            var one: [2]u8 = undefined;
            var l: usize = 1;
            if (cp < 0x80) {
                one[0] = @intCast(cp);
            } else {
                one[0] = 0xC0 | @as(u8, @intCast(cp >> 6));
                one[1] = 0x80 | @as(u8, @intCast(cp & 0x3F));
                l = 2;
            }
            if (self.token_map.get(one[0..l])) |id| try out.append(out_alloc, id);
            return;
        }

        // Byte-level -> unicode representation (raw bytes are the symbols).
        const mapped = try ta.alloc(u8, 2 * n);
        const starts = try ta.alloc(u32, n);
        const lens = try ta.alloc(u8, n);
        var mlen: usize = 0;
        for (piece, 0..) |b, i| {
            starts[i] = @intCast(mlen);
            const cp = byteToCp(b);
            if (cp < 0x80) {
                mapped[mlen] = @intCast(cp);
                lens[i] = 1;
                mlen += 1;
            } else {
                mapped[mlen] = 0xC0 | @as(u8, @intCast(cp >> 6));
                mapped[mlen + 1] = 0x80 | @as(u8, @intCast(cp & 0x3F));
                lens[i] = 2;
                mlen += 2;
            }
        }

        var nodes = std.ArrayList(Node).empty;
        try nodes.ensureTotalCapacity(ta, n);
        for (0..n) |i| {
            const s: usize = @intCast(starts[i]);
            const l: usize = lens[i];
            try nodes.append(ta, .{
                .text = mapped[s .. s + l],
                .prev = if (i == 0) -1 else @as(i32, @intCast(i - 1)),
                .next = if (i + 1 < n) @as(i32, @intCast(i + 1)) else -1,
                .pos = @intCast(i),
                .alive = true,
            });
        }

        var heap = std.ArrayList(HeapEntry).empty;
        try heap.ensureTotalCapacity(ta, 3 * n + 8);
        var i: usize = 0;
        while (i + 1 < n) : (i += 1) {
            if (self.pairRank(nodes.items[i].text, nodes.items[i + 1].text)) |r| {
                try heapPush(&heap, ta, .{
                    .rank = r,
                    .pos = nodes.items[i].pos,
                    .a = @intCast(i),
                    .b = @intCast(i + 1),
                });
            }
        }

        while (heap.items.len > 0) {
            const e = heapPop(&heap);
            const ia: usize = e.a;
            const ib: usize = e.b;
            if (ia >= nodes.items.len or ib >= nodes.items.len) continue;
            if (!nodes.items[ia].alive or !nodes.items[ib].alive) continue;
            if (nodes.items[ia].next != @as(i32, @intCast(ib))) continue;
            const r = self.pairRank(nodes.items[ia].text, nodes.items[ib].text) orelse continue;
            if (r != e.rank) continue;

            // Merge ib into ia.
            const la = nodes.items[ia].text.len;
            const lb = nodes.items[ib].text.len;
            const nt = try ta.alloc(u8, la + lb);
            @memcpy(nt[0..la], nodes.items[ia].text);
            @memcpy(nt[la..], nodes.items[ib].text);
            const bnext = nodes.items[ib].next;
            const aprev = nodes.items[ia].prev;

            nodes.items[ia].text = nt;
            nodes.items[ib].alive = false;
            nodes.items[ia].next = bnext;
            if (bnext >= 0) nodes.items[@intCast(bnext)].prev = @intCast(ia);

            if (aprev >= 0) {
                const ip: usize = @intCast(aprev);
                if (self.pairRank(nodes.items[ip].text, nt)) |rr| {
                    try heapPush(&heap, ta, .{
                        .rank = rr,
                        .pos = nodes.items[ip].pos,
                        .a = @intCast(aprev),
                        .b = e.a,
                    });
                }
            }
            if (bnext >= 0) {
                const inx: usize = @intCast(bnext);
                if (self.pairRank(nt, nodes.items[inx].text)) |rr| {
                    try heapPush(&heap, ta, .{
                        .rank = rr,
                        .pos = nodes.items[ia].pos,
                        .a = e.a,
                        .b = @intCast(bnext),
                    });
                }
            }
        }

        // Emit token ids.
        var idx: i32 = 0;
        while (idx >= 0) {
            const ni: usize = @intCast(idx);
            if (self.token_map.get(nodes.items[ni].text)) |id| {
                try out.append(out_alloc, id);
            }
            idx = nodes.items[ni].next;
        }
    }
};
