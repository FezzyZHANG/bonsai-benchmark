const std = @import("std");

const V = struct { text: []const u8, ids: []i32 };

pub fn main() !void {
    const alloc = std.heap.page_allocator;
    const s = try std.fmt.allocPrint(alloc, "x{d}", .{42});

    const bytes = try alloc.alloc(u8, 4);
    _ = try std.fmt.hexToBytes(bytes, "48656c6c");
    const v = try std.fmt.parseInt(u32, "123", 10);

    var list = std.ArrayList(u8).empty;
    try list.appendSlice(alloc, s);
    try list.appendSlice(alloc, bytes);

    const parsed = try std.json.parseFromSlice([]V, alloc, "[{\"text\":\"hi\",\"ids\":[1,2]}]", .{});
    defer parsed.deinit();

    const t0 = std.time.nanoTimestamp();
    const t1 = std.time.nanoTimestamp();
    const ns = t1 - t0;

    std.debug.print("{s} {d} {d} {d} {s} {d}\n", .{ s, bytes[0], v, list.items.len, parsed.value[0].text, ns });
}
