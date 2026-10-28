//! End-to-end smoke test for src/vk.zig.
//!
//!   * init Vulkan 1.4, print the device name and enabled feature set
//!   * upload a 512 MiB device-local buffer, download slices, verify contents
//!   * run a trivial compute shader over two storage buffers and verify results

const std = @import("std");
const vk = @import("vk");

const Ver = struct { major: u32, minor: u32, patch: u32 };

fn decodeVersion(v: u32) Ver {
    return .{ .major = v >> 22, .minor = (v >> 12) & 0x3ff, .patch = v & 0xfff };
}

fn nowNs() u64 {
    var ts: std.os.linux.timespec = undefined;
    _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("FAIL: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    const t_init = nowNs();
    var v = try vk.Vk.init(alloc);
    defer v.deinit();
    std.debug.print("vk init (instance+device+queue+pool): {d:.2} ms\n", .{@as(f64, @floatFromInt(nowNs() - t_init)) / 1e6});

    const ver = decodeVersion(v.apiVersion());
    std.debug.print("device: {s}\n", .{v.deviceName()});
    std.debug.print("apiVersion: {d}.{d}.{d} driver 0x{x} vendor 0x{x} device 0x{x}\n", .{
        ver.major, ver.minor, ver.patch, v.driver_version, v.vendor_id, v.device_id,
    });
    std.debug.print("compute queue family: {d}, max workgroup invocations: {d}, max storage buffer range: {d} MiB, max push constants: {d}\n", .{
        v.queue_family, v.max_compute_invocations, v.max_storage_buffer_range / (1024 * 1024), v.max_push_constants,
    });

    const c = v.caps;
    std.debug.print(
        "caps: f16={} i8={} sb16={} sb8={} dot={} timeline={} bda={} descidx={} scalar={} vmm={}\n",
        .{ c.shader_float16, c.shader_int8, c.storage_buffer16, c.storage_buffer8, c.integer_dot_product, c.timeline_semaphore, c.buffer_device_address, c.descriptor_indexing, c.scalar_block_layout, c.vulkan_memory_model },
    );
    std.debug.print(
        "caps: coopmat={} m5={} m7={} m8={} m9={} fc2={} expect={} rotate={} maxreconv={} quad={} untyped={} wgmem={} rob2={}\n",
        .{ c.cooperative_matrix, c.maintenance5, c.maintenance7, c.maintenance8, c.maintenance9, c.shader_float_controls2, c.expect_assume, c.subgroup_rotate, c.maximal_reconvergence, c.quad_control, c.untyped_pointers, c.workgroup_memory_explicit_layout, c.robustness2 },
    );
    std.debug.print("enabled extensions ({d}):", .{v.enabled_extension_count});
    for (v.enabled_extensions[0..v.enabled_extension_count]) |e| std.debug.print(" {s}", .{e});
    std.debug.print("\n", .{});

    // ------------------------------------------------------------------
    // 512 MiB upload / download round trip
    // ------------------------------------------------------------------
    const big: usize = 512 * 1024 * 1024;
    const src = try alloc.alloc(u8, big);
    defer alloc.free(src);
    @memset(src, 0xA5);
    std.mem.writeInt(u64, src[0..8], 0x0123456789ABCDEF, .little);
    std.mem.writeInt(u64, src[(1 << 20) - 8 ..][0..8], 0xFEEDFACECAFEBEEF, .little);
    std.mem.writeInt(u64, src[big - 8 ..][0..8], 0xDEADBEEF12345678, .little);

    const t0 = nowNs();
    const big_buf = try v.uploadDeviceLocal(src);
    defer v.destroyBuffer(big_buf);
    const up_ns = nowNs() - t0;
    std.debug.print("uploadDeviceLocal: {d} MiB in {d:.1} ms ({d:.1} GB/s), device_address=0x{x}\n", .{
        big / (1024 * 1024),
        @as(f64, @floatFromInt(up_ns)) / 1e6,
        @as(f64, @floatFromInt(big)) / @as(f64, @floatFromInt(up_ns)),
        big_buf.device_address,
    });

    const chunk: usize = 1024 * 1024;
    const out = try alloc.alloc(u8, chunk);
    defer alloc.free(out);
    const t1 = nowNs();
    try v.download(big_buf, out);
    const down_ns = nowNs() - t1;
    std.debug.print("download 1 MiB slice in {d:.1} ms\n", .{@as(f64, @floatFromInt(down_ns)) / 1e6});
    if (!std.mem.eql(u8, out, src[0..chunk])) fail("downloaded prefix mismatch", .{});

    var tail: [8]u8 = undefined;
    try v.downloadRange(big_buf, big - 8, &tail);
    const tail_val = std.mem.readInt(u64, &tail, .little);
    if (tail_val != 0xDEADBEEF12345678) fail("downloaded tail mismatch: 0x{x}", .{tail_val});

    var mid: [8]u8 = undefined;
    try v.downloadRange(big_buf, (1 << 20) - 8, &mid);
    const mid_val = std.mem.readInt(u64, &mid, .little);
    if (mid_val != 0xFEEDFACECAFEBEEF) fail("downloaded mid mismatch: 0x{x}", .{mid_val});
    std.debug.print("512 MiB round trip verified\n", .{});

    // ------------------------------------------------------------------
    // Compute dispatch
    // ------------------------------------------------------------------
    const n: u32 = 4096;
    const a_src = try alloc.alloc(u32, n);
    defer alloc.free(a_src);
    const b_src = try alloc.alloc(u32, n);
    defer alloc.free(b_src);
    for (a_src, 0..) |*x, i| x.* = @intCast(i);
    for (b_src, 0..) |*x, i| x.* = @intCast(1000 + i);

    const a_buf = try v.uploadDeviceLocal(std.mem.sliceAsBytes(a_src));
    defer v.destroyBuffer(a_buf);
    const b_buf = try v.uploadDeviceLocal(std.mem.sliceAsBytes(b_src));
    defer v.destroyBuffer(b_buf);

    const spv_bytes = @embedFile("add.spv");
    if (spv_bytes.len % 4 != 0) fail("spirv size not multiple of 4", .{});
    const code = try alloc.alloc(u32, spv_bytes.len / 4);
    defer alloc.free(code);
    @memcpy(std.mem.sliceAsBytes(code), spv_bytes);

    const pipe = try v.createComputePipeline(code, 2);
    defer v.destroyPipeline(pipe);
    try v.dispatch(pipe, &.{ a_buf, b_buf }, n / 64, 1, 1);

    const got = try alloc.alloc(u32, n);
    defer alloc.free(got);
    try v.download(a_buf, std.mem.sliceAsBytes(got));
    var bad: usize = 0;
    for (got, 0..) |x, i| {
        const expect: u32 = @as(u32, @intCast(i)) + 3 * @as(u32, @intCast(1000 + i));
        if (x != expect) {
            if (bad < 4) std.debug.print("mismatch at {d}: got {d} want {d}\n", .{ i, x, expect });
            bad += 1;
        }
    }
    if (bad != 0) fail("{d}/{d} compute results wrong", .{ bad, n });
    std.debug.print("compute dispatch verified: {d} elements, a[i]+3*b[i]\n", .{n});

    // ------------------------------------------------------------------
    // Push-constant dispatch
    // ------------------------------------------------------------------
    {
        const m: u32 = 256;
        const zeros = try alloc.alloc(u32, m);
        defer alloc.free(zeros);
        @memset(zeros, 0);
        const c_buf = try v.uploadDeviceLocal(std.mem.sliceAsBytes(zeros));
        defer v.destroyBuffer(c_buf);

        const push_bytes = @embedFile("push.spv");
        const pcode = try alloc.alloc(u32, push_bytes.len / 4);
        defer alloc.free(pcode);
        @memcpy(std.mem.sliceAsBytes(pcode), push_bytes);

        const ppipe = try v.createComputePipelineEx(pcode, 1, 4);
        defer v.destroyPipeline(ppipe);
        const addend: u32 = 7;
        try v.dispatchPush(ppipe, &.{c_buf}, std.mem.asBytes(&addend), m / 64, 1, 1);

        const pgot = try alloc.alloc(u32, m);
        defer alloc.free(pgot);
        try v.download(c_buf, std.mem.sliceAsBytes(pgot));
        for (pgot, 0..) |x, i| {
            if (x != 7) fail("push constant result wrong at {d}: {d}", .{ i, x });
        }
        std.debug.print("push-constant dispatch verified: {d} elements, a[i]+7\n", .{m});
    }

    std.debug.print("PASS\n", .{});
}
