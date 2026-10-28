const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vk = b.createModule(.{
        .root_source_file = b.path("src/vk.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const sdk = b.graph.environ_map.get("VULKAN_SDK") orelse "/opt/vulkan-sdk";
    const loader_dir = b.pathJoin(&.{ sdk, "lib/VulkanLoader/lib" });
    vk.addLibraryPath(.{ .cwd_relative = loader_dir });
    vk.linkSystemLibrary("vulkan", .{});

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root.addImport("vk", vk);

    const lib = b.addLibrary(.{
        .name = "bonsai",
        .linkage = .dynamic,
        .root_module = root,
    });
    b.installArtifact(lib);
}
