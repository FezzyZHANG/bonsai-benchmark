const std = @import("std");
const vk = @import("vk_bindings_raw");

pub fn main() void {
    std.debug.print("size VkPhysicalDeviceVulkan14Features = {d}\n", .{@sizeOf(vk.VkPhysicalDeviceVulkan14Features)});
    std.debug.print("size VkPhysicalDeviceVulkan12Features = {d}\n", .{@sizeOf(vk.VkPhysicalDeviceVulkan12Features)});
    std.debug.print("size VkPhysicalDeviceVulkan13Features = {d}\n", .{@sizeOf(vk.VkPhysicalDeviceVulkan13Features)});
    std.debug.print("size VkBufferCreateInfo = {d}\n", .{@sizeOf(vk.VkBufferCreateInfo)});
    var f: vk.VkPhysicalDeviceVulkan14Features = .{};
    f.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    std.debug.print("sType = {d}\n", .{f.sType});
}
