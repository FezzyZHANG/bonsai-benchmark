//! Minimal raw-Vulkan 1.4 bootstrap for a headless compute inference engine.
//!
//! No external dependencies: all Vulkan entry points and types come from
//! `vk_bindings.zig`, generated with `zig translate-c` from the Vulkan SDK
//! `vulkan/vulkan.h` header. The module links against the system Vulkan loader
//! (`-lvulkan`).
//!
//! Scope:
//!   * Vulkan 1.4 instance (retries 1.3/1.2 if the loader refuses 1.4)
//!   * first physical device exposing a compute queue
//!   * one logical device + one compute queue + one command pool/buffer
//!   * device-local buffer upload (staging + fence wait) and download
//!   * optional helper: single-shader compute pipeline + synchronous dispatch
//!
//! Feature/extension policy: everything optional is queried first with
//! vkGetPhysicalDeviceFeatures2 / vkGetPhysicalDeviceProperties2 pNext chains
//! and only enabled when the driver advertises it. Absent optional features
//! never fail initialization.

const std = @import("std");
const vk = @import("vk_bindings.zig");

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub const Buffer = struct {
    buffer: vk.VkBuffer = null,
    memory: vk.VkDeviceMemory = null,
    size: u64 = 0,
    device_address: u64 = 0,
    host_visible: bool = false,
};

pub const Pipeline = struct {
    pipeline: vk.VkPipeline = null,
    layout: vk.VkPipelineLayout = null,
    set_layout: vk.VkDescriptorSetLayout = null,
    descriptor_pool: vk.VkDescriptorPool = null,
    descriptor_set: vk.VkDescriptorSet = null,
    storage_buffer_count: u32 = 0,
    push_constant_size: u32 = 0,
};

/// Which of the requested Vulkan 1.4 / Roadmap-2026 compute features were
/// advertised by the driver and enabled on the logical device.
pub const Caps = struct {
    shader_float16: bool = false,
    shader_int8: bool = false,
    storage_buffer16: bool = false,
    storage_buffer8: bool = false,
    integer_dot_product: bool = false,
    timeline_semaphore: bool = false,
    buffer_device_address: bool = false,
    descriptor_indexing: bool = false,
    scalar_block_layout: bool = false,
    vulkan_memory_model: bool = false,

    cooperative_matrix: bool = false,
    maintenance5: bool = false,
    maintenance7: bool = false,
    maintenance8: bool = false,
    maintenance9: bool = false,
    shader_float_controls2: bool = false,
    expect_assume: bool = false,
    subgroup_rotate: bool = false,
    maximal_reconvergence: bool = false,
    quad_control: bool = false,
    untyped_pointers: bool = false,
    workgroup_memory_explicit_layout: bool = false,
    robustness2: bool = false,
};

/// VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, re-exported so users of this module do
/// not need to import the generated bindings themselves.
pub const buffer_usage_storage: u32 = @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

pub const Error = error{
    VulkanError,
    NoVulkanDevice,
    NoComputeQueue,
    NoSuitableMemoryType,
    EmptyBuffer,
    BufferTooSmall,
    TooManyStorageBuffers,
    PushConstantTooLarge,
    BufferNotHostVisible,
    BufferCountMismatch,
    OutOfMemory,
};

// ---------------------------------------------------------------------------
// Requested device extensions (enabled only when advertised)
// ---------------------------------------------------------------------------

const requested_extensions = [_][]const u8{
    "VK_KHR_cooperative_matrix",
    "VK_KHR_maintenance5",
    "VK_KHR_maintenance7",
    "VK_KHR_maintenance8",
    "VK_KHR_maintenance9",
    "VK_KHR_shader_float_controls2",
    "VK_KHR_shader_expect_assume",
    "VK_KHR_shader_subgroup_rotate",
    "VK_KHR_shader_maximal_reconvergence",
    "VK_KHR_shader_quad_control",
    "VK_KHR_shader_untyped_pointers",
    "VK_KHR_shader_subgroup_uniform_control_flow",
    "VK_KHR_workgroup_memory_explicit_layout",
    "VK_KHR_robustness2",
};

const Exts = struct {
    cooperative_matrix: bool = false,
    maintenance5: bool = false,
    maintenance7: bool = false,
    maintenance8: bool = false,
    maintenance9: bool = false,
    float_controls2: bool = false,
    expect_assume: bool = false,
    subgroup_rotate: bool = false,
    maximal_reconvergence: bool = false,
    quad_control: bool = false,
    untyped_pointers: bool = false,
    subgroup_uniform_control_flow: bool = false,
    workgroup_mem_explicit: bool = false,
    robustness2: bool = false,
};

// ---------------------------------------------------------------------------
// Feature struct chain
// ---------------------------------------------------------------------------

const FeatureSet = struct {
    core11: vk.VkPhysicalDeviceVulkan11Features = .{},
    core12: vk.VkPhysicalDeviceVulkan12Features = .{},
    core13: vk.VkPhysicalDeviceVulkan13Features = .{},
    core14: vk.VkPhysicalDeviceVulkan14Features = .{},
    coopmat: vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR = .{},
    maintenance7: vk.VkPhysicalDeviceMaintenance7FeaturesKHR = .{},
    maintenance8: vk.VkPhysicalDeviceMaintenance8FeaturesKHR = .{},
    maintenance9: vk.VkPhysicalDeviceMaintenance9FeaturesKHR = .{},
    maximal_reconvergence: vk.VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR = .{},
    quad_control: vk.VkPhysicalDeviceShaderQuadControlFeaturesKHR = .{},
    untyped_pointers: vk.VkPhysicalDeviceShaderUntypedPointersFeaturesKHR = .{},
    workgroup_mem_explicit: vk.VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR = .{},
    robustness2: vk.VkPhysicalDeviceRobustness2FeaturesKHR = .{},
};

/// Common view of the head of every Vulkan structure: sType + pNext.
const ChainHead = extern struct {
    s_type: u32,
    p_next: ?*anyopaque,
};

fn setPNext(ptr: *anyopaque, next: ?*anyopaque) void {
    const head: *ChainHead = @ptrCast(@alignCast(ptr));
    head.p_next = next;
}

fn linkChain(ptrs: []const *anyopaque) ?*anyopaque {
    for (ptrs, 0..) |p, i| {
        setPNext(p, if (i + 1 < ptrs.len) ptrs[i + 1] else null);
    }
    return if (ptrs.len == 0) null else ptrs[0];
}

fn prepareFeatureChain(set: *FeatureSet, exts: Exts, core: vk.VkPhysicalDeviceFeatures, api: u32) vk.VkPhysicalDeviceFeatures2 {
    set.core11.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    set.core12.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    set.core13.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    set.core14.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    set.coopmat.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    set.maintenance7.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_7_FEATURES_KHR;
    set.maintenance8.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR;
    set.maintenance9.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_9_FEATURES_KHR;
    set.maximal_reconvergence.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR;
    set.quad_control.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR;
    set.untyped_pointers.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR;
    set.workgroup_mem_explicit.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR;
    set.robustness2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR;

    var list: [16]*anyopaque = undefined;
    var n: usize = 0;

    // Only chain core feature structs the device's API version understands.
    list[n] = @ptrCast(&set.core11);
    n += 1;
    if (api >= vk.VK_API_VERSION_1_2) {
        list[n] = @ptrCast(&set.core12);
        n += 1;
    }
    if (api >= vk.VK_API_VERSION_1_3) {
        list[n] = @ptrCast(&set.core13);
        n += 1;
    }
    if (api >= vk.VK_API_VERSION_1_4) {
        list[n] = @ptrCast(&set.core14);
        n += 1;
    }
    // Extension feature structs are only chained when the extension is
    // advertised: unknown sTypes in a pNext chain are undefined behaviour.
    // VK_KHR_cooperative_matrix
    if (exts.cooperative_matrix) {
        list[n] = @ptrCast(&set.coopmat);
        n += 1;
    }
    if (exts.maintenance7) {
        list[n] = @ptrCast(&set.maintenance7);
        n += 1;
    }
    if (exts.maintenance8) {
        list[n] = @ptrCast(&set.maintenance8);
        n += 1;
    }
    if (exts.maintenance9) {
        list[n] = @ptrCast(&set.maintenance9);
        n += 1;
    }
    if (exts.maximal_reconvergence) {
        list[n] = @ptrCast(&set.maximal_reconvergence);
        n += 1;
    }
    if (exts.quad_control) {
        list[n] = @ptrCast(&set.quad_control);
        n += 1;
    }
    if (exts.untyped_pointers) {
        list[n] = @ptrCast(&set.untyped_pointers);
        n += 1;
    }
    if (exts.workgroup_mem_explicit) {
        list[n] = @ptrCast(&set.workgroup_mem_explicit);
        n += 1;
    }
    if (exts.robustness2) {
        list[n] = @ptrCast(&set.robustness2);
        n += 1;
    }

    return .{
        .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = linkChain(list[0..n]),
        .features = core,
    };
}

fn boolOf(v: vk.VkBool32) bool {
    return v != 0;
}

/// Copy only the requested feature flags from the queried (supported) set into
/// the set that will be passed to vkCreateDevice. Missing features stay off.
fn enableRequested(query: *const FeatureSet, create: *FeatureSet) Caps {
    var caps: Caps = .{};

    create.core11.storageBuffer16BitAccess = query.core11.storageBuffer16BitAccess;
    create.core11.uniformAndStorageBuffer16BitAccess = query.core11.uniformAndStorageBuffer16BitAccess;

    create.core12.shaderFloat16 = query.core12.shaderFloat16;
    create.core12.shaderInt8 = query.core12.shaderInt8;
    create.core12.storageBuffer8BitAccess = query.core12.storageBuffer8BitAccess;
    create.core12.uniformAndStorageBuffer8BitAccess = query.core12.uniformAndStorageBuffer8BitAccess;
    create.core12.descriptorIndexing = query.core12.descriptorIndexing;
    create.core12.timelineSemaphore = query.core12.timelineSemaphore;
    create.core12.bufferDeviceAddress = query.core12.bufferDeviceAddress;
    create.core12.scalarBlockLayout = query.core12.scalarBlockLayout;
    create.core12.vulkanMemoryModel = query.core12.vulkanMemoryModel;

    create.core13.shaderIntegerDotProduct = query.core13.shaderIntegerDotProduct;

    create.core14.shaderFloatControls2 = query.core14.shaderFloatControls2;
    create.core14.shaderExpectAssume = query.core14.shaderExpectAssume;
    create.core14.shaderSubgroupRotate = query.core14.shaderSubgroupRotate;
    create.core14.maintenance5 = query.core14.maintenance5;
    create.core14.maintenance6 = query.core14.maintenance6;

    create.coopmat.cooperativeMatrix = query.coopmat.cooperativeMatrix;
    create.coopmat.cooperativeMatrixRobustBufferAccess = query.coopmat.cooperativeMatrixRobustBufferAccess;

    create.maintenance7.maintenance7 = query.maintenance7.maintenance7;
    create.maintenance8.maintenance8 = query.maintenance8.maintenance8;
    create.maintenance9.maintenance9 = query.maintenance9.maintenance9;
    create.maximal_reconvergence.shaderMaximalReconvergence = query.maximal_reconvergence.shaderMaximalReconvergence;
    create.quad_control.shaderQuadControl = query.quad_control.shaderQuadControl;
    create.untyped_pointers.shaderUntypedPointers = query.untyped_pointers.shaderUntypedPointers;
    create.workgroup_mem_explicit.workgroupMemoryExplicitLayout = query.workgroup_mem_explicit.workgroupMemoryExplicitLayout;
    create.workgroup_mem_explicit.workgroupMemoryExplicitLayoutScalarBlockLayout = query.workgroup_mem_explicit.workgroupMemoryExplicitLayoutScalarBlockLayout;
    create.workgroup_mem_explicit.workgroupMemoryExplicitLayout8BitAccess = query.workgroup_mem_explicit.workgroupMemoryExplicitLayout8BitAccess;
    create.workgroup_mem_explicit.workgroupMemoryExplicitLayout16BitAccess = query.workgroup_mem_explicit.workgroupMemoryExplicitLayout16BitAccess;
    create.robustness2.robustBufferAccess2 = query.robustness2.robustBufferAccess2;
    create.robustness2.robustImageAccess2 = query.robustness2.robustImageAccess2;
    create.robustness2.nullDescriptor = query.robustness2.nullDescriptor;

    caps.shader_float16 = boolOf(query.core12.shaderFloat16);
    caps.shader_int8 = boolOf(query.core12.shaderInt8);
    caps.storage_buffer16 = boolOf(query.core11.storageBuffer16BitAccess);
    caps.storage_buffer8 = boolOf(query.core12.storageBuffer8BitAccess);
    caps.integer_dot_product = boolOf(query.core13.shaderIntegerDotProduct);
    caps.timeline_semaphore = boolOf(query.core12.timelineSemaphore);
    caps.buffer_device_address = boolOf(query.core12.bufferDeviceAddress);
    caps.descriptor_indexing = boolOf(query.core12.descriptorIndexing);
    caps.scalar_block_layout = boolOf(query.core12.scalarBlockLayout);
    caps.vulkan_memory_model = boolOf(query.core12.vulkanMemoryModel);
    caps.cooperative_matrix = boolOf(query.coopmat.cooperativeMatrix);
    caps.maintenance5 = boolOf(query.core14.maintenance5);
    caps.maintenance7 = boolOf(query.maintenance7.maintenance7);
    caps.maintenance8 = boolOf(query.maintenance8.maintenance8);
    caps.maintenance9 = boolOf(query.maintenance9.maintenance9);
    caps.shader_float_controls2 = boolOf(query.core14.shaderFloatControls2);
    caps.expect_assume = boolOf(query.core14.shaderExpectAssume);
    caps.subgroup_rotate = boolOf(query.core14.shaderSubgroupRotate);
    caps.maximal_reconvergence = boolOf(query.maximal_reconvergence.shaderMaximalReconvergence);
    caps.quad_control = boolOf(query.quad_control.shaderQuadControl);
    caps.untyped_pointers = boolOf(query.untyped_pointers.shaderUntypedPointers);
    caps.workgroup_memory_explicit_layout = boolOf(query.workgroup_mem_explicit.workgroupMemoryExplicitLayout);
    caps.robustness2 = boolOf(query.robustness2.robustBufferAccess2);
    return caps;
}

// ---------------------------------------------------------------------------
// Vk
// ---------------------------------------------------------------------------

pub const Vk = struct {
    alloc: std.mem.Allocator,

    instance: vk.VkInstance = null,
    physical: vk.VkPhysicalDevice = null,
    device: vk.VkDevice = null,
    queue: vk.VkQueue = null,
    queue_family: u32 = 0,

    cmd_pool: vk.VkCommandPool = null,
    cmd: vk.VkCommandBuffer = null,
    fence: vk.VkFence = null,

    mem_props: vk.VkPhysicalDeviceMemoryProperties = .{},
    name: [256]u8 = std.mem.zeroes([256]u8),
    api_version: u32 = 0,
    driver_version: u32 = 0,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    max_compute_invocations: u32 = 0,
    max_compute_workgroup_size: [3]u32 = .{ 0, 0, 0 },
    max_storage_buffer_range: u32 = 0,
    max_per_stage_storage_buffers: u32 = 0,
    max_push_constants: u32 = 0,
    timestamp_period: f32 = 0,

    caps: Caps = .{},
    exts: Exts = .{},
    enabled_extensions: [requested_extensions.len][]const u8 = undefined,
    enabled_extension_count: usize = 0,

    // Feature set used for vkCreateDevice. Kept as a field so its address
    // stays stable while the pNext chain is being rebuilt in createDevice().
    pending_features: FeatureSet = .{},
    // Core 1.0 features that must accompany a requested extension feature.
    pending_core_features: vk.VkPhysicalDeviceFeatures = .{},

    // Bumped on every successful submission; handy for callers that want to
    // know whether a buffer's contents are still in flight.
    submit_count: u64 = 0,

    // -----------------------------------------------------------------------
    // init / deinit
    // -----------------------------------------------------------------------

    pub fn init(alloc: std.mem.Allocator) !Vk {
        var self: Vk = .{ .alloc = alloc };
        errdefer self.deinit();

        try self.createInstance();

        var count: u32 = 0;
        try vkCheck("vkEnumeratePhysicalDevices(count)", vk.vkEnumeratePhysicalDevices(self.instance, &count, null));
        if (count == 0) return Error.NoVulkanDevice;
        const phys_list = try alloc.alloc(vk.VkPhysicalDevice, count);
        defer alloc.free(phys_list);
        var got: u32 = count;
        try vkCheck("vkEnumeratePhysicalDevices(list)", vk.vkEnumeratePhysicalDevices(self.instance, &got, phys_list.ptr));

        // First device exposing a compute queue.
        var chosen_family: u32 = 0;
        for (phys_list[0..got]) |phys| {
            if (self.findComputeFamily(phys)) |family| {
                self.physical = phys;
                chosen_family = family;
                break;
            }
        }
        if (self.physical == null) return Error.NoComputeQueue;
        self.queue_family = chosen_family;

        try self.queryPhysicalDevice();
        try self.createDevice();
        try self.createCommandPool();
        return self;
    }

    pub fn deinit(self: *Vk) void {
        if (self.device != null) {
            _ = vk.vkDeviceWaitIdle(self.device);
            if (self.fence != null) vk.vkDestroyFence(self.device, self.fence, null);
            if (self.cmd_pool != null) vk.vkDestroyCommandPool(self.device, self.cmd_pool, null);
            vk.vkDestroyDevice(self.device, null);
        }
        self.fence = null;
        self.cmd_pool = null;
        self.cmd = null;
        self.device = null;
        if (self.instance != null) {
            vk.vkDestroyInstance(self.instance, null);
        }
        self.instance = null;
        self.physical = null;
        self.queue = null;
    }

    fn createInstance(self: *Vk) !void {
        var app = vk.VkApplicationInfo{
            .sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "bonsai-vk",
            .applicationVersion = 1,
            .pEngineName = "bonsai-vk",
            .engineVersion = 1,
            .apiVersion = @intCast(vk.VK_API_VERSION_1_4),
        };

        // Vulkan 1.4 is the target; fall back only if the loader/driver refuses.
        const versions = [_]u32{
            @intCast(vk.VK_API_VERSION_1_4),
            @intCast(vk.VK_API_VERSION_1_3),
            @intCast(vk.VK_API_VERSION_1_2),
        };
        var last: vk.VkResult = vk.VK_SUCCESS;
        for (versions) |ver| {
            app.apiVersion = ver;
            var info = vk.VkInstanceCreateInfo{
                .sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pApplicationInfo = &app,
                .enabledExtensionCount = 0,
                .ppEnabledExtensionNames = null,
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = null,
            };
            var inst: vk.VkInstance = null;
            last = vk.vkCreateInstance(&info, null, &inst);
            if (last == vk.VK_SUCCESS) {
                self.instance = inst;
                return;
            }
        }
        std.log.err("vkCreateInstance failed: {}", .{last});
        return Error.VulkanError;
    }

    fn findComputeFamily(self: *Vk, phys: vk.VkPhysicalDevice) ?u32 {
        var count: u32 = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, null);
        if (count == 0) return null;
        const props = self.alloc.alloc(vk.VkQueueFamilyProperties, count) catch return null;
        defer self.alloc.free(props);
        var got: u32 = count;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(phys, &got, props.ptr);
        const compute_bit: vk.VkQueueFlags = @intCast(vk.VK_QUEUE_COMPUTE_BIT);
        for (props[0..got], 0..) |p, i| {
            if ((p.queueFlags & compute_bit) != 0 and p.queueCount > 0) return @intCast(i);
        }
        return null;
    }

    fn queryPhysicalDevice(self: *Vk) !void {
        const phys = self.physical.?;
        vk.vkGetPhysicalDeviceMemoryProperties(phys, &self.mem_props);

        // The device API version determines which core structs may legally
        // appear in the pNext chains below.
        var core_props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(phys, &core_props);
        self.api_version = core_props.apiVersion;

        // ---- properties (pNext chain) ----
        var props11 = vk.VkPhysicalDeviceVulkan11Properties{ .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
        var props12 = vk.VkPhysicalDeviceVulkan12Properties{ .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
        var props13 = vk.VkPhysicalDeviceVulkan13Properties{ .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
        var props14 = vk.VkPhysicalDeviceVulkan14Properties{ .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES };
        var prop_list: [4]*anyopaque = undefined;
        var pn: usize = 0;
        prop_list[pn] = @ptrCast(&props11);
        pn += 1;
        if (self.api_version >= vk.VK_API_VERSION_1_2) {
            prop_list[pn] = @ptrCast(&props12);
            pn += 1;
        }
        if (self.api_version >= vk.VK_API_VERSION_1_3) {
            prop_list[pn] = @ptrCast(&props13);
            pn += 1;
        }
        if (self.api_version >= vk.VK_API_VERSION_1_4) {
            prop_list[pn] = @ptrCast(&props14);
            pn += 1;
        }

        var props2 = vk.VkPhysicalDeviceProperties2{
            .sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        };
        props2.pNext = linkChain(prop_list[0..pn]);
        vk.vkGetPhysicalDeviceProperties2(phys, &props2);

        const props = props2.properties;
        self.name = props.deviceName;
        self.api_version = props.apiVersion;
        self.driver_version = props.driverVersion;
        self.vendor_id = props.vendorID;
        self.device_id = props.deviceID;
        self.max_compute_invocations = props.limits.maxComputeWorkGroupInvocations;
        self.max_compute_workgroup_size = props.limits.maxComputeWorkGroupSize;
        self.max_storage_buffer_range = props.limits.maxStorageBufferRange;
        self.max_per_stage_storage_buffers = props.limits.maxPerStageDescriptorStorageBuffers;
        self.max_push_constants = props.limits.maxPushConstantsSize;
        self.timestamp_period = props.limits.timestampPeriod;

        // ---- extension coverage ----
        self.exts = try self.queryExtensions(phys);

        // ---- features (pNext chain) ----
        var query_set: FeatureSet = .{};
        var features2 = prepareFeatureChain(&query_set, self.exts, .{}, self.api_version);
        vk.vkGetPhysicalDeviceFeatures2(phys, &features2);

        // ---- enable what is both requested and advertised ----
        var create_set: FeatureSet = .{};
        self.caps = enableRequested(&query_set, &create_set);

        // VUID-VkPhysicalDeviceRobustness2FeaturesKHR-robustBufferAccess2-04000:
        // robustBufferAccess2 requires core robustBufferAccess.
        self.pending_core_features = .{
            .robustBufferAccess = if (boolOf(create_set.robustness2.robustBufferAccess2))
                features2.features.robustBufferAccess
            else
                0,
        };

        // Stash the create-mode chain in self temporarily via the device
        // creation function (it rebuilds the chain with the same layout).
        self.pending_features = create_set;
    }

    fn queryExtensions(self: *Vk, phys: vk.VkPhysicalDevice) !Exts {
        var count: u32 = 0;
        const first = vk.vkEnumerateDeviceExtensionProperties(phys, null, &count, null);
        if (first < 0 or count == 0) return .{};
        const props = try self.alloc.alloc(vk.VkExtensionProperties, count);
        defer self.alloc.free(props);
        var got: u32 = count;
        const res = vk.vkEnumerateDeviceExtensionProperties(phys, null, &got, props.ptr);
        if (res < 0) return .{};

        var exts: Exts = .{};
        var name_buf = std.mem.zeroes([256]u8);
        for (props[0..got]) |p| {
            name_buf = p.extensionName;
            const end = std.mem.indexOfScalar(u8, name_buf[0..], 0) orelse name_buf.len;
            const name = name_buf[0..end];
            if (std.mem.eql(u8, name, "VK_KHR_cooperative_matrix")) exts.cooperative_matrix = true;
            if (std.mem.eql(u8, name, "VK_KHR_maintenance5")) exts.maintenance5 = true;
            if (std.mem.eql(u8, name, "VK_KHR_maintenance7")) exts.maintenance7 = true;
            if (std.mem.eql(u8, name, "VK_KHR_maintenance8")) exts.maintenance8 = true;
            if (std.mem.eql(u8, name, "VK_KHR_maintenance9")) exts.maintenance9 = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_float_controls2")) exts.float_controls2 = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_expect_assume")) exts.expect_assume = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_subgroup_rotate")) exts.subgroup_rotate = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_maximal_reconvergence")) exts.maximal_reconvergence = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_quad_control")) exts.quad_control = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_untyped_pointers")) exts.untyped_pointers = true;
            if (std.mem.eql(u8, name, "VK_KHR_shader_subgroup_uniform_control_flow")) exts.subgroup_uniform_control_flow = true;
            if (std.mem.eql(u8, name, "VK_KHR_workgroup_memory_explicit_layout")) exts.workgroup_mem_explicit = true;
            if (std.mem.eql(u8, name, "VK_KHR_robustness2")) exts.robustness2 = true;
        }
        return exts;
    }

    fn createDevice(self: *Vk) !void {
        const phys = self.physical.?;

        // Enable the advertised subset of the requested extensions.
        var name_ptrs: [requested_extensions.len][*c]const u8 = undefined;
        var ext_count: u32 = 0;
        self.enabled_extension_count = 0;
        const advertised = [_]struct { name: []const u8, on: bool }{
            .{ .name = "VK_KHR_cooperative_matrix", .on = self.exts.cooperative_matrix },
            .{ .name = "VK_KHR_maintenance5", .on = self.exts.maintenance5 },
            .{ .name = "VK_KHR_maintenance7", .on = self.exts.maintenance7 },
            .{ .name = "VK_KHR_maintenance8", .on = self.exts.maintenance8 },
            .{ .name = "VK_KHR_maintenance9", .on = self.exts.maintenance9 },
            .{ .name = "VK_KHR_shader_float_controls2", .on = self.exts.float_controls2 },
            .{ .name = "VK_KHR_shader_expect_assume", .on = self.exts.expect_assume },
            .{ .name = "VK_KHR_shader_subgroup_rotate", .on = self.exts.subgroup_rotate },
            .{ .name = "VK_KHR_shader_maximal_reconvergence", .on = self.exts.maximal_reconvergence },
            .{ .name = "VK_KHR_shader_quad_control", .on = self.exts.quad_control },
            .{ .name = "VK_KHR_shader_untyped_pointers", .on = self.exts.untyped_pointers },
            .{ .name = "VK_KHR_shader_subgroup_uniform_control_flow", .on = self.exts.subgroup_uniform_control_flow },
            .{ .name = "VK_KHR_workgroup_memory_explicit_layout", .on = self.exts.workgroup_mem_explicit },
            .{ .name = "VK_KHR_robustness2", .on = self.exts.robustness2 },
        };
        for (advertised) |e| {
            if (!e.on) continue;
            name_ptrs[ext_count] = @ptrCast(e.name.ptr);
            ext_count += 1;
            self.enabled_extensions[self.enabled_extension_count] = e.name;
            self.enabled_extension_count += 1;
        }

        var features2 = prepareFeatureChain(&self.pending_features, self.exts, self.pending_core_features, self.api_version);

        var priority: [1]f32 = .{1.0};
        const qci = vk.VkDeviceQueueCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = self.queue_family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };
        const dci = vk.VkDeviceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = @ptrCast(&features2),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &qci,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = ext_count,
            .ppEnabledExtensionNames = if (ext_count == 0) null else &name_ptrs,
            .pEnabledFeatures = null,
        };
        var dev: vk.VkDevice = null;
        const res = vk.vkCreateDevice(phys, &dci, null, &dev);
        if (res != vk.VK_SUCCESS) {
            std.log.err("vkCreateDevice failed: {}", .{res});
            return Error.VulkanError;
        }
        self.device = dev;
        vk.vkGetDeviceQueue(dev, self.queue_family, 0, &self.queue);
        // The create-mode chain is no longer needed; drop the (now dangling)
        // pNext links so the value stored in Vk is inert.
        self.pending_features = .{};
    }

    fn createCommandPool(self: *Vk) !void {
        const info = vk.VkCommandPoolCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = @intCast(vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),
            .queueFamilyIndex = self.queue_family,
        };
        var pool: vk.VkCommandPool = null;
        try vkCheck("vkCreateCommandPool", vk.vkCreateCommandPool(self.device, &info, null, &pool));
        errdefer vk.vkDestroyCommandPool(self.device, pool, null);
        self.cmd_pool = pool;

        const alloc_info = vk.VkCommandBufferAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = @intCast(vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY),
            .commandBufferCount = 1,
        };
        try vkCheck("vkAllocateCommandBuffers", vk.vkAllocateCommandBuffers(self.device, &alloc_info, &self.cmd));

        const fence_info = vk.VkFenceCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = 0,
        };
        try vkCheck("vkCreateFence", vk.vkCreateFence(self.device, &fence_info, null, &self.fence));
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    pub fn deviceName(self: *const Vk) []const u8 {
        const end = std.mem.indexOfScalar(u8, &self.name, 0) orelse self.name.len;
        return self.name[0..end];
    }

    pub fn apiVersion(self: *const Vk) u32 {
        return self.api_version;
    }

    pub fn hasBufferDeviceAddress(self: *const Vk) bool {
        return self.caps.buffer_device_address;
    }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    fn findMemoryType(self: *const Vk, type_bits: u32, required: vk.VkMemoryPropertyFlags, preferred: vk.VkMemoryPropertyFlags) !u32 {
        var fallback: ?u32 = null;
        var i: u32 = 0;
        while (i < self.mem_props.memoryTypeCount) : (i += 1) {
            if ((type_bits & (@as(u32, 1) << @intCast(i))) == 0) continue;
            const flags = self.mem_props.memoryTypes[i].propertyFlags;
            if ((flags & required) != required) continue;
            if ((flags & preferred) == preferred) return i;
            if (fallback == null) fallback = i;
        }
        return fallback orelse Error.NoSuitableMemoryType;
    }

    const CreateOpts = struct {
        size: u64,
        usage: vk.VkBufferUsageFlags,
        host_visible: bool = false,
        device_local: bool = false,
    };

    fn createBufferInternal(self: *Vk, opts: CreateOpts) !Buffer {
        if (opts.size == 0) return Error.EmptyBuffer;
        const info = vk.VkBufferCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = opts.size,
            .usage = opts.usage,
            .sharingMode = @intCast(vk.VK_SHARING_MODE_EXCLUSIVE),
        };
        var buf: vk.VkBuffer = null;
        try vkCheck("vkCreateBuffer", vk.vkCreateBuffer(self.device, &info, null, &buf));
        errdefer vk.vkDestroyBuffer(self.device, buf, null);

        var reqs: vk.VkMemoryRequirements = undefined;
        vk.vkGetBufferMemoryRequirements(self.device, buf, &reqs);

        const device_local_bit: vk.VkMemoryPropertyFlags = @intCast(vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        const host_visible_bit: vk.VkMemoryPropertyFlags = @intCast(vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        const host_coherent_bit: vk.VkMemoryPropertyFlags = @intCast(vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        var required: vk.VkMemoryPropertyFlags = 0;
        var preferred: vk.VkMemoryPropertyFlags = 0;
        if (opts.host_visible) {
            required = host_visible_bit | host_coherent_bit;
        }
        if (opts.device_local) {
            preferred = device_local_bit;
            if (!opts.host_visible) required = device_local_bit;
        }
        const mt = try self.findMemoryType(reqs.memoryTypeBits, required, preferred);

        const address_usage: vk.VkBufferUsageFlags = @intCast(vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        const wants_address = self.caps.buffer_device_address and (opts.usage & address_usage) != 0;

        var flags_info = vk.VkMemoryAllocateFlagsInfo{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .flags = @intCast(vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT),
            .deviceMask = 0,
        };
        var alloc_info = vk.VkMemoryAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = reqs.size,
            .memoryTypeIndex = mt,
        };
        if (wants_address) alloc_info.pNext = @ptrCast(&flags_info);

        var mem: vk.VkDeviceMemory = null;
        try vkCheck("vkAllocateMemory", vk.vkAllocateMemory(self.device, &alloc_info, null, &mem));
        errdefer vk.vkFreeMemory(self.device, mem, null);

        try vkCheck("vkBindBufferMemory", vk.vkBindBufferMemory(self.device, buf, mem, 0));

        var out = Buffer{
            .buffer = buf,
            .memory = mem,
            .size = opts.size,
            .host_visible = opts.host_visible,
        };
        if (wants_address) {
            const addr_info = vk.VkBufferDeviceAddressInfo{
                .sType = vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .buffer = buf,
            };
            out.device_address = vk.vkGetBufferDeviceAddress(self.device, &addr_info);
        }
        return out;
    }

    pub fn destroyBuffer(self: *Vk, buf: Buffer) void {
        if (self.device == null) return;
        if (buf.buffer != null) vk.vkDestroyBuffer(self.device, buf.buffer, null);
        if (buf.memory != null) vk.vkFreeMemory(self.device, buf.memory, null);
    }

    /// Create a device-local buffer and fill it from host memory through a
    /// staging buffer, waiting for the copy to complete.
    pub fn uploadDeviceLocal(self: *Vk, data: []const u8) !Buffer {
        if (data.len == 0) return Error.EmptyBuffer;
        const size: u64 = @intCast(data.len);

        const dst = try self.createBufferInternal(.{
            .size = size,
            .usage = @intCast(vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT),
            .device_local = true,
        });
        errdefer self.destroyBuffer(dst);

        const staging = try self.createBufferInternal(.{
            .size = size,
            .usage = @intCast(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
            .host_visible = true,
        });
        defer self.destroyBuffer(staging);

        const mapped = try self.mapMemory(staging.memory, size);
        @memcpy(mapped[0..data.len], data);
        self.unmapMemory(staging.memory);

        try self.copyAndWait(staging.buffer, dst.buffer, size);
        return dst;
    }

    /// Create a device-local buffer with explicit usage flags (no initial data).
    pub fn createDeviceBuffer(self: *Vk, size: u64, usage: vk.VkBufferUsageFlags) !Buffer {
        return self.createBufferInternal(.{ .size = size, .usage = usage, .device_local = true });
    }

    /// Create a host-visible, host-coherent buffer (mappable).
    pub fn createHostBuffer(self: *Vk, size: u64, usage: vk.VkBufferUsageFlags) !Buffer {
        return self.createBufferInternal(.{ .size = size, .usage = usage, .host_visible = true });
    }

    /// Write `data` into `buf` at `offset` (staging copy unless host visible).
    pub fn writeBuffer(self: *Vk, buf: Buffer, offset: u64, data: []const u8) !void {
        if (data.len == 0) return;
        const len: u64 = @intCast(data.len);
        if (offset + len > buf.size) return Error.BufferTooSmall;

        if (buf.host_visible) {
            const mapped = try self.mapMemory(buf.memory, buf.size);
            @memcpy(mapped[@intCast(offset)..][0..data.len], data);
            self.unmapMemory(buf.memory);
            return;
        }

        const staging = try self.createBufferInternal(.{
            .size = len,
            .usage = @intCast(vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT),
            .host_visible = true,
        });
        defer self.destroyBuffer(staging);
        const mapped = try self.mapMemory(staging.memory, len);
        @memcpy(mapped[0..data.len], data);
        self.unmapMemory(staging.memory);
        try self.copyAndWaitOffset(staging.buffer, buf.buffer, 0, offset, len);
    }

    pub fn mapHostBuffer(self: *Vk, buf: Buffer) ![*]u8 {
        if (!buf.host_visible) return Error.BufferNotHostVisible;
        return self.mapMemory(buf.memory, buf.size);
    }

    pub fn unmapHostBuffer(self: *Vk, buf: Buffer) void {
        if (!buf.host_visible) return;
        self.unmapMemory(buf.memory);
    }

    /// Copy `out.len` bytes from the start of `buf` back to host memory.
    pub fn download(self: *Vk, buf: Buffer, out: []u8) !void {
        try self.downloadRange(buf, 0, out);
    }

    pub fn downloadRange(self: *Vk, buf: Buffer, offset: u64, out: []u8) !void {
        if (out.len == 0) return;
        const len: u64 = @intCast(out.len);
        if (offset + len > buf.size) return Error.BufferTooSmall;

        const staging = try self.createBufferInternal(.{
            .size = len,
            .usage = @intCast(vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT),
            .host_visible = true,
        });
        defer self.destroyBuffer(staging);

        try self.copyAndWaitOffset(buf.buffer, staging.buffer, offset, 0, len);

        const mapped = try self.mapMemory(staging.memory, len);
        @memcpy(out, mapped[0..out.len]);
        self.unmapMemory(staging.memory);
    }

    fn mapMemory(self: *Vk, memory: vk.VkDeviceMemory, size: u64) ![*]u8 {
        var p: ?*anyopaque = null;
        try vkCheck("vkMapMemory", vk.vkMapMemory(self.device, memory, 0, size, 0, &p));
        if (p == null) return Error.VulkanError;
        return @ptrCast(p.?);
    }

    fn unmapMemory(self: *Vk, memory: vk.VkDeviceMemory) void {
        vk.vkUnmapMemory(self.device, memory);
    }

    // -----------------------------------------------------------------------
    // Command submission
    // -----------------------------------------------------------------------

    fn beginCmd(self: *Vk) !vk.VkCommandBuffer {
        try vkCheck("vkResetCommandBuffer", vk.vkResetCommandBuffer(self.cmd, 0));
        const begin = vk.VkCommandBufferBeginInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = @intCast(vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),
        };
        try vkCheck("vkBeginCommandBuffer", vk.vkBeginCommandBuffer(self.cmd, &begin));
        return self.cmd;
    }

    fn submitAndWait(self: *Vk) !void {
        var cmd = self.cmd;
        const submit = vk.VkSubmitInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };
        try vkCheck("vkQueueSubmit", vk.vkQueueSubmit(self.queue, 1, &submit, self.fence));
        try vkCheck(
            "vkWaitForFences",
            vk.vkWaitForFences(self.device, 1, &self.fence, 1, std.math.maxInt(u64)),
        );
        try vkCheck("vkResetFences", vk.vkResetFences(self.device, 1, &self.fence));
        self.submit_count += 1;
    }

    fn copyAndWait(self: *Vk, src: vk.VkBuffer, dst: vk.VkBuffer, size: u64) !void {
        try self.copyAndWaitOffset(src, dst, 0, 0, size);
    }

    fn copyAndWaitOffset(self: *Vk, src: vk.VkBuffer, dst: vk.VkBuffer, src_off: u64, dst_off: u64, size: u64) !void {
        const cmd = try self.beginCmd();
        const region = vk.VkBufferCopy{ .srcOffset = src_off, .dstOffset = dst_off, .size = size };
        vk.vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        try vkCheck("vkEndCommandBuffer", vk.vkEndCommandBuffer(cmd));
        try self.submitAndWait();
    }

    // -----------------------------------------------------------------------
    // Compute pipelines
    // -----------------------------------------------------------------------

    pub fn createComputePipeline(self: *Vk, spirv: []const u32, n_storage_buffers: u32) !Pipeline {
        return self.createComputePipelineEx(spirv, n_storage_buffers, 0);
    }

    /// Same as createComputePipeline but additionally declares a compute-stage
    /// push-constant block of `push_constant_bytes` bytes.
    pub fn createComputePipelineEx(self: *Vk, spirv: []const u32, n_storage_buffers: u32, push_constant_bytes: u32) !Pipeline {
        if (spirv.len == 0) return Error.VulkanError;
        if (push_constant_bytes > self.max_push_constants) return Error.PushConstantTooLarge;
        const max_per_stage = self.maxPerStageStorageBuffers();
        if (n_storage_buffers > max_per_stage) return Error.TooManyStorageBuffers;
        if (n_storage_buffers > 64) return Error.TooManyStorageBuffers;

        var bindings: [64]vk.VkDescriptorSetLayoutBinding = undefined;
        var i: u32 = 0;
        while (i < n_storage_buffers) : (i += 1) {
            bindings[i] = .{
                .binding = i,
                .descriptorType = @intCast(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                .descriptorCount = 1,
                .stageFlags = @intCast(vk.VK_SHADER_STAGE_COMPUTE_BIT),
                .pImmutableSamplers = null,
            };
        }

        var pipe: Pipeline = .{ .storage_buffer_count = n_storage_buffers };
        errdefer self.destroyPipeline(pipe);

        // Descriptor set layout (always one set; may contain 0 bindings).
        const dsl_info = vk.VkDescriptorSetLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = n_storage_buffers,
            .pBindings = if (n_storage_buffers == 0) null else &bindings,
        };
        try vkCheck(
            "vkCreateDescriptorSetLayout",
            vk.vkCreateDescriptorSetLayout(self.device, &dsl_info, null, &pipe.set_layout),
        );

        pipe.push_constant_size = push_constant_bytes;
        const push_range = vk.VkPushConstantRange{
            .stageFlags = @intCast(vk.VK_SHADER_STAGE_COMPUTE_BIT),
            .offset = 0,
            .size = push_constant_bytes,
        };
        const pipe_layout_info = vk.VkPipelineLayoutCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &pipe.set_layout,
            .pushConstantRangeCount = if (push_constant_bytes > 0) 1 else 0,
            .pPushConstantRanges = if (push_constant_bytes > 0) &push_range else null,
        };
        try vkCheck(
            "vkCreatePipelineLayout",
            vk.vkCreatePipelineLayout(self.device, &pipe_layout_info, null, &pipe.layout),
        );

        var module_info = vk.VkShaderModuleCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.len * @sizeOf(u32),
            .pCode = spirv.ptr,
        };
        var module: vk.VkShaderModule = null;
        try vkCheck("vkCreateShaderModule", vk.vkCreateShaderModule(self.device, &module_info, null, &module));
        defer vk.vkDestroyShaderModule(self.device, module, null);

        const stage = vk.VkPipelineShaderStageCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = @intCast(vk.VK_SHADER_STAGE_COMPUTE_BIT),
            .module = module,
            .pName = "main",
            .pSpecializationInfo = null,
        };
        const cp_info = vk.VkComputePipelineCreateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stage,
            .layout = pipe.layout,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };
        try vkCheck(
            "vkCreateComputePipelines",
            vk.vkCreateComputePipelines(self.device, null, 1, &cp_info, null, &pipe.pipeline),
        );

        if (n_storage_buffers > 0) {
            const pool_size = vk.VkDescriptorPoolSize{
                .type = @intCast(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                .descriptorCount = n_storage_buffers,
            };
            const pool_info = vk.VkDescriptorPoolCreateInfo{
                .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .flags = 0,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &pool_size,
            };
            try vkCheck(
                "vkCreateDescriptorPool",
                vk.vkCreateDescriptorPool(self.device, &pool_info, null, &pipe.descriptor_pool),
            );
            const set_alloc = vk.VkDescriptorSetAllocateInfo{
                .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pipe.descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &pipe.set_layout,
            };
            try vkCheck(
                "vkAllocateDescriptorSets",
                vk.vkAllocateDescriptorSets(self.device, &set_alloc, &pipe.descriptor_set),
            );
        }
        return pipe;
    }

    fn maxPerStageStorageBuffers(self: *const Vk) u32 {
        // RADV reports >= 16; the Vulkan minimum is 4. Cap at our array size.
        return @min(self.max_per_stage_storage_buffers, 64);
    }

    pub fn destroyPipeline(self: *Vk, pipe: Pipeline) void {
        if (self.device == null) return;
        if (pipe.descriptor_pool != null) vk.vkDestroyDescriptorPool(self.device, pipe.descriptor_pool, null);
        if (pipe.pipeline != null) vk.vkDestroyPipeline(self.device, pipe.pipeline, null);
        if (pipe.layout != null) vk.vkDestroyPipelineLayout(self.device, pipe.layout, null);
        if (pipe.set_layout != null) vk.vkDestroyDescriptorSetLayout(self.device, pipe.set_layout, null);
    }

    /// Bind `bufs` as storage buffers 0..n-1 of set 0 and dispatch, then wait
    /// for completion (synchronous helper).
    pub fn dispatch(self: *Vk, pipe: Pipeline, bufs: []const Buffer, gx: u32, gy: u32, gz: u32) !void {
        try self.dispatchPush(pipe, bufs, &.{}, gx, gy, gz);
    }

    /// Like dispatch() but uploads `push_data` as compute-stage push constants.
    pub fn dispatchPush(self: *Vk, pipe: Pipeline, bufs: []const Buffer, push_data: []const u8, gx: u32, gy: u32, gz: u32) !void {
        if (bufs.len != pipe.storage_buffer_count) return Error.BufferCountMismatch;
        if (push_data.len > pipe.push_constant_size) return Error.PushConstantTooLarge;

        if (bufs.len > 0) {
            var infos: [64]vk.VkDescriptorBufferInfo = undefined;
            var writes: [64]vk.VkWriteDescriptorSet = undefined;
            // A storage-buffer descriptor range may not exceed
            // maxStorageBufferRange; clamp defensively (0 means "not queried").
            const max_range: u64 = if (self.max_storage_buffer_range != 0)
                self.max_storage_buffer_range
            else
                std.math.maxInt(u64);
            for (bufs, 0..) |b, i| {
                infos[i] = .{ .buffer = b.buffer, .offset = 0, .range = @min(b.size, max_range) };
                writes[i] = .{
                    .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = pipe.descriptor_set,
                    .dstBinding = @intCast(i),
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = @intCast(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
                    .pBufferInfo = &infos[i],
                };
            }
            vk.vkUpdateDescriptorSets(self.device, @intCast(bufs.len), &writes, 0, null);
        }

        const cmd = try self.beginCmd();
        vk.vkCmdBindPipeline(cmd, @intCast(vk.VK_PIPELINE_BIND_POINT_COMPUTE), pipe.pipeline);
        if (bufs.len > 0) {
            vk.vkCmdBindDescriptorSets(
                cmd,
                @intCast(vk.VK_PIPELINE_BIND_POINT_COMPUTE),
                pipe.layout,
                0,
                1,
                &pipe.descriptor_set,
                0,
                null,
            );
        }
        if (pipe.push_constant_size > 0 and push_data.len > 0) {
            vk.vkCmdPushConstants(
                cmd,
                pipe.layout,
                @intCast(vk.VK_SHADER_STAGE_COMPUTE_BIT),
                0,
                @intCast(push_data.len),
                push_data.ptr,
            );
        }
        vk.vkCmdDispatch(cmd, gx, gy, gz);
        try vkCheck("vkEndCommandBuffer", vk.vkEndCommandBuffer(cmd));
        try self.submitAndWait();
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn vkCheck(comptime what: []const u8, res: vk.VkResult) !void {
    if (res < 0) {
        std.log.err("{s} failed with VkResult {}", .{ what, res });
        return Error.VulkanError;
    }
}
