# VK_STATUS - raw Vulkan 1.4 bootstrap (`src/vk.zig`)

Status: **PASS** on the target machine (AMD Radeon RX 9070 / RADV), including a
512 MiB device-local upload/download round trip, a compute dispatch and a
push-constant dispatch. Zero validation-layer messages.

## Files

| path | purpose |
|---|---|
| `src/vk.zig` | the module (instance, device, queue, command pool, buffers, compute pipelines) |
| `src/vk_bindings.zig` | Vulkan ABI declarations, generated with `zig translate-c` (1.5 MB, self-contained Zig, no `@cImport`) |
| `tools/vktest.zig` | smoke test / benchmark executable |
| `tools/add.comp`, `tools/add.spv` | two-storage-buffer test shader (`a[i] += 3*b[i]`) |
| `tools/push.comp`, `tools/push.spv` | push-constant test shader (`a[i] += pc.add`) |
| `tools/build_vktest.sh` | `zig build-exe` driver for the test (`debug` / `release`) |
| `tools/buildtest/build.zig` | throwaway project that proves the `build.zig` snippet below works |

Regenerate the bindings with:

```
zig translate-c -I "$VULKAN_SDK/include" "$VULKAN_SDK/include/vulkan/vulkan.h" > src/vk_bindings.zig
```

(`@cImport` is not used anywhere; no include paths are needed at build time.)

## Exact API

```zig
const vk = @import("vk");

pub const Buffer = struct {
    buffer: vk.VkBuffer,
    memory: vk.VkDeviceMemory,
    size: u64,
    device_address: u64,   // 0 unless bufferDeviceAddress is enabled
    host_visible: bool,
};

pub const Pipeline = struct {
    pipeline: vk.VkPipeline,
    layout: vk.VkPipelineLayout,
    set_layout: vk.VkDescriptorSetLayout,
    descriptor_pool: vk.VkDescriptorPool,
    descriptor_set: vk.VkDescriptorSet,
    storage_buffer_count: u32,
    push_constant_size: u32,
};

pub const Caps = struct { /* one bool per requested feature, see below */ };

pub const Vk = struct {
    pub fn init(alloc: std.mem.Allocator) !Vk;
    pub fn deinit(self: *Vk) void;

    pub fn deviceName(self: *const Vk) []const u8;
    pub fn apiVersion(self: *const Vk) u32;
    pub fn hasBufferDeviceAddress(self: *const Vk) bool;

    pub fn uploadDeviceLocal(self: *Vk, data: []const u8) !Buffer; // staging + fence wait
    pub fn download(self: *Vk, buf: Buffer, out: []u8) !void;      // from offset 0
    pub fn downloadRange(self: *Vk, buf: Buffer, offset: u64, out: []u8) !void;
    pub fn destroyBuffer(self: *Vk, buf: Buffer) void;

    pub fn createDeviceBuffer(self: *Vk, size: u64, usage: vk.VkBufferUsageFlags) !Buffer;
    pub fn createHostBuffer(self: *Vk, size: u64, usage: vk.VkBufferUsageFlags) !Buffer;
    pub fn writeBuffer(self: *Vk, buf: Buffer, offset: u64, data: []const u8) !void;
    pub fn mapHostBuffer(self: *Vk, buf: Buffer) ![*]u8;
    pub fn unmapHostBuffer(self: *Vk, buf: Buffer) void;

    pub fn createComputePipeline(self: *Vk, spirv: []const u32, n_storage_buffers: u32) !Pipeline;
    pub fn createComputePipelineEx(self: *Vk, spirv: []const u32, n_storage_buffers: u32,
                                   push_constant_bytes: u32) !Pipeline;
    pub fn destroyPipeline(self: *Vk, pipe: Pipeline) void;
    pub fn dispatch(self: *Vk, pipe: Pipeline, bufs: []const Buffer, gx: u32, gy: u32, gz: u32) !void;
    pub fn dispatchPush(self: *Vk, pipe: Pipeline, bufs: []const Buffer,
                        push_data: []const u8, gx: u32, gy: u32, gz: u32) !void;

    // observable state
    alloc: std.mem.Allocator,
    instance, physical, device, queue, queue_family,
    cmd_pool, cmd, fence,
    mem_props: vk.VkPhysicalDeviceMemoryProperties,
    api_version, driver_version, vendor_id, device_id,
    max_compute_invocations, max_compute_workgroup_size,
    max_storage_buffer_range, max_per_stage_storage_buffers, max_push_constants,
    caps: Caps, exts: Exts,
    enabled_extensions: [14][]const u8, enabled_extension_count: usize,
    submit_count: u64,
};
```

Semantics:

* `init` creates a Vulkan 1.4 instance (`VK_API_VERSION_1_4`, falling back to
  1.3/1.2 only if the loader refuses), enumerates physical devices, picks the
  **first device with a queue family exposing `VK_QUEUE_COMPUTE_BIT`**, queries
  memory/queue/extensions/features/properties through
  `vkGetPhysicalDeviceFeatures2` / `vkGetPhysicalDeviceProperties2` pNext
  chains, creates the logical device with the requested extensions that are
  advertised, gets the compute queue, and creates a command pool + one
  primary command buffer + one fence.
* Optional features that the driver does not advertise are simply left
  disabled; init never fails because of an absent optional feature.
* `uploadDeviceLocal` creates a `DEVICE_LOCAL` buffer (usage
  `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST | SHADER_DEVICE_ADDRESS`),
  uploads through a host-visible staging buffer, submits `vkCmdCopyBuffer` and
  waits on a fence. `download`/`downloadRange` do the reverse.
* `dispatch` binds each `Buffer` as storage buffer binding 0..n-1 of set 0 of
  `Pipeline`, records `vkCmdDispatch` and waits for the queue.
* All calls are synchronous and assume a single calling thread (the harness is
  single-threaded). `Buffer`/`Pipeline` objects must be destroyed by the caller
  before `deinit` (there is no hidden registry).

## build.zig snippet (verified)

The module is linked against the system Vulkan loader. This exact snippet was
compiled and run in `tools/buildtest/build.zig`:

```zig
const vk_dep = b.dependency(...); // none needed

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

// for an executable:
const exe = b.addExecutable(.{
    .name = "bonsai-vk",
    .root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    }),
});
exe.root_module.addImport("vk", vk);
// optional: bake the loader dir into RUNPATH so LD_LIBRARY_PATH is not needed
exe.root_module.addRPath(.{ .cwd_relative = loader_dir });
```

`addRPath` (capital P) is the Zig 0.17 spelling. Verified: the resulting binary
runs with `LD_LIBRARY_PATH` unset and `ldd` resolves
`libvulkan.so.1 => /opt/vulkan-sdk/lib/VulkanLoader/lib/libvulkan.so.1`.

For `libbonsai.so`, do the same on the shared-library root module
(`root_module.addImport("vk", vk)`); `linkSystemLibrary("vulkan")` on the `vk`
module propagates to whatever links it.

Equivalent one-shot command used by the test:

```
zig build-exe -O ReleaseFast --dep vk -Mroot=tools/vktest.zig \
    -O ReleaseFast -Mvk=src/vk.zig -lc \
    -L "$VULKAN_SDK/lib/VulkanLoader/lib" -lvulkan -femit-bin=tools/vktest
```

Runtime needs the loader on the library path, e.g.
`LD_LIBRARY_PATH=/opt/vulkan-sdk/lib/VulkanLoader/lib` (or an rpath); the
benchmark image already exports it.

## Device

```
device: AMD Radeon RX 9070 (RADV GFX1201)
apiVersion: 1.4.335  driver 0x6800008  vendor 0x1002  device 0x7550
compute queue family: 0
maxComputeWorkGroupInvocations: 1024
maxStorageBufferRange: 4095 MiB
maxPushConstantsSize: 256
```

The driver prints `radv is not a conformant Vulkan implementation, testing use
only.` on stderr; expected, not an error.

## Enabled features (all queried true and enabled)

Core (VkPhysicalDeviceVulkan11/12/13/14Features chain):
`shaderFloat16`, `shaderInt8`, `storageBuffer16BitAccess`,
`storageBuffer8BitAccess`, `shaderIntegerDotProduct`, `timelineSemaphore`,
`bufferDeviceAddress`, `descriptorIndexing`, `scalarBlockLayout`,
`vulkanMemoryModel` - **all true on this device**. The companions that came
back supported are enabled too
(`uniformAndStorageBuffer16BitAccess`, `uniformAndStorageBuffer8BitAccess`,
`shaderFloatControls2`, `shaderExpectAssume`, `shaderSubgroupRotate`,
`maintenance5`, `maintenance6`).

Extension feature structs (chained only when the extension is advertised):
`VK_KHR_cooperative_matrix` (cooperativeMatrix + robustBufferAccess),
`VK_KHR_maintenance7/8/9`, `VK_KHR_shader_maximal_reconvergence`,
`VK_KHR_shader_quad_control`, `VK_KHR_shader_untyped_pointers`,
`VK_KHR_workgroup_memory_explicit_layout` (all 4 flags),
`VK_KHR_robustness2` (robustBufferAccess2, robustImageAccess2, nullDescriptor).
Because `robustBufferAccess2` requires it, core `robustBufferAccess` is
auto-enabled (VUID-VkPhysicalDeviceRobustness2FeaturesKHR-robustBufferAccess2-04000).

Extensions enabled on the device (14, every one advertised):
`VK_KHR_cooperative_matrix`, `VK_KHR_maintenance5`, `VK_KHR_maintenance7`,
`VK_KHR_maintenance8`, `VK_KHR_maintenance9`,
`VK_KHR_shader_float_controls2`, `VK_KHR_shader_expect_assume`,
`VK_KHR_shader_subgroup_rotate`, `VK_KHR_shader_maximal_reconvergence`,
`VK_KHR_shader_quad_control`, `VK_KHR_shader_untyped_pointers`,
`VK_KHR_shader_subgroup_uniform_control_flow`,
`VK_KHR_workgroup_memory_explicit_layout`, `VK_KHR_robustness2`.

## Test output (ReleaseFast, exact)

```
WARNING: radv is not a conformant Vulkan implementation, testing use only.
vk init (instance+device+queue+pool): 17.33 ms
device: AMD Radeon RX 9070 (RADV GFX1201)
apiVersion: 1.4.335 driver 0x6800008 vendor 0x1002 device 0x7550
compute queue family: 0, max workgroup invocations: 1024, max storage buffer range: 4095 MiB, max push constants: 256
caps: f16=true i8=true sb16=true sb8=true dot=true timeline=true bda=true descidx=true scalar=true vmm=true
caps: coopmat=true m5=true m7=true m8=true m9=true fc2=true expect=true rotate=true maxreconv=true quad=true untyped=true wgmem=true rob2=true
enabled extensions (14): VK_KHR_cooperative_matrix VK_KHR_maintenance5 VK_KHR_maintenance7 VK_KHR_maintenance8 VK_KHR_maintenance9 VK_KHR_shader_float_controls2 VK_KHR_shader_expect_assume VK_KHR_shader_subgroup_rotate VK_KHR_shader_maximal_reconvergence VK_KHR_shader_quad_control VK_KHR_shader_untyped_pointers VK_KHR_shader_subgroup_uniform_control_flow VK_KHR_workgroup_memory_explicit_layout VK_KHR_robustness2
uploadDeviceLocal: 512 MiB in 107.1 ms (5.0 GB/s), device_address=0xffff800100200000
download 1 MiB slice in 4.2 ms
512 MiB round trip verified
compute dispatch verified: 4096 elements, a[i]+3*b[i]
push-constant dispatch verified: 256 elements, a[i]+7
PASS
```

Timings vary by a few percent run to run (observed: init 10-22 ms, 512 MiB
upload 107-111 ms, 1 MiB download 4.2-9.5 ms).

Verification details:

* 512 MiB uploaded, then the first 1 MiB, bytes at 1 MiB-8 and the last 8
  bytes were downloaded and compared (`0x0123456789ABCDEF`,
  `0xFEEDFACECAFEBEEF`, `0xDEADBEEF12345678`).
* Compute results checked element-wise on the host after download.
* With `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` the run reports
  **0** validation errors/warnings and still prints `PASS`.

## Measured numbers

| operation | value |
|---|---|
| `Vk.init` (instance + device + queue + pool) | 12-17 ms |
| 512 MiB `uploadDeviceLocal` | ~107 ms (5.0 GB/s, includes host staging write + PCIe copy) |
| 1 MiB `download` | 4.5-6 ms |
| 4096-element dispatch (2 storage buffers) | verified |
| 256-element push-constant dispatch | verified |
| validation layer messages | 0 |

## Notes / limitations

* One compute queue, one command buffer, one fence; submissions are serialized
  and waited on immediately (`submit_count` increments per submission). A
  future async path can add semaphores/fences without touching init.
* `deinit` destroys the fence, command pool, device and instance only; the
  caller owns buffers and pipelines.
* Device selection is "first physical device with a compute queue family", as
  specified; on this machine that is the RX 9070.
* Feature/property structs are version-gated on the physical device's
  `apiVersion`, so a <1.4 driver would not see 1.4 structs in the chain.
* `VK_KHR_cooperative_matrix` is enabled but no coopmat shader is used by the
  smoke test; the feature flag is what the engine needs before compiling
  `GL_KHR_cooperative_matrix` SPIR-V.
* The bindings file is machine-generated and contains commented translate-c
  warnings (`vk_video` bitfields); harmless.
