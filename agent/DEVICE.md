# Target GPU and Vulkan feature set

The benchmark runs on one machine:

- GPU: **AMD Radeon RX 9070** (RDNA4, `RADV GFX1201`), 16304 MiB VRAM
- Driver: Mesa 26.0.8 (RADV), device `apiVersion` **1.4.335**
- Loader: Vulkan SDK **1.4.363.0**
- The driver prints `radv is not a conformant Vulkan implementation, testing use only.`
  This is expected; it is the target device.

The engine must target **Vulkan 1.4** and the compute-relevant subset of
**Vulkan Roadmap 2026**. The full profile is not implementable on this GPU; the
three missing items are presentation/copy/host-image features that a headless
inference engine does not need.

## Full VP_KHR_roadmap_2026 gaps (not required)

| missing item | kind |
|---|---|
| `VK_KHR_copy_memory_indirect` + `indirectMemoryCopy` | device extension/feature |
| `VK_KHR_present_mode_fifo_latest_ready` + `presentModeFifoLatestReady` | device extension/feature |
| `hostImageCopy` | Vulkan 1.4 core feature |

The line-rasterization requirement is an OR-group; the device satisfies it via
rectangular/Bresenham/smooth/stippled-Bresenham. The required 2026 limits are
all met.

## Required compute feature set (all supported)

Core Vulkan 1.4 features, including:

- `shaderFloat16`, `shaderInt8`
- `storageBuffer16BitAccess`, `storageBuffer8BitAccess`
- `shaderIntegerDotProduct` (8-bit paths accelerated)
- `timelineSemaphore`, `bufferDeviceAddress`, `descriptorIndexing`
- `scalarBlockLayout`, `vulkanMemoryModel`

Extensions/features:

- `VK_KHR_cooperative_matrix` (revision 2; compute stage;
  `cooperativeMatrix = true`)
- `VK_KHR_maintenance5`, `VK_KHR_maintenance7`, `VK_KHR_maintenance8`,
  `VK_KHR_maintenance9`
- `VK_KHR_shader_float_controls2`
- `VK_KHR_shader_expect_assume`
- `VK_KHR_shader_subgroup_rotate`
- `VK_KHR_shader_subgroup_uniform_control_flow`
- `VK_KHR_shader_maximal_reconvergence`
- `VK_KHR_shader_quad_control`
- `VK_KHR_shader_untyped_pointers`
- `VK_KHR_workgroup_memory_explicit_layout`
- `VK_KHR_robustness2` (`robustBufferAccess2`, `robustImageAccess2`,
  `nullDescriptor`)
- `VK_KHR_pipeline_binary`, `VK_KHR_map_memory2`
- `VK_KHR_shader_clock`, `VK_KHR_compute_shader_derivatives`

The device reports `matrix cores: KHR_coopmat`, `int dot: 1`, `fp16: 1`,
`bf16: 1`. Cooperative matrix is supported in compute shaders; `glslc` from the
SDK compiles `GL_KHR_cooperative_matrix` f16/f32 SPIR-V.

## Querying the device

`vulkaninfo` is on `PATH`. The benchmark image also provides a captured
`docs/device/vulkaninfo.json` (`vulkaninfo -j`), the Roadmap 2026 profile
(`docs/device/VP_KHR_roadmap.json`), and the roadmap appendix
(`docs/device/Roadmap-2026.adoc`).

Suggested checks:

```bash
vulkaninfo --summary
vulkaninfo | grep -i -A20 "VkPhysicalDeviceVulkan14Features"
glslc --version
spirv-val --target-env vulkan1.4 shader.spv
```

## Practical notes

- Use the SDK loader:
  `-I "$VULKAN_SDK/include" -L "$VULKAN_SDK/lib/VulkanLoader/lib" -lvulkan`,
  and run with `LD_LIBRARY_PATH` including that directory.
- The SDK is at `$VULKAN_SDK` (`/opt/vulkan-sdk` in the benchmark image).
- Use `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` if automatic
  ICD discovery is a problem.
- Validation layers are installed; `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
  is useful while developing, but disable them for performance measurements.
- `vkprofiles` can inspect the profile file; the exact compute feature list
  above is the authoritative scope for this benchmark.
