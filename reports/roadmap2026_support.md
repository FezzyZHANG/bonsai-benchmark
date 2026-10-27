# Vulkan Roadmap 2026 support on the benchmark GPU

- GPU: AMD Radeon RX 9070 (RADV GFX1201), 16304 MiB
- Driver: Mesa 26.0.8 (RADV), Vulkan `apiVersion` 1.4.335
- Loader: Vulkan SDK 1.4.363.0
- Profile source: `VP_KHR_roadmap.json` shipped in the SDK (api-version 1.4.328)
- Check: `scripts/check_roadmap2026.py` against `vulkaninfo -j` (231 device extensions)

## Full VP_KHR_roadmap_2026: missing requirements

Three root causes, all outside a headless compute/inference workload:

| missing item | kind | relevance |
|---|---|---|
| `VK_KHR_copy_memory_indirect` | device extension | not needed |
| `VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR.indirectMemoryCopy` | feature | not needed |
| `VK_KHR_present_mode_fifo_latest_ready` | device extension | presentation only |
| `VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR.presentModeFifoLatestReady` | feature | presentation only |
| `VkPhysicalDeviceVulkan14Features.hostImageCopy` | Vulkan 1.4 core feature | not needed |

The requirement that is *not* a gap:

- Line rasterization is an OR-group of six capability sets. The device passes
  line0/line1/line2/line4 (rectangular, Bresenham, smooth, stippled-Bresenham);
  `stippledRectangularLines` and `stippledSmoothLines` are false but are not
  required because the profile accepts any one line capability set.
- `bufferImageGranularity` (device 1, profile 4096) and
  `maxViewportDimensions` (device 32768, profile 8192) satisfy the required
  limit semantics (upper bound / per-component minimum respectively).

## Compute-relevant Roadmap 2026 subset: fully supported

Core: Vulkan 1.4 (`shaderFloat16`, `shaderInt8`, `storageBuffer16/8BitAccess`,
`shaderIntegerDotProduct`, `timelineSemaphore`, `bufferDeviceAddress`, etc.).

Extensions/features relevant to a compute-only LLM engine, all present:

- `VK_KHR_cooperative_matrix` (rev 2; compute stage; `cooperativeMatrix = true`)
- `VK_KHR_shader_float_controls2`, `VK_KHR_shader_expect_assume`
- `VK_KHR_shader_subgroup_rotate`, `VK_KHR_shader_subgroup_uniform_control_flow`
- `VK_KHR_shader_maximal_reconvergence`, `VK_KHR_shader_quad_control`
- `VK_KHR_shader_untyped_pointers` (rev 1)
- `VK_KHR_workgroup_memory_explicit_layout`
- `VK_KHR_maintenance5`, `maintenance7`, `maintenance8`, `maintenance9`
- `VK_KHR_robustness2` (`robustBufferAccess2`, `robustImageAccess2`, `nullDescriptor`)
- `VK_KHR_pipeline_binary`, `VK_KHR_map_memory2`
- `VK_KHR_shader_clock`, `VK_KHR_compute_shader_derivatives`
- 8-bit integer dot-product paths are accelerated (`integerDotProduct8Bit*Accelerated = true`)

## Recommendation for the benchmark

Require the **compute-relevant Roadmap 2026 subset**, not the full profile.
The three missing root causes are presentation/copy/host-image features that a
headless inference engine never uses. Document the exact subset in the agent's
`DEVICE.md`, and optionally generate a custom Vulkan profile JSON
(`VP_BONSAI_COMPUTE_2026`) for the engine to validate at startup.

Target API: Vulkan 1.4 (device 1.4.335 >= profile 1.4.328).
