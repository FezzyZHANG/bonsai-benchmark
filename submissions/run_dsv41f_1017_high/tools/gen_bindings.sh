#!/bin/bash
zig translate-c -I /opt/vulkan-sdk/include /opt/vulkan-sdk/include/vulkan/vulkan.h > /workspace/src/vk_bindings_raw.zig 2> /workspace/tools/gen_bindings.err
echo "exit=$?"
ls -la /workspace/src/vk_bindings_raw.zig
wc -l /workspace/src/vk_bindings_raw.zig
