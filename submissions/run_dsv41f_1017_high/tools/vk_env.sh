#!/bin/bash
echo "=== loader libs ==="
ls -la /opt/vulkan-sdk/lib/VulkanLoader/lib 2>&1
echo "=== sdk lib ==="
ls -la /opt/vulkan-sdk/lib 2>&1
echo "=== find libvulkan ==="
find /opt/vulkan-sdk -name 'libvulkan*' 2>/dev/null
echo "=== include ==="
ls /opt/vulkan-sdk/include/vulkan 2>&1
echo "=== ldconfig ==="
ldconfig -p 2>/dev/null | grep vulkan
echo "=== glslc ==="
glslc --version 2>&1
echo "=== cmds ==="
command -v glslc spirv-val vulkaninfo zig
