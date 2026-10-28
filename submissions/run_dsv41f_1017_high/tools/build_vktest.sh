#!/bin/bash
# Build the Vulkan smoke test. Usage: bash tools/build_vktest.sh [debug|release]
set -e
MODE="${1:-debug}"
cd /workspace
if [ "$MODE" = "release" ]; then
  OPT=ReleaseFast
else
  OPT=Debug
fi
zig build-exe \
  -O "$OPT" --dep vk -Mroot=/workspace/tools/vktest.zig \
  -O "$OPT" -Mvk=/workspace/src/vk.zig \
  -lc \
  -L/opt/vulkan-sdk/lib/VulkanLoader/lib -lvulkan \
  -femit-bin=/workspace/tools/vktest
echo "built /workspace/tools/vktest ($OPT)"
