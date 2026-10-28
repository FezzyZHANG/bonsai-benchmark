#!/bin/bash
cd /workspace/src
for s in \
  VkApplicationInfo VkInstanceCreateInfo VkPhysicalDeviceFeatures2 VkPhysicalDeviceFeatures \
  VkPhysicalDeviceVulkan11Features VkPhysicalDeviceVulkan12Features VkPhysicalDeviceVulkan13Features \
  VkDeviceQueueCreateInfo VkDeviceCreateInfo VkExtensionProperties \
  VkQueueFamilyProperties VkPhysicalDeviceMemoryProperties VkMemoryType VkMemoryHeap VkMemoryRequirements \
  VkBufferCreateInfo VkMemoryAllocateInfo VkMemoryAllocateFlagsInfo VkBufferDeviceAddressInfo \
  VkBufferCopy VkCommandPoolCreateInfo VkCommandBufferAllocateInfo VkCommandBufferBeginInfo \
  VkSubmitInfo VkFenceCreateInfo VkShaderModuleCreateInfo \
  VkDescriptorSetLayoutBinding VkDescriptorSetLayoutCreateInfo VkPipelineLayoutCreateInfo \
  VkPipelineShaderStageCreateInfo VkComputePipelineCreateInfo \
  VkDescriptorPoolSize VkDescriptorPoolCreateInfo VkDescriptorSetAllocateInfo \
  VkDescriptorBufferInfo VkWriteDescriptorSet \
  VkPhysicalDeviceProperties VkPhysicalDeviceLimits VkPhysicalDeviceProperties2 \
  VkPhysicalDeviceVulkan12Properties VkPhysicalDeviceVulkan13Properties VkPhysicalDeviceVulkan14Properties \
  VkPhysicalDeviceCooperativeMatrixPropertiesKHR VkPhysicalDeviceCooperativeMatrixFeaturesKHR \
  VkPhysicalDeviceMaintenance7FeaturesKHR VkPhysicalDeviceMaintenance8FeaturesKHR VkPhysicalDeviceMaintenance9FeaturesKHR \
  VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR VkPhysicalDeviceShaderQuadControlFeaturesKHR \
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR \
  VkPhysicalDeviceRobustness2FeaturesKHR \
  VkGetDeviceBufferMemoryRequirementsInfoKHR VkDeviceBufferMemoryRequirements \
  ; do
  awk -v name="struct_${s} = extern struct" '
    $0 ~ ("pub const " name) {print "===== " name; p=1}
    p {print}
    p && /^};/ {p=0}
  ' vk_bindings_raw.zig
done > /workspace/tools/structs.txt
wc -l /workspace/tools/structs.txt
