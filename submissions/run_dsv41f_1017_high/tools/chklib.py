import glob, os
for pat in ["/usr/lib/x86_64-linux-gnu/libvulkan.so*","/opt/vulkan-sdk/lib/VulkanLoader/lib/*","/usr/lib/*/libvulkan.so*"]:
    for f in glob.glob(pat):
        print(f, os.path.exists(f))
print("VULKAN_SDK env:", os.environ.get("VULKAN_SDK"))
