#!/bin/bash
echo "=== ldd ==="
ldd /workspace/tools/buildtest/out/bin/vktest_build
echo "=== readelf runpath ==="
readelf -d /workspace/tools/buildtest/out/bin/vktest_build | grep -i "rpath\|runpath\|NEEDED"
echo "=== run without LD_LIBRARY_PATH ==="
env -u LD_LIBRARY_PATH /workspace/tools/buildtest/out/bin/vktest_build > /workspace/tools/rpath_test.log 2>&1
echo "exit=$?"
tail -6 /workspace/tools/rpath_test.log
