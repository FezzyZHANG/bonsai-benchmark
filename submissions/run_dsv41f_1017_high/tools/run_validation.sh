#!/bin/bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
/workspace/tools/vktest > /workspace/tools/validation.log 2>&1
echo "exit=$?"
echo "--- validation errors/warnings ---"
grep -c "VUID\|Validation Error\|Validation Warning" /workspace/tools/validation.log
grep -m 40 "VUID\|Validation Error\|Validation Warning" /workspace/tools/validation.log
echo "--- tail ---"
tail -20 /workspace/tools/validation.log
