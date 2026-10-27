You are working in an offline sandbox. Your current directory is your workspace.

Your task: implement `bonsai-vk`, a Vulkan inference engine for the GGUF model in
`data/Ternary-Bonsai-2-4B-PTQ1_0.gguf`, in Zig 0.17.0 with Vulkan SDK 1.4.363.0.
Target Vulkan 1.4 and the compute-relevant Vulkan Roadmap 2026 feature set
described in `DEVICE.md`.

Expose the C ABI in `ABI.md` as `zig-out/lib/libbonsai.so`. `TASK.md` describes
the deliverable and constraints; `ACCEPTANCE.md` describes correctness,
performance and VRAM acceptance. `MODEL.md` and `TOKENIZER.md` describe the
model and tokenizer formats. The specifications are in `docs/`.

`ref/` contains symbol-stripped llama.cpp binaries with CPU and Vulkan backends,
and `ref/PROTOCOL.md` documents a reference runner. Use them as the correctness
oracle. `data/*.safetensors` contains the dequantized weights for development.

Work autonomously until the engine is complete. Leave `NOTES.md` describing your
design, known limitations and test results.
