You are working in an offline sandbox. Your current directory is your workspace.

Your task: implement `bonsai-vk`, a Vulkan inference engine for the GGUF model in
`data/Ternary-Bonsai-2-4B-PTQ1_0.gguf`, in Zig 0.17.0 with Vulkan SDK 1.4.363.0.
Target Vulkan 1.4 and the compute-relevant Vulkan Roadmap 2026 feature set
described in `DEVICE.md`.

Expose the C ABI in `ABI.md` as `zig-out/lib/libbonsai.so`. `TASK.md` describes
the deliverable and constraints; `ACCEPTANCE.md` describes correctness,
performance and VRAM acceptance. `MODEL.md` lists the available model
artifacts and the oracle; the architecture is intentionally not documented.
`TOKENIZER.md` describes the tokenizer format. The specifications are in
`docs/`.

`ref/` contains symbol-stripped llama.cpp binaries with CPU and Vulkan backends,
and `ref/PROTOCOL.md` documents a reference runner. Use them as the correctness
oracle. `data/*.safetensors` contains the dequantized weights for development.

You have a wall-clock budget: run `time-left` to see the remaining time and
read the time-budget section of `TASK.md`. Only the workspace at the deadline is
graded, so save your work continuously.

## Bash in this sandbox

Claude Code's Bash command classifier sometimes denies safe commands. Work
within it:

1. Keep commands simple: split `&&`, `;`, pipelines, loops and heredocs into
   separate Bash calls.
2. Do not set environment variables in Bash commands, especially to paths
   outside the workspace; use the environment already provided.
3. If a safe command is still denied, write the commands into a script in the
   workspace and run the script (for example `bash tools/check.sh`).
4. Tool results over ~30 KB are truncated and the full output is stored outside
   the workspace, where you cannot read it. Redirect large output to a file in
   the workspace (for example `cmd > out.txt`) and read that file.

The podman container is the security boundary. These are usability workarounds,
not a way to reach the network or files outside the workspace.

Work autonomously until the engine is complete. Leave `NOTES.md` describing your
design, known limitations and test results.
